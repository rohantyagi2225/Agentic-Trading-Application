from typing import Dict, List, Optional
from collections import defaultdict

from sqlalchemy import Date, func, select, cast
from sqlalchemy.orm import Session

from backend.db.models.portfolio_position import PortfolioPosition
from backend.db.models.trade import Trade
from backend.cache.cache_service import CacheService
from backend.analytics import portfolio_analytics


class PortfolioService:
    """
    Service for computing high-level portfolio metrics from positions.
    Optional Redis cache for summary.
    """

    def __init__(self, db: Session, cache: Optional[CacheService] = None) -> None:
        self._db = db
        self._cache = cache

    def get_portfolio_summary(self) -> Dict[str, float]:

        if self._cache and self._cache.is_available:
            cached = self._cache.get_portfolio_summary()
            if cached is not None:
                return cached

        total_positions_stmt = select(func.count(PortfolioPosition.id))
        total_positions = int(self._db.execute(total_positions_stmt).scalar_one() or 0)

        exposure_stmt = select(
            func.coalesce(func.sum(func.abs(PortfolioPosition.quantity)), 0.0)
        )

        exposure = float(self._db.execute(exposure_stmt).scalar_one() or 0.0)

        out = {
            "total_positions": total_positions,
            "total_exposure": exposure,
            "portfolio_value": exposure,
        }
        if self._cache and self._cache.is_available:
            self._cache.set_portfolio_summary(out)
        return out

    def get_returns_from_trades(
        self,
        initial_equity: float = 100_000.0,
    ) -> List[float]:
        """
        Build a list of daily returns from trade history.
        Daily PnL = (SELL proceeds) - (BUY cost) per day; return = daily_pnl / equity_start.
        """
        rows = (
            self._db.query(
                cast(Trade.timestamp, Date).label("day"),
                Trade.action,
                (Trade.quantity * Trade.price).label("value"),
            )
            .order_by(Trade.timestamp)
            .all()
        )
        if not rows:
            return []
        daily_pnl: Dict[str, float] = defaultdict(float)
        for r in rows:
            day_key = r.day.isoformat() if hasattr(r.day, "isoformat") else str(r.day)
            if (r.action or "").upper() == "SELL":
                daily_pnl[day_key] += float(r.value)
            else:
                daily_pnl[day_key] -= float(r.value)
        ordered_days = sorted(daily_pnl.keys())
        cum_pnl = 0.0
        returns: List[float] = []
        for d in ordered_days:
            pnl = daily_pnl[d]
            equity_start = initial_equity + cum_pnl
            cum_pnl += pnl
            if equity_start > 0:
                returns.append(pnl / equity_start)
            else:
                returns.append(0.0)
        return returns

    def get_portfolio_summary_with_analytics(
        self,
        initial_equity: float = 100_000.0,
    ) -> Dict[str, float]:
        """
        Portfolio summary plus analytics (sharpe, sortino, volatility, max_drawdown)
        when enough return history exists from trades.
        """
        summary = self.get_portfolio_summary()
        returns = self.get_returns_from_trades(initial_equity=initial_equity)
        if len(returns) >= 2:
            analytics = portfolio_analytics(returns, periods_per_year=252)
            summary["sharpe"] = analytics["sharpe"]
            summary["sortino"] = analytics["sortino"]
            summary["volatility"] = analytics["volatility"]
            summary["max_drawdown"] = analytics["max_drawdown"]
        return summary

    def get_total_exposure(self) -> float:

        stmt = select(
            func.coalesce(func.sum(func.abs(PortfolioPosition.quantity)), 0.0)
        )

        return float(self._db.execute(stmt).scalar_one() or 0.0)

    def get_portfolio_value(self) -> float:

        stmt = select(
            func.coalesce(func.sum(func.abs(PortfolioPosition.quantity)), 0.0)
        )

        return float(self._db.execute(stmt).scalar_one() or 0.0)    