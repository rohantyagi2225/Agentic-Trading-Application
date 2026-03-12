"""
Aggregate analytics from a return series (and optional benchmark).
"""

from typing import Dict, List, Optional

from backend.analytics.sharpe import sharpe_ratio
from backend.analytics.sortino import sortino_ratio
from backend.analytics.volatility import annualized_volatility
from backend.analytics.max_drawdown import max_drawdown
from backend.analytics.alpha_beta import alpha_beta


def portfolio_analytics(
    returns: List[float],
    benchmark_returns: Optional[List[float]] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute full analytics for a return series.

    :param returns: Period returns (e.g. daily).
    :param benchmark_returns: Optional benchmark (same length) for alpha/beta.
    :param risk_free_rate: Period risk-free rate.
    :param periods_per_year: 252 for daily.
    :return: Dict with sharpe, sortino, volatility, max_drawdown; alpha and beta if benchmark given.
    """
    out: Dict[str, float] = {
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "volatility": annualized_volatility(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
    }
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        alpha, beta = alpha_beta(
            returns, benchmark_returns, risk_free_rate, periods_per_year
        )
        out["alpha"] = alpha
        out["beta"] = beta
    return out
