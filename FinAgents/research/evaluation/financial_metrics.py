"""Extended Financial Metrics for Research Evaluation.

This module provides comprehensive financial metrics calculation including
advanced risk-adjusted returns, period breakdowns, and regime-specific analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FinancialMetricsReport:
    """Comprehensive financial metrics report.

    Attributes
    ----------
    sharpe_ratio : float
        Annualized Sharpe ratio (risk-adjusted return).
    sortino_ratio : float
        Annualized Sortino ratio (downside risk-adjusted return).
    calmar_ratio : float
        Annualized return divided by maximum drawdown.
    information_ratio : float
        Risk-adjusted excess return over benchmark.
    omega_ratio : float
        Probability-weighted ratio of gains to losses.
    total_return : float
        Total return over the period.
    annualized_return : float
        Annualized return rate.
    max_drawdown : float
        Maximum peak-to-trough decline.
    volatility : float
        Annualized volatility.
    win_rate : float
        Percentage of profitable trades.
    profit_factor : float
        Gross profits divided by gross losses.
    period_breakdown : dict
        Returns broken down by month/quarter/year.
    regime_performance : dict
        Performance metrics per market regime.
    cost_adjusted_return : float
        Return after accounting for transaction costs.
    raw_return : float
        Return before transaction costs.
    num_trades : int
        Total number of trades executed.
    avg_holding_period_days : float
        Average holding period in days.
    """

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    omega_ratio: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    period_breakdown: Dict[str, List[float]] = field(default_factory=dict)
    regime_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cost_adjusted_return: float = 0.0
    raw_return: float = 0.0
    num_trades: int = 0
    avg_holding_period_days: float = 0.0


class FinancialMetricsCalculator:
    """Calculate comprehensive financial metrics.

    Provides methods for computing standard and advanced financial metrics
    including risk-adjusted returns, drawdowns, and regime-specific analysis.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate for Sharpe ratio calculation (default 0.04).
    """

    def __init__(self, risk_free_rate: float = 0.04) -> None:
        """Initialize the financial metrics calculator.

        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate (default 4%).
        """
        self.risk_free_rate = risk_free_rate

    def compute_all_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        trades: Optional[List[Dict[str, Any]]] = None,
        transaction_costs: float = 0.001,
        regime_labels: Optional[List[str]] = None,
        dates: Optional[List[datetime]] = None,
    ) -> FinancialMetricsReport:
        """Compute comprehensive financial metrics.

        Parameters
        ----------
        returns : np.ndarray
            Array of period returns.
        benchmark_returns : np.ndarray, optional
            Array of benchmark returns for comparison metrics.
        trades : list of dict, optional
            List of trade records with 'pnl', 'entry_time', 'exit_time'.
        transaction_costs : float
            Average transaction cost per trade (default 0.1%).
        regime_labels : list of str, optional
            Market regime labels for each return period.
        dates : list of datetime, optional
            Dates corresponding to each return period.

        Returns
        -------
        FinancialMetricsReport
            Comprehensive metrics report.
        """
        if len(returns) == 0:
            return FinancialMetricsReport()

        returns = np.asarray(returns)
        
        # Basic return metrics
        total_return = float(np.sum(returns))
        annualized_return = self._annualize_return(returns)
        volatility = float(np.std(returns) * np.sqrt(252))

        # Risk-adjusted metrics
        sharpe_ratio = self._sharpe_ratio(returns)
        sortino_ratio = self._sortino_ratio(returns)
        calmar_ratio = self.calmar_ratio(returns)
        
        # Benchmark comparison metrics
        information_ratio = 0.0
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            information_ratio = self.information_ratio(returns, np.asarray(benchmark_returns))

        # Omega ratio
        omega_ratio = self.omega_ratio(returns)

        # Maximum drawdown
        max_drawdown = self._max_drawdown(returns)

        # Trade-based metrics
        win_rate = 0.0
        profit_factor = 0.0
        num_trades = 0
        avg_holding_period_days = 0.0
        
        if trades:
            num_trades = len(trades)
            pnls = [t.get("pnl", 0) for t in trades]
            winning = [p for p in pnls if p > 0]
            losing = [p for p in pnls if p < 0]
            
            if pnls:
                win_rate = len(winning) / len(pnls) if pnls else 0.0
            
            total_gains = sum(winning)
            total_losses = abs(sum(losing))
            profit_factor = total_gains / total_losses if total_losses > 0 else float('inf') if total_gains > 0 else 0.0
            
            # Calculate average holding period
            holding_periods = []
            for trade in trades:
                if "entry_time" in trade and "exit_time" in trade:
                    try:
                        entry = trade["entry_time"]
                        exit_time = trade["exit_time"]
                        if isinstance(entry, str):
                            entry = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                        if isinstance(exit_time, str):
                            exit_time = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
                        holding_periods.append((exit_time - entry).days)
                    except (ValueError, TypeError):
                        pass
            
            avg_holding_period_days = np.mean(holding_periods) if holding_periods else 0.0

        # Cost-adjusted returns
        cost_adjusted_returns = self.cost_adjusted_returns(
            returns, num_trades, transaction_costs
        )
        cost_adjusted_return = float(np.sum(cost_adjusted_returns))

        # Period breakdown
        period_breakdown = self.period_breakdown(returns, dates)

        # Regime performance
        regime_performance = {}
        if regime_labels and len(regime_labels) == len(returns):
            regime_performance = self.regime_performance(returns, regime_labels)

        return FinancialMetricsReport(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            omega_ratio=omega_ratio,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
            period_breakdown=period_breakdown,
            regime_performance=regime_performance,
            cost_adjusted_return=cost_adjusted_return,
            raw_return=total_return,
            num_trades=num_trades,
            avg_holding_period_days=avg_holding_period_days,
        )

    def calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown).

        Parameters
        ----------
        returns : np.ndarray
            Array of period returns.

        Returns
        -------
        float
            Calmar ratio. Returns inf if max drawdown is zero.
        """
        if len(returns) == 0:
            return 0.0

        max_dd = self._max_drawdown(returns)
        if max_dd == 0:
            return float('inf') if np.sum(returns) > 0 else 0.0

        annualized_return = self._annualize_return(returns)
        return float(annualized_return / max_dd)

    def information_ratio(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        """Calculate information ratio (risk-adjusted excess return).

        IR = mean(active_return) / std(active_return) * sqrt(252)

        Parameters
        ----------
        returns : np.ndarray
            Strategy returns.
        benchmark_returns : np.ndarray
            Benchmark returns.

        Returns
        -------
        float
            Annualized information ratio.
        """
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        returns = np.asarray(returns)
        benchmark_returns = np.asarray(benchmark_returns)

        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        if min_len == 0:
            return 0.0

        active_returns = returns - benchmark_returns
        
        mean_active = np.mean(active_returns)
        std_active = np.std(active_returns)

        if std_active == 0:
            return 0.0

        return float(mean_active / std_active * np.sqrt(252))

    def omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega ratio.

        Omega = sum(max(0, r - threshold)) / sum(max(0, threshold - r))

        Parameters
        ----------
        returns : np.ndarray
            Array of period returns.
        threshold : float
            Return threshold (default 0.0).

        Returns
        -------
        float
            Omega ratio. Returns inf if denominator is zero.
        """
        if len(returns) == 0:
            return 0.0

        returns = np.asarray(returns)
        
        gains = np.sum(np.maximum(0, returns - threshold))
        losses = np.sum(np.maximum(0, threshold - returns))

        if losses == 0:
            return float('inf') if gains > 0 else 0.0

        return float(gains / losses)

    def period_breakdown(
        self,
        returns: np.ndarray,
        dates: Optional[List[datetime]] = None
    ) -> Dict[str, List[float]]:
        """Break down returns by time period.

        If dates are provided, groups by actual month/quarter/year.
        Otherwise, splits returns into approximate periods based on
        standard trading calendar assumptions.

        Parameters
        ----------
        returns : np.ndarray
            Array of period returns.
        dates : list of datetime, optional
            Dates corresponding to each return.

        Returns
        -------
        dict
            {"monthly": [...], "quarterly": [...], "annual": [...]}
        """
        if len(returns) == 0:
            return {"monthly": [], "quarterly": [], "annual": []}

        returns = np.asarray(returns)

        if dates is not None and len(dates) == len(returns):
            return self._period_breakdown_by_dates(returns, dates)
        else:
            return self._period_breakdown_by_chunks(returns)

    def _period_breakdown_by_dates(
        self, returns: np.ndarray, dates: List[datetime]
    ) -> Dict[str, List[float]]:
        """Group returns by actual dates."""
        monthly: Dict[str, List[float]] = {}
        quarterly: Dict[str, List[float]] = {}
        annual: Dict[str, List[float]] = {}

        for i, ret in enumerate(returns):
            if i >= len(dates):
                break
            date = dates[i]
            if isinstance(date, str):
                try:
                    date = datetime.fromisoformat(date.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue

            # Monthly key: YYYY-MM
            month_key = f"{date.year}-{date.month:02d}"
            monthly.setdefault(month_key, []).append(ret)

            # Quarterly key: YYYY-QN
            quarter = (date.month - 1) // 3 + 1
            quarter_key = f"{date.year}-Q{quarter}"
            quarterly.setdefault(quarter_key, []).append(ret)

            # Annual key: YYYY
            year_key = str(date.year)
            annual.setdefault(year_key, []).append(ret)

        # Aggregate returns
        monthly_returns = [float(np.sum(v)) for v in monthly.values()]
        quarterly_returns = [float(np.sum(v)) for v in quarterly.values()]
        annual_returns = [float(np.sum(v)) for v in annual.values()]

        return {
            "monthly": monthly_returns,
            "quarterly": quarterly_returns,
            "annual": annual_returns,
        }

    def _period_breakdown_by_chunks(
        self, returns: np.ndarray
    ) -> Dict[str, List[float]]:
        """Split returns into approximate periods."""
        n = len(returns)
        
        # Monthly (approximately 21 trading days)
        monthly = []
        for i in range(0, n, 21):
            chunk = returns[i:i + 21]
            monthly.append(float(np.sum(chunk)))
        
        # Quarterly (approximately 63 trading days)
        quarterly = []
        for i in range(0, n, 63):
            chunk = returns[i:i + 63]
            quarterly.append(float(np.sum(chunk)))
        
        # Annual (approximately 252 trading days)
        annual = []
        for i in range(0, n, 252):
            chunk = returns[i:i + 252]
            annual.append(float(np.sum(chunk)))

        return {
            "monthly": monthly,
            "quarterly": quarterly,
            "annual": annual,
        }

    def regime_performance(
        self, returns: np.ndarray, regime_labels: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics by market regime.

        Parameters
        ----------
        returns : np.ndarray
            Array of period returns.
        regime_labels : list of str
            Regime label for each return period.

        Returns
        -------
        dict
            {regime: {return, sharpe, volatility, num_periods, win_rate}}
        """
        if len(returns) == 0 or len(regime_labels) == 0:
            return {}

        returns = np.asarray(returns)
        min_len = min(len(returns), len(regime_labels))
        returns = returns[:min_len]
        regime_labels = regime_labels[:min_len]

        # Group returns by regime
        regime_returns: Dict[str, List[float]] = {}
        for i, label in enumerate(regime_labels):
            regime_returns.setdefault(label, []).append(returns[i])

        # Calculate metrics per regime
        result = {}
        for regime, rets in regime_returns.items():
            rets_arr = np.array(rets)
            n = len(rets_arr)
            
            if n == 0:
                continue

            regime_total_return = float(np.sum(rets_arr))
            regime_sharpe = self._sharpe_ratio(rets_arr)
            regime_volatility = float(np.std(rets_arr) * np.sqrt(252)) if n > 1 else 0.0
            regime_win_rate = float(np.sum(rets_arr > 0) / n) if n > 0 else 0.0

            result[regime] = {
                "return": regime_total_return,
                "sharpe": regime_sharpe,
                "volatility": regime_volatility,
                "num_periods": n,
                "win_rate": regime_win_rate,
            }

        return result

    def cost_adjusted_returns(
        self,
        returns: np.ndarray,
        num_trades: int,
        avg_cost_per_trade: float = 0.001,
    ) -> np.ndarray:
        """Adjust returns for transaction costs.

        Distributes transaction costs proportionally across all return periods.

        Parameters
        ----------
        returns : np.ndarray
            Array of period returns.
        num_trades : int
            Number of trades executed.
        avg_cost_per_trade : float
            Average cost per trade as fraction of trade value.

        Returns
        -------
        np.ndarray
            Returns adjusted for transaction costs.
        """
        if len(returns) == 0 or num_trades == 0:
            return np.array(returns) if len(returns) > 0 else np.array([])

        returns = np.asarray(returns)
        
        # Total cost as fraction of portfolio
        total_cost = num_trades * avg_cost_per_trade
        
        # Distribute cost proportionally across periods
        cost_per_period = total_cost / len(returns)
        
        return returns - cost_per_period

    def _sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        returns = np.asarray(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Daily risk-free rate
        daily_rf = self.risk_free_rate / 252
        excess_return = mean_return - daily_rf

        return float(excess_return / std_return * np.sqrt(252))

    def _sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        returns = np.asarray(returns)
        mean_return = np.mean(returns)

        # Downside deviation
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0

        downside_std = np.std(negative_returns)

        if downside_std == 0:
            return 0.0

        daily_rf = self.risk_free_rate / 252
        excess_return = mean_return - daily_rf

        return float(excess_return / downside_std * np.sqrt(252))

    def _max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        returns = np.asarray(returns)
        
        # Cumulative returns (equity curve)
        cumulative = np.cumprod(1 + returns)
        
        # Running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Drawdowns
        drawdowns = (running_max - cumulative) / running_max
        
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _annualize_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0

        returns = np.asarray(returns)
        total_return = np.prod(1 + returns) - 1
        n_periods = len(returns)

        if n_periods == 0:
            return 0.0

        # Annualize assuming daily returns
        return float((1 + total_return) ** (252 / n_periods) - 1)
