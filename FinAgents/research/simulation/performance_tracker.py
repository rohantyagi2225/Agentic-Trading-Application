"""Real-Time Performance Metrics Tracking.

This module provides comprehensive performance tracking including
risk-adjusted returns, drawdowns, agent attribution, and benchmark comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AgentAttribution:
    """Attribution metrics for a single agent.

    Attributes
    ----------
    agent_id : str
        Agent identifier.
    total_pnl : float
        Total profit/loss attributed to this agent.
    num_trades : int
        Number of trades made by this agent.
    win_rate : float
        Percentage of profitable trades.
    avg_pnl_per_trade : float
        Average PnL per trade.
    contribution_pct : float
        Percentage of total PnL attributable to this agent.
    """

    agent_id: str
    total_pnl: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    contribution_pct: float = 0.0


@dataclass
class BenchmarkComparison:
    """Comparison metrics against a benchmark.

    Attributes
    ----------
    strategy_return : float
        Total return of the strategy.
    benchmark_return : float
        Total return of the benchmark.
    alpha : float
        Excess return over benchmark.
    beta : float
        Sensitivity to benchmark movements.
    information_ratio : float
        Risk-adjusted excess return.
    tracking_error : float
        Standard deviation of excess returns.
    """

    strategy_return: float = 0.0
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0


class PerformanceTracker:
    """Tracks and computes real-time performance metrics.

    Maintains portfolio values, returns, trades, and agent attributions
    to compute comprehensive performance statistics.

    Attributes
    ----------
    initial_capital : float
        Starting portfolio value.
    portfolio_values : list
        Portfolio values over time.
    returns : list
        Period returns.
    trades : list
        Record of all trades.
    """

    def __init__(
        self, initial_capital: float, benchmark_type: str = "buy_and_hold"
    ) -> None:
        """Initialize the performance tracker.

        Parameters
        ----------
        initial_capital : float
            Starting portfolio value.
        benchmark_type : str
            Type of benchmark ('buy_and_hold' supported).
        """
        self.initial_capital = initial_capital
        self.benchmark_type = benchmark_type

        # Time series data
        self.portfolio_values: List[float] = [initial_capital]
        self.benchmark_values: List[float] = [initial_capital]
        self.returns: List[float] = []
        self.benchmark_returns: List[float] = []

        # Trade tracking
        self.trades: List[Dict[str, Any]] = []

        # Agent attribution
        self._agent_pnl: Dict[str, List[float]] = {}
        self._agent_trades: Dict[str, int] = {}

        # Position tracking for trade PnL
        self._positions: Dict[str, Dict[str, Any]] = {}

    def record_step(
        self,
        portfolio_value: float,
        benchmark_value: Optional[float] = None,
        trades: Optional[List[Dict[str, Any]]] = None,
        agent_attributions: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a single step's performance data.

        Parameters
        ----------
        portfolio_value : float
            Current portfolio value.
        benchmark_value : float, optional
            Current benchmark value.
        trades : list of dict, optional
            Trades executed this step.
        agent_attributions : dict, optional
            Per-agent PnL contributions this step.
        """
        # Compute returns
        prev_value = self.portfolio_values[-1]
        period_return = (portfolio_value - prev_value) / prev_value if prev_value != 0 else 0.0

        self.portfolio_values.append(portfolio_value)
        self.returns.append(period_return)

        # Record benchmark
        if benchmark_value is not None:
            prev_benchmark = self.benchmark_values[-1]
            benchmark_return = (
                (benchmark_value - prev_benchmark) / prev_benchmark
                if prev_benchmark != 0
                else 0.0
            )
            self.benchmark_values.append(benchmark_value)
            self.benchmark_returns.append(benchmark_return)
        else:
            self.benchmark_values.append(self.benchmark_values[-1])
            self.benchmark_returns.append(0.0)

        # Record trades
        if trades:
            for trade in trades:
                self.trades.append(trade)

                # Track per-agent performance
                agent_id = trade.get("agent_id", "unknown")
                pnl = trade.get("pnl", 0.0)

                if agent_id not in self._agent_pnl:
                    self._agent_pnl[agent_id] = []
                    self._agent_trades[agent_id] = 0

                self._agent_pnl[agent_id].append(pnl)
                self._agent_trades[agent_id] += 1

        # Record agent attributions
        if agent_attributions:
            for agent_id, pnl in agent_attributions.items():
                if agent_id not in self._agent_pnl:
                    self._agent_pnl[agent_id] = []
                self._agent_pnl[agent_id].append(pnl)

    def get_sharpe_ratio(self, window: Optional[int] = None) -> float:
        """Calculate Sharpe ratio.

        Parameters
        ----------
        window : int, optional
            Rolling window size. If None, use full history.

        Returns
        -------
        float
            Annualized Sharpe ratio.
        """
        if not self.returns:
            return 0.0

        returns = np.array(self.returns)
        if window is not None and len(returns) >= window:
            returns = returns[-window:]

        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualize (252 trading days)
        return float(mean_return / std_return * np.sqrt(252))

    def get_max_drawdown(self) -> Tuple[float, int, int]:
        """Calculate maximum drawdown.

        Returns
        -------
        tuple
            (max_drawdown_pct, peak_step, trough_step)
        """
        if len(self.portfolio_values) < 2:
            return 0.0, 0, 0

        values = np.array(self.portfolio_values)
        peak = values[0]
        peak_idx = 0
        max_dd = 0.0
        max_dd_peak_idx = 0
        max_dd_trough_idx = 0

        for i, value in enumerate(values):
            if value > peak:
                peak = value
                peak_idx = i

            drawdown = (peak - value) / peak if peak != 0 else 0.0

            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_peak_idx = peak_idx
                max_dd_trough_idx = i

        return float(max_dd), max_dd_peak_idx, max_dd_trough_idx

    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak.

        Returns
        -------
        float
            Current drawdown as a percentage.
        """
        if len(self.portfolio_values) < 2:
            return 0.0

        peak = max(self.portfolio_values)
        current = self.portfolio_values[-1]

        if peak == 0:
            return 0.0

        return float((peak - current) / peak)

    def get_total_return(self) -> float:
        """Calculate total return.

        Returns
        -------
        float
            Total return as a percentage.
        """
        if not self.portfolio_values:
            return 0.0

        final = self.portfolio_values[-1]
        return float((final - self.initial_capital) / self.initial_capital)

    def get_volatility(self, annualized: bool = True) -> float:
        """Calculate volatility.

        Parameters
        ----------
        annualized : bool
            Whether to annualize the volatility.

        Returns
        -------
        float
            Volatility.
        """
        if len(self.returns) < 2:
            return 0.0

        vol = np.std(self.returns)

        if annualized:
            vol *= np.sqrt(252)

        return float(vol)

    def get_sortino_ratio(self) -> float:
        """Calculate Sortino ratio using downside deviation.

        Returns
        -------
        float
            Annualized Sortino ratio.
        """
        if not self.returns:
            return 0.0

        returns = np.array(self.returns)
        mean_return = np.mean(returns)

        # Downside deviation (only negative returns)
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return float("inf") if mean_return > 0 else 0.0

        downside_std = np.std(negative_returns)

        if downside_std == 0:
            return 0.0

        return float(mean_return / downside_std * np.sqrt(252))

    def get_calmar_ratio(self) -> float:
        """Calculate Calmar ratio.

        Returns
        -------
        float
            Calmar ratio (annualized return / max drawdown).
        """
        max_dd, _, _ = self.get_max_drawdown()

        if max_dd == 0:
            return float("inf")

        annualized_return = self.get_total_return() * (252 / max(1, len(self.returns)))

        return float(annualized_return / max_dd)

    def get_win_rate(self) -> float:
        """Calculate win rate for trades.

        Returns
        -------
        float
            Percentage of profitable trades.
        """
        if not self.trades:
            return 0.0

        profitable = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        return float(profitable / len(self.trades))

    def get_profit_factor(self) -> float:
        """Calculate profit factor.

        Returns
        -------
        float
            Sum of gains / abs(sum of losses).
        """
        if not self.trades:
            return 0.0

        gains = sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) > 0)
        losses = abs(sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) < 0))

        if losses == 0:
            return float("inf") if gains > 0 else 0.0

        return float(gains / losses)

    def get_agent_attribution(self) -> Dict[str, AgentAttribution]:
        """Get per-agent performance attribution.

        Returns
        -------
        dict
            AgentAttribution per agent ID.
        """
        total_pnl = sum(
            sum(pnls) for pnls in self._agent_pnl.values()
        )
        attributions = {}

        for agent_id, pnls in self._agent_pnl.items():
            agent_total = sum(pnls)
            num_trades = self._agent_trades.get(agent_id, len(pnls))
            wins = sum(1 for p in pnls if p > 0)

            attr = AgentAttribution(
                agent_id=agent_id,
                total_pnl=agent_total,
                num_trades=num_trades,
                win_rate=wins / num_trades if num_trades > 0 else 0.0,
                avg_pnl_per_trade=agent_total / num_trades if num_trades > 0 else 0.0,
                contribution_pct=agent_total / total_pnl if total_pnl != 0 else 0.0,
            )
            attributions[agent_id] = attr

        return attributions

    def get_benchmark_comparison(self) -> BenchmarkComparison:
        """Get comparison metrics against benchmark.

        Returns
        -------
        BenchmarkComparison
            Benchmark comparison metrics.
        """
        strategy_return = self.get_total_return()
        benchmark_return = (
            (self.benchmark_values[-1] - self.initial_capital) / self.initial_capital
            if self.benchmark_values
            else 0.0
        )

        # Alpha = strategy return - beta * benchmark return
        beta = self._calculate_beta()
        alpha = strategy_return - beta * benchmark_return

        # Information ratio
        excess_returns = np.array(self.returns) - np.array(self.benchmark_returns)
        tracking_error = np.std(excess_returns) * np.sqrt(252) if len(excess_returns) > 1 else 0.0

        information_ratio = (
            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            if len(excess_returns) > 1 and np.std(excess_returns) > 0
            else 0.0
        )

        return BenchmarkComparison(
            strategy_return=strategy_return,
            benchmark_return=benchmark_return,
            alpha=float(alpha),
            beta=float(beta),
            information_ratio=float(information_ratio),
            tracking_error=float(tracking_error),
        )

    def _calculate_beta(self) -> float:
        """Calculate beta against benchmark.

        Returns
        -------
        float
            Beta coefficient.
        """
        if len(self.returns) < 2 or len(self.benchmark_returns) < 2:
            return 1.0

        returns = np.array(self.returns)
        benchmark_returns = np.array(self.benchmark_returns)

        # Covariance / variance
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        variance = np.var(benchmark_returns)

        if variance == 0:
            return 1.0

        return float(covariance / variance)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics.

        Returns
        -------
        dict
            Complete metrics dictionary.
        """
        max_dd, peak_step, trough_step = self.get_max_drawdown()

        return {
            "total_return": self.get_total_return(),
            "sharpe_ratio": self.get_sharpe_ratio(),
            "sortino_ratio": self.get_sortino_ratio(),
            "calmar_ratio": self.get_calmar_ratio(),
            "max_drawdown": max_dd,
            "max_drawdown_peak_step": peak_step,
            "max_drawdown_trough_step": trough_step,
            "current_drawdown": self.get_current_drawdown(),
            "volatility": self.get_volatility(),
            "win_rate": self.get_win_rate(),
            "profit_factor": self.get_profit_factor(),
            "num_trades": len(self.trades),
            "benchmark_comparison": self.get_benchmark_comparison().__dict__,
            "agent_attribution": {
                k: v.__dict__ for k, v in self.get_agent_attribution().items()
            },
            "equity_curve": self.get_equity_curve(),
        }

    def get_equity_curve(self) -> List[float]:
        """Get portfolio values over time.

        Returns
        -------
        list
            Portfolio values at each recorded step.
        """
        return self.portfolio_values.copy()

    def get_drawdown_curve(self) -> List[float]:
        """Get drawdown at each step.

        Returns
        -------
        list
            Drawdown percentage at each recorded step.
        """
        if len(self.portfolio_values) < 2:
            return [0.0]

        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdowns = (peak - values) / peak

        return drawdowns.tolist()

    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self.portfolio_values = [self.initial_capital]
        self.benchmark_values = [self.initial_capital]
        self.returns = []
        self.benchmark_returns = []
        self.trades = []
        self._agent_pnl = {}
        self._agent_trades = {}
        self._positions = {}
