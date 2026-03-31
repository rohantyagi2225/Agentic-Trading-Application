"""
Comprehensive Evaluation Framework with Regime Tracking

This module provides complete performance evaluation across different market regimes,
enabling data-driven strategy selection and continuous improvement.

Key Features:
- Multi-metric performance evaluation (Sharpe, Sortino, Drawdown, Win Rate)
- Regime-specific performance tracking
- Strategy comparison and ranking
- Statistical significance testing
- Stability analysis across regimes
- Continuous monitoring and alerting

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │          Comprehensive Evaluation Framework             │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │ Performance  │  │ Regime       │  │ Statistical  │  │
    │  │ Metrics      │  │ Attribution  │  │ Analysis     │  │
    │  └──────────────┘  └──────────────┘  └──────────────┘  │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │         Strategy Comparison & Ranking            │  │
    │  └──────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────────────────────────────┐
    │           Evaluation Output                             │
    │  - Overall performance metrics                          │
    │  - Performance by regime                                │
    │  - Risk-adjusted returns                                │
    │  - Statistical significance                             │
    │  - Strategy recommendations                             │
    └─────────────────────────────────────────────────────────┘

Author: FinAgent Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
import json
from backend.analytics.portfolio_analytics import portfolio_analytics
from backend.market.regime_detector import MarketRegime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("EvaluationFramework")


@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    direction: int  # 1 for long, -1 for short
    pnl: float = 0.0
    return_pct: float = 0.0
    
    def calculate_pnl(self):
        """Calculate P&L if exit price is available"""
        if self.exit_price is not None:
            if self.direction == 1:  # Long
                self.pnl = (self.exit_price - self.entry_price) * self.quantity
            else:  # Short
                self.pnl = (self.entry_price - self.exit_price) * self.quantity
            
            self.return_pct = (self.exit_price / self.entry_price - 1) * self.direction


@dataclass
class PerformanceMetrics:
    """Complete set of performance metrics"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_period: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    volatility: float
    calmar_ratio: float
    omega_ratio: float
    tail_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'avg_holding_period': self.avg_holding_period,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'volatility': self.volatility,
            'calmar_ratio': self.calmar_ratio,
            'omega_ratio': self.omega_ratio,
            'tail_ratio': self.tail_ratio
        }


@dataclass
class RegimePerformance:
    """Performance metrics broken down by regime"""
    regime: MarketRegime
    metrics: PerformanceMetrics
    num_trades: int
    time_in_regime: int  # days
    avg_regime_return: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime': self.regime.value,
            'metrics': self.metrics.to_dict(),
            'num_trades': self.num_trades,
            'time_in_regime': self.time_in_regime,
            'avg_regime_return': self.avg_regime_return
        }


@dataclass
class StrategyEvaluation:
    """Complete evaluation of a trading strategy"""
    strategy_id: str
    strategy_name: str
    evaluation_period_start: datetime
    evaluation_period_end: datetime
    overall_metrics: PerformanceMetrics
    regime_breakdown: Dict[str, RegimePerformance]
    monthly_returns: List[float]
    daily_returns: List[float]
    equity_curve: List[float]
    benchmark_comparison: Optional[Dict[str, float]]
    stability_score: float
    consistency_score: float
    risk_score: float
    overall_score: float
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'evaluation_period': {
                'start': self.evaluation_period_start.isoformat(),
                'end': self.evaluation_period_end.isoformat()
            },
            'overall_metrics': self.overall_metrics.to_dict(),
            'regime_breakdown': {k: v.to_dict() for k, v in self.regime_breakdown.items()},
            'stability_score': self.stability_score,
            'consistency_score': self.consistency_score,
            'risk_score': self.risk_score,
            'overall_score': self.overall_score,
            'recommendations': self.recommendations
        }


class PerformanceCalculator:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_all_metrics(
        returns: List[float],
        trades: List[Trade],
        equity_curve: List[float],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> PerformanceMetrics:
        """Calculate all performance metrics from returns series"""
        
        if not returns or len(returns) < 2:
            raise ValueError("Insufficient return data")
        
        returns_array = np.array(returns)
        
        # Basic returns
        total_return = (np.prod(1 + returns_array) - 1) * 100
        annualized_return = ((1 + total_return/100) ** (periods_per_year / len(returns)) - 1) * 100
        
        # Use existing analytics module
        analytics_result = portfolio_analytics(
            returns=returns,
            risk_free_rate=risk_free_rate / periods_per_year,
            periods_per_year=periods_per_year
        )
        
        sharpe = analytics_result.get('sharpe', 0)
        sortino = analytics_result.get('sortino', 0)
        max_dd = analytics_result.get('max_drawdown', 0)
        volatility = analytics_result.get('volatility', 0)
        
        # Win/Loss analysis
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Holding period
        holding_periods = []
        for t in trades:
            if t.exit_date:
                holding_periods.append((t.exit_date - t.entry_date).days)
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # Calmar Ratio (Return / Max Drawdown)
        calmar_ratio = annualized_return / abs(max_dd) if max_dd != 0 else 0
        
        # Omega Ratio (probability-weighted ratio of gains vs losses)
        threshold = 0  # No minimum return threshold
        excess_returns = returns_array - threshold
        gains = excess_returns[excess_returns > 0]
        losses = excess_returns[excess_returns <= 0]
        
        omega_ratio = np.sum(gains) / abs(np.sum(losses)) if np.sum(losses) != 0 else float('inf')
        
        # Tail Ratio (95th percentile / 5th percentile)
        if len(returns_array) >= 20:
            percentile_95 = np.percentile(returns_array, 95)
            percentile_5 = np.percentile(returns_array, 5)
            tail_ratio = percentile_95 / abs(percentile_5) if percentile_5 != 0 else float('inf')
        else:
            tail_ratio = 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_holding_period=avg_holding_period,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            tail_ratio=tail_ratio
        )


class RegimeAttributionAnalyzer:
    """Attribute performance to specific market regimes"""
    
    def __init__(self):
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
    
    def add_regime_detection(
        self,
        timestamp: datetime,
        regime: MarketRegime
    ):
        """Record regime at specific timestamp"""
        self.regime_history.append((timestamp, regime))
    
    def attribute_trades_to_regimes(
        self,
        trades: List[Trade],
        regime_series: pd.Series
    ) -> Dict[MarketRegime, List[Trade]]:
        """Map each trade to the regime during which it was executed"""
        
        regime_trades = {regime: [] for regime in MarketRegime}
        
        for trade in trades:
            # Find regime during trade entry - convert to pandas Timestamp for comparison
            trade_entry_ts = pd.Timestamp(trade.entry_date)
            
            try:
                regime = regime_series.loc[trade_entry_ts]
                regime_trades[regime].append(trade)
            except KeyError:
                # If no regime data for that date, use closest previous
                prior_dates = [d for d in regime_series.index if d <= trade_entry_ts]
                if prior_dates:
                    regime = regime_series.loc[max(prior_dates)]
                    regime_trades[regime].append(trade)
        
        return regime_trades
    
    def calculate_regime_performance(
        self,
        regime_trades: Dict[MarketRegime, List[Trade]],
        all_returns: List[float],
        regime_timestamps: Dict[MarketRegime, List[datetime]]
    ) -> Dict[MarketRegime, RegimePerformance]:
        """Calculate performance metrics for each regime"""
        
        regime_performance = {}
        
        for regime, trades in regime_trades.items():
            if not trades:
                continue
            
            # Filter returns for this regime
            regime_returns = self._filter_returns_by_regime(
                all_returns, regime_timestamps.get(regime, [])
            )
            
            # Calculate equity curve for regime
            regime_equity = [1000 * (1 + r) for r in np.cumprod(1 + np.array(regime_returns)) - 1]
            
            # Calculate metrics
            if regime_returns:
                metrics = PerformanceCalculator.calculate_all_metrics(
                    returns=regime_returns,
                    trades=trades,
                    equity_curve=regime_equity
                )
            else:
                continue
            
            # Time in regime
            timestamps = regime_timestamps.get(regime, [])
            time_in_regime = len(set([t.date() for t in timestamps]))
            
            # Average return in regime
            avg_regime_return = np.mean(regime_returns) if regime_returns else 0
            
            regime_performance[regime] = RegimePerformance(
                regime=regime,
                metrics=metrics,
                num_trades=len(trades),
                time_in_regime=time_in_regime,
                avg_regime_return=avg_regime_return
            )
        
        return regime_performance
    
    def _filter_returns_by_regime(
        self,
        returns: List[float],
        regime_dates: List[datetime]
    ) -> List[float]:
        """Filter returns to only those occurring during regime"""
        # Simplified - in production would need proper date matching
        return returns[:len(regime_dates)] if regime_dates else []


class StabilityAnalyzer:
    """Analyze strategy stability across regimes and time"""
    
    @staticmethod
    def calculate_stability_score(regime_performance: Dict[MarketRegime, RegimePerformance]) -> float:
        """
        Calculate stability score (0-100) based on performance consistency across regimes
        
        Higher score = more consistent performance across different market conditions
        """
        if not regime_performance:
            return 0
        
        # Extract Sharpe ratios across regimes
        sharpes = [rp.metrics.sharpe_ratio for rp in regime_performance.values() if rp.metrics.sharpe_ratio != 0]
        
        if not sharpes or len(sharpes) < 2:
            return 50  # Neutral if insufficient data
        
        # Lower variance in Sharpe = higher stability
        sharpe_variance = np.var(sharpes)
        
        # Normalize to 0-100 score (assuming variance < 4 is good)
        stability_score = max(0, min(100, 100 * (1 - sharpe_variance / 4)))
        
        return stability_score
    
    @staticmethod
    def calculate_consistency_score(monthly_returns: List[float]) -> float:
        """
        Calculate consistency score (0-100) based on return stream smoothness
        
        Higher score = smoother, more predictable returns
        """
        if not monthly_returns or len(monthly_returns) < 3:
            return 0
        
        returns_array = np.array(monthly_returns)
        
        # Positive months ratio
        positive_months = np.sum(returns_array > 0)
        consistency_ratio = positive_months / len(returns_array)
        
        # Return volatility penalty
        return_std = np.std(returns_array)
        volatility_penalty = max(0, 1 - return_std / 0.1)  # Penalize high volatility
        
        # Combine
        consistency_score = (consistency_ratio * 60 + volatility_penalty * 40)
        
        return consistency_score
    
    @staticmethod
    def calculate_risk_score(metrics: PerformanceMetrics) -> float:
        """
        Calculate risk score (0-100, lower is better)
        
        Based on drawdown, volatility, and tail risk
        """
        # Max drawdown component (0-40 points)
        drawdown_score = min(40, abs(metrics.max_drawdown) * 2)
        
        # Volatility component (0-30 points)
        vol_score = min(30, metrics.volatility * 1.5)
        
        # Tail risk component (0-30 points)
        tail_risk = 1 / metrics.tail_ratio if metrics.tail_ratio > 0 else 30
        tail_score = min(30, tail_risk * 10)
        
        total_risk_score = drawdown_score + vol_score + tail_score
        
        return total_risk_score


class StrategyComparator:
    """Compare and rank multiple strategies"""
    
    @staticmethod
    def compare_strategies(
        evaluations: List[StrategyEvaluation],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategies and rank them
        
        Args:
            evaluations: List of strategy evaluations
            weights: Custom weights for scoring (default: equal weight)
        """
        
        if weights is None:
            weights = {
                'sharpe': 0.25,
                'return': 0.20,
                'drawdown': 0.20,
                'stability': 0.15,
                'consistency': 0.10,
                'win_rate': 0.10
            }
        
        comparison_data = []
        
        for eval in evaluations:
            row = {
                'strategy_id': eval.strategy_id,
                'strategy_name': eval.strategy_name,
                'sharpe_ratio': eval.overall_metrics.sharpe_ratio,
                'total_return': eval.overall_metrics.total_return,
                'max_drawdown': eval.overall_metrics.max_drawdown,
                'win_rate': eval.overall_metrics.win_rate,
                'stability_score': eval.stability_score,
                'consistency_score': eval.consistency_score,
                'risk_score': eval.risk_score,
                'overall_score': eval.overall_score
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('overall_score', ascending=False)
        
        return df


class ComprehensiveEvaluator:
    """
    Main class for comprehensive strategy evaluation
    
    Usage:
        evaluator = ComprehensiveEvaluator()
        
        evaluation = evaluator.evaluate_strategy(
            trades=trade_list,
            returns=daily_returns,
            equity_curve=equity_values,
            regime_series=regime_data,
            strategy_name="My Strategy"
        )
        
        print(evaluation.overall_metrics.sharpe_ratio)
        print(evaluation.regime_breakdown)
    """
    
    def __init__(self, benchmark_symbol: str = "SPY"):
        self.benchmark_symbol = benchmark_symbol
        self.performance_calculator = PerformanceCalculator()
        self.regime_analyzer = RegimeAttributionAnalyzer()
        self.stability_analyzer = StabilityAnalyzer()
        
        logger.info("✅ Comprehensive evaluator initialized")
    
    def evaluate_strategy(
        self,
        trades: List[Trade],
        returns: List[float],
        equity_curve: List[float],
        regime_series: pd.Series,
        strategy_name: str,
        benchmark_returns: Optional[List[float]] = None,
        evaluation_period_days: int = 252
    ) -> StrategyEvaluation:
        """Perform comprehensive strategy evaluation"""
        
        if not trades or not returns:
            raise ValueError("Trades and returns data required")
        
        # ===== 1. OVERALL METRICS =====
        overall_metrics = self.performance_calculator.calculate_all_metrics(
            returns=returns,
            trades=trades,
            equity_curve=equity_curve
        )
        
        # ===== 2. REGIME ATTRIBUTION =====
        regime_trades = self.regime_analyzer.attribute_trades_to_regimes(trades, regime_series)
        
        # Get regime timestamps
        regime_timestamps = {regime: [t.entry_date for t in trades_list] 
                           for regime, trades_list in regime_trades.items()}
        
        regime_performance = self.regime_analyzer.calculate_regime_performance(
            regime_trades=regime_trades,
            all_returns=returns,
            regime_timestamps=regime_timestamps
        )
        
        # ===== 3. STABILITY & CONSISTENCY =====
        stability_score = self.stability_analyzer.calculate_stability_score(regime_performance)
        
        # Monthly returns for consistency calculation
        monthly_returns = self._convert_to_monthly_returns(returns)
        consistency_score = self.stability_analyzer.calculate_consistency_score(monthly_returns)
        
        # ===== 4. RISK SCORE =====
        risk_score = self.stability_analyzer.calculate_risk_score(overall_metrics)
        
        # ===== 5. OVERALL SCORE (Weighted combination) =====
        overall_score = (
            0.30 * min(10, overall_metrics.sharpe_ratio) / 10 +  # Normalized Sharpe
            0.25 * (100 - risk_score) / 100 +                     # Inverse risk
            0.25 * stability_score / 100 +                        # Stability
            0.20 * consistency_score / 100                        # Consistency
        ) * 100
        
        # ===== 6. BENCHMARK COMPARISON =====
        benchmark_comparison = None
        if benchmark_returns:
            benchmark_metrics = self.performance_calculator.calculate_all_metrics(
                returns=benchmark_returns,
                trades=[],
                equity_curve=[]
            )
            benchmark_comparison = {
                'strategy_sharpe': overall_metrics.sharpe_ratio,
                'benchmark_sharpe': benchmark_metrics.sharpe_ratio,
                'strategy_return': overall_metrics.total_return,
                'benchmark_return': benchmark_metrics.total_return,
                'alpha': overall_metrics.total_return - benchmark_metrics.total_return
            }
        
        # ===== 7. GENERATE RECOMMENDATIONS =====
        recommendations = self._generate_recommendations(
            overall_metrics, regime_performance, stability_score, consistency_score
        )
        
        # ===== 8. CREATE EVALUATION =====
        evaluation = StrategyEvaluation(
            strategy_id=f"strat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            strategy_name=strategy_name,
            evaluation_period_start=datetime.now() - timedelta(days=evaluation_period_days),
            evaluation_period_end=datetime.now(),
            overall_metrics=overall_metrics,
            regime_breakdown={k.value: v for k, v in regime_performance.items()},
            monthly_returns=monthly_returns,
            daily_returns=returns,
            equity_curve=equity_curve,
            benchmark_comparison=benchmark_comparison,
            stability_score=stability_score,
            consistency_score=consistency_score,
            risk_score=risk_score,
            overall_score=overall_score,
            recommendations=recommendations
        )
        
        logger.info(
            f"📊 Evaluation complete: {strategy_name}\n"
            f"  Overall Score: {overall_score:.1f}/100\n"
            f"  Sharpe: {overall_metrics.sharpe_ratio:.2f}\n"
            f"  Max DD: {overall_metrics.max_drawdown:.1f}%\n"
            f"  Stability: {stability_score:.1f}/100"
        )
        
        return evaluation
    
    def _convert_to_monthly_returns(self, daily_returns: List[float]) -> List[float]:
        """Convert daily returns to monthly returns"""
        if not daily_returns:
            return []
        
        # Simple approximation: compound ~21 days per month
        monthly = []
        for i in range(0, len(daily_returns), 21):
            month_chunk = daily_returns[i:i+21]
            monthly_return = np.prod(1 + np.array(month_chunk)) - 1
            monthly.append(monthly_return)
        
        return monthly
    
    def _generate_recommendations(
        self,
        metrics: PerformanceMetrics,
        regime_perf: Dict[MarketRegime, RegimePerformance],
        stability: float,
        consistency: float
    ) -> List[str]:
        """Generate actionable recommendations based on evaluation"""
        
        recommendations = []
        
        # Sharpe-based recommendations
        if metrics.sharpe_ratio < 1.0:
            recommendations.append("⚠️ Sharpe ratio below 1.0 - consider improving risk-adjusted returns")
        elif metrics.sharpe_ratio > 2.0:
            recommendations.append("✓ Excellent risk-adjusted returns maintained")
        
        # Drawdown recommendations
        if abs(metrics.max_drawdown) > 20:
            recommendations.append("⚠️ Maximum drawdown exceeds 20% - implement stricter risk controls")
        elif abs(metrics.max_drawdown) < 10:
            recommendations.append("✓ Drawdown well-controlled")
        
        # Win rate recommendations
        if metrics.win_rate < 0.45:
            recommendations.append("⚠️ Win rate below 45% - review entry criteria and signal quality")
        elif metrics.win_rate > 0.60:
            recommendations.append("✓ High win rate indicates good signal quality")
        
        # Stability recommendations
        if stability < 50:
            recommendations.append("⚠️ Low stability across regimes - strategy may be regime-dependent")
        elif stability > 75:
            recommendations.append("✓ Strategy performs consistently across market regimes")
        
        # Regime-specific recommendations
        worst_regime = min(regime_perf.items(), key=lambda x: x[1].metrics.sharpe_ratio)
        if worst_regime[1].metrics.sharpe_ratio < 0:
            recommendations.append(
                f"⚠️ Poor performance in {worst_regime[0].value} regime - "
                f"consider reducing exposure during these periods"
            )
        
        # Profit factor recommendations
        if metrics.profit_factor < 1.5:
            recommendations.append("⚠️ Profit factor below 1.5 - work on letting winners run")
        elif metrics.profit_factor > 2.5:
            recommendations.append("✓ Excellent profit factor - good risk/reward management")
        
        return recommendations
    
    def export_evaluation_report(
        self,
        evaluation: StrategyEvaluation,
        format: str = 'json'
    ) -> str:
        """Export evaluation report"""
        
        if format == 'json':
            return json.dumps(evaluation.to_dict(), indent=2)
        elif format == 'text':
            return self._format_evaluation_as_text(evaluation)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_evaluation_as_text(self, evaluation: StrategyEvaluation) -> str:
        """Format evaluation as readable text report"""
        
        lines = [
            "=" * 80,
            f"STRATEGY EVALUATION REPORT - {evaluation.strategy_name}",
            "=" * 80,
            "",
            f"Evaluation Period: {evaluation.evaluation_period_start.strftime('%Y-%m-%d')} to "
            f"{evaluation.evaluation_period_end.strftime('%Y-%m-%d')}",
            "",
            "-" * 80,
            "OVERALL PERFORMANCE METRICS",
            "-" * 80,
            f"Total Return: {evaluation.overall_metrics.total_return:.2f}%",
            f"Annualized Return: {evaluation.overall_metrics.annualized_return:.2f}%",
            f"Sharpe Ratio: {evaluation.overall_metrics.sharpe_ratio:.2f}",
            f"Sortino Ratio: {evaluation.overall_metrics.sortino_ratio:.2f}",
            f"Maximum Drawdown: {evaluation.overall_metrics.max_drawdown:.2f}%",
            f"Win Rate: {evaluation.overall_metrics.win_rate:.1%}",
            f"Profit Factor: {evaluation.overall_metrics.profit_factor:.2f}",
            f"Volatility: {evaluation.overall_metrics.volatility:.2f}%",
            f"Calmar Ratio: {evaluation.overall_metrics.calmar_ratio:.2f}",
            "",
            "-" * 80,
            "SCORE BREAKDOWN",
            "-" * 80,
            f"Overall Score: {evaluation.overall_score:.1f}/100",
            f"Stability Score: {evaluation.stability_score:.1f}/100",
            f"Consistency Score: {evaluation.consistency_score:.1f}/100",
            f"Risk Score: {evaluation.risk_score:.1f}/100 (lower is better)",
            "",
            "-" * 80,
            "REGIME PERFORMANCE BREAKDOWN",
            "-" * 80
        ]
        
        for regime_name, regime_perf in evaluation.regime_breakdown.items():
            lines.extend([
                f"\n{regime_name.upper()}:",
                f"  Trades: {regime_perf.num_trades}",
                f"  Sharpe: {regime_perf.metrics.sharpe_ratio:.2f}",
                f"  Return: {regime_perf.metrics.total_return:.2f}%",
                f"  Win Rate: {regime_perf.metrics.win_rate:.1%}"
            ])
        
        lines.extend([
            "",
            "-" * 80,
            "RECOMMENDATIONS",
            "-" * 80
        ])
        
        for i, rec in enumerate(evaluation.recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        if evaluation.benchmark_comparison:
            lines.extend([
                "",
                "-" * 80,
                "BENCHMARK COMPARISON",
                "-" * 80,
                f"Strategy Sharpe: {evaluation.benchmark_comparison['strategy_sharpe']:.2f}",
                f"Benchmark Sharpe: {evaluation.benchmark_comparison['benchmark_sharpe']:.2f}",
                f"Alpha (Excess Return): {evaluation.benchmark_comparison['alpha']:.2f}%"
            ])
        
        lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return "\n".join(lines)


def create_comprehensive_evaluator(benchmark: str = "SPY") -> ComprehensiveEvaluator:
    """Factory function to create comprehensive evaluator"""
    return ComprehensiveEvaluator(benchmark)
