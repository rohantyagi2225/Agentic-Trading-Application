"""
Comprehensive Evaluation Framework
===================================

Multi-dimensional evaluation of the multi-agent system:

Financial Metrics:
- Returns (total, annualized)
- Risk-adjusted (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Win rate and profit factor

AI Metrics:
- Decision accuracy
- Explanation quality
- Learning progression
- Coordination effectiveness

Comparison Framework:
- Enhanced vs base system
- Ablation studies
- Sensitivity analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    financial_metrics: Dict[str, float]
    ai_metrics: Dict[str, float]
    comparison_metrics: Optional[Dict[str, float]]
    statistical_tests: Dict[str, Any]
    summary: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class EvaluationFramework:
    """
    Comprehensive evaluation framework for multi-agent trading systems
    
    Evaluates financial performance, AI decision quality, and
    provides statistical comparisons.
    """
    
    def __init__(self, framework_id: str = "main"):
        """
        Initialize evaluation framework
        
        Args:
            framework_id: Unique identifier
        """
        self.framework_id = framework_id
        
        # Historical results storage
        self.results_history: List[EvaluationResult] = []
        
        # Benchmark data
        self.benchmarks = {
            "sp500": {"return": 0.10, "volatility": 0.15, "sharpe": 0.67},
            "naive_strategy": {"return": 0.05, "volatility": 0.20, "sharpe": 0.25},
        }
        
        logger.info(f"EvaluationFramework {framework_id} initialized")
    
    def evaluate_system(
        self,
        portfolio_values: List[float],
        trade_log: List[Dict[str, Any]],
        decisions: List[Dict[str, Any]],
        explanations: Optional[List[Any]] = None,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> EvaluationResult:
        """
        Perform comprehensive system evaluation
        
        Args:
            portfolio_values: Time series of portfolio values
            trade_log: List of executed trades
            decisions: List of agent decisions
            explanations: Generated explanations
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Complete evaluation result
        """
        # Financial metrics
        financial_metrics = self._calculate_financial_metrics(portfolio_values, trade_log)
        
        # AI metrics
        ai_metrics = self._calculate_ai_metrics(decisions, explanations)
        
        # Comparison with benchmarks
        comparison_metrics = None
        statistical_tests = {}
        
        if benchmark_returns is not None:
            comparison_metrics = self._compare_with_benchmark(
                portfolio_values, benchmark_returns
            )
            statistical_tests = self._perform_statistical_tests(
                portfolio_values, benchmark_returns
            )
        
        # Generate summary
        summary = self._generate_summary(financial_metrics, ai_metrics, comparison_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(financial_metrics, ai_metrics)
        
        result = EvaluationResult(
            financial_metrics=financial_metrics,
            ai_metrics=ai_metrics,
            comparison_metrics=comparison_metrics,
            statistical_tests=statistical_tests,
            summary=summary,
            recommendations=recommendations,
        )
        
        self.results_history.append(result)
        
        return result
    
    def _calculate_financial_metrics(
        self,
        portfolio_values: List[float],
        trade_log: List[Dict[str, Any]],
        risk_free_rate: float = 0.02,
    ) -> Dict[str, float]:
        """Calculate comprehensive financial metrics"""
        if len(portfolio_values) < 2:
            return self._empty_financial_metrics()
        
        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        # Basic return metrics
        total_return = (values[-1] - values[0]) / values[0]
        n_periods = len(values)
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1 if n_periods > 0 else 0
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_dev = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Risk-adjusted metrics
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_dev * np.sqrt(252) if downside_dev > 0 else 0
        
        # Drawdown analysis
        rolling_max = np.maximum.accumulate(values)
        drawdowns = (values - rolling_max) / rolling_max
        max_drawdown = abs(np.min(drawdowns))
        
        # Trade-based metrics
        if trade_log:
            winning_trades = [t for t in trade_log if t.get("pnl", 0) > 0]
            losing_trades = [t for t in trade_log if t.get("pnl", 0) < 0]
            
            win_rate = len(winning_trades) / len(trade_log)
            
            gross_profit = sum(t["pnl"] for t in winning_trades)
            gross_loss = abs(sum(t["pnl"] for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
            win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
            win_loss_ratio = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "win_loss_ratio": win_loss_ratio,
            "calmar_ratio": calmar_ratio,
            "num_trades": len(trade_log),
        }
    
    def _calculate_ai_metrics(
        self,
        decisions: List[Dict[str, Any]],
        explanations: Optional[List[Any]],
    ) -> Dict[str, float]:
        """Calculate AI system performance metrics"""
        if not decisions:
            return self._empty_ai_metrics()
        
        # Decision confidence
        confidences = [d.get("confidence", 0.5) for d in decisions]
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        # Decision consistency
        actions = [d.get("decision", "HOLD") for d in decisions]
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        most_common_action = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else "HOLD"
        consistency_ratio = action_counts.get(most_common_action, 0) / len(actions)
        
        # Agent agreement rate (if multi-agent)
        agent_agreements = []
        for d in decisions:
            supporting = len(d.get("supporting_agents", []))
            opposing = len(d.get("opposing_agents", []))
            total = supporting + opposing
            if total > 0:
                agreement_rate = supporting / total
                agent_agreements.append(agreement_rate)
        
        avg_agent_agreement = np.mean(agent_agreements) if agent_agreements else 0.5
        
        # Explanation quality (if available)
        explanation_score = 0.5
        if explanations:
            # Simple heuristic: length and structure
            exp_lengths = [len(str(e)) for e in explanations]
            explanation_score = min(np.mean(exp_lengths) / 500, 1.0)  # Normalize
        
        # Learning rate (improvement over time)
        learning_rate = 0.0
        if len(decisions) > 10:
            first_half_conf = np.mean([d.get("confidence", 0.5) for d in decisions[:len(decisions)//2]])
            second_half_conf = np.mean([d.get("confidence", 0.5) for d in decisions[len(decisions)//2:]])
            learning_rate = second_half_conf - first_half_conf
        
        return {
            "avg_decision_confidence": avg_confidence,
            "confidence_stability": 1 - confidence_std,
            "decision_consistency": consistency_ratio,
            "agent_agreement_rate": avg_agent_agreement,
            "explanation_quality": explanation_score,
            "learning_rate": learning_rate,
            "total_decisions": len(decisions),
        }
    
    def _compare_with_benchmark(
        self,
        portfolio_values: List[float],
        benchmark_returns: np.ndarray,
    ) -> Dict[str, float]:
        """Compare performance with benchmark"""
        port_returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        
        # Excess returns (alpha)
        excess_returns = port_returns - benchmark_returns
        
        alpha = np.mean(excess_returns) * 252  # Annualized
        beta = np.cov(port_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0
        
        # Information ratio
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        
        # Outperformance rate
        outperformance_rate = np.mean(excess_returns > 0)
        
        return {
            "alpha": alpha,
            "beta": beta,
            "information_ratio": information_ratio,
            "outperformance_rate": outperformance_rate,
            "tracking_error": tracking_error,
        }
    
    def _perform_statistical_tests(
        self,
        portfolio_values: List[float],
        benchmark_returns: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        port_returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        
        # T-test for mean returns
        t_stat, p_value = stats.ttest_1samp(port_returns, 0)
        
        # Test against benchmark
        excess_returns = port_returns - benchmark_returns
        t_stat_alpha, p_value_alpha = stats.ttest_1samp(excess_returns, 0)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(port_returns)
        
        return {
            "returns_t_statistic": t_stat,
            "returns_p_value": p_value,
            "alpha_t_statistic": t_stat_alpha,
            "alpha_p_value": p_value_alpha,
            "normality_jb_statistic": jb_stat,
            "normality_jb_p_value": jb_pvalue,
            "significant_alpha": p_value_alpha < 0.05,
        }
    
    def _generate_summary(
        self,
        financial_metrics: Dict[str, float],
        ai_metrics: Dict[str, float],
        comparison_metrics: Optional[Dict[str, float]],
    ) -> str:
        """Generate human-readable summary"""
        sharpe = financial_metrics.get("sharpe_ratio", 0)
        total_return = financial_metrics.get("total_return", 0)
        max_dd = financial_metrics.get("max_drawdown", 0)
        
        if sharpe > 1.5:
            risk_adj_perf = "Excellent"
        elif sharpe > 1.0:
            risk_adj_perf = "Good"
        elif sharpe > 0.5:
            risk_adj_perf = "Moderate"
        else:
            risk_adj_perf = "Poor"
        
        summary = f"""
PERFORMANCE SUMMARY
==================

Overall Performance:
- Total Return: {total_return*100:.2f}%
- Risk-Adjusted Performance: {risk_adj_perf} (Sharpe: {sharpe:.2f})
- Maximum Drawdown: {max_dd*100:.2f}%

Trading Effectiveness:
- Win Rate: {financial_metrics.get('win_rate', 0)*100:.1f}%
- Profit Factor: {financial_metrics.get('profit_factor', 0):.2f}
- Number of Trades: {financial_metrics.get('num_trades', 0)}

AI System Quality:
- Average Decision Confidence: {ai_metrics.get('avg_decision_confidence', 0)*100:.1f}%
- Agent Agreement Rate: {ai_metrics.get('agent_agreement_rate', 0)*100:.1f}%
- Learning Rate: {ai_metrics.get('learning_rate', 0)*100:.2f}%
"""
        
        if comparison_metrics:
            alpha = comparison_metrics.get("alpha", 0)
            summary += f"""
Benchmark Comparison:
- Alpha (Excess Return): {alpha*100:.2f}% annualized
- Outperformance Rate: {comparison_metrics.get('outperformance_rate', 0)*100:.1f}%
"""
        
        return summary.strip()
    
    def _generate_recommendations(
        self,
        financial_metrics: Dict[str, float],
        ai_metrics: Dict[str, float],
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Risk management
        if financial_metrics.get("max_drawdown", 0) > 0.20:
            recommendations.append("Consider implementing stricter risk controls to reduce drawdowns")
        
        if financial_metrics.get("volatility", 0) > 0.30:
            recommendations.append("Portfolio volatility is high - consider diversification")
        
        # Trading strategy
        if financial_metrics.get("win_rate", 0) < 0.45:
            recommendations.append("Win rate below 45% - review entry criteria and signal quality")
        
        if financial_metrics.get("profit_factor", 0) < 1.2:
            recommendations.append("Profit factor below 1.2 - improve risk-reward ratios")
        
        # AI system
        if ai_metrics.get("agent_agreement_rate", 0) < 0.6:
            recommendations.append("Low agent agreement - review coordination mechanisms")
        
        if ai_metrics.get("learning_rate", 0) < 0:
            recommendations.append("Negative learning trend - investigate performance degradation")
        
        if ai_metrics.get("avg_decision_confidence", 0) < 0.5:
            recommendations.append("Low decision confidence - enhance signal generation")
        
        return recommendations
    
    def _empty_financial_metrics(self) -> Dict[str, float]:
        """Return empty financial metrics"""
        return {
            "total_return": 0,
            "annualized_return": 0,
            "volatility": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "win_loss_ratio": 0,
            "calmar_ratio": 0,
            "num_trades": 0,
        }
    
    def _empty_ai_metrics(self) -> Dict[str, float]:
        """Return empty AI metrics"""
        return {
            "avg_decision_confidence": 0,
            "confidence_stability": 0,
            "decision_consistency": 0,
            "agent_agreement_rate": 0,
            "explanation_quality": 0,
            "learning_rate": 0,
            "total_decisions": 0,
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get framework state"""
        return {
            "framework_id": self.framework_id,
            "evaluations_performed": len(self.results_history),
        }
