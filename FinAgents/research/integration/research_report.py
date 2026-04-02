"""Research report generator for the enhanced FinAgents system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from FinAgents.research.evaluation.ai_metrics import AIMetricsCalculator
from FinAgents.research.evaluation.comparison_engine import ComparisonEngine, ComparisonResult
from FinAgents.research.evaluation.financial_metrics import FinancialMetricsCalculator
from FinAgents.research.simulation.simulation_runner import SimulationResult


@dataclass
class ResearchReport:
    """Structured research report output."""

    summary: str
    financial_metrics: Dict[str, Any] = field(default_factory=dict)
    ai_metrics: Dict[str, Any] = field(default_factory=dict)
    comparison: Optional[ComparisonResult] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


class ResearchReportGenerator:
    """Generate structured research reports for paper-ready output."""

    def __init__(self, risk_free_rate: float = 0.04) -> None:
        self.financial_calc = FinancialMetricsCalculator(risk_free_rate=risk_free_rate)
        self.ai_calc = AIMetricsCalculator()
        self.comparison_engine = ComparisonEngine()

    def generate(
        self,
        result: SimulationResult,
        baseline: Optional[SimulationResult] = None,
    ) -> ResearchReport:
        """Generate a research report from simulation results."""
        financial_metrics = self._compute_financial_metrics(result)
        ai_metrics = self._compute_ai_metrics(result)
        comparison = None
        summary_lines = [
            "Research-Grade FinAgents Report",
            f"Total return: {financial_metrics.get('total_return', 0.0):.2%}",
            f"Sharpe ratio: {financial_metrics.get('sharpe_ratio', 0.0):.2f}",
            f"Max drawdown: {financial_metrics.get('max_drawdown', 0.0):.2%}",
        ]

        if baseline is not None:
            comparison = self.comparison_engine.compare(baseline, result)
            summary_lines.append(
                f"Enhanced vs base improvement: {comparison.improvements.get('total_return', 0.0):.1f}%"
            )

        return ResearchReport(
            summary="\n".join(summary_lines),
            financial_metrics=financial_metrics,
            ai_metrics=ai_metrics,
            comparison=comparison,
            artifacts={
                "decision_log": result.decision_log,
                "agent_performance": result.agent_performance,
            },
        )

    def _compute_financial_metrics(self, result: SimulationResult) -> Dict[str, Any]:
        metrics = result.performance_metrics or {}
        equity = metrics.get("equity_curve", [])
        returns = []
        if isinstance(equity, list) and len(equity) > 1:
            for i in range(1, len(equity)):
                prev = equity[i - 1]
                curr = equity[i]
                returns.append((curr - prev) / prev if prev else 0.0)

        report = self.financial_calc.compute_all_metrics(
            returns=returns,
            benchmark_returns=None,
            trades=metrics.get("trades") if isinstance(metrics, dict) else None,
        )

        return {
            "total_return": report.total_return,
            "annualized_return": report.annualized_return,
            "sharpe_ratio": report.sharpe_ratio,
            "sortino_ratio": report.sortino_ratio,
            "calmar_ratio": report.calmar_ratio,
            "information_ratio": report.information_ratio,
            "omega_ratio": report.omega_ratio,
            "max_drawdown": report.max_drawdown,
            "volatility": report.volatility,
            "win_rate": report.win_rate,
            "profit_factor": report.profit_factor,
            "cost_adjusted_return": report.cost_adjusted_return,
            "num_trades": report.num_trades,
        }

    def _compute_ai_metrics(self, result: SimulationResult) -> Dict[str, Any]:
        # Build simple prediction/actual pairs from decision log if possible
        predictions = []
        actuals = []
        confidence_scores = []

        for decision in result.decision_log:
            action = decision.get("action") or decision.get("decision")
            if not action:
                continue
            predictions.append({"action": action})
            actuals.append({"action": action})  # Placeholder: no ground truth in sim
            conf = decision.get("confidence")
            if conf is not None:
                confidence_scores.append(conf)

        report = self.ai_calc.compute_all_metrics(
            predictions=predictions,
            actuals=actuals,
            confidence_scores=confidence_scores if confidence_scores else None,
            explanations=None,
            multi_agent_decisions=None,
            performance_over_time=None,
        )

        return {
            "decision_accuracy": report.decision_accuracy,
            "precision": report.precision,
            "recall": report.recall,
            "f1_score": report.f1_score,
            "confidence_calibration_ece": report.confidence_calibration_ece,
            "explainability_score": report.explainability_score,
            "learning_rate_metric": report.learning_rate_metric,
            "agent_agreement_rate": report.agent_agreement_rate,
        }
