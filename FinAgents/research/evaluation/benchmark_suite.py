"""Benchmark suite for research-grade evaluation.

Defines standard market scenarios (bull, bear, sideways, high-volatility,
crash-recovery) and provides utilities to run simulations and compile
comparison reports between base and enhanced systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from FinAgents.research.evaluation.comparison_engine import ComparisonEngine
from FinAgents.research.evaluation.financial_metrics import FinancialMetricsCalculator
from FinAgents.research.simulation.simulation_runner import SimulationResult, SimulationRunner
from FinAgents.research.simulation.market_environment import MarketRegime


@dataclass
class BenchmarkScenario:
    """Definition of a benchmark scenario."""

    name: str
    description: str
    num_steps: int = 252
    market_config: Dict[str, Any] = field(default_factory=dict)
    event_config: Dict[str, Any] = field(default_factory=dict)
    forced_regime: Optional[MarketRegime] = None


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results for a system."""

    scenario_results: Dict[str, SimulationResult] = field(default_factory=dict)
    metrics_by_scenario: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: str = ""


class BenchmarkSuite:
    """Run benchmark scenarios for base and enhanced systems."""

    def __init__(
        self,
        scenarios: Optional[List[BenchmarkScenario]] = None,
        risk_free_rate: float = 0.04,
    ) -> None:
        self.scenarios = scenarios or self.default_scenarios()
        self.metrics_calculator = FinancialMetricsCalculator(risk_free_rate=risk_free_rate)
        self.comparison_engine = ComparisonEngine()

    @staticmethod
    def default_scenarios() -> List[BenchmarkScenario]:
        """Create default benchmark scenarios."""
        return [
            BenchmarkScenario(
                name="bull_market",
                description="Sustained upward drift with moderate volatility.",
                market_config={"drift": 0.0004, "base_volatility": 0.015},
                forced_regime=MarketRegime.BULL,
            ),
            BenchmarkScenario(
                name="bear_market",
                description="Downward drift with elevated volatility.",
                market_config={"drift": -0.0004, "base_volatility": 0.03},
                forced_regime=MarketRegime.BEAR,
            ),
            BenchmarkScenario(
                name="sideways_market",
                description="Low drift, low volatility range-bound market.",
                market_config={"drift": 0.0, "base_volatility": 0.01},
                forced_regime=MarketRegime.SIDEWAYS,
            ),
            BenchmarkScenario(
                name="high_volatility",
                description="High volatility with mixed directional drift.",
                market_config={"drift": 0.0001, "base_volatility": 0.05},
                forced_regime=MarketRegime.HIGH_VOLATILITY,
            ),
            BenchmarkScenario(
                name="crash_recovery",
                description="Crash-like volatility with stronger jump magnitude.",
                market_config={
                    "drift": -0.0002,
                    "base_volatility": 0.06,
                    "jump_probability": 0.03,
                    "jump_magnitude": 0.12,
                },
                forced_regime=MarketRegime.CRASH,
            ),
        ]

    def run(
        self,
        runner_factory: Callable[[BenchmarkScenario], SimulationRunner],
        label: str = "system",
    ) -> BenchmarkReport:
        """Run all scenarios for a single system.

        Parameters
        ----------
        runner_factory:
            Callable that returns a configured SimulationRunner for a scenario.
        label:
            Label used in summary.
        """
        scenario_results: Dict[str, SimulationResult] = {}
        metrics_by_scenario: Dict[str, Dict[str, Any]] = {}

        for scenario in self.scenarios:
            runner = runner_factory(scenario)
            if runner.config.num_steps != scenario.num_steps:
                runner.config.num_steps = scenario.num_steps

            # Force regime if requested
            if scenario.forced_regime is not None:
                for symbol in runner.config.symbols:
                    runner.market.set_regime(symbol, scenario.forced_regime)

            result = runner.run(verbose=False)
            scenario_results[scenario.name] = result
            metrics_by_scenario[scenario.name] = self._compute_financial_metrics(result)

        summary = self._build_summary(metrics_by_scenario, label=label)
        return BenchmarkReport(
            scenario_results=scenario_results,
            metrics_by_scenario=metrics_by_scenario,
            summary=summary,
        )

    def run_comparison(
        self,
        base_factory: Callable[[BenchmarkScenario], SimulationRunner],
        enhanced_factory: Callable[[BenchmarkScenario], SimulationRunner],
    ) -> Dict[str, Any]:
        """Run comparison benchmarks for base vs enhanced systems."""
        base_report = self.run(base_factory, label="base")
        enhanced_report = self.run(enhanced_factory, label="enhanced")

        comparisons: Dict[str, Any] = {}
        for scenario in self.scenarios:
            base_result = base_report.scenario_results.get(scenario.name)
            enhanced_result = enhanced_report.scenario_results.get(scenario.name)
            if base_result and enhanced_result:
                comparisons[scenario.name] = self.comparison_engine.compare(
                    base_result, enhanced_result
                )

        return {
            "base": base_report,
            "enhanced": enhanced_report,
            "comparisons": comparisons,
        }

    def _compute_financial_metrics(self, result: SimulationResult) -> Dict[str, Any]:
        """Compute financial metrics for a simulation result."""
        metrics = result.performance_metrics or {}
        equity = metrics.get("equity_curve", [])
        returns = []
        if isinstance(equity, list) and len(equity) > 1:
            equity_arr = np.array(equity)
            returns = (np.diff(equity_arr) / equity_arr[:-1]).tolist()

        report = self.metrics_calculator.compute_all_metrics(
            returns=np.asarray(returns) if returns else np.array([]),
            benchmark_returns=None,
            trades=metrics.get("trades") if isinstance(metrics, dict) else None,
        )

        return {
            "total_return": report.total_return,
            "annualized_return": report.annualized_return,
            "sharpe_ratio": report.sharpe_ratio,
            "sortino_ratio": report.sortino_ratio,
            "calmar_ratio": report.calmar_ratio,
            "max_drawdown": report.max_drawdown,
            "volatility": report.volatility,
            "win_rate": report.win_rate,
            "profit_factor": report.profit_factor,
            "num_trades": report.num_trades,
        }

    def _build_summary(self, metrics_by_scenario: Dict[str, Dict[str, Any]], label: str) -> str:
        """Build a short summary of benchmark results."""
        if not metrics_by_scenario:
            return f"No benchmark results available for {label}."

        lines = [f"Benchmark summary for {label}:"]
        for scenario, metrics in metrics_by_scenario.items():
            lines.append(
                f"- {scenario}: return={metrics.get('total_return', 0.0):.2%}, "
                f"sharpe={metrics.get('sharpe_ratio', 0.0):.2f}, "
                f"max_dd={metrics.get('max_drawdown', 0.0):.2%}"
            )
        return "\n".join(lines)
