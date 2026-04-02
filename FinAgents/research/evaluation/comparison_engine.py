"""Comparison Engine for Base vs Enhanced System Evaluation.

This module provides tools for comparing base and enhanced trading systems,
including statistical testing and improvement attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ComparisonResult:
    """Result of comparing base and enhanced systems.

    Attributes
    ----------
    base_metrics : dict
        Metrics from the base system.
    enhanced_metrics : dict
        Metrics from the enhanced system.
    improvements : dict
        Metric name -> improvement percentage.
    statistical_tests : dict
        Statistical test results per metric.
    improvement_attribution : dict
        Enhancement name -> contribution percentage.
    summary : str
        Narrative summary of comparison.
    visualization_data : dict
        Data formatted for visualization.
    """

    base_metrics: Dict[str, Any] = field(default_factory=dict)
    enhanced_metrics: Dict[str, Any] = field(default_factory=dict)
    improvements: Dict[str, float] = field(default_factory=dict)
    statistical_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    improvement_attribution: Dict[str, float] = field(default_factory=dict)
    summary: str = ""
    visualization_data: Dict[str, Any] = field(default_factory=dict)


class ComparisonEngine:
    """Engine for comparing base and enhanced trading systems.

    Provides comprehensive comparison including statistical significance
    testing and improvement attribution analysis.

    Parameters
    ----------
    config : dict, optional
        Configuration parameters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the comparison engine.

        Parameters
        ----------
        config : dict, optional
            Configuration parameters.
        """
        self.config = config or {}

    def compare(
        self,
        base_result: Any,
        enhanced_result: Any,
    ) -> ComparisonResult:
        """Perform full comparison of two simulation results.

        Parameters
        ----------
        base_result : SimulationResult
            Result from base system simulation.
        enhanced_result : SimulationResult
            Result from enhanced system simulation.

        Returns
        -------
        ComparisonResult
            Comprehensive comparison result.
        """
        # Extract metrics from results
        base_metrics = self._extract_metrics(base_result)
        enhanced_metrics = self._extract_metrics(enhanced_result)

        # Compute improvements
        improvements = self._compute_improvements(base_metrics, enhanced_metrics)

        # Extract returns for statistical testing
        base_returns = self._extract_returns(base_result)
        enhanced_returns = self._extract_returns(enhanced_result)

        # Run statistical tests
        statistical_tests = {}
        if len(base_returns) > 0 and len(enhanced_returns) > 0:
            statistical_tests["returns"] = self.statistical_significance(
                np.array(base_returns), np.array(enhanced_returns)
            )

        # Improvement attribution
        attribution = self.improvement_attribution(base_result, enhanced_result)

        # Generate summary
        summary = self._generate_summary(
            base_metrics, enhanced_metrics, improvements, statistical_tests
        )

        # Visualization data
        visualization_data = self.generate_visualization_data(
            ComparisonResult(
                base_metrics=base_metrics,
                enhanced_metrics=enhanced_metrics,
                improvements=improvements,
            )
        )

        return ComparisonResult(
            base_metrics=base_metrics,
            enhanced_metrics=enhanced_metrics,
            improvements=improvements,
            statistical_tests=statistical_tests,
            improvement_attribution=attribution,
            summary=summary,
            visualization_data=visualization_data,
        )

    def statistical_significance(
        self,
        base_returns: np.ndarray,
        enhanced_returns: np.ndarray,
    ) -> Dict[str, Any]:
        """Test statistical significance of improvement.

        Performs paired t-test and bootstrap confidence interval analysis.

        Parameters
        ----------
        base_returns : np.ndarray
            Returns from base system.
        enhanced_returns : np.ndarray
            Returns from enhanced system.

        Returns
        -------
        dict
            {
                "paired_t_test": {"t_stat": float, "p_value": float, "significant": bool},
                "bootstrap": {"mean_diff": float, "ci_lower": float, "ci_upper": float, "significant": bool}
            }
        """
        result = {
            "paired_t_test": {"t_stat": 0.0, "p_value": 1.0, "significant": False},
            "bootstrap": {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "significant": False},
        }

        if len(base_returns) == 0 or len(enhanced_returns) == 0:
            return result

        base_returns = np.asarray(base_returns)
        enhanced_returns = np.asarray(enhanced_returns)

        min_len = min(len(base_returns), len(enhanced_returns))
        base_returns = base_returns[:min_len]
        enhanced_returns = enhanced_returns[:min_len]

        if min_len < 2:
            return result

        # Paired t-test (manual implementation)
        differences = enhanced_returns - base_returns
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        if std_diff == 0:
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            # Two-tailed p-value using t-distribution approximation
            # Using normal approximation for large n
            if n > 30:
                # Normal approximation
                p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
            else:
                # Simple t-distribution approximation
                p_value = self._t_distribution_pvalue(abs(t_stat), n - 1)

        significant = p_value < 0.05
        result["paired_t_test"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(significant),
        }

        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_means = []

        rng = np.random.default_rng(42)
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = rng.choice(n, size=n, replace=True)
            sample_diff = differences[indices]
            bootstrap_means.append(np.mean(sample_diff))

        bootstrap_means = np.array(bootstrap_means)
        mean_diff_bootstrap = np.mean(bootstrap_means)
        ci_lower = float(np.percentile(bootstrap_means, 2.5))
        ci_upper = float(np.percentile(bootstrap_means, 97.5))

        # Significant if CI doesn't contain 0
        significant_bootstrap = not (ci_lower <= 0 <= ci_upper)

        result["bootstrap"] = {
            "mean_diff": float(mean_diff_bootstrap),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": bool(significant_bootstrap),
        }

        return result

    def improvement_attribution(
        self,
        base_result: Any,
        enhanced_result: Any,
        enhancement_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Attribute improvement to specific enhancements.

        Parameters
        ----------
        base_result : SimulationResult
            Result from base system.
        enhanced_result : SimulationResult
            Result from enhanced system.
        enhancement_names : list of str, optional
            Names of enhancement components.

        Returns
        -------
        dict
            Enhancement name -> contribution percentage.
        """
        if enhancement_names is None:
            enhancement_names = [
                "multimodal",
                "coordination",
                "memory_learning",
                "risk_compliance",
                "explainability",
            ]

        base_metrics = self._extract_metrics(base_result)
        enhanced_metrics = self._extract_metrics(enhanced_result)

        base_return = base_metrics.get("total_return", 0.0)
        enhanced_return = enhanced_metrics.get("total_return", 0.0)

        total_improvement = enhanced_return - base_return

        if total_improvement == 0:
            # No improvement - equal attribution
            equal_share = 100.0 / len(enhancement_names) if enhancement_names else 0.0
            return {name: equal_share for name in enhancement_names}

        # Simple equal attribution model
        # In a real system, this would use ablation studies
        attribution = {}
        share = 100.0 / len(enhancement_names)

        for name in enhancement_names:
            attribution[name] = share

        return attribution

    def generate_comparison_table(self, comparison: ComparisonResult) -> str:
        """Generate formatted text comparison table.

        Parameters
        ----------
        comparison : ComparisonResult
            Comparison result to format.

        Returns
        -------
        str
            Formatted table string.
        """
        # Define key metrics to display
        metrics_to_show = [
            ("Total Return", "total_return"),
            ("Sharpe Ratio", "sharpe_ratio"),
            ("Sortino Ratio", "sortino_ratio"),
            ("Max Drawdown", "max_drawdown"),
            ("Win Rate", "win_rate"),
            ("Profit Factor", "profit_factor"),
        ]

        lines = []
        lines.append("=" * 80)
        lines.append(f"{'Metric':<25} | {'Base':>12} | {'Enhanced':>12} | {'Improvement':>15}")
        lines.append("-" * 80)

        for display_name, metric_key in metrics_to_show:
            base_val = comparison.base_metrics.get(metric_key, 0.0)
            enhanced_val = comparison.enhanced_metrics.get(metric_key, 0.0)
            improvement = comparison.improvements.get(metric_key, 0.0)

            # Format based on metric type
            if metric_key in ["max_drawdown", "total_return", "win_rate"]:
                base_str = f"{base_val:.2%}"
                enhanced_str = f"{enhanced_val:.2%}"
            elif metric_key in ["sharpe_ratio", "sortino_ratio", "profit_factor"]:
                base_str = f"{base_val:.3f}"
                enhanced_str = f"{enhanced_val:.3f}"
            else:
                base_str = f"{base_val:.4f}"
                enhanced_str = f"{enhanced_val:.4f}"

            improvement_str = f"{improvement:+.1f}%"

            lines.append(
                f"{display_name:<25} | {base_str:>12} | {enhanced_str:>12} | {improvement_str:>15}"
            )

        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_visualization_data(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """Generate data suitable for charting.

        Parameters
        ----------
        comparison : ComparisonResult
            Comparison result.

        Returns
        -------
        dict
            {
                "equity_curves": {"base": list, "enhanced": list},
                "drawdown_curves": {"base": list, "enhanced": list},
                "metric_bars": {"metrics": list, "base_values": list, "enhanced_values": list},
                "improvement_pie": {"labels": list, "values": list}
            }
        """
        # Extract equity curves
        base_equity = comparison.base_metrics.get("equity_curve", [1.0])
        enhanced_equity = comparison.enhanced_metrics.get("equity_curve", [1.0])

        # Extract drawdown curves
        base_drawdown = comparison.base_metrics.get("drawdown_curve", [0.0])
        enhanced_drawdown = comparison.enhanced_metrics.get("drawdown_curve", [0.0])

        # Metric bars
        metrics_for_bars = ["sharpe_ratio", "sortino_ratio", "total_return", "win_rate"]
        metric_labels = ["Sharpe", "Sortino", "Return", "Win Rate"]
        base_values = [
            comparison.base_metrics.get(m, 0.0) for m in metrics_for_bars
        ]
        enhanced_values = [
            comparison.enhanced_metrics.get(m, 0.0) for m in metrics_for_bars
        ]

        # Improvement pie
        attribution = comparison.improvement_attribution
        pie_labels = list(attribution.keys())
        pie_values = list(attribution.values())

        return {
            "equity_curves": {
                "base": list(base_equity) if isinstance(base_equity, (list, np.ndarray)) else [1.0],
                "enhanced": list(enhanced_equity) if isinstance(enhanced_equity, (list, np.ndarray)) else [1.0],
            },
            "drawdown_curves": {
                "base": list(base_drawdown) if isinstance(base_drawdown, (list, np.ndarray)) else [0.0],
                "enhanced": list(enhanced_drawdown) if isinstance(enhanced_drawdown, (list, np.ndarray)) else [0.0],
            },
            "metric_bars": {
                "metrics": metric_labels,
                "base_values": base_values,
                "enhanced_values": enhanced_values,
            },
            "improvement_pie": {
                "labels": pie_labels,
                "values": pie_values,
            },
        }

    def _extract_metrics(self, result: Any) -> Dict[str, Any]:
        """Extract metrics from a SimulationResult."""
        if result is None:
            return {}

        # Handle SimulationResult objects
        if hasattr(result, "performance_metrics"):
            metrics = dict(result.performance_metrics)
            metrics["total_return"] = result.total_return_pct / 100 if hasattr(result, "total_return_pct") else 0.0
            return metrics

        # Handle dict-like results
        if isinstance(result, dict):
            return result

        return {}

    def _extract_returns(self, result: Any) -> List[float]:
        """Extract returns from a SimulationResult."""
        if result is None:
            return []

        # Handle SimulationResult with performance metrics
        if hasattr(result, "performance_metrics"):
            metrics = result.performance_metrics
            if isinstance(metrics, dict):
                # Try to get equity curve and compute returns
                equity = metrics.get("equity_curve", [])
                if isinstance(equity, (list, np.ndarray)) and len(equity) > 1:
                    equity = np.array(equity)
                    returns = np.diff(equity) / equity[:-1]
                    return returns.tolist()

        # Handle dict with returns
        if isinstance(result, dict):
            if "returns" in result:
                return list(result["returns"])
            if "equity_curve" in result:
                equity = np.array(result["equity_curve"])
                if len(equity) > 1:
                    returns = np.diff(equity) / equity[:-1]
                    return returns.tolist()

        return []

    def _compute_improvements(
        self,
        base_metrics: Dict[str, Any],
        enhanced_metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute improvement percentages for each metric."""
        improvements = {}

        # Key metrics to compare
        key_metrics = [
            "total_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",  # Lower is better
            "win_rate",
            "profit_factor",
            "calmar_ratio",
        ]

        for metric in key_metrics:
            base_val = base_metrics.get(metric, 0.0)
            enhanced_val = enhanced_metrics.get(metric, 0.0)

            if base_val == 0:
                if enhanced_val != 0:
                    improvements[metric] = 100.0 if enhanced_val > 0 else -100.0
                else:
                    improvements[metric] = 0.0
            else:
                improvement = (enhanced_val - base_val) / abs(base_val) * 100

                # For max_drawdown, improvement is reduction
                if metric == "max_drawdown":
                    improvement = -improvement

                improvements[metric] = float(improvement)

        return improvements

    def _generate_summary(
        self,
        base_metrics: Dict[str, Any],
        enhanced_metrics: Dict[str, Any],
        improvements: Dict[str, float],
        statistical_tests: Dict[str, Any],
    ) -> str:
        """Generate narrative summary of comparison."""
        lines = []

        # Overall assessment
        total_return_improvement = improvements.get("total_return", 0.0)
        sharpe_improvement = improvements.get("sharpe_ratio", 0.0)

        if total_return_improvement > 0:
            lines.append(
                f"The enhanced system outperformed the base system "
                f"with a {total_return_improvement:.1f}% improvement in total return."
            )
        else:
            lines.append(
                f"The enhanced system underperformed the base system "
                f"by {abs(total_return_improvement):.1f}% in total return."
            )

        # Risk-adjusted performance
        if sharpe_improvement > 0:
            lines.append(
                f"Risk-adjusted returns improved with Sharpe ratio increasing "
                f"from {base_metrics.get('sharpe_ratio', 0):.2f} to "
                f"{enhanced_metrics.get('sharpe_ratio', 0):.2f} ({sharpe_improvement:.1f}% improvement)."
            )

        # Statistical significance
        if "returns" in statistical_tests:
            t_test = statistical_tests["returns"].get("paired_t_test", {})
            if t_test.get("significant", False):
                lines.append(
                    "The improvement is statistically significant at the 5% level "
                    f"(p-value: {t_test.get('p_value', 1):.4f})."
                )
            else:
                lines.append(
                    "The improvement is not statistically significant at the 5% level."
                )

        return " ".join(lines)

    def _normal_cdf(self, x: float) -> float:
        """Compute standard normal CDF using approximation."""
        # Abramowitz and Stegun approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / np.sqrt(2)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

        return 0.5 * (1.0 + sign * y)

    def _t_distribution_pvalue(self, t_stat: float, df: int) -> float:
        """Approximate p-value for t-distribution.

        Uses a simple approximation based on the normal distribution
        with a correction for small sample sizes.
        """
        # For large df, t-distribution approaches normal
        if df >= 30:
            return 2 * (1 - self._normal_cdf(t_stat))

        # Simple correction for small df
        # This is a rough approximation
        correction = np.sqrt(df / (df - 2)) if df > 2 else 1.0
        adjusted_t = t_stat / correction

        return 2 * (1 - self._normal_cdf(adjusted_t))
