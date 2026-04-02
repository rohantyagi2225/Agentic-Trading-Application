"""Research Evaluation Framework.

This module provides research-grade financial and AI metrics, comparison engine,
and benchmark suite for evaluating trading agent systems.

Components
----------
FinancialMetricsCalculator
    Extended financial metrics including Calmar, Information, and Omega ratios.
AIMetricsCalculator
    AI/agent-specific metrics including decision accuracy and calibration.
ComparisonEngine
    Base vs Enhanced system comparison with statistical testing.
BenchmarkSuite
    Standard benchmark scenarios for systematic evaluation.
"""

from FinAgents.research.evaluation.financial_metrics import (
    FinancialMetricsReport,
    FinancialMetricsCalculator,
)
from FinAgents.research.evaluation.ai_metrics import (
    AIMetricsReport,
    AIMetricsCalculator,
)
from FinAgents.research.evaluation.comparison_engine import (
    ComparisonResult,
    ComparisonEngine,
)
from FinAgents.research.evaluation.benchmark_suite import (
    BenchmarkScenario,
    BenchmarkReport,
    BenchmarkSuite,
)

__all__ = [
    # Financial metrics
    "FinancialMetricsReport",
    "FinancialMetricsCalculator",
    # AI metrics
    "AIMetricsReport",
    "AIMetricsCalculator",
    # Comparison engine
    "ComparisonResult",
    "ComparisonEngine",
    # Benchmark suite
    "BenchmarkScenario",
    "BenchmarkReport",
    "BenchmarkSuite",
]
