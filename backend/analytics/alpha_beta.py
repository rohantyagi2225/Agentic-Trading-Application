"""
Alpha and Beta vs a benchmark (CAPM-style).
"""

import math
from typing import List, Tuple


def alpha_beta(
    returns: List[float],
    benchmark_returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Tuple[float, float]:
    """
    CAPM alpha and beta from period returns vs benchmark.

    Beta = Cov(R, R_b) / Var(R_b)
    Alpha (period) = mean(R) - Rf - beta * (mean(R_b) - Rf); then annualized.

    :param returns: Portfolio period returns.
    :param benchmark_returns: Benchmark period returns (same length as returns).
    :param risk_free_rate: Period risk-free rate.
    :param periods_per_year: For annualizing alpha (e.g. 252 for daily).
    :return: (alpha_annualized, beta). Alpha in decimal; beta dimensionless.
    """
    if not returns or not benchmark_returns or len(returns) != len(benchmark_returns):
        return 0.0, 0.0
    n = len(returns)
    if n < 2:
        return 0.0, 0.0
    mean_r = sum(returns) / n
    mean_b = sum(benchmark_returns) / n
    cov = sum((r - mean_r) * (b - mean_b) for r, b in zip(returns, benchmark_returns)) / (n - 1)
    var_b = sum((b - mean_b) ** 2 for b in benchmark_returns) / (n - 1)
    if var_b <= 0:
        return 0.0, 0.0
    beta = cov / var_b
    alpha_period = (mean_r - risk_free_rate) - beta * (mean_b - risk_free_rate)
    alpha_annual = alpha_period * periods_per_year
    return float(alpha_annual), float(beta)
