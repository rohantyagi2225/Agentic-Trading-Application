"""
Sortino ratio: excess return per unit of downside volatility.
"""

import math
from typing import List


def sortino_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sortino ratio from period returns.

    Downside deviation uses only returns below the target (risk-free rate).
    Formula: (mean(R - Rf) / downside_std) * sqrt(periods_per_year)

    :param returns: List of period returns (e.g. daily).
    :param risk_free_rate: Period risk-free rate.
    :param periods_per_year: 252 for daily, 12 for monthly.
    :return: Annualized Sortino; 0.0 if insufficient data or zero downside deviation.
    """
    if not returns or len(returns) < 2:
        return 0.0
    n = len(returns)
    mean_excess = sum(r - risk_free_rate for r in returns) / n
    downside_squared = [
        (r - risk_free_rate) ** 2
        for r in returns
        if r < risk_free_rate
    ]
    if not downside_squared:
        return 0.0
    downside_var = sum(downside_squared) / len(downside_squared)
    if downside_var <= 0:
        return 0.0
    downside_std = math.sqrt(downside_var)
    return float((mean_excess / downside_std) * math.sqrt(periods_per_year))
