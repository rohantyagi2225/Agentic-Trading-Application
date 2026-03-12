"""
Volatility: annualized standard deviation of period returns.
"""

import math
from typing import List


def annualized_volatility(
    returns: List[float],
    periods_per_year: int = 252,
) -> float:
    """
    Annualized volatility (standard deviation) from period returns.

    :param returns: List of period returns (e.g. daily).
    :param periods_per_year: 252 for daily, 12 for monthly.
    :return: Annualized volatility; 0.0 if insufficient data or zero variance.
    """
    if not returns or len(returns) < 2:
        return 0.0
    n = len(returns)
    mean_ret = sum(returns) / n
    variance = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
    if variance <= 0:
        return 0.0
    return float(math.sqrt(variance * periods_per_year))
