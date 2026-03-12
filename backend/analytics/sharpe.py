"""
Sharpe ratio: risk-adjusted return (excess return over risk-free per unit of volatility).
"""

from typing import List


def sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe ratio from period returns (e.g. daily).

    Formula: (mean(R - Rf) / std(R)) * sqrt(periods_per_year)

    :param returns: List of period returns (e.g. 0.01 for 1%).
    :param risk_free_rate: Period risk-free rate (same frequency as returns).
    :param periods_per_year: 252 for daily, 12 for monthly.
    :return: Annualized Sharpe; 0.0 if insufficient data or zero volatility.
    """
    if not returns or len(returns) < 2:
        return 0.0
    n = len(returns)
    mean_excess = sum(r - risk_free_rate for r in returns) / n
    variance = sum((r - risk_free_rate - mean_excess) ** 2 for r in returns) / (n - 1)
    if variance <= 0:
        return 0.0
    import math
    std = math.sqrt(variance)
    return float((mean_excess / std) * math.sqrt(periods_per_year))
