"""
Max drawdown: largest peak-to-trough decline in cumulative returns.
"""

from typing import List


def max_drawdown(returns: List[float]) -> float:
    """
    Maximum drawdown from period returns.

    Builds cumulative wealth (1 + r1)(1 + r2)... and tracks (peak - current) / peak.
    Returns a negative number (e.g. -0.15 for 15% drawdown).

    :param returns: List of period returns (e.g. daily).
    :return: Max drawdown as a decimal; 0.0 if no returns or no drawdown.
    """
    if not returns:
        return 0.0
    peak = 1.0
    max_dd = 0.0
    wealth = 1.0
    for r in returns:
        wealth *= 1.0 + r
        if wealth > peak:
            peak = wealth
        if peak > 0:
            dd = (peak - wealth) / peak
            if dd > max_dd:
                max_dd = dd
    return float(-max_dd)
