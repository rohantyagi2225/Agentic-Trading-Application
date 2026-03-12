import pytest
from backend.risk.risk_engine import RiskEngine


def test_risk_rejects_large_position():
    risk = RiskEngine()

    portfolio_value = 100000
    current_exposure = 0
    position_size = 200000  # intentionally too large

    approved, reason = risk.validate_trade(
        portfolio_value,
        current_exposure,
        position_size,
    )

    assert approved is False
    assert isinstance(reason, str)


def test_risk_allows_small_position():
    risk = RiskEngine()

    portfolio_value = 100000
    current_exposure = 0
    position_size = 1000

    approved, reason = risk.validate_trade(
        portfolio_value,
        current_exposure,
        position_size,
    )

    assert approved is True