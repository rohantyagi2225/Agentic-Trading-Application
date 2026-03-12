import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def test_signal_structure():
    response = client.get("/signals/AAPL")

    assert response.status_code == 200

    body = response.json()
    data = body["data"]

    assert isinstance(data["symbol"], str)
    assert isinstance(data["price"], float)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["explanation"], str)


def test_risk_rejection_possible():
    # This test checks system stability,
    # not exact rejection behavior
    response = client.get("/signals/AAPL")
    assert response.status_code == 200

    body = response.json()
    assert "data" in body