import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()

    assert "message" in data
    assert data["health"] == "/health"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_live_signal_success():
    response = client.get("/signals/AAPL")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "success"

    data = body["data"]
    assert "symbol" in data
    assert "price" in data
    assert "signal" in data
    assert "confidence" in data
    assert "explanation" in data