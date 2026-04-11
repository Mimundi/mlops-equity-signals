"""Tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data


def test_predict_no_model(client):
    """Test prediction endpoint when no model is loaded."""
    payload = {
        "symbol": "AAPL",
        "open": 150.0,
        "high": 155.0,
        "low": 149.0,
        "close": 153.0,
        "adj_close": 153.0,
        "volume": 5000000.0,
    }
    response = client.post("/predict", json=payload)
    # Should return 503 since model isn't loaded in test
    assert response.status_code == 503
