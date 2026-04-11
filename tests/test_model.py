"""Tests for model training module."""

import pandas as pd
import numpy as np
import pytest
from src.models.train import time_based_split, evaluate_model, compute_financial_metrics


@pytest.fixture
def sample_featured_data():
    """Create sample featured data for model tests."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Date": dates,
        "Symbol": "TEST",
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "feature_3": np.random.randn(n),
        "daily_return": np.random.randn(n) * 0.02,
        "signal": np.random.randint(0, 2, n),
    })
    return df


def test_time_based_split(sample_featured_data):
    feature_cols = ["feature_1", "feature_2", "feature_3"]
    X_train, y_train, X_val, y_val, X_test, y_test, test_df = time_based_split(
        sample_featured_data, feature_cols, test_size=0.2, val_size=0.1
    )
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(sample_featured_data)
    assert len(X_test) > 0
    assert len(X_val) > 0
    assert len(X_train) > len(X_val)


def test_evaluate_model():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.4, 0.8, 0.2, 0.6, 0.7, 0.3])

    metrics = evaluate_model(y_true, y_pred, y_prob)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_compute_financial_metrics():
    np.random.seed(42)
    test_df = pd.DataFrame({
        "daily_return": np.random.randn(100) * 0.02,
    })
    y_pred = np.random.randint(0, 2, 100)

    metrics = compute_financial_metrics(test_df, y_pred)
    assert "strategy_total_return" in metrics
    assert "buyhold_total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "excess_return" in metrics
