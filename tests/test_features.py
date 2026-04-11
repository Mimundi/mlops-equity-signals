"""Tests for feature engineering module."""

import pandas as pd
import numpy as np
import pytest
from src.features.build_features import (
    compute_returns, compute_momentum, compute_volatility,
    compute_moving_averages, compute_rsi, compute_target,
    build_features, get_feature_columns
)


@pytest.fixture
def sample_ticker_data():
    """Create sample OHLCV data for one ticker."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Date": dates,
        "Symbol": "TEST",
        "Open": prices + np.random.randn(n) * 0.2,
        "High": prices + abs(np.random.randn(n) * 0.5),
        "Low": prices - abs(np.random.randn(n) * 0.5),
        "Close": prices + np.random.randn(n) * 0.1,
        "Adj Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    })
    return df


@pytest.fixture
def sample_config():
    return {
        "features": {
            "tickers": ["TEST"],
            "momentum_windows": [5, 10],
            "volatility_windows": [10],
            "ma_windows": [5, 10],
            "rsi_window": 14,
            "forward_return_days": 5,
            "signal_threshold": 0.0,
        }
    }


def test_compute_returns(sample_ticker_data):
    result = compute_returns(sample_ticker_data)
    assert "daily_return" in result.columns
    assert "log_return" in result.columns
    assert result["daily_return"].iloc[0] != result["daily_return"].iloc[0]  # First is NaN


def test_compute_momentum(sample_ticker_data):
    result = compute_returns(sample_ticker_data)
    result = compute_momentum(result, [5, 10])
    assert "momentum_5d" in result.columns
    assert "momentum_10d" in result.columns


def test_compute_volatility(sample_ticker_data):
    result = compute_returns(sample_ticker_data)
    result = compute_volatility(result, [10])
    assert "volatility_10d" in result.columns
    # Volatility should be non-negative where not NaN
    valid = result["volatility_10d"].dropna()
    assert (valid >= 0).all()


def test_compute_moving_averages(sample_ticker_data):
    result = compute_moving_averages(sample_ticker_data, [5, 10])
    assert "sma_5d" in result.columns
    assert "price_to_sma_5d" in result.columns


def test_compute_rsi(sample_ticker_data):
    result = compute_rsi(sample_ticker_data, window=14)
    assert "rsi_14" in result.columns
    valid = result["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_compute_target(sample_ticker_data):
    result = compute_target(sample_ticker_data, forward_days=5, threshold=0.0)
    assert "forward_return" in result.columns
    assert "signal" in result.columns
    assert set(result["signal"].dropna().unique()).issubset({0, 1})


def test_build_features(sample_ticker_data, sample_config):
    # Build features needs a multi-ticker df with groupby
    df = sample_ticker_data.copy()
    result = build_features(df, sample_config)
    feature_cols = get_feature_columns(result)
    assert len(feature_cols) > 0
    assert "signal" in result.columns
    assert result.isnull().sum().sum() == 0  # No NaNs after build


def test_get_feature_columns(sample_ticker_data, sample_config):
    df = sample_ticker_data.copy()
    result = build_features(df, sample_config)
    feature_cols = get_feature_columns(result)
    # Should not include metadata or target columns
    assert "Date" not in feature_cols
    assert "Symbol" not in feature_cols
    assert "signal" not in feature_cols
    assert "forward_return" not in feature_cols
