"""Tests for data ingestion module."""

import pandas as pd
import numpy as np
import pytest
from src.data.ingest import load_stock_data, filter_tickers, validate_data


@pytest.fixture
def sample_stock_data(tmp_path):
    """Create a small sample stock CSV for testing."""
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    data = []
    for symbol in ["AAPL", "MSFT"]:
        for date in dates:
            data.append({
                "Date": date,
                "Symbol": symbol,
                "Open": np.random.uniform(100, 200),
                "High": np.random.uniform(100, 200),
                "Low": np.random.uniform(100, 200),
                "Close": np.random.uniform(100, 200),
                "Adj Close": np.random.uniform(100, 200),
                "Volume": np.random.randint(1000000, 10000000),
            })
    df = pd.DataFrame(data)
    filepath = tmp_path / "test_stocks.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


def test_load_stock_data(sample_stock_data):
    df = load_stock_data(sample_stock_data)
    assert len(df) == 40
    assert "Date" in df.columns
    assert "Symbol" in df.columns
    assert df["Symbol"].nunique() == 2


def test_filter_tickers(sample_stock_data):
    df = load_stock_data(sample_stock_data)
    filtered = filter_tickers(df, ["AAPL"])
    assert filtered["Symbol"].nunique() == 1
    assert filtered["Symbol"].iloc[0] == "AAPL"


def test_filter_tickers_none(sample_stock_data):
    df = load_stock_data(sample_stock_data)
    result = filter_tickers(df, None)
    assert len(result) == len(df)


def test_validate_data(sample_stock_data):
    df = load_stock_data(sample_stock_data)
    report = validate_data(df)
    assert report["total_rows"] == 40
    assert report["tickers"] == 2
    assert report["duplicate_rows"] == 0
