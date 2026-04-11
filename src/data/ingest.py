"""
Data ingestion module.
Loads raw S&P 500 stock data, index data, and company metadata.
Performs initial validation and cleaning.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_stock_data(filepath: str) -> pd.DataFrame:
    """Load and validate the S&P 500 stock OHLCV data."""
    logger.info(f"Loading stock data from {filepath}")
    df = pd.read_csv(filepath, parse_dates=["Date"])

    # Drop rows where all OHLCV values are missing
    ohlcv_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df.dropna(subset=ohlcv_cols, how="all")

    # Sort by symbol and date
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    logger.info(
        f"Loaded {len(df)} rows, {df['Symbol'].nunique()} tickers, "
        f"date range: {df['Date'].min()} to {df['Date'].max()}"
    )
    return df


def load_index_data(filepath: str) -> pd.DataFrame:
    """Load the S&P 500 index benchmark data."""
    logger.info(f"Loading index data from {filepath}")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={"S&P500": "sp500"})
    logger.info(f"Loaded {len(df)} index data points")
    return df


def load_company_data(filepath: str) -> pd.DataFrame:
    """Load S&P 500 company metadata."""
    logger.info(f"Loading company metadata from {filepath}")
    df = pd.read_csv(filepath)
    # Keep relevant columns
    cols = [
        "Symbol", "Shortname", "Sector", "Industry",
        "Currentprice", "Marketcap", "Weight"
    ]
    df = df[[c for c in cols if c in df.columns]]
    logger.info(f"Loaded metadata for {len(df)} companies")
    return df


def filter_tickers(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Filter stock data to a specific set of tickers."""
    if tickers is None:
        return df
    filtered = df[df["Symbol"].isin(tickers)].copy()
    logger.info(f"Filtered to {filtered['Symbol'].nunique()} tickers: {tickers}")
    return filtered


def validate_data(df: pd.DataFrame) -> dict:
    """Run basic data quality checks and return a report."""
    report = {
        "total_rows": len(df),
        "tickers": df["Symbol"].nunique(),
        "date_range": (str(df["Date"].min()), str(df["Date"].max())),
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicate_rows": df.duplicated(subset=["Date", "Symbol"]).sum(),
    }
    logger.info(f"Validation: {report['total_rows']} rows, {report['tickers']} tickers, "
                f"{report['duplicate_rows']} duplicates")
    return report


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    stocks = load_stock_data(config["data"]["raw_stocks"])
    index = load_index_data(config["data"]["raw_index"])
    companies = load_company_data(config["data"]["raw_companies"])

    stocks = filter_tickers(stocks, config["features"]["tickers"])
    report = validate_data(stocks)
    print("\nData Validation Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
