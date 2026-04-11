"""
Feature engineering module.
Builds financially meaningful features: momentum, volatility,
moving averages, RSI, and return-based indicators.
Defines the target variable (short-term buy/sell signal).
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_returns(group: pd.DataFrame) -> pd.DataFrame:
    """Compute daily and log returns."""
    group = group.copy()
    group["daily_return"] = group["Adj Close"].pct_change()
    group["log_return"] = np.log(group["Adj Close"] / group["Adj Close"].shift(1))
    return group


def compute_momentum(group: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Compute momentum as cumulative return over lookback windows."""
    group = group.copy()
    for w in windows:
        group[f"momentum_{w}d"] = group["Adj Close"].pct_change(periods=w)
    return group


def compute_volatility(group: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Compute rolling volatility (std of daily returns)."""
    group = group.copy()
    for w in windows:
        group[f"volatility_{w}d"] = group["daily_return"].rolling(window=w).std()
    return group


def compute_moving_averages(group: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Compute simple moving averages and price-to-MA ratios."""
    group = group.copy()
    for w in windows:
        group[f"sma_{w}d"] = group["Adj Close"].rolling(window=w).mean()
        group[f"price_to_sma_{w}d"] = group["Adj Close"] / group[f"sma_{w}d"]
    return group


def compute_rsi(group: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index."""
    group = group.copy()
    delta = group["Adj Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    group[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return group


def compute_volume_features(group: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based features."""
    group = group.copy()
    group["volume_sma_10"] = group["Volume"].rolling(window=10).mean()
    group["volume_ratio"] = group["Volume"] / group["volume_sma_10"]
    group["volume_change"] = group["Volume"].pct_change()
    return group


def compute_price_features(group: pd.DataFrame) -> pd.DataFrame:
    """Compute additional price-based features."""
    group = group.copy()
    group["high_low_range"] = (group["High"] - group["Low"]) / group["Close"]
    group["close_open_range"] = (group["Close"] - group["Open"]) / group["Open"]
    group["upper_shadow"] = (group["High"] - group[["Open", "Close"]].max(axis=1)) / group["Close"]
    group["lower_shadow"] = (group[["Open", "Close"]].min(axis=1) - group["Low"]) / group["Close"]
    return group


def compute_target(group: pd.DataFrame, forward_days: int = 5,
                   threshold: float = 0.0) -> pd.DataFrame:
    """
    Compute target variable: 1 if forward return > threshold, else 0.
    This is a binary buy/sell signal.
    """
    group = group.copy()
    group["forward_return"] = (
        group["Adj Close"].shift(-forward_days) / group["Adj Close"] - 1
    )
    group["signal"] = (group["forward_return"] > threshold).astype(int)
    return group


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Applies all feature computations per ticker, then drops NaN rows.
    """
    logger.info("Starting feature engineering...")
    feat_cfg = config["features"]

    results = []
    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values("Date").copy()

        # Core features
        group = compute_returns(group)
        group = compute_momentum(group, feat_cfg["momentum_windows"])
        group = compute_volatility(group, feat_cfg["volatility_windows"])
        group = compute_moving_averages(group, feat_cfg["ma_windows"])
        group = compute_rsi(group, feat_cfg["rsi_window"])
        group = compute_volume_features(group)
        group = compute_price_features(group)

        # Target
        group = compute_target(
            group,
            forward_days=feat_cfg["forward_return_days"],
            threshold=feat_cfg["signal_threshold"]
        )
        results.append(group)

    featured = pd.concat(results, ignore_index=True)

    # Drop rows with NaN from rolling calculations and forward target
    pre_drop = len(featured)
    featured = featured.dropna().reset_index(drop=True)
    logger.info(
        f"Feature engineering complete. "
        f"Rows: {pre_drop} -> {len(featured)} (dropped {pre_drop - len(featured)} NaN rows). "
        f"Features: {len(get_feature_columns(featured))}"
    )
    return featured


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature column names (excludes metadata and target)."""
    exclude = [
        "Date", "Symbol", "Open", "High", "Low", "Close", "Adj Close",
        "Volume", "forward_return", "signal"
    ]
    return [c for c in df.columns if c not in exclude]


def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save the processed feature dataset."""
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    import yaml
    from src.data.ingest import load_stock_data, filter_tickers

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    stocks = load_stock_data(config["data"]["raw_stocks"])
    stocks = filter_tickers(stocks, config["features"]["tickers"])
    featured = build_features(stocks, config)

    print(f"\nFeature columns ({len(get_feature_columns(featured))}):")
    for col in get_feature_columns(featured):
        print(f"  {col}")

    print(f"\nTarget distribution:\n{featured['signal'].value_counts(normalize=True)}")

    save_processed_data(featured, f"{config['data']['processed_dir']}/featured_data.csv")
