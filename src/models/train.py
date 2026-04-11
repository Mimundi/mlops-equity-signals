"""
Model training module.
Trains baseline and advanced models, logs experiments to MLflow,
and selects the best model based on evaluation metrics.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def time_based_split(df: pd.DataFrame, feature_cols: list,
                     test_size: float = 0.2, val_size: float = 0.1):
    """
    Split data chronologically to avoid look-ahead bias.
    Returns train, validation, and test sets.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - val_size))

    train = df.iloc[:val_idx]
    val = df.iloc[val_idx:test_idx]
    test = df.iloc[test_idx:]

    X_train, y_train = train[feature_cols], train["signal"]
    X_val, y_val = val[feature_cols], val["signal"]
    X_test, y_test = test[feature_cols], test["signal"]

    logger.info(
        f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, test


def evaluate_model(y_true, y_pred, y_prob=None) -> dict:
    """Compute classification and financial evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0
    return metrics


def compute_financial_metrics(test_df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """
    Compute financial performance metrics for the trading signal.
    Compares signal returns against buy-and-hold benchmark.
    """
    test = test_df.copy()
    test["predicted_signal"] = y_pred
    test["strategy_return"] = test["predicted_signal"] * test["daily_return"]
    test["cumulative_strategy"] = (1 + test["strategy_return"]).cumprod()
    test["cumulative_buyhold"] = (1 + test["daily_return"]).cumprod()

    strategy_total = test["cumulative_strategy"].iloc[-1] - 1
    buyhold_total = test["cumulative_buyhold"].iloc[-1] - 1

    # Annualized Sharpe ratio (approx 252 trading days)
    if test["strategy_return"].std() > 0:
        sharpe = (
            test["strategy_return"].mean() / test["strategy_return"].std()
        ) * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        "strategy_total_return": round(strategy_total, 4),
        "buyhold_total_return": round(buyhold_total, 4),
        "excess_return": round(strategy_total - buyhold_total, 4),
        "sharpe_ratio": round(sharpe, 4),
    }


def get_models(config: dict) -> dict:
    """Define the model zoo to train and compare."""
    xgb_params = config["model"]["xgb_params"]
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=config["model"]["random_state"]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            random_state=config["model"]["random_state"], n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, random_state=config["model"]["random_state"]
        ),
        "xgboost": XGBClassifier(
            **xgb_params,
            random_state=config["model"]["random_state"],
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1
        ),
    }


def train_and_log(config: dict, df: pd.DataFrame, feature_cols: list) -> str:
    """
    Train all models, log to MLflow, and return the best model path.
    """
    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test, test_df = time_based_split(
        df, feature_cols,
        test_size=config["model"]["test_size"],
        val_size=config["model"]["val_size"]
    )

    models = get_models(config)
    best_model = None
    best_f1 = -1
    best_model_name = ""

    for name, model in models.items():
        logger.info(f"Training {name}...")
        with mlflow.start_run(run_name=name):
            # Train
            model.fit(X_train, y_train)

            # Predict on validation
            val_pred = model.predict(X_val)
            val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
            val_metrics = evaluate_model(y_val, val_pred, val_prob)

            # Predict on test
            test_pred = model.predict(X_test)
            test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            test_metrics = evaluate_model(y_test, test_pred, test_prob)

            # Financial metrics
            fin_metrics = compute_financial_metrics(test_df, test_pred)

            # Log parameters
            mlflow.log_params({
                "model_type": name,
                "n_features": len(feature_cols),
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
            })

            # Log validation metrics
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)

            # Log test metrics
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            # Log financial metrics
            for k, v in fin_metrics.items():
                mlflow.log_metric(k, v)

            # Log model
            if name == "xgboost":
                mlflow.xgboost.log_model(model, name)
            else:
                mlflow.sklearn.log_model(model, name)

            logger.info(
                f"{name} - Val F1: {val_metrics['f1']:.4f}, "
                f"Test F1: {test_metrics['f1']:.4f}, "
                f"Sharpe: {fin_metrics['sharpe_ratio']:.4f}"
            )

            # Track best model by validation F1
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_model = model
                best_model_name = name

    # Save best model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)

    # Save feature columns for inference
    joblib.dump(feature_cols, model_dir / "feature_columns.joblib")

    logger.info(f"\nBest model: {best_model_name} (Val F1: {best_f1:.4f})")
    logger.info(f"Model saved to {model_path}")

    return str(model_path)


if __name__ == "__main__":
    import yaml
    from src.data.ingest import load_stock_data, filter_tickers
    from src.features.build_features import build_features, get_feature_columns

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    stocks = load_stock_data(config["data"]["raw_stocks"])
    stocks = filter_tickers(stocks, config["features"]["tickers"])
    featured = build_features(stocks, config)
    feature_cols = get_feature_columns(featured)

    best_path = train_and_log(config, featured, feature_cols)
    print(f"\nBest model saved to: {best_path}")
