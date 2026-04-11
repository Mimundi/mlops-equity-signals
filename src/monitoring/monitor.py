"""
Monitoring module.
Tracks prediction distributions, data drift, and signal performance.
Generates monitoring reports and alerts.
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalMonitor:
    """Monitors deployed model performance and data drift."""

    def __init__(self, config: dict):
        self.drift_threshold = config["monitoring"]["drift_threshold"]
        self.accuracy_threshold = config["monitoring"]["accuracy_threshold"]
        self.log_dir = Path(config["monitoring"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_prediction(self, symbol: str, features: dict,
                       prediction: int, confidence: float):
        """Log a single prediction for later analysis."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "prediction": prediction,
            "confidence": confidence,
            **features,
        }
        log_file = self.log_dir / "prediction_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def compute_drift(self, reference_df: pd.DataFrame,
                      current_df: pd.DataFrame,
                      feature_cols: list) -> dict:
        """
        Compute feature drift between reference (training) and current data
        using Population Stability Index (PSI).
        """
        drift_report = {}
        for col in feature_cols:
            if col not in reference_df.columns or col not in current_df.columns:
                continue
            psi = self._psi(reference_df[col].dropna(), current_df[col].dropna())
            drift_report[col] = {
                "psi": round(psi, 6),
                "drifted": psi > self.drift_threshold,
                "ref_mean": round(reference_df[col].mean(), 4),
                "cur_mean": round(current_df[col].mean(), 4),
            }
        drifted_features = [k for k, v in drift_report.items() if v["drifted"]]
        logger.info(
            f"Drift check: {len(drifted_features)}/{len(feature_cols)} "
            f"features drifted (threshold={self.drift_threshold})"
        )
        return drift_report

    def _psi(self, reference: pd.Series, current: pd.Series,
             n_bins: int = 10) -> float:
        """Compute Population Stability Index between two distributions."""
        eps = 1e-6
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            n_bins + 1
        )
        ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference) + eps
        cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current) + eps
        psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))
        return psi

    def evaluate_signal_performance(self, predictions_df: pd.DataFrame) -> dict:
        """Evaluate trading signal performance over a monitoring window."""
        if "actual_signal" not in predictions_df.columns:
            return {"error": "No actual outcomes available yet"}

        accuracy = accuracy_score(
            predictions_df["actual_signal"], predictions_df["prediction"]
        )
        f1 = f1_score(
            predictions_df["actual_signal"], predictions_df["prediction"],
            zero_division=0
        )

        # Signal-based returns
        if "daily_return" in predictions_df.columns:
            strategy_return = (
                predictions_df["prediction"] * predictions_df["daily_return"]
            ).sum()
            buyhold_return = predictions_df["daily_return"].sum()
        else:
            strategy_return = None
            buyhold_return = None

        report = {
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "accuracy_alert": accuracy < self.accuracy_threshold,
            "strategy_return": round(strategy_return, 4) if strategy_return else None,
            "buyhold_return": round(buyhold_return, 4) if buyhold_return else None,
            "n_predictions": len(predictions_df),
            "signal_distribution": predictions_df["prediction"].value_counts().to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        if report["accuracy_alert"]:
            logger.warning(
                f"ALERT: Model accuracy {accuracy:.4f} below "
                f"threshold {self.accuracy_threshold}"
            )

        return report

    def generate_monitoring_report(self, drift_report: dict,
                                   performance_report: dict) -> dict:
        """Combine drift and performance into a full monitoring report."""
        drifted = [k for k, v in drift_report.items()
                   if isinstance(v, dict) and v.get("drifted")]
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_summary": {
                "total_features_checked": len(drift_report),
                "features_drifted": len(drifted),
                "drifted_features": drifted,
            },
            "performance": performance_report,
            "needs_retraining": (
                len(drifted) > len(drift_report) * 0.3
                or performance_report.get("accuracy_alert", False)
            ),
        }


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    monitor = SignalMonitor(config)
    print("Monitor initialized. Ready for drift checks and performance evaluation.")
