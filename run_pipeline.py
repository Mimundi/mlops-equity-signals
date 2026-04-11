"""
Pipeline orchestrator.
Runs the full MLOps pipeline end-to-end:
  1. Data ingestion
  2. Feature engineering
  3. Model training & evaluation (with MLflow)
  4. Save artifacts
"""

import yaml
import logging
import argparse
from pathlib import Path

from src.data.ingest import load_stock_data, load_index_data, load_company_data, filter_tickers, validate_data
from src.features.build_features import build_features, get_feature_columns, save_processed_data
from src.models.train import train_and_log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str = "configs/config.yaml", skip_training: bool = False):
    """Execute the full pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING MLOPS EQUITY SIGNAL PIPELINE")
    logger.info("=" * 60)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- Step 1: Data Ingestion ---
    logger.info("\n--- STEP 1: Data Ingestion ---")
    stocks = load_stock_data(config["data"]["raw_stocks"])
    index = load_index_data(config["data"]["raw_index"])
    companies = load_company_data(config["data"]["raw_companies"])

    # Filter to selected tickers
    stocks = filter_tickers(stocks, config["features"]["tickers"])

    # Validate
    report = validate_data(stocks)
    logger.info(f"Validation report: {report}")

    # --- Step 2: Feature Engineering ---
    logger.info("\n--- STEP 2: Feature Engineering ---")
    featured = build_features(stocks, config)
    feature_cols = get_feature_columns(featured)
    logger.info(f"Generated {len(feature_cols)} features")

    # Save processed data
    processed_path = Path(config["data"]["processed_dir"])
    processed_path.mkdir(parents=True, exist_ok=True)
    save_processed_data(featured, str(processed_path / "featured_data.csv"))

    if skip_training:
        logger.info("Skipping training (--skip-training flag set)")
        return

    # --- Step 3: Model Training & Evaluation ---
    logger.info("\n--- STEP 3: Model Training & Evaluation ---")
    best_model_path = train_and_log(config, featured, feature_cols)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Best model: {best_model_path}")
    logger.info(f"Processed data: {processed_path / 'featured_data.csv'}")
    logger.info("=" * 60)

    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MLOps equity signal pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, skip_training=args.skip_training)
