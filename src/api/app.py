"""
FastAPI application for serving equity trading signals.
Exposes endpoints for health checks and signal prediction.
"""

import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Equity Signal API",
    description="MLOps pipeline for short-term equity signal generation and evaluation",
    version="1.0.0",
)

# Load model and feature columns at startup
MODEL_PATH = Path("models/best_model.joblib")
FEATURES_PATH = Path("models/feature_columns.joblib")

model = None
feature_columns = None


@app.on_event("startup")
def load_model():
    global model, feature_columns
    if MODEL_PATH.exists() and FEATURES_PATH.exists():
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(FEATURES_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
        logger.info(f"Features: {len(feature_columns)} columns")
    else:
        logger.warning("Model files not found. Train the model first.")


# --- Request / Response Schemas ---

class StockInput(BaseModel):
    """Input schema: raw OHLCV data for a single ticker."""
    symbol: str = Field(..., description="Ticker symbol", example="AAPL")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    adj_close: float = Field(..., description="Adjusted close price")
    volume: float = Field(..., description="Trading volume")
    # Pre-computed features (optional: API can accept raw or pre-computed)
    daily_return: Optional[float] = None
    log_return: Optional[float] = None
    momentum_5d: Optional[float] = None
    momentum_10d: Optional[float] = None
    momentum_21d: Optional[float] = None
    volatility_10d: Optional[float] = None
    volatility_21d: Optional[float] = None
    sma_5d: Optional[float] = None
    sma_10d: Optional[float] = None
    sma_21d: Optional[float] = None
    sma_50d: Optional[float] = None
    price_to_sma_5d: Optional[float] = None
    price_to_sma_10d: Optional[float] = None
    price_to_sma_21d: Optional[float] = None
    price_to_sma_50d: Optional[float] = None
    rsi_14: Optional[float] = None
    volume_sma_10: Optional[float] = None
    volume_ratio: Optional[float] = None
    volume_change: Optional[float] = None
    high_low_range: Optional[float] = None
    close_open_range: Optional[float] = None
    upper_shadow: Optional[float] = None
    lower_shadow: Optional[float] = None


class PredictionResponse(BaseModel):
    """Output schema: predicted signal and confidence."""
    symbol: str
    signal: int = Field(..., description="1 = Buy, 0 = Sell/Hold")
    confidence: float = Field(..., description="Model confidence (probability)")
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
    timestamp: str


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy" if model is not None else "no_model",
        model_loaded=model is not None,
        n_features=len(feature_columns) if feature_columns else 0,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: StockInput):
    """Generate a trading signal for the given stock data."""
    if model is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build feature vector from input
        input_dict = input_data.model_dump()
        feature_values = {}
        for col in feature_columns:
            if col in input_dict and input_dict[col] is not None:
                feature_values[col] = input_dict[col]
            else:
                feature_values[col] = 0.0  # Default for missing features

        features_df = pd.DataFrame([feature_values])[feature_columns]

        # Predict
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0]

        return PredictionResponse(
            symbol=input_data.symbol,
            signal=int(prediction),
            confidence=round(float(probability[int(prediction)]), 4),
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "Equity Signal API - MLOps Pipeline",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
