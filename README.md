# MLOps Pipeline for Short-Term Equity Signal Generation and Evaluation

An end-to-end MLOps pipeline that generates, evaluates, and maintains short-term equity trading signals using machine learning. Built with reproducibility, automation, and continuous monitoring at its core.

## Project Overview

Financial markets produce large amounts of data, but deriving reliable and deployable trading signals remains challenging. This project addresses that gap by combining machine learning with modern DevOps/MLOps practices to build a scalable, reproducible pipeline for short-term equity signal generation.

The system ingests S&P 500 stock data, engineers financially meaningful features (momentum, volatility, RSI, moving averages), trains and evaluates multiple ML models, deploys the best model as a REST API, and continuously monitors signal performance and data drift.

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Raw Data    │───▶│  Feature Engine   │───▶│  Model Training │
│  (S&P 500)  │    │  (26+ features)   │    │  (4 models)     │
└─────────────┘    └──────────────────┘    └────────┬────────┘
                                                     │
                           ┌─────────────────────────┘
                           ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Monitoring  │◀──│  FastAPI Service  │◀──│  MLflow Registry│
│  (Drift/Perf)│    │  (/predict)      │    │  (Experiments)  │
└─────────────┘    └──────────────────┘    └─────────────────┘
```

## Tools & Technologies

| Category | Tool |
|---|---|
| Language | Python 3.11 |
| ML Models | Scikit-learn, XGBoost |
| Feature Engineering | Pandas, NumPy |
| Experiment Tracking | MLflow |
| Data/Model Versioning | DVC |
| API Deployment | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Monitoring | Custom drift detection (PSI), Evidently |
| Version Control | Git, GitHub |

## Project Structure

```
mlops-equity-signals/
├── .github/workflows/      # CI/CD pipelines
│   ├── ci.yml              # Lint, test, build
│   └── cd.yml              # Deploy on merge to main
├── configs/
│   └── config.yaml         # Pipeline configuration
├── data/
│   ├── raw/                # Raw S&P 500 CSVs (DVC tracked)
│   ├── processed/          # Feature-engineered data
│   └── predictions/        # Prediction logs for monitoring
├── docs/                   # Additional documentation
├── models/                 # Trained model artifacts
├── notebooks/              # EDA and experimentation
├── src/
│   ├── api/app.py          # FastAPI prediction service
│   ├── data/ingest.py      # Data loading and validation
│   ├── features/build_features.py  # Feature engineering
│   ├── models/train.py     # Training with MLflow
│   └── monitoring/monitor.py  # Drift and performance monitoring
├── tests/                  # Unit and integration tests
├── docker-compose.yml      # Multi-service Docker setup
├── Dockerfile              # API container
├── dvc.yaml                # DVC pipeline stages
├── requirements.txt        # Python dependencies
└── run_pipeline.py         # End-to-end pipeline orchestrator
```

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/<your-username>/mlops-equity-signals.git
cd mlops-equity-signals
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Data

Place the Kaggle S&P 500 dataset CSVs into `data/raw/`:
- `sp500_stocks.csv`
- `sp500_index.csv`
- `sp500_companies.csv`

### 3. Run the Pipeline

```bash
python run_pipeline.py
```

This will:
1. Ingest and validate the raw data
2. Engineer 26+ features per ticker
3. Train 4 models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
4. Log all experiments to MLflow
5. Save the best model to `models/`

### 4. View Experiments in MLflow

```bash
mlflow ui --backend-store-uri mlruns
```

Open http://localhost:5000 to compare model runs.

### 5. Start the API

```bash
uvicorn src.api.app:app --reload
```

Open http://localhost:8000/docs for interactive API documentation.

### 6. Run with Docker

```bash
docker-compose up --build
```

This starts both the prediction API (port 8000) and MLflow UI (port 5000).

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check and model status |
| POST | `/predict` | Generate trading signal |

### Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "open": 150.0,
    "high": 155.0,
    "low": 149.0,
    "close": 153.0,
    "adj_close": 153.0,
    "volume": 5000000,
    "momentum_5d": 0.03,
    "volatility_10d": 0.015,
    "rsi_14": 55.0
  }'
```

## Running Tests

```bash
pytest tests/ -v
```

## CI/CD Pipeline

The GitHub Actions workflows automate:

**CI (on push/PR):** Code formatting check (Black) → Linting (Flake8) → Unit tests (pytest) → Docker build verification

**CD (on merge to main):** Full test suite → Docker image build → API smoke test → (Optional) Push to registry

## Monitoring

The monitoring module tracks:
- **Data drift**: Population Stability Index (PSI) per feature
- **Model performance**: Accuracy, F1, and financial returns over time
- **Retraining alerts**: Triggered when drift exceeds thresholds or accuracy drops

## Dataset

Source: [S&P 500 Stocks (daily updated) — Kaggle](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks)

Three files used:
- `sp500_stocks.csv` — Daily OHLCV data for S&P 500 constituents (2010–2024)
- `sp500_index.csv` — Daily S&P 500 index values (benchmark)
- `sp500_companies.csv` — Company metadata (sector, industry, market cap)

## Author

Michael Imundi

## License

This project is for educational purposes as part of a DevOps/MLOps course project.
