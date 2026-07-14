# Product Recommendation System

> **End-to-End MLOps Product Recommendation System** — A production-quality AI-powered platform
> demonstrating the complete Machine Learning Operations lifecycle.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.18-orange.svg)](https://mlflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-red.svg)](https://scikit-learn.org)

---

## Overview

This system implements a full MLOps lifecycle for product recommendations:

```
Dataset → Validation → Transformation → Feature Engineering
→ Model Training → MLflow Tracking → Evaluation → Registry
→ Prediction API → Responsive Frontend → Admin Dashboard
```

### Recommendation Engines (7)
| Engine | Algorithm |
|--------|-----------|
| Content-Based | TF-IDF + Cosine Similarity |
| Frequently Bought Together | Co-occurrence Matrix |
| Trending | Time-weighted Popularity |
| Popularity | Rating × Review Count Score |
| Category Similarity | Category Path Matching |
| Price Similarity | Price Bucket Distance |
| Brand Similarity | Brand + Category Intersection |
| **Hybrid** | Weighted Ensemble of all above |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your Kaggle credentials (optional for initial setup)
```

### 3. Setup Dataset

**Option A — With Kaggle credentials** (recommended):
```bash
# Add KAGGLE_USERNAME and KAGGLE_KEY to .env first
python scripts/setup_dataset.py
```

**Option B — Development fallback** (no credentials needed):
```bash
python scripts/generate_fallback.py
# Then set use_fallback: true in src/config/config.yaml
```

### 4. Train the Models
```bash
python -m src.pipelines.training_pipeline
```

### 5. Start the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Access the Application
| Service | URL |
|---------|-----|
| Frontend | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Admin Dashboard | http://localhost:8000/admin.html |
| MLflow UI | http://localhost:5000 |

---

## Project Structure

```
Product-Recommendation-System/
├── app/                    FastAPI application
│   ├── api/v1/             REST API routes
│   ├── services/           Business logic layer
│   ├── database/           SQLAlchemy ORM (18 tables)
│   ├── schemas/            Pydantic request/response models
│   ├── middleware/         Logging, rate limiting, monitoring
│   └── main.py             Application factory
│
├── src/                    Core ML & utilities
│   ├── components/         Pipeline components (ingestion → registry)
│   ├── engines/            7 recommendation engines + hybrid
│   ├── monitoring/         Drift detection, system metrics
│   ├── pipelines/          Training & prediction pipelines
│   ├── logger/             7 domain loggers
│   ├── exception/          Custom exception hierarchy
│   ├── config/             YAML config + typed loaders
│   ├── entity/             Config & artifact dataclasses
│   └── utils/              Shared utilities
│
├── frontend/               Responsive web UI
│   ├── index.html          Landing page (10 recommendation sections)
│   ├── product.html        Product detail + XAI panel
│   ├── search.html         Search + typo correction
│   ├── category.html       Category browser
│   └── admin.html          Admin dashboard
│
├── data/                   Dataset directory
│   ├── raw/                Place amazon.csv here
│   ├── processed/          Pipeline output
│   └── external/           sample_products.csv (dev fallback)
│
├── models/                 Trained model artifacts
├── mlruns/                 MLflow experiment data
├── logs/                   Domain-separated log files
├── scripts/                Setup, training, deployment scripts
├── docker/                 Dockerfile + docker-compose
├── tests/                  Unit + integration tests
└── docs/                   API, deployment, MLOps documentation
```

---

## Architecture

```
User Browser
    │
    ▼
FastAPI (port 8000)
    │
    ├── Recommendation Service ──→ 7 Engines ──→ Confidence Scoring ──→ XAI
    ├── Search Service ──────────→ TF-IDF + rapidfuzz typo correction
    ├── Session Service ─────────→ Anonymous personalization
    ├── Analytics Service ───────→ CTR, Conversion, Acceptance tracking
    ├── Training Service ────────→ Background pipeline + MLflow
    └── Monitoring Service ──────→ Drift, latency, system metrics
    │
    └── SQLite (18 tables) ←──────────────────────────────────────────┘
```

---

## API Reference

Base URL: `http://localhost:8000/api/v1`

| Category | Endpoints |
|----------|-----------|
| Products | `GET /products`, `GET /products/{id}`, `GET /categories`, `GET /search` |
| Recommendations | `GET /recommendations/similar/{id}`, `/fbt/{id}`, `/trending`, `/top-rated`, etc. |
| Explain | `GET /recommendations/explain/{source_id}/{target_id}` |
| Training | `POST /training/run`, `GET /training/status`, `GET /experiments` |
| Admin | `POST /admin/strategy`, `POST /admin/clear-cache`, `POST /admin/recompute-similarity` |
| Monitoring | `GET /monitoring/latency`, `/cache`, `/drift`, `/recommendations` |
| Analytics | `GET /analytics/ctr`, `/most-clicked`, `/conversion-rate` |
| Health | `GET /health`, `/health/live`, `/health/ready` |

Full documentation: [docs/API.md](docs/API.md) · Swagger: http://localhost:8000/docs

---

## MLOps Workflow

```
Dataset ──→ Validate ──→ Transform ──→ Feature Eng
    ↓
Train (TF-IDF + 7 Engines)
    ↓
MLflow Tracking (params, metrics, artifacts)
    ↓
Evaluate (Precision@K, Coverage, Diversity, Novelty)
    ↓
Model Registry (Staging → Production via Admin)
    ↓
Prediction API ──→ Caching ──→ Monitoring ──→ Drift Detection
```

---

## Deployment

### Docker
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Manual
```bash
# Train first
python -m src.pipelines.training_pipeline

# Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Start MLflow UI (separate terminal)
mlflow ui --host 0.0.0.0 --port 5000
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full deployment guide.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| ML | scikit-learn, NumPy, Pandas |
| Experiment Tracking | MLflow |
| Backend | FastAPI + Uvicorn |
| Database | SQLite + SQLAlchemy |
| Search | rapidfuzz |
| Frontend | HTML5 + CSS3 + JavaScript |
| Charts | Plotly.js |
| Rate Limiting | SlowAPI |
| Monitoring | psutil + custom trackers |
| Containerization | Docker + Docker Compose |
| Testing | pytest + pytest-asyncio |

---

## License

This project is developed for academic and educational purposes.
