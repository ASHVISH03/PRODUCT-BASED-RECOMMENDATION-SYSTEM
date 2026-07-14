# ============================================================
# Product Recommendation System — Makefile
# Developer convenience commands for Windows (PowerShell)
# ============================================================

.PHONY: help install setup-dataset train start test lint format clean docker-up docker-down

help:
	@echo "Product Recommendation System — Available Commands"
	@echo "=================================================="
	@echo "  make install         Install all dependencies"
	@echo "  make setup-dataset   Download Kaggle dataset (requires .env)"
	@echo "  make generate-data   Generate synthetic fallback dataset"
	@echo "  make train           Run full training pipeline"
	@echo "  make start           Start FastAPI server (development)"
	@echo "  make test            Run all tests"
	@echo "  make lint            Run linting checks"
	@echo "  make format          Format code with black + isort"
	@echo "  make clean           Remove generated artifacts"
	@echo "  make docker-up       Start full stack with Docker Compose"
	@echo "  make docker-down     Stop Docker Compose stack"

install:
	pip install -r requirements.txt
	pip install -e .

setup-dataset:
	python scripts/setup_dataset.py

generate-data:
	python scripts/generate_fallback.py

train:
	python -m src.pipelines.training_pipeline

start:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

start-prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	flake8 src/ app/ --max-line-length=100 --exclude=venv
	mypy src/ app/ --ignore-missing-imports

format:
	black src/ app/ scripts/ --line-length=100
	isort src/ app/ scripts/ --profile=black

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null; true

clean-models:
	rm -f models/*.pkl models/*.joblib models/metadata.json

clean-data:
	rm -rf data/processed/ data/interim/

clean-logs:
	rm -rf logs/*.log

clean-all: clean clean-models clean-data clean-logs

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f api

init-git:
	git init
	git add .
	git commit -m "chore: initial project structure"
