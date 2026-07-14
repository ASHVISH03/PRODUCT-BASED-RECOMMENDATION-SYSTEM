# ============================================================
# Multi-Stage Dockerfile
# Stage 1: builder — installs dependencies
# Stage 2: runtime — lean production image
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.13-slim AS runtime

# Metadata
LABEL maintainer="Product Recommendation System"
LABEL version="1.0.0"
LABEL description="End-to-End MLOps Product Recommendation System"

# Create non-root user for security
RUN groupadd -r prs && useradd -r -g prs -d /app -s /bin/bash prs

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY setup.py .
COPY pyproject.toml .

# Create required directories with correct permissions
RUN mkdir -p \
    data/raw \
    data/processed \
    data/interim \
    data/external \
    models \
    artifacts \
    mlruns \
    logs \
    && chown -R prs:prs /app

# Install the package in development mode
RUN pip install -e . --no-deps

# Switch to non-root user
USER prs

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/v1/health/live', timeout=5); r.raise_for_status()"

# Environment defaults (override via docker-compose or -e flags)
ENV APP_ENV=production \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    LOG_LEVEL=INFO \
    MLFLOW_TRACKING_URI=mlruns

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
