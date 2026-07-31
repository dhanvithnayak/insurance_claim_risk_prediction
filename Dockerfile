# Production Dockerfile for Auto Insurance Claim Risk Prediction Service
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security: run as non-privileged system user
RUN useradd -m -u 10001 -s /bin/bash appuser

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application assets, templates, and serialized model pipeline
COPY app.py model_pipeline.joblib ./
COPY templates/ ./templates/

# Ensure non-root ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Production multi-worker ASGI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
