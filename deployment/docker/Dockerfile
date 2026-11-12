# ==============================================================================
# Industrial Optical Inspection (AOI) — Production Container
# Multi-stage Python 3.12 Dockerfile for FastAPI microservice and model inference
# ==============================================================================

# Stage 1: Build & Dependency Resolution
FROM python:3.12-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production Runtime
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install runtime system libraries required by OpenCV and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source code, configurations, and database migrations
COPY backend /app/backend
COPY src /app/src
COPY frontend /app/frontend
COPY alembic.ini /app/alembic.ini
COPY pyproject.toml /app/pyproject.toml
COPY deployment/docker/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Create persistent storage directories
RUN mkdir -p /app/data/inspection_media /app/models

# Expose microservice HTTP port
EXPOSE 8000

# Health check to ensure microservice and database connection are alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]

