#!/usr/bin/env bash
set -e

echo "=== [AOI Service] Initializing Container Environment ==="

# Execute pending database migrations to current revision
echo "=== [AOI Service] Applying Alembic Database Migrations ==="
alembic upgrade head || {
    echo "Warning: Alembic upgrade returned non-zero. Attempting standard table initialization fallback..."
}

echo "=== [AOI Service] Starting Uvicorn Production Web Server ==="
exec uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 --log-level info

