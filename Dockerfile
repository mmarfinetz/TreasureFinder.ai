# Multi-stage build for TreasureHunter production deployment
# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libspatialindex-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application files (except treasure_hunter_module.py initially)
COPY treasure_api.py .
COPY gee_lazy_init_patch.py .
COPY convert_notebook.py .
COPY frontend/ ./frontend/
# Copy notebooks needed for conversion
COPY TreasurHunter.ipynb .
COPY satellite.ipynb .
COPY satellite_300mile.ipynb .
COPY satellite_production_modular_unified.ipynb .

# Create necessary directories
RUN mkdir -p saved_models logs

# Convert notebook to module (required for API)
RUN python convert_notebook.py && \
    test -f treasure_hunter_module.py || \
    (echo "ERROR: Failed to generate treasure_hunter_module.py" && exit 1)

# Copy the patched treasure_hunter_module.py to overwrite the auto-generated one
COPY treasure_hunter_module.py .

# Set environment variables
ENV PRODUCTION_MODE=true
ENV FLASK_APP=treasure_api.py
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Health check (respect Railway's PORT env) - use lightweight endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/healthz || exit 1

# Run with gunicorn for production, binding to dynamic PORT if provided
# Use fewer workers by default to fit small-memory hosts like Railway free tier
# Override via env: WORKERS, THREADS
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers ${WORKERS:-1} --threads ${THREADS:-2} --timeout 120 --log-level info treasure_api:app"]