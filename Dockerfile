# Multi-stage build for TreasureHunter production deployment
# Stage 1: Builder
# Pin to Docker Hub official image to ensure anonymous pulls in CI/CD (e.g., Railway)
FROM docker.io/library/python:3.10-slim as builder

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
FROM docker.io/library/python:3.10-slim

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
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages

# Prepare cache and weights directories for DOFA
RUN mkdir -p /app/.cache /app/weights && chmod -R 777 /app/.cache /app/weights

# Prefer bundled weights from the repository if present
# This allows offline builds (e.g., Railway) to succeed without remote downloads
COPY weights/ /app/weights/

# Optional: download DOFA weights at build time (only if not bundled)
# Provide URL and SHA256 via build args at deploy time; no defaults
ARG DOFA_WEIGHTS_URL
ARG DOFA_WEIGHTS_SHA256
ARG HF_TOKEN
ARG SKIP_REMOTE_WEIGHTS=true
RUN set -eu; \
    # If a bundled weight exists, trust it by default (skip SHA check)
    if [ -s "/app/weights/dofa.pth" ]; then \
      DOFA_SIZE=$(stat -c%s /app/weights/dofa.pth || echo 0); \
      echo "Found bundled DOFA weights (${DOFA_SIZE} bytes). Skipping SHA verification."; \
      if [ "${DOFA_SIZE}" -lt 1048576 ]; then \
        echo "Bundled DOFA weights file too small; removing to allow remote download..."; \
        rm -f /app/weights/dofa.pth; \
      fi; \
    fi; \
    # If no bundled weights, optionally download if URL+SHA provided (unless skipping)
    if [ ! -s "/app/weights/dofa.pth" ]; then \
      if [ "${SKIP_REMOTE_WEIGHTS:-true}" = "true" ]; then \
        echo "Skipping remote DOFA download (SKIP_REMOTE_WEIGHTS=true)"; \
        exit 0; \
      fi; \
      # Sanitize inputs to avoid accidental trailing characters (e.g., semicolons, quotes, CRLF)
      SANITIZED_URL=$(printf "%s" "${DOFA_WEIGHTS_URL:-}" | tr -d '\r' | sed -e 's/[[:space:]]*$//' -e 's/[";]*$//'); \
      SANITIZED_SHA=$(printf "%s" "${DOFA_WEIGHTS_SHA256:-}" | tr -d '\r' | tr -d '[:space:]'); \
      if [ -n "${SANITIZED_URL}" ] && [ -n "${SANITIZED_SHA}" ]; then \
        echo "Downloading DOFA weights from remote..."; \
        RESOLVED_URL=$(printf "%s" "${SANITIZED_URL}" | sed 's|/blob/|/resolve/|'); \
        if [ -n "${HF_TOKEN:-}" ]; then \
          curl --retry 5 --retry-delay 5 --retry-all-errors -fSL \
            -H "Authorization: Bearer ${HF_TOKEN}" \
            -H "Accept: application/octet-stream" \
            -o /app/weights/dofa.pth "${RESOLVED_URL}"; \
        else \
          curl --retry 5 --retry-delay 5 --retry-all-errors -fSL \
            -o /app/weights/dofa.pth "${RESOLVED_URL}"; \
        fi; \
        DOFA_SIZE=$(stat -c%s /app/weights/dofa.pth || echo 0); \
        if [ "${DOFA_SIZE}" -lt 1048576 ]; then \
          echo "Downloaded DOFA weights file too small (${DOFA_SIZE} bytes). Failing build."; \
          exit 1; \
        fi; \
        echo "${SANITIZED_SHA}  /app/weights/dofa.pth" | sha256sum -c -; \
      else \
        echo "Skipping DOFA weight download (no URL/SHA provided and no bundled file)"; \
      fi; \
    fi

# Optional: download CNN weights at build time
ARG CNN_WEIGHTS_URL
ARG CNN_WEIGHTS_SHA256
RUN set -e; \
    if [ -n "$CNN_WEIGHTS_URL" ] && [ -n "$CNN_WEIGHTS_SHA256" ]; then \
      echo "Downloading CNN weights..."; \
      curl -fSL -o /app/weights/archaeo_cnn.pth "$CNN_WEIGHTS_URL"; \
      echo "$CNN_WEIGHTS_SHA256  /app/weights/archaeo_cnn.pth" | sha256sum -c -; \
    else \
      echo "Skipping CNN weight download (CNN_WEIGHTS_URL/SHA256 not provided)"; \
    fi

# Copy application files (except treasure_hunter_module.py initially)
COPY treasure_api.py .
COPY gee_lazy_init_patch.py .
COPY convert_notebook.py .
COPY frontend/ ./frontend/
# DOFA model definitions
COPY models/ ./models/
# Create necessary directories
RUN mkdir -p saved_models logs

# Copy pre-converted module directly (skip notebook conversion in CI)
COPY treasure_hunter_module.py .

# Set environment variables
ENV PRODUCTION_MODE=true
ENV FLASK_APP=treasure_api.py
ENV PYTHONUNBUFFERED=1

# DOFA runtime configuration
ENV DOFA_LOCAL_WEIGHTS=/app/weights/dofa.pth \
    USE_DOFA=true \
    TORCH_HOME=/app/.cache \
    TORCH_SHOW_DOWNLOAD_PROGRESS=0 \
    CNN_WEIGHTS_PATH=/app/weights/archaeo_cnn.pth

# Expose port
EXPOSE 5000

# Health check (respect Railway's PORT env) - use lightweight endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/healthz || exit 1

# Add a lightweight startup script to handle optional GEE credentials at runtime
RUN set -eu; \
    echo '#!/bin/sh' > /app/start.sh; \
    echo 'set -eu' >> /app/start.sh; \
    echo '' >> /app/start.sh; \
    echo 'echo "🚀 Starting TreasureHunter API..."' >> /app/start.sh; \
    echo '# Decode base64 credentials if provided' >> /app/start.sh; \
    echo 'if [ -n "${GOOGLE_CREDENTIALS_B64:-}" ]; then' >> /app/start.sh; \
    echo '  echo "$GOOGLE_CREDENTIALS_B64" | base64 -d > /app/gee_sa.json 2>/dev/null || true' >> /app/start.sh; \
    echo '  if [ -s /app/gee_sa.json ]; then' >> /app/start.sh; \
    echo '    export GOOGLE_APPLICATION_CREDENTIALS=/app/gee_sa.json' >> /app/start.sh; \
    echo '    echo "✅ Set GOOGLE_APPLICATION_CREDENTIALS"' >> /app/start.sh; \
    echo '  fi' >> /app/start.sh; \
    echo 'elif [ -n "${GEE_SERVICE_ACCOUNT_JSON:-}" ] && [ ! -f /app/gee_sa.json ]; then' >> /app/start.sh; \
    echo '  echo "$GEE_SERVICE_ACCOUNT_JSON" > /app/gee_sa.json' >> /app/start.sh; \
    echo '  export GOOGLE_APPLICATION_CREDENTIALS=/app/gee_sa.json' >> /app/start.sh; \
    echo '  echo "✅ Set GOOGLE_APPLICATION_CREDENTIALS from JSON"' >> /app/start.sh; \
    echo 'fi' >> /app/start.sh; \
    echo '' >> /app/start.sh; \
    echo 'echo "📊 Env: PORT=${PORT:-5000} WORKERS=${WORKERS:-1} THREADS=${THREADS:-2}"' >> /app/start.sh; \
    echo 'exec gunicorn \\' >> /app/start.sh; \
    echo '  --bind 0.0.0.0:${PORT:-5000} \\' >> /app/start.sh; \
    echo '  --workers ${WORKERS:-1} \\' >> /app/start.sh; \
    echo '  --threads ${THREADS:-2} \\' >> /app/start.sh; \
    echo '  --timeout 300 \\' >> /app/start.sh; \
    echo '  --log-level info \\' >> /app/start.sh; \
    echo '  treasure_api:app' >> /app/start.sh; \
    chmod +x /app/start.sh

# Use the startup script as the default command
CMD ["/app/start.sh"]
