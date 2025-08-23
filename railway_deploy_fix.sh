#!/bin/bash
# Railway deployment fix script
# Applies patches to fix 502 Bad Gateway and GEE initialization issues

set -e

echo "🚂 Railway Deployment Fix Script"
echo "================================"

# Check if we're in Railway environment
if [ -n "$RAILWAY_ENVIRONMENT" ] || [ -n "$RAILWAY_PROJECT_ID" ]; then
    echo "✅ Running in Railway environment"
    IS_RAILWAY=true
else
    echo "📍 Running locally (not in Railway)"
    IS_RAILWAY=false
fi

# Step 1: Apply the lazy initialization patch
echo ""
echo "📦 Step 1: Applying GEE lazy initialization patch..."

# Backup original files
if [ -f treasure_hunter_module.py ]; then
    cp treasure_hunter_module.py treasure_hunter_module.py.backup
    echo "  ✅ Backed up treasure_hunter_module.py"
fi

if [ -f treasure_api.py ]; then
    cp treasure_api.py treasure_api.py.backup
    echo "  ✅ Backed up treasure_api.py"
fi

# Copy fixed files
if [ -f gee_lazy_init_patch.py ]; then
    echo "  ✅ GEE lazy init patch available"
else
    echo "  ❌ gee_lazy_init_patch.py not found!"
    exit 1
fi

if [ -f treasure_api_fixed.py ]; then
    cp treasure_api_fixed.py treasure_api.py
    echo "  ✅ Applied fixed API with health checks"
else
    echo "  ⚠️  treasure_api_fixed.py not found, keeping original"
fi

# Step 2: Handle credentials in Railway
echo ""
echo "🔐 Step 2: Setting up GEE credentials..."

if [ -n "$GOOGLE_CREDENTIALS_B64" ]; then
    echo "  ✅ Found GOOGLE_CREDENTIALS_B64 in environment"
    
    # Decode and save to file for GEE
    if [ "$IS_RAILWAY" = true ]; then
        # In Railway, use /app directory
        echo "$GOOGLE_CREDENTIALS_B64" | base64 -d > /app/gee_sa.json 2>/dev/null || \
        echo "$GOOGLE_CREDENTIALS_B64" | base64 -D > /app/gee_sa.json 2>/dev/null || \
        echo "  ⚠️  Failed to decode credentials"
        
        if [ -f /app/gee_sa.json ]; then
            export GOOGLE_APPLICATION_CREDENTIALS=/app/gee_sa.json
            echo "  ✅ Set GOOGLE_APPLICATION_CREDENTIALS=/app/gee_sa.json"
        fi
    else
        # Local environment
        echo "$GOOGLE_CREDENTIALS_B64" | base64 -d > ./gee_sa.json 2>/dev/null || \
        echo "$GOOGLE_CREDENTIALS_B64" | base64 -D > ./gee_sa.json 2>/dev/null
        
        if [ -f ./gee_sa.json ]; then
            export GOOGLE_APPLICATION_CREDENTIALS=./gee_sa.json
            echo "  ✅ Set GOOGLE_APPLICATION_CREDENTIALS=./gee_sa.json"
        fi
    fi
elif [ -n "$GEE_SERVICE_ACCOUNT_JSON" ]; then
    echo "  ✅ Found GEE_SERVICE_ACCOUNT_JSON in environment"
elif [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "  ✅ Using existing GOOGLE_APPLICATION_CREDENTIALS: $GOOGLE_APPLICATION_CREDENTIALS"
else
    echo "  ⚠️  No GEE credentials found in environment"
fi

# Step 3: Set project ID
echo ""
echo "📋 Step 3: Setting GEE project ID..."

if [ -n "$GEE_PROJECT_ID" ]; then
    echo "  ✅ GEE_PROJECT_ID: $GEE_PROJECT_ID"
elif [ -n "$GOOGLE_EARTH_ENGINE_PROJECT" ]; then
    echo "  ✅ GOOGLE_EARTH_ENGINE_PROJECT: $GOOGLE_EARTH_ENGINE_PROJECT"
    export GEE_PROJECT_ID=$GOOGLE_EARTH_ENGINE_PROJECT
elif [ -n "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "  ✅ GOOGLE_CLOUD_PROJECT: $GOOGLE_CLOUD_PROJECT"
    export GEE_PROJECT_ID=$GOOGLE_CLOUD_PROJECT
else
    echo "  ⚠️  No GEE project ID found"
fi

# Step 4: Test the setup
echo ""
echo "🧪 Step 4: Testing configuration..."

python3 - <<'EOF'
import os
import sys

print("  Python version:", sys.version)
print("  Working directory:", os.getcwd())

# Test imports
try:
    from gee_lazy_init_patch import lazy_initialize_earth_engine
    print("  ✅ GEE lazy init patch imported")
except ImportError as e:
    print(f"  ❌ Failed to import patch: {e}")
    sys.exit(1)

# Test GEE initialization (lazy)
print("  🚀 Testing lazy GEE initialization...")
if lazy_initialize_earth_engine():
    print("  ✅ GEE initialization successful!")
else:
    print("  ⚠️  GEE initialization failed (will use fallback methods)")

print("  ✅ Configuration test complete")
EOF

if [ $? -ne 0 ]; then
    echo "  ❌ Configuration test failed"
    exit 1
fi

# Step 5: Start the application
echo ""
echo "🚀 Step 5: Starting application..."

if [ "$IS_RAILWAY" = true ]; then
    # Railway production mode
    echo "  Starting with gunicorn (Railway mode)..."
    echo "  Port: ${PORT:-5000}"
    echo "  Workers: ${WORKERS:-1}"
    echo "  Threads: ${THREADS:-2}"
    echo "  Timeout: 120s"
    
    exec gunicorn \
        --bind 0.0.0.0:${PORT:-5000} \
        --workers ${WORKERS:-1} \
        --threads ${THREADS:-2} \
        --timeout 120 \
        --log-level info \
        --access-logfile - \
        --error-logfile - \
        treasure_api:app
else
    # Local development mode
    echo "  Starting with Flask development server..."
    python3 treasure_api.py
fi