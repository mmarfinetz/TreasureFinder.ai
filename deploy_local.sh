#!/bin/bash

# Local Deployment Script for TreasureFinder.ai API
# This script sets up and runs the API locally

set -e  # Exit on error

echo "🚀 TreasureFinder.ai Local Deployment"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "✓ Python version: $python_version"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Install dependencies
echo ""
echo "📦 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo "✓ Installed from requirements.txt"
else
    echo "Installing core dependencies..."
    pip install -q flask flask-cors pandas numpy folium matplotlib requests
    pip install -q earthengine-api geopy scikit-learn xgboost
    echo "✓ Core dependencies installed"
fi

# Step 2: Convert notebooks to modules
echo ""
echo "📓 Converting notebooks to modules..."
if [ -f "convert_notebook.py" ]; then
    python3 convert_notebook.py
else
    echo "⚠️  convert_notebook.py not found - attempting manual conversion..."
    python3 << 'EOF'
import json
from pathlib import Path

notebook_path = Path('TreasurHunter.ipynb')
if notebook_path.exists():
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    code = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if not source.strip().startswith('!') and not source.strip().startswith('%'):
                code.append(source)
    
    with open('treasure_hunter_module.py', 'w') as f:
        f.write('\n'.join(code))
    print("✓ Manual conversion successful")
else:
    print("⚠️  TreasurHunter.ipynb not found")
EOF
fi

# Step 3: Check for API keys
echo ""
echo "🔑 Checking configuration..."

if [ -z "$MAPBOX_ACCESS_TOKEN" ]; then
    echo "⚠️  MAPBOX_ACCESS_TOKEN not set"
    echo "   To add satellite imagery support:"
    echo "   export MAPBOX_ACCESS_TOKEN='your_token_here'"
    echo "   Get a free token at: https://account.mapbox.com/"
else
    echo "✓ Mapbox token configured"
fi

if [ -z "$GEE_PROJECT_ID" ]; then
    echo "⚠️  Google Earth Engine not configured (optional)"
else
    echo "✓ Google Earth Engine configured"
fi

# Step 4: Check if port is available
PORT=${PORT:-5000}
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo ""
    echo "⚠️  Port $PORT is already in use!"
    echo "   Kill the process: kill -9 \$(lsof -t -i:$PORT)"
    echo "   Or use a different port: PORT=8080 ./deploy_local.sh"
    exit 1
fi

# Step 5: Create a simple test script
echo ""
echo "📝 Creating test endpoint..."
cat > test_api.py << 'EOF'
import requests
import json
import sys

API_URL = f"http://localhost:{sys.argv[1] if len(sys.argv) > 1 else 5000}"

print(f"Testing API at {API_URL}")

# Test health
try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    if r.status_code == 200:
        print("✓ Health check passed")
    else:
        print(f"✗ Health check failed: {r.status_code}")
except Exception as e:
    print(f"✗ Cannot connect to API: {e}")
    sys.exit(1)

# Test analysis
try:
    data = {"lat": 37.7749, "lon": -122.4194}
    r = requests.post(f"{API_URL}/analyze", json=data, timeout=10)
    if r.status_code == 200:
        result = r.json()
        print(f"✓ Analysis endpoint working")
        if 'anomaly_score' in result:
            print(f"  Anomaly score: {result['anomaly_score']:.3f}")
    else:
        print(f"✗ Analysis failed: {r.status_code}")
except Exception as e:
    print(f"✗ Analysis error: {e}")
EOF

# Step 6: Start the API
echo ""
echo "🌐 Starting API server on http://localhost:$PORT"
echo "======================================"
echo ""
echo "API Endpoints:"
echo "  GET  http://localhost:$PORT/health         - Health check"
echo "  POST http://localhost:$PORT/analyze        - Analyze location"
echo "  POST http://localhost:$PORT/scan_region    - Scan region"
echo "  POST http://localhost:$PORT/geode_probability - Geode detection"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Export port for Flask
export FLASK_APP=treasure_api.py
export FLASK_ENV=development
export PORT=$PORT

# Run the API
if [ -f "treasure_api.py" ]; then
    python3 treasure_api.py
else
    echo "❌ treasure_api.py not found!"
    echo "   Please ensure you're in the correct directory"
    exit 1
fi