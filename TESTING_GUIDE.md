# Testing Guide for TreasureFinder.ai

## Quick Start Testing

### 1. Minimal Test (No API Keys Required)

```bash
# Install minimal dependencies
pip install numpy pandas folium matplotlib requests

# Run Python test
python3 << 'EOF'
import sys
sys.path.append('.')

# Import the module
from treasure_hunter_module import *

# Test with statistical fallback (no satellite data needed)
print("Testing basic functionality...")
result = calculate_geode_probability(43.0, -111.0)  # Dugway, Utah
print(f"Geode probability: {result:.3f}")

# Test mineral segmentation loading
try:
    segmenter = load_mineral_segmenter()
    print("✅ Mineral segmenter loaded successfully")
except:
    print("⚠️ Mineral segmenter not available (expected if model file missing)")

print("\nSystem is working! Add API keys for full functionality.")
EOF
```

### 2. Test with Mapbox (Easiest Satellite Option)

```bash
# Get a free Mapbox token from: https://account.mapbox.com/access-tokens/
export MAPBOX_ACCESS_TOKEN="your_token_here"

python3 << 'EOF'
import os
import sys
sys.path.append('.')
from treasure_hunter_module import *

# This will use Mapbox for RGB satellite imagery
results = main_analysis(
    'Test Location', 
    (37.7749, -122.4194),  # San Francisco
    radius_km=5,
    num_points=10
)

print(f"Found {len(results)} anomalies")
print("Check treasure_map.html for results!")
EOF
```

### 3. Full Test with Google Earth Engine

```bash
# First authenticate Earth Engine (one-time setup)
python3 << 'EOF'
import ee
ee.Authenticate()  # Opens browser for OAuth
ee.Initialize(project='your-project-id')  # Replace with your GCP project
print("Earth Engine authenticated!")
EOF

# Now run full analysis
python3 << 'EOF'
import sys
sys.path.append('.')
from treasure_hunter_module import *

# Test geode detection
print("Testing geode detection...")
result = analyze_satellite_anomalies(43.0, -111.0)
print(f"Anomaly score: {result['anomaly_score']:.3f}")
print(f"Method used: {result['method']}")

# Test archaeological detection  
print("\nTesting archaeological site detection...")
results = main_analysis(
    'Oak Island',
    (44.5133, -64.2947),
    radius_km=10,
    num_points=20,
    use_enhanced_features=True
)

print(f"Analysis complete! Found {len(results)} potential sites")
print("Results saved to:")
print("  - treasure_map.html (interactive map)")
print("  - simple_treasure_map.html (data table)")
EOF
```

## Testing Individual Components

### Test Satellite Connectivity

```python
from treasure_hunter_module import *

# Test coordinates
TEST_LAT, TEST_LON = 37.7749, -122.4194  # San Francisco

# Check which providers are available
providers = []
try:
    import ee
    ee.Initialize()
    providers.append("Google Earth Engine")
except:
    pass

if os.environ.get('MAPBOX_ACCESS_TOKEN'):
    providers.append("Mapbox")

print(f"Available providers: {providers if providers else 'None - will use statistical fallback'}")

# Test fetch
try:
    image_data = fetch_satellite_image(TEST_LAT, TEST_LON, size=256)
    print(f"✅ Satellite fetch successful! Shape: {image_data.shape}")
except Exception as e:
    print(f"⚠️ Satellite fetch failed: {e}")
    print("Will use statistical analysis instead")
```

### Test Geode Detection

```python
from treasure_hunter_module import *

# Known geode locations for testing
test_sites = [
    (43.0, -111.0, "Dugway Geode Beds, Utah"),
    (32.8, -113.7, "Hauser Geode Beds, California"),
    (44.86, -110.50, "Yellowstone, Wyoming"),
]

for lat, lon, name in test_sites:
    score = calculate_geode_probability(lat, lon)
    print(f"{name}: {score:.3f}")
```

### Test Mineral Segmentation

```python
from treasure_hunter_module import *

# Test the unified interface
result = combined_analysis(
    lat=43.0, 
    lon=-111.0,
    analysis_type="both"  # or "archaeological" or "geological"
)

if "archaeology" in result:
    print(f"Archaeological score: {result['archaeology']['score']:.3f}")

if "geology" in result:
    print(f"Geological score: {result['geology']['score']:.3f}")
    if 'minerals' in result['geology']:
        print(f"Detected minerals: {result['geology']['minerals']}")
```

### Test CNN Model

```python
from treasure_hunter_module import *

# Test if CNN is available
try:
    import torch
    model = SatelliteAnomalyCNN()
    model.load_state_dict(torch.load('satellite_cnn.pth', map_location='cpu'))
    print("✅ CNN model loaded successfully")
    
    # Test inference
    test_image = torch.randn(1, 6, 256, 256)
    with torch.no_grad():
        output = model(test_image)
        score = torch.sigmoid(output).item()
    print(f"Test inference score: {score:.3f}")
except:
    print("⚠️ CNN not available - will use statistical fallback")
```

## Testing Web Interface

### Flask API

```bash
# Start the API server
python treasure_api.py

# In another terminal, test endpoints
curl http://localhost:5000/health

# Test analysis endpoint
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"lat": 37.7749, "lon": -122.4194}'

# Test region scan
curl -X POST http://localhost:5000/scan_region \
  -H "Content-Type: application/json" \
  -d '{"center_lat": 43.0, "center_lon": -111.0, "radius_miles": 10}'
```

### Streamlit App

```bash
# Install Streamlit
pip install streamlit

# Run the app
streamlit run streamlit_app.py

# Open browser to http://localhost:8501
```

## Testing with Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Test the service
curl http://localhost:5000/health

# Stop when done
docker-compose down
```

## Testing Large-Scale Analysis

```python
from treasure_hunter_module import *

# Test 300-mile radius scan (from satellite_300mile.ipynb)
def test_large_scan():
    center_lat, center_lon = 36.1069, -112.1129  # Grand Canyon area
    
    # This will scan a large area - be patient!
    results = intelligent_site_prediction(
        center_lat, 
        center_lon,
        target_type='archaeological',
        search_radius_miles=50  # Start smaller for testing
    )
    
    print(f"Scanned area, found {len(results)} potential sites")
    return results

# Run if you have good satellite connectivity
# results = test_large_scan()
```

## Troubleshooting

### No Satellite Data Available

If you see this error, you need to configure at least one provider:

1. **Easiest**: Add a Mapbox token
   ```bash
   export MAPBOX_ACCESS_TOKEN="pk.xxx..."  # Get from mapbox.com
   ```

2. **Best**: Configure Google Earth Engine
   ```python
   import ee
   ee.Authenticate()  # One-time browser auth
   ee.Initialize(project='your-project-id')
   ```

### Import Errors

```bash
# Check if module was converted properly
ls -la treasure_hunter_module.py

# If missing, convert from notebook
python3 << 'EOF'
import json

# Read the notebook
with open('TreasurHunter.ipynb', 'r') as f:
    notebook = json.load(f)

# Extract code cells
code = []
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        code.append(''.join(cell['source']))

# Write module
with open('treasure_hunter_module.py', 'w') as f:
    f.write('\n'.join(code))
print("Module created!")
EOF
```

### CNN Model Not Found

The CNN model is optional. The system will fall back to statistical analysis:

```python
# To train a new CNN model (requires PyTorch)
from treasure_hunter_module import *

# This will train on synthetic data
run_training_pipeline(num_samples=100, epochs=10)
# Model saved to satellite_cnn.pth
```

## Expected Output

### Successful Test Output

```
Testing basic functionality...
Geode probability: 0.762
✅ Mineral segmenter loaded successfully

Testing geode detection...
Anomaly score: 0.834
Method used: statistical

Testing archaeological site detection...
Analyzing region: Oak Island with radius 10 km
Sampled 20 points using Poisson disk distribution
Found 3 anomalies above threshold 0.7
Analysis complete! Found 3 potential sites
Results saved to:
  - treasure_map.html (interactive map)
  - simple_treasure_map.html (data table)
```

### Files Created

After testing, you should see:
- `treasure_map.html` - Interactive Folium map with markers
- `simple_treasure_map.html` - Table view of results
- `geode_detection_model.pkl` - Trained ML model (if training was run)
- `satellite_cnn.pth` - Trained CNN model (if training was run)

## Performance Benchmarks

- Single location analysis: ~2-5 seconds (with satellite data)
- 10-point region scan: ~20-30 seconds  
- 100-point region scan: ~3-5 minutes
- Statistical fallback: <1 second per point

## Next Steps

1. **Add more API keys** for better data:
   - Sentinel Hub for additional spectral bands
   - Planet Labs for high-resolution imagery
   - USGS/Mindat for geological context

2. **Train custom models** on your specific use case:
   - Collect labeled data for your region
   - Fine-tune the CNN for your targets
   - Improve geode detection with local samples

3. **Deploy to production**:
   - Use the Docker setup for cloud deployment
   - Set up HTTPS with nginx
   - Add authentication for API endpoints