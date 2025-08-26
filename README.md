## DOFA Segmenter Usage

Environment flags:
- `USE_DOFA=true|false` (default false)
- `DOFA_BACKBONE=tiny|small|base` (default tiny)
- `DOFA_HUB_REPO=DofA/DOFA` (default)
- `DOFA_LOCAL_WEIGHTS=/abs/path/to/dofa.pt` (optional)

Production requirements:
- Set `PRODUCTION_MODE=true` and provide `DOFA_LOCAL_WEIGHTS` pointing to a local weight file in the image or attached volume.
- In production, online downloads are disabled; Torch Hub is dev-only. Missing local weights cause a clear RuntimeError.
- Disable Torch Hub progress logs via `TORCH_SHOW_DOWNLOAD_PROGRESS=0` (default in Dockerfile.fixed).
- Deterministic inference: models run in `eval()` with no per-request seeding.

Railway environment (recommended):
- Set `USE_DOFA=true` and `DOFA_LOCAL_WEIGHTS=/app/weights/dofa.pth`.
- Optionally set build args `DOFA_WEIGHTS_URL` and `DOFA_WEIGHTS_SHA256` to fetch weights at build time; see Dockerfile.fixed.

API single point:
```bash
curl -s -X POST http://localhost:5000/api/analyze/single \
 -H 'Content-Type: application/json' \
 -d '{"latitude":44.5133,"longitude":-64.2947,"use_dofa":true,"return_mask":true}' | jq .
```

Python quick test:
```python
import os, treasure_hunter_module as thm
os.environ['USE_DOFA']='true'
thm.initialize_earth_engine()
thm.load_dofa_segmenter()
res = thm.analyze_satellite_anomalies(44.5133, -64.2947, use_dofa=True, return_mask=True)
print(res['method'], res['anomaly_score'], 'mask_b64' in res or 'dofa_mask' in res)
```

Notes:
- Real-data-only: If imagery is unavailable, DOFA path raises.
- Deterministic inference: no random seeds in prediction.
- Performance: GPU is recommended; CPU works for small tiles.
## Prerequisites and Dataset Preparation

To experiment with the new DOFA segmenter and geospatial utilities, install the
required libraries and fetch the training data:

```bash
pip install dofa torchgeo torch torchvision

# optional extras used by the notebooks
pip install geopandas folium xgboost
```

### Real Training Data

This repository includes real archaeological training data from the ArchaeoScape dataset:
- **Location**: `frontend/training_data/`
- **Contents**: 600+ georeferenced TIFF images with bounding box annotations
- **Categories**: DTM, Hillshade, Local_dominance, Open_Positive, Sky_View_Factor, Slope
- **Labels**: Archaeological features (roundhouses)

### Using the Training Data

```bash
# Validate data integrity (requires PRODUCTION_MODE=true)
export PRODUCTION_MODE=true
python train_from_annotations.py --dry-run

# Train CNN with real data
python train_from_annotations.py \
  --data-root frontend/training_data \
  --datasets DTM,Hillshade,Slope \
  --epochs 30 \
  --batch-size 32

# For custom TIFF datasets, point to your data directory
python train_from_annotations.py \
  --data-root /path/to/your/tiff/data \
  --train-csv train_annotations.csv \
  --val-csv valid_annotations.csv
```

### Data Acquisition for New Regions

For satellite imagery of new regions, configure one of these providers:

1. **Google Earth Engine** (Recommended - Free with quota)
   ```bash
   export GEE_PROJECT_ID="your-project-id"
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   ```
   Setup: https://developers.google.com/earth-engine/guides/service_account

2. **Mapbox** (RGB only, easy setup)
   ```bash
   export MAPBOX_ACCESS_TOKEN="your-token"
   ```
   Get token: https://account.mapbox.com/access-tokens/

3. **Sentinel Hub** (Advanced features - NOT YET IMPLEMENTED)
   ```bash
   export SENTINELHUB_CLIENT_ID="your-client-id"
   export SENTINELHUB_CLIENT_SECRET="your-secret"
   ```
   Note: Full implementation pending

4. **Planet Labs** (High resolution - NOT YET IMPLEMENTED)
   ```bash
   export PLANET_API_KEY="your-api-key"
   ```
   Note: Full implementation pending

## Data Validation

### Validate Training Data Integrity

```bash
# Quick validation - checks if CSV references resolve to real files
python train_from_annotations.py --dry-run

# Full validation with sampling (N=50 random samples)
python -c "
import os, csv, random
data_root = 'frontend/training_data'
for dataset in ['DTM', 'Hillshade', 'Slope']:
    csv_path = os.path.join(data_root, dataset, 'train_annotations.csv')
    if not os.path.exists(csv_path): continue
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    sample = random.sample(rows, min(50, len(rows)))
    missing = sum(1 for r in sample if not os.path.exists(os.path.join(data_root, dataset, r[0])))
    print(f'{dataset}: {len(rows)} annotations, {missing}/{len(sample)} missing in sample')
"
```

## DOFA Segmenter Usage

### Training

```bash
python dofa_segmenter.py train \
  --data-root data/dofa \
  --epochs 100 \
  --batch-size 4 \
  --checkpoint runs/dofa_latest.pth
```

### Evaluation

```bash
python dofa_segmenter.py eval \
  --data-root data/dofa \
  --checkpoint runs/dofa_latest.pth
```

### Inference

```bash
python dofa_segmenter.py infer \
  --checkpoint runs/dofa_latest.pth \
  --image-path examples/sample_scene.tif \
  --output-path outputs/sample_mask.tif
```

## Model Orchestration in Region Scanning

`scan_region_comprehensive` orchestrates two specialised models:

1. **Archaeological model** – the DOFA segmenter or the legacy
   `SatelliteAnomalyCNN` scores each tile for man‑made structures.
2. **Mineral model** – `calculate_geode_probability` extracts spectral and
   terrain cues for geode likelihood.

The function calls `combined_analysis` at every sampled point. This helper
fetches imagery, runs both models, merges the archaeological mask with the
geologic probability, and returns a unified record. `predict_discovery_zones`
builds on this by scanning larger regions, keeping points whose combined score
passes a threshold, and ranking them for follow‑up.

How `TreasurHunter.ipynb` works end‑to‑end, focusing on the concrete algorithms, data flow, and decision logic.

### Production Mode Requirements

**IMPORTANT**: In production, the system enforces:
- **No mock data**: `MOCK_DATA` must be unset or `false`
- **Real providers only**: At least one satellite provider must be configured
- **Data validation**: All training data paths must exist
- **Hard failures**: Missing credentials or data cause immediate errors

```bash
# Enable production mode
export PRODUCTION_MODE=true
export MOCK_DATA=false

# Verify configuration
python -c "from treasure_api import validate_real_providers; print(validate_real_providers())"
```
- **Secrets/bootstrap**:
  - If in Colab, pulls secrets into env vars (`GEE_PROJECT_ID`, `SENTINEL_HUB_API_KEY`, `PLANET_API_KEY`, `MAPBOX_ACCESS_TOKEN`).
  - Initializes optional dependencies and reports status: PyTorch, XGBoost, Folium, GeoPandas, Earth Engine, Requests.

### Data sources and acquisition
- **Primary**: Google Earth Engine (GEE).
  - Tries `ee.Initialize(project=GEE_PROJECT_ID)`, else `ee.Authenticate()` then initialize, else default.
- **Fallback**: Mapbox Static Images API (RGB only) with simulated NIR; Sentinel Hub and Planet Labs are placeholders.
- If no provider is available, functions error out (no mock data allowed in production).

### Satellite fetch algorithm
- Entry: `fetch_satellite_image(lat, lon, size=IMAGE_SIZE)`
- Preconditions: Errors if `MOCK_DATA` set. Requires at least EE or Requests+Mapbox.
- **EE path**:
  1. AOI: `ee.Geometry.Point(lon, lat)` → `buffer(500).bounds()`.
  2. Collection: `COPERNICUS/S2_SR_HARMONIZED` filtered by AOI, 2023–2024, clouds < 20%, bands `[B4,B3,B2,B8,B11,B12]` (RGB, NIR, SWIR1, SWIR2).
  3. Median composite.
  4. Preferred extraction: `ee.data.computePixels` with `NUMPY_NDARRAY` and an affine transform to deliver a `size x size` grid at ~10m sampling.
  5. Conversion: ensure tensor shape to `(NUM_CHANNELS, size, size)`; per-band min–max normalization; pad channels if fewer than `NUM_CHANNELS`.
  6. Fallbacks if computePixels fails:
     - `image.sample(...)` then rasterize samples → smooth with Gaussian to enforce spatial coherence.
     - `image.reduceRegion(...)` → per-band means → broadcast to uniform tensor.
- **Mapbox path**:
  - API: `https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/...`
  - RGB only, converts to NumPy, pads to `NUM_CHANNELS`.

### Anomaly detection logic
- Entry: `analyze_satellite_anomalies(lat, lon, ...)`
- Flow:
  1. `fetch_satellite_image(lat, lon, IMAGE_SIZE)` → `(NUM_CHANNELS, 256, 256)` tensor.
  2. **CNN path (if available)**: `analyze_with_cnn(image_data)` → score ∈ [0, 1].
  3. **Statistical fallback**: `statistical_anomaly_detection(lat, lon)` → score ∈ [0, 1].
  4. Returns `{'lat', 'lon', 'anomaly_score', 'confidence', 'method', 'timestamp'}`.
- Statistical fallback is heuristic (noise + priors for known locations).

### Archaeological CNN training (optional)
- `create_training_dataset(num_samples)`: generates synthetic or fetched data for positive (Angkor Wat, Petra...) and negative (NYC, Tokyo...) sites.
- `train_cnn_model(X, y, epochs, batch_size, lr)`: trains `SatelliteAnomalyCNN` with `BCELoss`.
- Saves to `satellite_cnn.pth`.

### Geode/mineral probability calculation
- Entry: `calculate_geode_probability(lat, lon)`
- Key factors:
  1. **Geology**: Known volcanic regions → high score.
  2. **Climate**: Arid regions → high score.
  3. **Terrain**: Elevation 1000–2500m → high score.
  4. **Near known sites**: ≤50 km → bonus.
- Returns weighted average ∈ [0, 1].

### Region analysis flow
- Entry: `analyze_region(center_lat, center_lon, radius_km, num_points)`
- Algorithm:
  1. Poisson disk sampling around center → `num_points` locations.
  2. For each: `analyze_satellite_anomalies(lat, lon)` → anomaly scores.
  3. Filters by `score > ANOMALY_THRESHOLD`.
  4. Sorts by score, returns top points as DataFrame.

### Map generation (Folium)
- `create_enhanced_map(results_df, center_lat, center_lon)`: for each result, adds CircleMarker colored by score (red=high, green=low).
- `create_prediction_map(...)`: heatmap overlay + numbered labels.

### Main analysis pipeline
- Entry: `main_analysis(region_name, (center_lat, center_lon), radius_km, num_points, use_enhanced_features)`
- Orchestrates: calls `analyze_region`, applies enhanced scoring, creates map, saves to `treasure_map.html` and `simple_treasure_map.html`.

### Discovery prediction heuristic
- Entry: `intelligent_site_prediction(center_lat, center_lon, target_type, search_radius_miles)`
- Algorithm:
  - Sets adaptive search radius, grid density, thresholds by target type.
  - Calls `predict_discovery_zones`, filters by type, creates a prediction map via `create_prediction_map` (heatmap + ranked labels).

### Training module (optional, requires PyTorch)
- Goal: train `SatelliteAnomalyCNN` on a synthetic dataset derived from known positive/negative coordinates with real fetch (if available) or synthetic generation when fetch fails.
- `create_training_dataset(num_samples)`:
  - Positive sites: prominent archaeological locations (+ random small offsets).
  - Negative sites: major cities (+ offsets).
  - If fetch fails: synthetic image generation with distinct geometric patterns (positives) vs smoothed/natural noise (negatives).
- `train_cnn_model(X, y, epochs, batch_size, lr)`:
  - Train/test split, `BCELoss`, Adam, `ReduceLROnPlateau`, gradient clipping, tracks loss/accuracy.
- `save_trained_model`, `load_trained_model` for persistence.
- `run_training_pipeline` orchestrates dataset, training, saving, and updates global `satellite_cnn`.

### Key thresholds and constants
- `ANOMALY_THRESHOLD=0.7`, `MIN_CONFIDENCE=0.5`, `MAX_RADIUS_MILES=50`, `IMAGE_SIZE=256`, `NUM_CHANNELS=6`, `DEFAULT_ZOOM=11`.

### Error handling and constraints
- No mock data in production; if providers are missing or EE fails and Mapbox missing, functions raise `RuntimeError`.
- Sentinel Hub and Planet Labs paths are explicitly `NotImplementedError` placeholders.

### Typical usage entry points
- Archaeological flow: `main_analysis(region_name, (lat, lon), radius_km, num_points)` → runs `analyze_region` → outputs `treasure_map.html` and `simple_treasure_map.html`.
- Dual flow: `scan_region_comprehensive(...)` + `create_comprehensive_map(...)`.
- Prediction flow: `intelligent_site_prediction(...)` with saved map.

### Mineral segmentation (geological)
- `datasets/mineral_spectral/` references a labeled spectral dataset of mineral deposits derived from the USGS Spectral Library. A tiny sample is included for testing.
- `training/mineral_segmentation.py` fine-tunes a lightweight DOFA segmenter on these spectral bands and saves a model `mineral_segmenter.pt`.
- `combined_analysis` loads this model (via `load_mineral_segmenter`) to provide geology predictions alongside archaeology results through a shared interface.
