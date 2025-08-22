## Prerequisites and Dataset Preparation

To experiment with the new DOFA segmenter and geospatial utilities, install the
required libraries and fetch the training data:

```bash
pip install dofa torchgeo torch torchvision

# optional extras used by the notebooks
pip install geopandas folium xgboost
```

Download and unpack the DOFA training set (adjust paths as needed):

```bash
mkdir -p data/dofa
wget -O data/dofa/dataset.zip https://example.com/datasets/dofa_dataset.zip
unzip data/dofa/dataset.zip -d data/dofa
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

### Production guard and environment setup
- **Strict production mode**: Sets `PRODUCTION_MODE=true` and asserts the absence of `ALLOW_TEST_MODE`, `DEBUG`, `MOCK_DATA`.
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