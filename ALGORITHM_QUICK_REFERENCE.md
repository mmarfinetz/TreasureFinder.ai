# GeoFinder Algorithm Implementation - Quick Reference Guide

## Algorithm Locations Cheat Sheet

### Feature Extraction
```
NDVI, NDWI, BSI          → satellite_production_module.py:278-340
Iron Oxide & Clay Ratios → satellite_production_module.py:281-313
Elevation, Slope, Aspect → satellite_production_module.py:315-317
```

### Geological Data
```
USGS Lithology Query     → satellite_production_module.py:402-551
Mindat Occurrence Metrics → satellite_production_module.py:349-395
Fault Proximity Calculation → satellite_production_module.py:562-699
```

### Geode Detection
```
ML-based Detector        → satellite_production_module.py:1336-1481
Heuristic Score Formula  → satellite_production_module.py:1436-1449
Anomaly Score Computation → satellite_production_module.py:706-722
```

### ML Models
```
GeodeMLTrainer (3 models) → satellite_production_module.py:915-1161
  - Logistic Regression
  - XGBoost Classifier
  - Random Forest Classifier
```

### CNN Models
```
ImprovedCNN              → train_quick_demo.py:44-130
Enhanced CNN             → train_enhanced.py (multiple)
DOFASegmenter            → models/dofa_segmenter.py:29-100+
```

### Training Pipelines
```
Basic Training           → train_from_annotations.py:129-403
Enhanced Training        → train_enhanced.py (751 lines)
Optimized Training       → train_optimized_full.py (501 lines)
```

### API
```
Flask REST API           → treasure_api.py (1,039 lines)
Health Check             → treasure_api.py:169-176
Analyze Single           → treasure_api.py (~500 lines)
Analyze Region           → treasure_api.py (~400 lines)
```

---

## Hardcoded Values Quick Reference

### Earth Engine Parameters
```python
CLOUD_COVER_THRESHOLD = 0.20          # 20% (line 295)
MAX_COLLECTION_SIZE = 50              # (line 297)
BUFFER_RADIUS_M = 500                 # meters (line 289)
SCALE_M = 30                          # meters (line 323)
MAX_PIXELS = 1_000_000                # (line 324)
DATE_RANGE = '2021-01-01' to '2024-12-31'  # (line 294)
```

### Geode Detection Weights
```python
exposed_rock = 0.30      # BSI-based
iron_content = 0.25      # Red/NIR ratio
clay_index = 0.20        # SWIR1/SWIR2 ratio
low_veg = 0.15           # Inverse NDVI
terrain_complexity = 0.10 # Slope / 45°
proximity_bonus = 0.15    # Max bonus (line 1460)
```

### Geode Detection Thresholds
```python
ELEVATION_MAX = 3000     # meters (line 714)
SLOPE_MAX = 45           # degrees (line 1441)
PROXIMITY_RADIUS = 50    # miles (line 1460)
GEODE_PROBABILITY_THRESHOLD = 0.4  # (line 1561)
```

### CNN Hyperparameters
```python
FOCAL_LOSS_ALPHA = 1.0   # (train_quick_demo.py:26)
FOCAL_LOSS_GAMMA = 2.0   # (train_quick_demo.py:26)
BATCH_SIZE = 32          # (hyperparameter_config.yaml:13)
EPOCHS = 100             # (hyperparameter_config.yaml:14)
LEARNING_RATE = 3e-4     # (train_enhanced.py:50)
WEIGHT_DECAY = 1e-4      # (train_enhanced.py:51)
EARLY_STOPPING_PATIENCE = 15  # (train_enhanced.py:55)
```

### ML Model Hyperparameters
```python
# Logistic Regression
max_iterations = 1000
class_weight = 'balanced'
calibration = 'isotonic'

# XGBoost
n_estimators = 100
max_depth = 5
learning_rate = 0.1
scale_pos_weight = auto

# Random Forest
n_estimators = 100
max_depth = 10
class_weight = 'balanced'
calibration = 'sigmoid'
```

---

## Feature List for ML Models

### Satellite Features (8)
1. NDVI - Normalized Difference Vegetation Index
2. NDWI - Normalized Difference Water Index
3. BSI - Bare Soil Index
4. iron_oxide_ratio - Red / NIR
5. clay_minerals - SWIR1 / SWIR2
6. elevation - SRTM elevation (meters)
7. slope - Terrain slope (degrees)
8. aspect - Terrain aspect (degrees)

### Geological Features (10)
9. mindat_distance_km - Distance to nearest mineral occurrence
10. mindat_count - Count of mineral occurrences nearby
11. basalt_presence - Binary (0 or 1)
12. limestone_presence - Binary (0 or 1)
13. volcanic_proximity_km - Distance to known volcanic region
14. sedimentary_score - 0.0-0.8 score
15. nearest_fault_km - Distance to nearest fault
16. fault_density - Faults per 100 km²
17. seismic_activity_score - 0.0-1.0 score
18. recent_earthquakes - Count (2020-present, mag >= 2.0)

**Total: 18 features for ML prediction**

---

## Data Pipelines

### Geode Detection Pipeline
```
Input: (lat, lon)
  ↓
Extract Satellite Features (8 features)
  ↓
Query External Data (USGS, Mindat, Earthquakes)
  ↓
Try ML Model (if available)
  ├→ If available: Predict with confidence
  └→ If unavailable: Use heuristic weighting
  ↓
Add Proximity Bonus (known geode sites)
  ↓
Output: Dict with probability, method, indicators, confidence
```

### ML Training Pipeline
```
Input: Training data with labels
  ↓
Feature Preparation: StandardScaler + NaN filling
  ↓
Train 3 Models in Parallel:
  ├→ Logistic Regression
  ├→ XGBoost (if available)
  └→ Random Forest
  ↓
Calibrate: CalibratedClassifierCV
  ↓
Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC
  ↓
Select Best Model by F1 Score
  ↓
Save: Pickle to disk
```

### CNN Training Pipeline
```
Input: Satellite imagery patches + bounding boxes
  ↓
Data Augmentation: Crops, flips, rotations, color jitter
  ↓
Batch Processing: Adam optimizer + Learning rate scheduling
  ↓
Loss Function: Focal Loss (for class imbalance)
  ↓
Monitor: Validation loss, early stopping (patience=15)
  ↓
Checkpointing: Save best model
  ↓
Output: Trained weights + metadata
```

---

## Error Handling by Component

### ✅ Good Error Handling
- Satellite feature extraction (validates all 8 features present)
- ProductionConfig initialization (credential validation)
- API coordinate validation (-90 to 90 lat, -180 to 180 lon)
- Training data CSV parsing (malformed row skipping)
- Fallback chain for geological data

### ❌ Poor Error Handling
- External API calls (Mindat, USGS, Earthquakes) - NO RETRY LOGIC
- ML model loading (logs warning, continues)
- CNN input validation (no shape checking)
- NaN value handling (fill with 0, no validation)
- Timeout configuration (using requests default)

---

## Configuration Files

### Required Environment Variables
```bash
# Google Earth Engine
export GEE_PROJECT_ID=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Optional Satellite Providers
export MAPBOX_ACCESS_TOKEN=your_token
export SENTINELHUB_CLIENT_ID=your_id
export SENTINELHUB_CLIENT_SECRET=your_secret
export PLANET_API_KEY=your_key

# Optional External Data APIs
export MINDAT_API_KEY=your_key
export USGS_API_KEY=your_key

# Production Mode
export PRODUCTION_MODE=true
```

### Hyperparameter Configuration File
```
Location: hyperparameter_config.yaml (112 lines)

Key Sections:
- model: Architecture, dropout, attention
- training: Batch size, epochs, LR, weight decay
- augmentation: Flip, rotation, color jitter, mixup
- loss: Type (focal/cross_entropy), smoothing
- scheduler: Type (cosine/plateau/exponential)
- early_stopping: Patience, min_delta
- data: Image size, normalization, workers
- class_balance: Weighted sampling, class weights
```

---

## Known Issues & TODOs

### 🔴 Critical - Fix This Week
- [ ] Add retry logic with exponential backoff to mindat API
- [ ] Add retry logic to USGS API calls  
- [ ] Add retry logic to earthquake API calls
- [ ] Extract hardcoded thresholds to config file

### 🟡 Medium - Fix This Month
- [ ] Expand training dataset (5→20+ samples)
- [ ] Add unit tests for feature extraction
- [ ] Add tensor shape validation in CNN
- [ ] Document weight scheme decisions
- [ ] Add structured logging with context

### 🟢 Low - Fix Next Quarter
- [ ] Multi-scale feature extraction
- [ ] Regional threshold customization
- [ ] Active learning pipeline
- [ ] Model monitoring and versioning

---

## Testing Commands

```bash
# Run existing tests
python -m pytest tests/test_retry_logic.py -v

# Run full pipeline test
python test_full_pipeline.py

# Validate production configuration
python validate_production.py

# Train CNN model
python train_quick_demo.py --data-root frontend/training_data

# Train ML models
python -c "from satellite_production_module import train_and_evaluate_models; train_and_evaluate_models()"

# Run API
export PORT=5000
python treasure_api.py

# Health check
curl -s http://localhost:5000/healthz | jq .
```

---

## Performance Metrics

### Earth Engine
- Cloud filtering: 20% threshold
- Image count: 50 images → 1 median composite
- Processing time: ~5-10 seconds per location
- API quota: 1,000,000 max pixels per request

### ML Models
- Training samples: ~10 (too few!)
- Features: 18
- Models: 3 (auto-selected by F1)
- Training time: <1 minute
- Prediction time: <100ms

### CNN Models
- Input size: 128×128 (configurable)
- Training time: Hours (GPU recommended)
- Inference time: ~100ms per image
- Batch processing: 32 samples

---

## Contributing Guidelines

When modifying algorithms:
1. Update hardcoded values in config file, not code
2. Add input validation before predictions
3. Implement retry logic for external API calls
4. Add comprehensive docstrings
5. Update this reference guide
6. Run validation scripts before committing

---

**Last Updated:** 2025-11-03  
**For Details:** See `ALGORITHM_REVIEW.md` (26 KB)

