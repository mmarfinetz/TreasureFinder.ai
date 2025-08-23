#!/usr/bin/env python3
"""
Fixed TreasureHunter Web API with lazy GEE initialization and proper health checks.
This version won't block on startup and provides better Railway compatibility.
"""

import os
import sys
import json
import traceback
import threading
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

# Load environment variables from a .env file if present
try:
    from dotenv import load_dotenv, find_dotenv
    _env_path = os.environ.get('ENV_FILE') or find_dotenv()
    if _env_path:
        load_dotenv(_env_path, override=False)
    else:
        load_dotenv(override=False)
except Exception:
    pass

# Set production mode environment
os.environ['PRODUCTION_MODE'] = 'true'
if 'MOCK_DATA' in os.environ:
    os.environ.pop('MOCK_DATA', None)

# Import the lazy initialization patch
try:
    from gee_lazy_init_patch import lazy_initialize_earth_engine, EE_AVAILABLE
except ImportError:
    print("⚠️ GEE lazy init patch not found, will use eager loading")
    EE_AVAILABLE = False
    def lazy_initialize_earth_engine():
        return False

# Import TreasureHunter functions
NOTEBOOK_FUNCTIONS_AVAILABLE = False
try:
    from treasure_hunter_module import (
        main_analysis,
        analyze_satellite_anomalies,
        combined_analysis,
        scan_region_comprehensive,
        predict_discovery_zones,
        EXAMPLE_LOCATIONS
    )
    NOTEBOOK_FUNCTIONS_AVAILABLE = True
except ImportError:
    print("⚠️ TreasureHunter module not found - will use API-only mode")
    EXAMPLE_LOCATIONS = {
        'giza': (29.9792, 31.1342),
        'machu_picchu': (-13.1631, -72.5450),
        'angkor_wat': (13.4125, 103.8670),
        'easter_island': (-27.1127, -109.3497),
        'stonehenge': (51.1789, -1.8262),
        'petra': (30.3285, 35.4444),
        'chichen_itza': (20.6843, -88.5678),
        'oak_island': (44.5133, -64.2947),
    }

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Global configuration
API_VERSION = "1.1"
MAX_ANALYSIS_POINTS = 100
MAX_RADIUS_KM = 500

# Model training state
training_sessions = {}
current_model = None
model_metadata = {
    'name': None,
    'accuracy': None,
    'trained_date': None
}

# Model storage directory
MODELS_DIR = 'saved_models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def validate_coordinates(lat, lon):
    """Validate latitude and longitude values."""
    try:
        lat = float(lat)
        lon = float(lon)
        
        if not (-90 <= lat <= 90):
            return False, "Latitude must be between -90 and 90"
        if not (-180 <= lon <= 180):
            return False, "Longitude must be between -180 and 180"
        
        return True, (lat, lon)
    except (ValueError, TypeError):
        return False, "Invalid coordinate format"

@app.route('/healthz')
def health_check():
    """Basic health check that doesn't require GEE."""
    return jsonify({
        'status': 'healthy',
        'service': 'treasure-api',
        'version': API_VERSION,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/healthz/gee')
def health_check_gee():
    """Health check that tests GEE connectivity."""
    try:
        # Try to initialize GEE lazily
        if lazy_initialize_earth_engine():
            # Test with a simple operation
            import ee
            result = ee.Number(1).getInfo()
            return jsonify({
                'status': 'healthy',
                'gee_available': True,
                'test_result': result,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'status': 'degraded',
                'gee_available': False,
                'error': 'GEE initialization failed',
                'timestamp': datetime.now().isoformat()
            }), 503
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'gee_available': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/api/status')
def api_status():
    """Get API status and configuration."""
    # Initialize GEE if not already done (non-blocking)
    gee_status = "available" if lazy_initialize_earth_engine() else "unavailable"
    
    # Check for satellite providers
    providers = []
    if os.environ.get('MAPBOX_ACCESS_TOKEN'):
        providers.append('Mapbox')
    if gee_status == "available":
        providers.append('Google Earth Engine')
    if os.environ.get('SENTINEL_HUB_CLIENT_ID'):
        providers.append('Sentinel Hub')
    if os.environ.get('PLANET_API_KEY'):
        providers.append('Planet Labs')
    
    return jsonify({
        'status': 'running',
        'version': API_VERSION,
        'notebook_functions': NOTEBOOK_FUNCTIONS_AVAILABLE,
        'earth_engine': gee_status,
        'providers': providers,
        'capabilities': {
            'single_point_analysis': NOTEBOOK_FUNCTIONS_AVAILABLE,
            'regional_analysis': NOTEBOOK_FUNCTIONS_AVAILABLE,
            'predictive_discovery': NOTEBOOK_FUNCTIONS_AVAILABLE,
            'model_training': bool(current_model or NOTEBOOK_FUNCTIONS_AVAILABLE),
            'geocoding': True
        },
        'limits': {
            'max_analysis_points': MAX_ANALYSIS_POINTS,
            'max_radius_km': MAX_RADIUS_KM
        },
        'environment': {
            'railway': bool(os.environ.get('RAILWAY_ENVIRONMENT')),
            'production': os.environ.get('PRODUCTION_MODE') == 'true'
        }
    })

@app.route('/api/example-locations')
def get_example_locations():
    """Get list of example locations for testing."""
    return jsonify({
        'locations': EXAMPLE_LOCATIONS,
        'total': len(EXAMPLE_LOCATIONS)
    })

@app.route('/api/analyze/single', methods=['POST'])
def analyze_single():
    """Analyze a single location for anomalies."""
    
    if not NOTEBOOK_FUNCTIONS_AVAILABLE:
        return jsonify({
            'error': 'Analysis functions not available. Please ensure TreasureHunter module is loaded.'
        }), 503
    
    try:
        data = request.get_json()
        
        # Validate required parameters
        lat = data.get('latitude')
        lon = data.get('longitude')
        
        if lat is None or lon is None:
            return jsonify({'error': 'Missing latitude or longitude'}), 400
        
        # Validate coordinates
        valid, result = validate_coordinates(lat, lon)
        if not valid:
            return jsonify({'error': result}), 400
        
        lat, lon = result
        
        # Initialize GEE if needed (lazy)
        if not lazy_initialize_earth_engine():
            print("⚠️ GEE not available, will use fallback methods")
        
        # Perform analysis
        result = analyze_satellite_anomalies(lat, lon)
        
        return jsonify({
            'success': True,
            'location': {
                'latitude': lat,
                'longitude': lon
            },
            'analysis': result
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

@app.route('/api/analyze/region', methods=['POST'])
def analyze_region():
    """Analyze a region for anomalies."""
    
    if not NOTEBOOK_FUNCTIONS_AVAILABLE:
        return jsonify({
            'error': 'Analysis functions not available. Please ensure TreasureHunter module is loaded.'
        }), 503
    
    try:
        data = request.get_json()
        
        # Extract and validate parameters
        lat = data.get('latitude')
        lon = data.get('longitude')
        radius_km = data.get('radius_km', 10)
        num_points = data.get('num_points', 20)
        
        if lat is None or lon is None:
            return jsonify({'error': 'Missing latitude or longitude'}), 400
        
        # Validate coordinates
        valid, result = validate_coordinates(lat, lon)
        if not valid:
            return jsonify({'error': result}), 400
        
        lat, lon = result
        
        # Validate radius and points
        if radius_km > MAX_RADIUS_KM:
            return jsonify({'error': f'Radius exceeds maximum of {MAX_RADIUS_KM} km'}), 400
        
        if num_points > MAX_ANALYSIS_POINTS:
            return jsonify({'error': f'Number of points exceeds maximum of {MAX_ANALYSIS_POINTS}'}), 400
        
        # Initialize GEE if needed (lazy)
        if not lazy_initialize_earth_engine():
            print("⚠️ GEE not available, will use fallback methods")
        
        # Perform regional analysis
        region_name = data.get('region_name', f'Region_{lat:.2f}_{lon:.2f}')
        results = main_analysis(region_name, (lat, lon), radius_km, num_points)
        
        return jsonify({
            'success': True,
            'region': {
                'name': region_name,
                'center_latitude': lat,
                'center_longitude': lon,
                'radius_km': radius_km,
                'num_points': num_points
            },
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Regional analysis failed: {str(e)}',
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

# Serve frontend
@app.route('/')
def serve_frontend():
    """Serve the frontend application."""
    if os.path.exists('frontend/index.html'):
        return send_from_directory('frontend', 'index.html')
    else:
        return jsonify({
            'message': 'TreasureHunter API',
            'version': API_VERSION,
            'docs': '/api/status'
        })

if __name__ == '__main__':
    # Check if running in debug mode
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print("🏴‍☠️ TreasureHunter Web API Starting (Fixed Version)...")
    print(f"Debug mode: {debug_mode}")
    print(f"Notebook functions available: {NOTEBOOK_FUNCTIONS_AVAILABLE}")
    print(f"API Version: {API_VERSION}")
    
    # Don't initialize GEE at startup - it will be done lazily
    print("📌 GEE will be initialized on first use (lazy loading)")
    
    if not NOTEBOOK_FUNCTIONS_AVAILABLE:
        print("\n⚠️ WARNING: Analysis functions not available!")
        print("Convert TreasureHunter.ipynb to treasure_hunter_module.py")
        print("Run: python convert_notebook.py")
    
    print("\n🌐 API Endpoints:")
    print("  GET  /healthz                    - Basic health check")
    print("  GET  /healthz/gee                - GEE health check")
    print("  GET  /api/status                 - API status")
    print("  GET  /api/example-locations      - Example locations")
    print("  POST /api/analyze/single         - Single point analysis")
    print("  POST /api/analyze/region         - Regional analysis")
    print("\n🚀 Starting server...")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"🔌 Binding to 0.0.0.0:{port}")
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug_mode,
        threaded=True
    )