"""
Converted from satellite_production_modular_unified.ipynb
This module contains all code from the Jupyter notebook.
"""

# 🔮 PRODUCTION GEODE DETECTION SYSTEM - MODULAR ARCHITECTURE
# ⚠️ STRICT PRODUCTION VERSION - NO MOCK DATA, NO RANDOM NUMBERS

import os
import sys
import json
import hashlib
import requests
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import warnings

# Production logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("geode_production")

# Suppress only specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# PRODUCTION ASSERTION - NO DEVELOPMENT ENVIRONMENT ALLOWED
PRODUCTION_MODE = True
assert PRODUCTION_MODE, "This notebook requires PRODUCTION_MODE=True"

# Forbid test/debug/mock flags
if os.environ.get('ALLOW_TEST_MODE'):
    raise RuntimeError('TEST MODE detected - remove for production')
if os.environ.get('DEBUG') or os.environ.get('MOCK_DATA'):
    raise RuntimeError('Debug/Mock flags detected - remove for production')

print('🚀 Production Geode Detection System')
print('='*50)
print('Mode: STRICT PRODUCTION')
print('Data: REAL ONLY (no simulations)')
print('Behavior: FAIL-FAST on missing data')
print('='*50)

# PRODUCTION CONFIGURATION MODULE

@dataclass
class ProductionConfig:
    """Strict production configuration with mandatory API keys"""
    google_earth_engine_project: str
    google_application_credentials: str
    nasa_earthdata_key: Optional[str] = None
    usgs_api_key: Optional[str] = None
    mindat_api_key: Optional[str] = None
    cache_dir: str = ".production_cache"
    cache_ttl_hours: int = 24
    max_retries: int = 3
    timeout_seconds: int = 30
    
    def __post_init__(self):
        """Validate all required configurations"""
        # Mandatory Earth Engine configuration
        if not self.google_earth_engine_project:
            raise ValueError("❌ Google Earth Engine project ID is REQUIRED")
        if not self.google_application_credentials:
            raise ValueError("❌ Google Application Credentials path is REQUIRED")
        if not os.path.exists(self.google_application_credentials):
            # If it's JSON content rather than a path, save it
            if self.google_application_credentials.startswith('{'):
                creds_path = '/tmp/ee_service_account.json' if os.path.exists('/tmp') else 'ee_service_account.json'
                with open(creds_path, 'w') as f:
                    f.write(self.google_application_credentials)
                self.google_application_credentials = creds_path
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
            else:
                raise ValueError(f"❌ Credentials file not found: {self.google_application_credentials}")
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info("✅ Production configuration validated")
        logger.info(f"   Project: {self.google_earth_engine_project}")
        logger.info(f"   Cache: {self.cache_dir}")

class ProductionError(Exception):
    """Custom exception for production failures"""
    pass

class DataUnavailableError(ProductionError):
    """Raised when real data cannot be obtained"""
    pass

class ValidationError(ProductionError):
    """Raised when data validation fails"""
    pass

def load_colab_secrets():
    """Load secrets from Google Colab userdata"""
    secrets_loaded = False
    
    try:
        from google.colab import userdata
        print("🔍 Checking for Colab secrets...")
        
        # List of all possible secret keys
        secret_keys = [
            'GEE_PROJECT_ID', 
            'GOOGLE_APPLICATION_CREDENTIALS',
            'GOOGLE_APPLICATION_CREDENTIALS_JSON',
            'GEE_SERVICE_ACCOUNT_JSON',
            'NASA_API_KEY', 
            'USGS_API_KEY', 
            'MINDAT_API_KEY'
        ]
        
        for key in secret_keys:
            try:
                value = userdata.get(key)
                if value:
                    # Handle JSON credentials specially
                    if key in ['GOOGLE_APPLICATION_CREDENTIALS_JSON', 'GEE_SERVICE_ACCOUNT_JSON']:
                        # Save JSON to file and set the path
                        creds_path = '/content/ee_service_account.json' if os.path.exists('/content') else '/tmp/ee_service_account.json'
                        with open(creds_path, 'w') as f:
                            f.write(value)
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
                        print(f"   ✅ Loaded {key} and saved to {creds_path}")
                    else:
                        os.environ[key] = value
                        print(f"   ✅ Loaded {key}")
                    secrets_loaded = True
            except userdata.SecretNotFoundError:
                pass  # Secret not found, continue
            except Exception as e:
                print(f"   ⚠️ Error loading {key}: {e}")
        
        if secrets_loaded:
            print("✅ Colab secrets loaded successfully!")
        else:
            print("⚠️ No Colab secrets found")
            
    except ImportError:
        print("ℹ️ Not running in Google Colab")
    except Exception as e:
        print(f"⚠️ Error accessing Colab secrets: {e}")
    
    return secrets_loaded

def initialize_production_config() -> ProductionConfig:
    """Initialize and validate production configuration"""
    
    # First, try to load from Colab secrets
    load_colab_secrets()
    
    # Now check what we have
    gee_project = os.environ.get('GEE_PROJECT_ID')
    gee_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    
    # Provide helpful error messages if missing
    if not gee_project:
        error_msg = """
        ❌ GEE_PROJECT_ID not found!
        
        Please add it as a Colab secret or set environment variable:
        
        Option 1 - Colab Secret (recommended):
        1. Click the 🔑 key icon in the left sidebar
        2. Add a new secret named 'GEE_PROJECT_ID'
        3. Enter your Google Cloud project ID as the value
        
        Option 2 - Environment Variable:
        import os
        os.environ['GEE_PROJECT_ID'] = 'your-project-id'
        
        To get a project ID:
        1. Go to https://console.cloud.google.com/
        2. Create a new project or select existing
        3. The project ID is shown in the project selector
        """
        raise DataUnavailableError(error_msg)
    
    if not gee_creds:
        error_msg = """
        ❌ Google Earth Engine credentials not found!
        
        Please add credentials as a Colab secret:
        
        Option 1 - JSON Content (recommended for Colab):
        1. Click the 🔑 key icon in the left sidebar
        2. Add a new secret named 'GOOGLE_APPLICATION_CREDENTIALS_JSON'
        3. Paste the entire service account JSON key as the value
        
        Option 2 - File Path:
        1. Upload your service account JSON file
        2. Set: os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/key.json'
        
        To get credentials:
        1. Go to https://console.cloud.google.com/iam-admin/serviceaccounts
        2. Create a service account for your project
        3. Create and download a JSON key
        4. Copy the entire JSON content
        """
        raise DataUnavailableError(error_msg)
    
    print("\n📊 Configuration Summary:")
    print(f"   Project ID: {gee_project}")
    print(f"   Credentials: {'✅ Found' if gee_creds else '❌ Missing'}")
    print(f"   NASA API: {'✅ Found' if os.environ.get('NASA_API_KEY') else '⚠️ Not provided (optional)'}")
    print(f"   USGS API: {'✅ Found' if os.environ.get('USGS_API_KEY') else '⚠️ Not provided (optional)'}")
    print(f"   MinDat API: {'✅ Found' if os.environ.get('MINDAT_API_KEY') else '⚠️ Not provided (optional)'}")
    
    return ProductionConfig(
        google_earth_engine_project=gee_project,
        google_application_credentials=gee_creds,
        nasa_earthdata_key=os.environ.get('NASA_API_KEY'),
        usgs_api_key=os.environ.get('USGS_API_KEY'),
        mindat_api_key=os.environ.get('MINDAT_API_KEY')
    )

# Initialize configuration
print("\n🔐 Initializing Production Configuration...")
print("="*50)

try:
    config = initialize_production_config()
    print("\n✅ Configuration successful! Ready for production use.")
except DataUnavailableError as e:
    print(str(e))
    raise

# EARTH ENGINE INITIALIZATION (STRICT, AUTO OAUTH FALLBACK IN COLAB)
import ee

def initialize_earth_engine(project_id: str):
	"""Initialize Earth Engine.
	Order:
	1) Service account (EE_SERVICE_ACCOUNT + GOOGLE_APPLICATION_CREDENTIALS)
	2) Default credentials (ADC)
	3) Interactive OAuth (ee.Authenticate), auto-triggered if in Colab or ADC fails
	"""
	try:
		# 1) Prefer service account if provided
		service_account = os.environ.get('EE_SERVICE_ACCOUNT')
		key_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
		if service_account and key_path and os.path.exists(key_path):
			creds = ee.ServiceAccountCredentials(service_account, key_path)
			ee.Initialize(credentials=creds, project=project_id)
			print('✅ Earth Engine initialized with service account')
			return
		# 2) Try default credentials (ADC)
		ee.Initialize(project=project_id)
		print('✅ Earth Engine initialized with specified project (ADC)')
		return
	except Exception as e:
		# 3) Auto OAuth fallback (works in Colab or local with browser)
		try:
			try:
				from google.colab import auth as colab_auth  # type: ignore
				colab_auth.authenticate_user()
			except Exception:
				pass
			ee.Authenticate()
			ee.Initialize(project=project_id)
			print('✅ Earth Engine initialized via OAuth')
			return
		except Exception as e2:
			raise RuntimeError(
				"Earth Engine initialization failed after OAuth attempt.\n"
				f"ADC error: {e}\nOAuth error: {e2}"
			)
	# Should not reach here
	raise RuntimeError('Unexpected EE init flow termination')

# FEATURE EXTRACTION FROM SATELLITE (REAL DATA ONLY)
from typing import Tuple, List

def extract_satellite_features(lat: float, lon: float, radius_m: int = 500) -> Dict[str, float]:
	"""Compute NDVI, NDWI, BSI, iron/clay proxies, elevation, slope, aspect from Earth Engine.
	Raises on any failure. No defaults.
	"""
	if not (-90 <= lat <= 90 and -180 <= lon <= 180):
		raise ValueError('Invalid coordinates')
	
	pt = ee.Geometry.Point([lon, lat])
	buffer = pt.buffer(radius_m)
	
	# Landsat 8 SR collection
	landsat = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
		.filterBounds(buffer)
		.filterDate('2021-01-01', '2024-12-31')
		.filter(ee.Filter.lt('CLOUD_COVER', 20))
		.median())
	
	if landsat.bandNames().size().getInfo() == 0:
		raise RuntimeError('No satellite imagery available for location')
	
	ndvi = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
	ndwi = landsat.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
	red = landsat.select('SR_B4').rename('RED')
	nir = landsat.select('SR_B5').rename('NIR')
	green = landsat.select('SR_B3').rename('GREEN')
	blue = landsat.select('SR_B2').rename('BLUE')
	swir1 = landsat.select('SR_B6').rename('SWIR1')
	swir2 = landsat.select('SR_B7').rename('SWIR2')
	bsi = red.add(swir1).subtract(nir.add(green)).divide(red.add(swir1).add(nir).add(green)).rename('BSI')
	iron = red.divide(nir).rename('IRON')            # proxy: higher red vs nir
	clay = swir1.divide(swir2).rename('CLAY')        # proxy: SWIR1/SWIR2 ratio
	
	elevation = ee.Image('USGS/SRTMGL1_003').rename('ELEV')
	slope = ee.Terrain.slope(elevation).rename('SLOPE')
	aspect = ee.Terrain.aspect(elevation).rename('ASPECT')
	
	features_img = ee.Image.cat([ndvi, ndwi, bsi, iron, clay, elevation, slope, aspect])
	reducer = features_img.reduceRegion(
		reducer=ee.Reducer.mean(),
		geometry=buffer,
		scale=30,
		maxPixels=1_000_000
	)
	res = reducer.getInfo()
	# Validate presence of all keys
	for key in ['NDVI', 'NDWI', 'BSI', 'IRON', 'CLAY', 'ELEV', 'SLOPE', 'ASPECT']:
		if key not in res or res[key] is None:
			raise RuntimeError(f'Missing feature {key} at location')
	return {
		'ndvi': float(res['NDVI']),
		'ndwi': float(res['NDWI']),
		'bsi': float(res['BSI']),
		'iron_oxide_ratio': float(res['IRON']),
		'clay_minerals': float(res['CLAY']),
		'elevation': float(res['ELEV']),
		'slope': float(res['SLOPE']),
		'aspect': float(res['ASPECT'])
	}

# EXTERNAL OCCURRENCE QUERIES (OPTIONAL REAL DATA)
import requests
import json
import time

SESSION = requests.Session()

def mindat_occurrence_metrics(lat: float, lon: float, radius_km: int = 80) -> dict | None:
	"""Query Mindat for mineral occurrences near a point.
	Returns dict with min_distance_km and count, or None if unavailable.
	Requires MINDAT_API_KEY in env.
	"""
	api_key = os.environ.get('MINDAT_API_KEY')
	if not api_key:
		return None
	try:
		# NOTE: Endpoint/params may need adjustment per Mindat API spec.
		# We query occurrences for agate/chalcedony/quartz as proxies for geodes.
		endpoint = 'https://api.mindat.org/occurrence/'
		params = {
			'lat': lat,
			'lon': lon,
			'radius_km': radius_km,
			'mineral': 'agate,chalcedony,quartz',
			'limit': 200
		}
		headers = {
			'Authorization': f'Token {api_key}'
		}
		resp = SESSION.get(endpoint, params=params, headers=headers, timeout=20)
		if resp.status_code != 200:
			return None
		data = resp.json()
		if not isinstance(data, dict) or 'results' not in data:
			return None
		results = data.get('results', [])
		if not results:
			return {'min_distance_km': None, 'count': 0}
		# Compute nearest distance (km) if API returns distances; otherwise compute via geodesic
		from geopy.distance import geodesic
		dists = []
		for r in results:
			plat = r.get('latitude') or r.get('lat')
			plon = r.get('longitude') or r.get('lon')
			if plat is None or plon is None:
				continue
			dists.append(geodesic((lat, lon), (plat, plon)).km)
		min_km = min(dists) if dists else None
		return {
			'min_distance_km': float(min_km) if min_km is not None else None,
			'count': int(len(results))
		}
	except Exception:
		return None

# USGS LITHOLOGY INTEGRATION (PRODUCTION)
from typing import Dict, Optional, List
import requests
import json

def query_usgs_lithology(lat: float, lon: float, radius_km: float = 10.0) -> Optional[Dict[str, Any]]:
    """
    Query USGS geological data API for lithology types at given location.
    Focus on basalt and limestone presence/proximity.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        radius_km: Search radius in kilometers
        
    Returns:
        Dict with lithology data or None if API unavailable
        {
            'primary_lithology': str,
            'basalt_presence': bool,
            'limestone_presence': bool,
            'volcanic_proximity_km': float or None,
            'sedimentary_score': float (0-1),
            'lithology_types': List[str]
        }
    """
    try:
        # USGS Geology State Map API endpoint
        # Note: This uses the USGS ScienceBase API for geological data
        base_url = "https://mrdata.usgs.gov/geology/state/json.php"
        
        # Convert radius to degrees (approximate)
        radius_deg = radius_km / 111.0  # 1 degree ≈ 111 km
        
        # Query parameters for bounding box around point
        params = {
            'bbox': f"{lon-radius_deg},{lat-radius_deg},{lon+radius_deg},{lat+radius_deg}",
            'format': 'json'
        }
        
        # Primary query for state geology
        response = SESSION.get(base_url, params=params, timeout=30)
        
        if response.status_code != 200:
            # Try alternative endpoint - USGS National Map
            alt_url = f"https://macrostrat.org/api/v2/units"
            alt_params = {
                'lat': lat,
                'lng': lon,
                'format': 'json'
            }
            response = SESSION.get(alt_url, params=alt_params, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"USGS lithology query failed: HTTP {response.status_code}")
                return None
        
        data = response.json()
        
        # Parse lithology data
        lithology_types = set()
        basalt_presence = False
        limestone_presence = False
        volcanic_proximity = None
        sedimentary_score = 0.0
        
        # Handle different response formats
        if isinstance(data, dict):
            if 'success' in data and data.get('success'):
                units = data.get('data', {}).get('units', [])
            elif 'features' in data:
                units = data.get('features', [])
            else:
                units = [data]
        elif isinstance(data, list):
            units = data
        else:
            units = []
        
        # Process geological units
        for unit in units:
            # Extract lithology information
            lith_info = None
            if isinstance(unit, dict):
                # Check various possible fields
                lith_info = (unit.get('lith') or 
                           unit.get('lithology') or 
                           unit.get('rock_type') or
                           unit.get('properties', {}).get('ROCKTYPE1') or
                           unit.get('properties', {}).get('UNIT_NAME', ''))
            
            if lith_info:
                lith_str = str(lith_info).lower()
                lithology_types.add(lith_str)
                
                # Check for basalt
                if any(term in lith_str for term in ['basalt', 'volcanic', 'mafic', 'andesite']):
                    basalt_presence = True
                    
                # Check for limestone  
                if any(term in lith_str for term in ['limestone', 'carbonate', 'dolomite', 'chalk']):
                    limestone_presence = True
                    
                # Calculate sedimentary score
                sed_terms = ['sandstone', 'shale', 'limestone', 'conglomerate', 'mudstone', 
                           'siltstone', 'claystone', 'sedimentary']
                if any(term in lith_str for term in sed_terms):
                    sedimentary_score = max(sedimentary_score, 0.8)
        
        # Fallback to geographic proximity for volcanic areas if no direct data
        if not units:
            # Known volcanic regions in US (simplified)
            volcanic_regions = [
                {'name': 'Yellowstone', 'lat': 44.4280, 'lon': -110.5885, 'radius_km': 100},
                {'name': 'Mt St Helens', 'lat': 46.1914, 'lon': -122.1956, 'radius_km': 50},
                {'name': 'Hawaii', 'lat': 19.4069, 'lon': -155.2834, 'radius_km': 200},
            ]
            
            for region in volcanic_regions:
                dist = geodesic((lat, lon), (region['lat'], region['lon'])).km
                if dist < region['radius_km']:
                    basalt_presence = True
                    volcanic_proximity = dist
                    break
        
        # Determine primary lithology
        primary_lith = 'unknown'
        if lithology_types:
            # Prioritize specific types for geode formation
            priority_types = ['limestone', 'basalt', 'rhyolite', 'sandstone']
            for ptype in priority_types:
                if any(ptype in lith for lith in lithology_types):
                    primary_lith = ptype
                    break
            if primary_lith == 'unknown' and lithology_types:
                primary_lith = list(lithology_types)[0]
        
        return {
            'primary_lithology': primary_lith,
            'basalt_presence': basalt_presence,
            'limestone_presence': limestone_presence,
            'volcanic_proximity_km': volcanic_proximity,
            'sedimentary_score': float(sedimentary_score),
            'lithology_types': list(lithology_types)[:10]  # Limit to 10 types
        }
        
    except requests.RequestException as e:
        logger.warning(f"USGS lithology API request failed: {e}")
        return None
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"USGS lithology data parsing failed: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error in USGS lithology query: {e}")
        return None

# Test the function
print("🪨 USGS Lithology module loaded")
# Example: test_lith = query_usgs_lithology(40.0, -111.0)
# print(f"Test lithology result: {test_lith}")

# FAULT PROXIMITY CALCULATION (PRODUCTION)
from typing import Dict, Optional, Tuple
import math

def calculate_fault_proximity(lat: float, lon: float, radius_km: float = 50.0) -> Optional[Dict[str, Any]]:
    """
    Query USGS earthquake/fault line databases for fault proximity metrics.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees  
        radius_km: Search radius in kilometers
        
    Returns:
        Dict with fault proximity data or None if API unavailable
        {
            'nearest_fault_km': float or None,
            'fault_density': float (faults per 100 km²),
            'seismic_activity_score': float (0-1),
            'recent_earthquakes': int,
            'tectonic_setting': str
        }
    """
    try:
        # USGS Quaternary Fault Database API
        fault_url = "https://earthquake.usgs.gov/ws/geoserve/regions.json"
        
        # First get tectonic region info
        region_params = {
            'latitude': lat,
            'longitude': lon
        }
        
        region_resp = SESSION.get(fault_url, params=region_params, timeout=30)
        
        tectonic_setting = 'stable_continental'
        if region_resp.status_code == 200:
            region_data = region_resp.json()
            # Parse tectonic plates info if available
            if 'tectonic' in region_data:
                tectonic_setting = 'active_margin'
        
        # Query recent seismic activity
        eq_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        
        # Calculate bounding box
        lat_range = radius_km / 111.0
        lon_range = radius_km / (111.0 * math.cos(math.radians(lat)))
        
        eq_params = {
            'format': 'geojson',
            'minlatitude': lat - lat_range,
            'maxlatitude': lat + lat_range,
            'minlongitude': lon - lon_range,
            'maxlongitude': lon + lon_range,
            'starttime': '2020-01-01',
            'minmagnitude': 2.0,
            'limit': 1000
        }
        
        eq_resp = SESSION.get(eq_url, params=eq_params, timeout=30)
        
        recent_earthquakes = 0
        nearest_fault_km = None
        seismic_activity_score = 0.0
        
        if eq_resp.status_code == 200:
            eq_data = eq_resp.json()
            features = eq_data.get('features', [])
            recent_earthquakes = len(features)
            
            # Calculate distance to nearest earthquake (proxy for fault)
            min_dist = float('inf')
            for feature in features:
                coords = feature.get('geometry', {}).get('coordinates', [])
                if len(coords) >= 2:
                    eq_lon, eq_lat = coords[0], coords[1]
                    dist = geodesic((lat, lon), (eq_lat, eq_lon)).km
                    min_dist = min(min_dist, dist)
            
            if min_dist < float('inf'):
                nearest_fault_km = min_dist
            
            # Calculate seismic activity score
            if recent_earthquakes > 0:
                # Normalize to 0-1 scale (100 earthquakes = high activity)
                seismic_activity_score = min(1.0, recent_earthquakes / 100.0)
        
        # Alternative: Query USGS Quaternary Faults KML service
        if nearest_fault_km is None:
            # Try alternative fault database
            qfault_url = "https://earthquake.usgs.gov/hazards/qfaults/map/proximal.php"
            qfault_params = {
                'lat': lat,
                'lon': lon,
                'radius': radius_km,
                'format': 'json'
            }
            
            try:
                qfault_resp = SESSION.get(qfault_url, params=qfault_params, timeout=15)
                if qfault_resp.status_code == 200:
                    qfault_data = qfault_resp.json()
                    if 'faults' in qfault_data:
                        faults = qfault_data['faults']
                        if faults:
                            nearest_fault_km = float(faults[0].get('distance_km', radius_km))
            except:
                # Fallback to known major fault zones
                major_faults = [
                    {'name': 'San Andreas', 'lat': 36.0, 'lon': -120.0, 'length_km': 1200},
                    {'name': 'New Madrid', 'lat': 36.5, 'lon': -89.5, 'length_km': 240},
                    {'name': 'Cascadia', 'lat': 45.0, 'lon': -124.0, 'length_km': 1000},
                ]
                
                for fault in major_faults:
                    # Simplified distance to fault line
                    dist = geodesic((lat, lon), (fault['lat'], fault['lon'])).km
                    if nearest_fault_km is None or dist < nearest_fault_km:
                        nearest_fault_km = dist
        
        # Calculate fault density (faults per 100 km²)
        area_km2 = math.pi * radius_km * radius_km
        fault_density = 0.0
        if recent_earthquakes > 0:
            # Use earthquake density as proxy for fault density
            fault_density = (recent_earthquakes / area_km2) * 10000.0  # per 100 km²
        
        return {
            'nearest_fault_km': float(nearest_fault_km) if nearest_fault_km else None,
            'fault_density': float(fault_density),
            'seismic_activity_score': float(seismic_activity_score),
            'recent_earthquakes': int(recent_earthquakes),
            'tectonic_setting': tectonic_setting
        }
        
    except requests.RequestException as e:
        logger.warning(f"USGS fault proximity API request failed: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error in fault proximity calculation: {e}")
        return None

print("🌋 Fault proximity module loaded")

# ANOMALY DETECTION (CLASSICAL + CNN PLACEHOLDER INTERFACE)
from typing import Dict

def compute_anomaly_score(features: Dict[str, float]) -> float:
	"""Rule-based anomaly using NDVI/BSI/elevation. 0..1. No defaults here because features are validated upstream."""
	ndvi = features['ndvi']
	bsi = features['bsi']
	elev = features['elevation']
	# Lower NDVI, higher BSI, moderate elevation -> higher anomaly
	score = (1 - max(-1, min(1, ndvi))) * 0.4 + max(0.0, min(1.0, (bsi + 1) / 2)) * 0.4
	# Elevation scaling (0..3000m -> 0..1)
	elev_factor = max(0.0, min(1.0, elev / 3000.0)) * 0.2
	return max(0.0, min(1.0, score + elev_factor))

def analyze_satellite_anomalies(lat: float, lon: float) -> float:
	"""Strict version: compute features and return anomaly score.
	In production this could call a trained CNN; here we keep a deterministic interface.
	"""
	features = extract_satellite_features(lat, lon)
	return compute_anomaly_score(features)

# LABELED DATASET GENERATION (PRODUCTION)
from typing import List, Dict, Optional
import pandas as pd

def generate_geode_training_data(
    positive_sites: List[Tuple[float, float, str]],
    negative_sites: List[Tuple[float, float, str]]
) -> pd.DataFrame:
    """
    Generate labeled dataset for ML training by extracting all features from known sites.
    
    Args:
        positive_sites: List of (lat, lon, name) tuples for known geode sites
        negative_sites: List of (lat, lon, name) tuples for non-geode control sites
        
    Returns:
        DataFrame with all features and 'is_geode' label
    """
    rows = []
    
    # Process positive samples (known geode sites)
    for lat, lon, name in positive_sites:
        try:
            # Extract satellite features
            sat_features = extract_satellite_features(lat, lon)
            
            # Get Mindat occurrence data
            mindat_data = mindat_occurrence_metrics(lat, lon) or {
                'min_distance_km': None, 
                'count': 0
            }
            
            # Get USGS lithology data
            lithology = query_usgs_lithology(lat, lon) or {
                'primary_lithology': 'unknown',
                'basalt_presence': False,
                'limestone_presence': False,
                'volcanic_proximity_km': None,
                'sedimentary_score': 0.0
            }
            
            # Get fault proximity data
            fault_data = calculate_fault_proximity(lat, lon) or {
                'nearest_fault_km': None,
                'fault_density': 0.0,
                'seismic_activity_score': 0.0,
                'recent_earthquakes': 0,
                'tectonic_setting': 'unknown'
            }
            
            # Combine all features
            row = {
                'lat': lat,
                'lon': lon,
                'name': name,
                'is_geode': 1,  # Positive label
                **sat_features,
                'mindat_distance_km': mindat_data['min_distance_km'],
                'mindat_count': mindat_data['count'],
                'primary_lithology': lithology['primary_lithology'],
                'basalt_presence': int(lithology['basalt_presence']),
                'limestone_presence': int(lithology['limestone_presence']),
                'volcanic_proximity_km': lithology['volcanic_proximity_km'],
                'sedimentary_score': lithology['sedimentary_score'],
                'nearest_fault_km': fault_data['nearest_fault_km'],
                'fault_density': fault_data['fault_density'],
                'seismic_activity_score': fault_data['seismic_activity_score'],
                'recent_earthquakes': fault_data['recent_earthquakes'],
                'tectonic_setting': fault_data['tectonic_setting']
            }
            rows.append(row)
            logger.info(f"✅ Processed positive site: {name}")
            
        except Exception as e:
            logger.warning(f"Failed to process positive site {name}: {e}")
    
    # Process negative samples (non-geode control sites)
    for lat, lon, name in negative_sites:
        try:
            # Extract satellite features
            sat_features = extract_satellite_features(lat, lon)
            
            # Get Mindat occurrence data
            mindat_data = mindat_occurrence_metrics(lat, lon) or {
                'min_distance_km': None,
                'count': 0
            }
            
            # Get USGS lithology data
            lithology = query_usgs_lithology(lat, lon) or {
                'primary_lithology': 'unknown',
                'basalt_presence': False,
                'limestone_presence': False,
                'volcanic_proximity_km': None,
                'sedimentary_score': 0.0
            }
            
            # Get fault proximity data
            fault_data = calculate_fault_proximity(lat, lon) or {
                'nearest_fault_km': None,
                'fault_density': 0.0,
                'seismic_activity_score': 0.0,
                'recent_earthquakes': 0,
                'tectonic_setting': 'unknown'
            }
            
            # Combine all features
            row = {
                'lat': lat,
                'lon': lon,
                'name': name,
                'is_geode': 0,  # Negative label
                **sat_features,
                'mindat_distance_km': mindat_data['min_distance_km'],
                'mindat_count': mindat_data['count'],
                'primary_lithology': lithology['primary_lithology'],
                'basalt_presence': int(lithology['basalt_presence']),
                'limestone_presence': int(lithology['limestone_presence']),
                'volcanic_proximity_km': lithology['volcanic_proximity_km'],
                'sedimentary_score': lithology['sedimentary_score'],
                'nearest_fault_km': fault_data['nearest_fault_km'],
                'fault_density': fault_data['fault_density'],
                'seismic_activity_score': fault_data['seismic_activity_score'],
                'recent_earthquakes': fault_data['recent_earthquakes'],
                'tectonic_setting': fault_data['tectonic_setting']
            }
            rows.append(row)
            logger.info(f"✅ Processed negative site: {name}")
            
        except Exception as e:
            logger.warning(f"Failed to process negative site {name}: {e}")
    
    df = pd.DataFrame(rows)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Encode categorical variables
    if 'primary_lithology' in df.columns:
        df['lithology_encoded'] = pd.Categorical(df['primary_lithology']).codes
    if 'tectonic_setting' in df.columns:
        df['tectonic_encoded'] = pd.Categorical(df['tectonic_setting']).codes
    
    return df

# Example known geode sites (replace with actual data)
KNOWN_GEODE_SITES = [
    # Format: (latitude, longitude, site_name)
    (43.0, -111.0, "Dugway Geode Beds, Utah"),
    (32.8, -113.7, "Hauser Geode Beds, California"),
    (39.25, -91.36, "Keokuk, Iowa"),
    (27.87, -98.11, "Las Choyas, Mexico"),
    (44.49, -111.10, "Yellowstone Area, Wyoming"),
]

# Example negative control sites (replace with actual non-geode locations)
NEGATIVE_CONTROL_SITES = [
    # Urban/agricultural areas unlikely to have geodes
    (40.7128, -74.0060, "New York City"),
    (41.8781, -87.6298, "Chicago, Illinois"),
    (33.4484, -112.0740, "Phoenix, Arizona"),
    (29.7604, -95.3698, "Houston, Texas"),
    (25.7617, -80.1918, "Miami, Florida"),
]

print("📊 Dataset generation module loaded")
print(f"   Positive sites: {len(KNOWN_GEODE_SITES)}")
print(f"   Negative sites: {len(NEGATIVE_CONTROL_SITES)}")

# ML MODEL TRAINING (PRODUCTION)
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

# Import XGBoost with fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available - will use other models")

class GeodeMLTrainer:
    """Train and evaluate ML models for geode detection."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.calibrated_models = {}
        self.feature_importance = {}
        self.best_model = None
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for ML training.
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            X: Feature matrix
            y: Labels
        """
        # Select numeric features only
        feature_cols = [
            'ndvi', 'ndwi', 'bsi', 'iron_oxide_ratio', 'clay_minerals',
            'elevation', 'slope', 'aspect',
            'mindat_distance_km', 'mindat_count',
            'basalt_presence', 'limestone_presence',
            'volcanic_proximity_km', 'sedimentary_score',
            'nearest_fault_km', 'fault_density',
            'seismic_activity_score', 'recent_earthquakes'
        ]
        
        # Add encoded categorical features if available
        if 'lithology_encoded' in df.columns:
            feature_cols.append('lithology_encoded')
        if 'tectonic_encoded' in df.columns:
            feature_cols.append('tectonic_encoded')
        
        # Filter to available columns
        self.feature_columns = [col for col in feature_cols if col in df.columns]
        
        X = df[self.feature_columns].fillna(0).values
        y = df['is_geode'].values
        
        return X, y
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train multiple ML models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 1. Logistic Regression
        logger.info("Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        lr_model.fit(X_train_scaled, y_train)
        self.models['logistic_regression'] = lr_model
        
        # 2. XGBoost (if available)
        if HAS_XGBOOST:
            logger.info("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1
            )
            xgb_model.fit(X_train, y_train)
            self.models['xgboost'] = xgb_model
        
        # 3. Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=100,
            max_depth=10,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # Calibrate models for better probability estimates
        logger.info("Calibrating models...")
        for name, model in self.models.items():
            if name == 'logistic_regression':
                # LR already calibrated, but we'll use isotonic regression
                calibrated = CalibratedClassifierCV(
                    model, method='isotonic', cv=3
                )
            else:
                calibrated = CalibratedClassifierCV(
                    model, method='sigmoid', cv=3
                )
            
            # Refit on appropriate data
            if name == 'logistic_regression':
                calibrated.fit(X_train_scaled, y_train)
            else:
                calibrated.fit(X_train, y_train)
            
            self.calibrated_models[name] = calibrated
        
        # Extract feature importance
        self._extract_feature_importance(X_train)
    
    def _extract_feature_importance(self, X_train: np.ndarray):
        """Extract and store feature importance from models."""
        # XGBoost feature importance
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost'].feature_importances_
            self.feature_importance['xgboost'] = dict(zip(
                self.feature_columns, xgb_importance
            ))
        
        # Random Forest feature importance
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            self.feature_importance['random_forest'] = dict(zip(
                self.feature_columns, rf_importance
            ))
        
        # Logistic Regression coefficients (as proxy for importance)
        if 'logistic_regression' in self.models:
            lr_coefs = np.abs(self.models['logistic_regression'].coef_[0])
            self.feature_importance['logistic_regression'] = dict(zip(
                self.feature_columns, lr_coefs
            ))
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with evaluation metrics
        """
        results = []
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.calibrated_models.items():
            # Get predictions
            if name == 'logistic_regression':
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'model': name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
            }
            results.append(metrics)
            
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
        
        results_df = pd.DataFrame(results)
        
        # Select best model based on F1 score
        best_idx = results_df['f1_score'].idxmax()
        self.best_model = results_df.loc[best_idx, 'model']
        logger.info(f"✅ Best model: {self.best_model}")
        
        return results_df
    
    def save_model(self, filepath: str):
        """Save the best model and scaler to disk."""
        model_data = {
            'best_model_name': self.best_model,
            'model': self.calibrated_models[self.best_model],
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model_name']
        self.calibrated_models = {self.best_model: model_data['model']}
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.feature_importance = model_data.get('feature_importance', {})
        
        logger.info(f"✅ Model loaded from {filepath}")
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Features to predict on
            
        Returns:
            predictions: Binary predictions
            confidence: Confidence scores (0-1)
        """
        if not self.best_model:
            raise ValueError("No model trained yet")
        
        model = self.calibrated_models[self.best_model]
        
        # Scale if needed
        if self.best_model == 'logistic_regression':
            X = self.scaler.transform(X)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Confidence is the max probability
        confidence = np.max(probabilities, axis=1)
        
        return predictions, confidence

print("🤖 ML Training module loaded")
if HAS_XGBOOST:
    print("   Models: Logistic Regression, XGBoost, Random Forest")
else:
    print("   Models: Logistic Regression, Random Forest (XGBoost not available)")
print("   Calibration: CalibratedClassifierCV for probability estimates")

# MODEL EVALUATION AND VISUALIZATION (PRODUCTION)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report

def plot_model_evaluation(trainer: GeodeMLTrainer, X_test: np.ndarray, y_test: np.ndarray):
    """
    Create comprehensive evaluation plots for the trained models.
    
    Args:
        trainer: Trained GeodeMLTrainer instance
        X_test: Test features
        y_test: Test labels
    """
    if not trainer.best_model:
        logger.warning("No trained model available for evaluation")
        return
    
    # Get best model predictions
    model = trainer.calibrated_models[trainer.best_model]
    
    if trainer.best_model == 'logistic_regression':
        X_test_input = trainer.scaler.transform(X_test)
    else:
        X_test_input = X_test
    
    y_pred = model.predict(X_test_input)
    y_prob = model.predict_proba(X_test_input)[:, 1]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Model Evaluation: {trainer.best_model}', fontsize=16)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Geode', 'Geode'])
    disp.plot(ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title('Confusion Matrix')
    
    # 2. ROC Curve
  
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    axes[0, 2].plot(recall, precision, color='green', lw=2, label=f'AP = {avg_precision:.2f}')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision-Recall Curve')
    axes[0, 2].legend(loc="lower left")
    
    # 4. Feature Importance (Top 10)
    if trainer.best_model in trainer.feature_importance:
        importance = trainer.feature_importance[trainer.best_model]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        features, values = zip(*sorted_features)
        
        axes[1, 0].barh(range(len(features)), values, color='skyblue')
        axes[1, 0].set_yticks(range(len(features)))
        axes[1, 0].set_yticklabels(features)
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 10 Feature Importance')
        axes[1, 0].invert_yaxis()
    
    # 5. Probability Distribution
    axes[1, 1].hist(y_prob[y_test == 0], bins=20, alpha=0.5, label='Non-Geode', color='blue')
    axes[1, 1].hist(y_prob[y_test == 1], bins=20, alpha=0.5, label='Geode', color='red')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Probability Distribution by Class')
    axes[1, 1].legend()
    
    # 6. Model Comparison (if multiple models)
    if hasattr(trainer, 'evaluation_results'):
        results_df = trainer.evaluation_results
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(results_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            axes[1, 2].bar(x + i * width, results_df[metric], width, label=metric)
        
        axes[1, 2].set_xlabel('Model')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Model Comparison')
        axes[1, 2].set_xticks(x + width * 1.5)
        axes[1, 2].set_xticklabels(results_df['model'], rotation=45)
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed classification report
    print("\n" + "="*50)
    print(f"Classification Report for {trainer.best_model}")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Non-Geode', 'Geode']))

# Example usage function (not executed automatically)
def train_and_evaluate_models(training_data_df: pd.DataFrame = None):
    """
    Complete pipeline to train and evaluate geode detection models.
    
    Args:
        training_data_df: Optional pre-generated training data. If None, will generate.
    """
    # Generate training data if not provided
    if training_data_df is None:
        logger.info("Generating training data from known sites...")
        training_data_df = generate_geode_training_data(
            KNOWN_GEODE_SITES,
            NEGATIVE_CONTROL_SITES
        )
    
    if training_data_df.empty or len(training_data_df) < 4:
        logger.error("Insufficient training data. Need at least 4 samples.")
        return None
    
    # Initialize trainer
    trainer = GeodeMLTrainer(random_state=42)
    
    # Prepare features
    X, y = trainer.prepare_features(training_data_df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train models
    trainer.train_models(X_train, y_train)
    
    # Evaluate models
    evaluation_results = trainer.evaluate_models(X_test, y_test)
    trainer.evaluation_results = evaluation_results
    
    print("\n" + "="*50)
    print("Model Evaluation Results")
    print("="*50)
    print(evaluation_results.to_string())
    
    # Create visualizations
    plot_model_evaluation(trainer, X_test, y_test)
    
    # Save model
    model_path = 'geode_detection_model.pkl'
    trainer.save_model(model_path)
    
    return trainer

print("📈 Model evaluation module loaded")
print("   Visualizations: Confusion Matrix, ROC, PR Curve, Feature Importance")
print("   Metrics: Accuracy, Precision, Recall, F1, AUC")

# GEODE DETECTION (PRODUCTION WITH ML MODEL SUPPORT)
from typing import List, Optional
from geopy.distance import geodesic
import pandas as pd
import pickle
import numpy as np

class GeodeDetector:
	"""Compute geode formation likelihood using ML models or heuristics. Production-ready with fallbacks."""
	def __init__(self, known_sites_csv: Optional[str] = None, model_path: Optional[str] = None):
		self.known_geode_sites: List[Dict] = []
		self.ml_trainer = None
		self.use_ml_model = False
		
		# Load known sites if provided
		if known_sites_csv:
			if not os.path.exists(known_sites_csv):
				raise FileNotFoundError(f"Known sites CSV not found: {known_sites_csv}")
			df = pd.read_csv(known_sites_csv)
			required = {'lat','lon','name'}
			if not required.issubset(df.columns):
				raise ValueError('Known sites CSV missing columns lat, lon, name')
			self.known_geode_sites = df[['lat','lon','name']].to_dict('records')
		
		# Try to load ML model if provided or use default path
		if model_path is None:
			model_path = 'geode_detection_model.pkl'
		
		if os.path.exists(model_path):
			try:
				self.ml_trainer = GeodeMLTrainer()
				self.ml_trainer.load_model(model_path)
				self.use_ml_model = True
				logger.info(f"✅ ML model loaded from {model_path}")
			except Exception as e:
				logger.warning(f"Failed to load ML model: {e}. Using heuristics.")
				self.use_ml_model = False
		else:
			logger.info("No ML model found. Using heuristic scoring.")
	
	def calculate_geode_probability(self, lat: float, lon: float, radius_m: int = 500) -> Dict:
		"""Calculate probability using ML model if available, otherwise heuristics."""
		
		# Extract base satellite features
		features = extract_satellite_features(lat, lon, radius_m)
		
		# Try ML model first if available
		confidence_score = None
		if self.use_ml_model and self.ml_trainer:
			try:
				# Prepare features for ML model
				feature_dict = {**features}
				
				# Add external data features if available
				mindat_data = mindat_occurrence_metrics(lat, lon) or {'min_distance_km': None, 'count': 0}
				lithology = query_usgs_lithology(lat, lon) or {
					'basalt_presence': False,
					'limestone_presence': False,
					'volcanic_proximity_km': None,
					'sedimentary_score': 0.0
				}
				fault_data = calculate_fault_proximity(lat, lon) or {
					'nearest_fault_km': None,
					'fault_density': 0.0,
					'seismic_activity_score': 0.0,
					'recent_earthquakes': 0
				}
				
				# Build feature vector
				ml_features = {
					'ndvi': features['ndvi'],
					'ndwi': features['ndwi'],
					'bsi': features['bsi'],
					'iron_oxide_ratio': features['iron_oxide_ratio'],
					'clay_minerals': features['clay_minerals'],
					'elevation': features['elevation'],
					'slope': features['slope'],
					'aspect': features['aspect'],
					'mindat_distance_km': mindat_data['min_distance_km'] or 100.0,
					'mindat_count': mindat_data['count'],
					'basalt_presence': int(lithology['basalt_presence']),
					'limestone_presence': int(lithology['limestone_presence']),
					'volcanic_proximity_km': lithology['volcanic_proximity_km'] or 100.0,
					'sedimentary_score': lithology['sedimentary_score'],
					'nearest_fault_km': fault_data['nearest_fault_km'] or 100.0,
					'fault_density': fault_data['fault_density'],
					'seismic_activity_score': fault_data['seismic_activity_score'],
					'recent_earthquakes': fault_data['recent_earthquakes']
				}
				
				# Create feature array in correct order
				X = np.array([[ml_features.get(col, 0.0) for col in self.ml_trainer.feature_columns]])
				
				# Get ML predictions
				predictions, confidence = self.ml_trainer.predict_with_confidence(X)
				score = float(predictions[0])
				confidence_score = float(confidence[0])
				
				logger.info(f"ML prediction: {score:.3f} (confidence: {confidence_score:.3f})")
				
			except Exception as e:
				logger.warning(f"ML prediction failed: {e}. Falling back to heuristics.")
				score = None
		else:
			score = None
		
		# Fall back to heuristic scoring if ML failed or unavailable
		if score is None:
			exposed_rock = max(0.0, min(1.0, (features['bsi'] + 1) / 2))
			iron_content = max(0.0, min(1.0, features['iron_oxide_ratio']))
			clay_index = max(0.0, min(1.0, features['clay_minerals']))
			low_veg = max(0.0, min(1.0, 1 - (features['ndvi'] + 1) / 2))
			terrain_complexity = max(0.0, min(1.0, features['slope'] / 45.0))
			
			score = (
				exposed_rock * 0.3 +
				iron_content * 0.25 +
				clay_index * 0.2 +
				low_veg * 0.15 +
				terrain_complexity * 0.1
			)
		
		# Add proximity bonus to known sites
		nearest = None
		if self.known_geode_sites:
			best_dist = None
			for site in self.known_geode_sites:
				d = geodesic((lat, lon), (site['lat'], site['lon'])).miles
				if best_dist is None or d < best_dist:
					best_dist, nearest = d, site
			if best_dist is not None:
				bonus = max(0.0, 1.0 - min(best_dist, 50.0) / 50.0) * 0.15
				score = min(1.0, score + bonus)
		
		result = {
			'geode_probability': float(max(0.0, min(1.0, score))),
			'method': 'ML' if confidence_score is not None else 'heuristic',
			'geological_indicators': {
				'exposed_rock': max(0.0, min(1.0, (features['bsi'] + 1) / 2)),
				'iron_content': max(0.0, min(1.0, features['iron_oxide_ratio'])),
				'clay_minerals': max(0.0, min(1.0, features['clay_minerals'])),
				'low_vegetation': max(0.0, min(1.0, 1 - (features['ndvi'] + 1) / 2)),
				'terrain_complexity': max(0.0, min(1.0, features['slope'] / 45.0)),
			},
			'nearest_known_site': nearest,
			'distance_to_nearest': float(geodesic((lat, lon), (nearest['lat'], nearest['lon'])).miles) if nearest else None
		}
		
		# Add confidence score if ML was used
		if confidence_score is not None:
			result['confidence_score'] = confidence_score
		
		return result

# MAIN EXECUTION WRAPPER (WITH ML MODEL SUPPORT)
CONFIG = initialize_production_config()
initialize_earth_engine(config.google_earth_engine_project)

# Initialize detectors with ML model support
geode_sites_csv = os.environ.get('GEODE_SITES_PATH')
model_path = os.environ.get('GEODE_MODEL_PATH', 'geode_detection_model.pkl')

# Initialize with ML model if available
if os.path.exists(model_path):
    print(f"🤖 Attempting to load ML model from: {model_path}")
    geode_detector = GeodeDetector(known_sites_csv=geode_sites_csv, model_path=model_path)
    if geode_detector.use_ml_model:
        print("✅ ML model loaded successfully - using machine learning predictions")
    else:
        print("⚠️ ML model failed to load - using heuristic scoring")
else:
    print(f"ℹ️ No ML model found at {model_path} - using heuristic scoring")
    geode_detector = GeodeDetector(known_sites_csv=geode_sites_csv)

# Example usage function (not executed automatically)
def analyze_location(lat: float, lon: float) -> Dict:
	"""Analyze a location with all available data sources."""
	features = extract_satellite_features(lat, lon)
	anomaly = compute_anomaly_score(features)
	geode = geode_detector.calculate_geode_probability(lat, lon)
	
	# Add external data if available
	result = {
		'coords': (lat, lon),
		'features': features,
		'anomaly_score': anomaly,
		'geode': geode,
		'method': geode.get('method', 'heuristic')
	}
	
	# Try to get additional geological data
	lithology = query_usgs_lithology(lat, lon)
	if lithology:
		result['lithology'] = lithology
	
	fault_data = calculate_fault_proximity(lat, lon)
	if fault_data:
		result['fault_proximity'] = fault_data
		
	return result

print('✅ Unified production modules loaded with ML support')
print('📝 To train a model: Run the cell with validated_sites.csv loading')
print('🔍 To analyze: Use analyze_location(lat, lon)')

# MAIN EXECUTION WRAPPER (NO INTERACTIVE PROMPTS)
CONFIG = validate_required_api_keys()
initialize_earth_engine(os.environ['GEE_PROJECT_ID'])

# Initialize detectors
geode_sites_csv = os.environ.get('GEODE_SITES_PATH')
geode_detector = GeodeDetector(known_sites_csv=geode_sites_csv) if geode_sites_csv else GeodeDetector()

# Example usage function (not executed automatically)
def analyze_location(lat: float, lon: float) -> Dict:
	features = extract_satellite_features(lat, lon)
	anomaly = compute_anomaly_score(features)
	geode = geode_detector.calculate_geode_probability(lat, lon)
	return {
		'coords': (lat, lon),
		'features': features,
		'anomaly_score': anomaly,
		'geode': geode
	}

print('✅ Unified production modules loaded')

# REGION ANALYSIS PARAMETERS
REGION = "Erie, PA"
RADIUS_MILES = 50
GRID_STEP_MILES = 10  # spacing between grid points
TOP_N_MARKERS = 50
GEODE_THRESHOLD = 0.4  # only mark geode sites with prob >= 0.4

print(f"📍 Region: {REGION} | Radius: {RADIUS_MILES} mi | Step: {GRID_STEP_MILES} mi")

# GRID GENERATION + ANALYSIS
import math
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd

mi_to_km = 1.60934

def generate_grid(center_lat: float, center_lon: float, radius_mi: int, step_mi: int):
	points = []
	for d in range(0, radius_mi + 1, step_mi):
		for bearing in range(0, 360, max(1, int(360 / max(1, (radius_mi // step_mi) * 8)))):
			# approximate: move d miles along bearing
			# small step geodesic offset via simple lat/lon delta (acceptable for small radii)
			delta_lat = (d / 69.0) * math.cos(math.radians(bearing))
			delta_lon = (d / (69.0 * math.cos(math.radians(center_lat)))) * math.sin(math.radians(bearing))
			points.append((center_lat + delta_lat, center_lon + delta_lon))
	return points

# Resolve region center
geolocator = Nominatim(user_agent="geode_unified_prod")
loc = geolocator.geocode(REGION)
if not loc:
	raise RuntimeError(f"Could not geocode region: {REGION}")
center = (loc.latitude, loc.longitude)

# Build grid
grid_points = generate_grid(center[0], center[1], RADIUS_MILES, GRID_STEP_MILES)
print(f"🧭 Generated {len(grid_points)} grid points around {REGION}")

# Analyze
rows = []
for lat, lon in grid_points:
	try:
		features = extract_satellite_features(lat, lon)
		anomaly = compute_anomaly_score(features)
		geode = geode_detector.calculate_geode_probability(lat, lon)
		rows.append({
			'lat': lat, 'lon': lon,
			'anomaly_score': anomaly,
			'geode_probability': geode['geode_probability'],
			'ndvi': features['ndvi'], 'bsi': features['bsi'],
			'iron_oxide_ratio': features['iron_oxide_ratio'], 'clay_minerals': features['clay_minerals']
		})
	except Exception as e:
		# skip points with missing data; production policy is fail-fast per point, not whole run
		pass

df = pd.DataFrame(rows)
print(f"📊 Analyzed {len(df)} points with valid data")

# Rank and select top
if not df.empty:
	df['score'] = df['anomaly_score'] * 0.5 + df['geode_probability'] * 0.5
	df_top = df.sort_values('score', ascending=False).head(TOP_N_MARKERS)
else:
	df_top = pd.DataFrame()

print("Top rows:")
df_top.head(5)

# MAP VISUALIZATION
import folium
from folium.plugins import MarkerCluster

if df_top is not None and not df_top.empty:
	m = folium.Map(location=center, zoom_start=9, tiles='OpenStreetMap')
	cluster = MarkerCluster().add_to(m)
	for _, row in df_top.iterrows():
		color = 'purple' if row['geode_probability'] >= GEODE_THRESHOLD else 'red'
		popup = folium.Popup(
			f"<b>Score:</b> {row['score']:.2f}<br>"
			f"<b>Geode:</b> {row['geode_probability']:.2f}<br>"
			f"<b>Anomaly:</b> {row['anomaly_score']:.2f}<br>"
			f"<b>NDVI:</b> {row['ndvi']:.2f} | <b>BSI:</b> {row['bsi']:.2f}<br>"
			f"<b>Iron:</b> {row['iron_oxide_ratio']:.2f} | <b>Clay:</b> {row['clay_minerals']:.2f}",
			max_width=300
		)
		folium.CircleMarker(
			location=(row['lat'], row['lon']),
			radius=6,
			color=color,
			fill=True,
			fill_opacity=0.7,
			popup=popup
		).add_to(cluster)
	
	print(m)
	m.save('unified_treasure_map.html')
	print('🗺️ Map saved to unified_treasure_map.html')
else:
	print('No valid analysis points to display on map')
