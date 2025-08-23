#!/usr/bin/env python3
"""
Quick test script for TreasureFinder.ai
Run this to verify your installation is working.
"""

import os
import sys
import json
from pathlib import Path

# Load environment variables from .env if present (does not override existing env)
try:
    from dotenv import load_dotenv, find_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        # Fallback: search upwards for a .env
        found = find_dotenv()
        if found:
            load_dotenv(found, override=False)
except Exception:
    # If python-dotenv isn't available or any error occurs, proceed without it
    pass

def check_dependencies():
    """Check which dependencies and APIs are available."""
    print("🔍 Checking dependencies...")
    
    deps = {
        'numpy': False,
        'pandas': False,
        'folium': False,
        'matplotlib': False,
        'requests': False,
        'earthengine-api': False,
        'torch': False,
        'xgboost': False,
    }
    
    for dep in deps:
        try:
            if dep == 'earthengine-api':
                __import__('ee')
            else:
                __import__(dep.replace('-api', ''))
            deps[dep] = True
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} (optional)")
    
    return deps

def check_api_keys():
    """Check which API keys are configured."""
    print("\n🔑 Checking API keys...")
    
    apis = {
        'MAPBOX_ACCESS_TOKEN': 'Mapbox (satellite imagery)',
        'GEE_PROJECT_ID': 'Google Earth Engine',
        'MINDAT_API_KEY': 'Mindat (mineral data)',
        'SENTINEL_HUB_CLIENT_ID': 'Sentinel Hub',
        'PLANET_API_KEY': 'Planet Labs',
    }
    
    configured = {}
    for key, desc in apis.items():
        if os.environ.get(key):
            print(f"  ✅ {desc}")
            configured[key] = True
        else:
            print(f"  ⚠️  {desc} not configured")
            configured[key] = False
    
    return configured

def test_earth_engine_auth():
    """Actually test Earth Engine authentication and API access."""
    print("\n🌍 Testing Earth Engine Authentication...")
    
    try:
        import ee
        
        # Check for credentials
        project_id = os.environ.get('GEE_PROJECT_ID') or os.environ.get('GOOGLE_EARTH_ENGINE_PROJECT')
        if not project_id:
            print("  ❌ No GEE_PROJECT_ID found in environment")
            return False
        
        print(f"  📋 Project ID: {project_id}")
        
        # Try to initialize
        try:
            # First check if already initialized
            test_point = ee.Geometry.Point(-122.4194, 37.7749)
            test_collection = ee.ImageCollection('COPERNICUS/S2').filterBounds(test_point).limit(1)
            size = test_collection.size().getInfo()
            print(f"  ✅ Earth Engine already initialized, found {size} test image(s)")
            return True
        except:
            pass
        
        # Try different authentication methods
        auth_methods = []
        
        # Method 1: Service account via env
        if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            auth_methods.append('service_account_file')
            print(f"  🔐 Using service account: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        elif os.environ.get('GEE_SERVICE_ACCOUNT_JSON') or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON'):
            auth_methods.append('service_account_json')
            print("  🔐 Using service account JSON from environment")
        
        # Try to authenticate
        if not auth_methods:
            print("  ⚠️  No authentication method configured")
            print("     Set GOOGLE_APPLICATION_CREDENTIALS or GEE_SERVICE_ACCOUNT_JSON")
            
            # Try default credentials as last resort
            try:
                ee.Initialize(project=project_id)
                auth_methods.append('default')
            except:
                print("  ❌ Failed to authenticate with default credentials")
                return False
        else:
            try:
                ee.Initialize(project=project_id)
            except Exception as e:
                print(f"  ❌ Failed to initialize: {str(e)}")
                return False
        
        # Test actual API access
        print("\n  🧪 Testing Earth Engine API access...")
        
        # Test 1: Simple geometry operation
        try:
            point = ee.Geometry.Point(-122.4194, 37.7749)
            buffer = point.buffer(1000)
            area = buffer.area().getInfo()
            print(f"  ✅ Geometry operations work (buffer area: {area:.0f} m²)")
        except Exception as e:
            print(f"  ❌ Geometry test failed: {e}")
            return False
        
        # Test 2: Image collection query
        try:
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(point) \
                .filterDate('2024-01-01', '2024-12-31') \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                .limit(10)
            
            count = collection.size().getInfo()
            print(f"  ✅ Image collection query works ({count} images found)")
            
            if count > 0:
                # Get first image metadata
                first = ee.Image(collection.first())
                bands = first.bandNames().getInfo()
                print(f"  ✅ Can access image metadata ({len(bands)} bands)")
        except Exception as e:
            print(f"  ❌ Collection query failed: {e}")
            return False
        
        # Test 3: Actual pixel data retrieval
        try:
            if count > 0:
                # Try to get actual pixel values
                sample = first.select(['B4', 'B3', 'B2']).sample(
                    region=point.buffer(100),
                    scale=10,
                    numPixels=10
                )
                sample_data = sample.first().getInfo()
                if sample_data and 'properties' in sample_data:
                    print(f"  ✅ Can retrieve pixel data (sampled {len(sample_data['properties'])} bands)")
                else:
                    print("  ⚠️  No pixel data retrieved (might be cloudy)")
        except Exception as e:
            print(f"  ⚠️  Pixel retrieval test skipped: {e}")
        
        print("\n  🎉 Earth Engine fully authenticated and working!")
        return True
        
    except ImportError:
        print("  ❌ earthengine-api not installed")
        print("     Run: pip install earthengine-api")
        return False
    except Exception as e:
        print(f"  ❌ Earth Engine test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functions without external dependencies."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from treasure_hunter_module import calculate_geode_probability
        
        # Test known geode location
        result = calculate_geode_probability(43.0, -111.0)  # Dugway, Utah
        score = result['geode_probability'] if isinstance(result, dict) else float(result)
        print(f"  ✅ Geode probability calculation: {score:.3f}")
        
        if score > 0.5:
            print(f"     (Correctly identified known geode site!)")
        
        return True
    except Exception as e:
        print(f"  ❌ Basic test failed: {e}")
        return False

def test_satellite_fetch():
    """Test satellite data fetching if available."""
    print("\n🛰️  Testing satellite data fetch...")
    
    try:
        from treasure_hunter_module import fetch_satellite_image
        
        # Test coordinates (San Francisco)
        data = fetch_satellite_image(37.7749, -122.4194, size=256)
        print(f"  ✅ Satellite fetch successful! Shape: {data.shape}")
        return True
    except Exception as e:
        print(f"  ⚠️  Satellite fetch not available: {str(e)[:50]}...")
        print(f"     (System will use statistical fallback)")
        return False

def test_analysis_pipeline():
    """Test the full analysis pipeline."""
    print("\n🔬 Testing analysis pipeline...")
    
    try:
        from treasure_hunter_module import analyze_satellite_anomalies
        
        result = analyze_satellite_anomalies(37.7749, -122.4194)
        print(f"  ✅ Analysis successful!")
        print(f"     Anomaly score: {result['anomaly_score']:.3f}")
        print(f"     Method: {result['method']}")
        print(f"     Confidence: {result['confidence']:.3f}")
        return True
    except Exception as e:
        print(f"  ❌ Analysis failed: {e}")
        return False

def test_mineral_segmentation():
    """Test mineral segmentation if model is available."""
    print("\n⛏️  Testing mineral segmentation...")
    
    try:
        from treasure_hunter_module import load_mineral_segmenter, combined_analysis
        
        # Try to load the segmenter
        segmenter = load_mineral_segmenter()
        print(f"  ✅ Mineral segmenter loaded")
        
        # Test combined analysis
        result = combined_analysis(43.0, -111.0, analysis_type="geological")
        if 'geology' in result:
            print(f"     Geological score: {result['geology']['score']:.3f}")
        
        return True
    except Exception as e:
        print(f"  ⚠️  Mineral segmentation not available: {str(e)[:50]}...")
        return False

def test_cnn_model():
    """Test CNN model if available."""
    print("\n🤖 Testing CNN model...")
    
    try:
        import torch
        from treasure_hunter_module import SatelliteAnomalyCNN
        
        if Path('satellite_cnn.pth').exists():
            model = SatelliteAnomalyCNN()
            model.load_state_dict(torch.load('satellite_cnn.pth', map_location='cpu'))
            print(f"  ✅ CNN model loaded successfully")
            
            # Test inference
            test_input = torch.randn(1, 6, 256, 256)
            with torch.no_grad():
                output = model(test_input)
                score = torch.sigmoid(output).item()
            print(f"     Test inference score: {score:.3f}")
            return True
        else:
            print(f"  ⚠️  CNN model file not found (satellite_cnn.pth)")
            return False
    except ImportError:
        print(f"  ⚠️  PyTorch not installed (CNN requires torch)")
        return False
    except Exception as e:
        print(f"  ❌ CNN test failed: {e}")
        return False

def run_mini_analysis():
    """Run a mini analysis to generate output files."""
    print("\n🗺️  Running mini analysis...")
    
    try:
        from treasure_hunter_module import main_analysis
        
        results = main_analysis(
            'Test Location',
            (37.7749, -122.4194),  # San Francisco
            radius_km=2,
            num_points=5
        )
        
        print(f"  ✅ Analysis complete! Found {len(results)} anomalies")
        
        if Path('treasure_map.html').exists():
            print(f"     📍 Map saved to: treasure_map.html")
        if Path('simple_treasure_map.html').exists():
            print(f"     📊 Table saved to: simple_treasure_map.html")
        
        return True
    except Exception as e:
        print(f"  ❌ Mini analysis failed: {e}")
        return False

def main():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("TreasureFinder.ai Quick Test")
    print("=" * 60)
    
    # Check environment
    deps = check_dependencies()
    apis = check_api_keys()
    
    # Test Earth Engine authentication FIRST
    ee_auth = test_earth_engine_auth()
    
    # Run tests
    results = {
        'earth_engine': ee_auth,
        'basic': test_basic_functionality(),
        'satellite': test_satellite_fetch(),
        'analysis': test_analysis_pipeline(),
        'mineral': test_mineral_segmentation(),
        'cnn': test_cnn_model(),
    }
    
    # Try mini analysis if basic tests pass
    if results['basic'] and results['analysis']:
        results['mini_analysis'] = run_mini_analysis()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    working = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n✅ {working}/{total} components working")
    
    # Specific Earth Engine status
    if not results.get('earth_engine', False):
        print("\n⚠️  Earth Engine NOT authenticated!")
        print("   This is CRITICAL for satellite analysis.")
        print("   To fix:")
        print("   1. Set GEE_PROJECT_ID environment variable")
        print("   2. Set one of:")
        print("      - GOOGLE_APPLICATION_CREDENTIALS (path to JSON key)")
        print("      - GEE_SERVICE_ACCOUNT_JSON (JSON content)")
        print("   3. Or run: earthengine authenticate (for interactive auth)")
    
    if not any(apis.values()):
        print("\n⚠️  No API keys configured!")
        print("   Add at least MAPBOX_ACCESS_TOKEN for satellite imagery:")
        print("   export MAPBOX_ACCESS_TOKEN='your_token_here'")
        print("   Get a free token at: https://account.mapbox.com/")
    
    if working == total:
        print("\n🎉 All systems operational! You're ready to hunt for treasure!")
    elif results['basic']:
        print("\n✨ Basic functionality working! Add API keys for full features.")
    else:
        print("\n❌ Some issues detected. Check the errors above.")
        print("   You may need to install dependencies:")
        print("   pip install -r requirements.txt")
    
    print("\n📚 For detailed testing instructions, see TESTING_GUIDE.md")
    
    return working == total

if __name__ == "__main__":
    sys.exit(0 if main() else 1)