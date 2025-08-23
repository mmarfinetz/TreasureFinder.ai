#!/usr/bin/env python3
"""
Railway Google Earth Engine Configuration Helper

This script helps diagnose and fix GEE initialization issues on Railway deployment.
Run this to test your Railway environment variables and get proper base64 encoding.
"""

import os
import sys
import base64
import json
import ee

def test_base64_encoding(json_file_path=None):
    """Test base64 encoding/decoding of service account JSON"""
    
    if json_file_path and os.path.exists(json_file_path):
        print(f"📄 Reading service account key from: {json_file_path}")
        with open(json_file_path, 'r') as f:
            json_content = f.read()
    else:
        # Try to get from environment
        json_content = (
            os.environ.get('GEE_SERVICE_ACCOUNT_JSON') or
            os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        )
        if not json_content:
            print("❌ No service account JSON found in environment or file")
            return False
    
    # Validate JSON
    try:
        data = json.loads(json_content)
        client_email = data.get('client_email', 'unknown')
        print(f"✅ Valid JSON for service account: {client_email}")
    except Exception as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    
    # Test base64 encoding
    print("\n🔧 Testing base64 encoding methods:")
    
    # Method 1: Standard encoding (may have newlines)
    b64_standard = base64.b64encode(json_content.encode()).decode()
    print(f"1. Standard base64 length: {len(b64_standard)} chars")
    has_newlines = 'yes' if '\n' in b64_standard else 'no'
    print(f"   Has newlines: {has_newlines}")
    
    # Method 2: Railway-safe encoding (no newlines/spaces)
    b64_safe = base64.b64encode(json_content.encode()).decode().replace('\n', '').replace(' ', '')
    print(f"2. Railway-safe base64 length: {len(b64_safe)} chars")
    
    # Test decoding both ways
    for method, encoded in [("standard", b64_standard), ("railway-safe", b64_safe)]:
        try:
            # Test with stripping
            decoded = base64.b64decode(encoded.strip().replace('\n', '').replace(' ', '')).decode()
            json.loads(decoded)  # Validate it's still valid JSON
            print(f"   ✅ {method} decoding works")
        except Exception as e:
            print(f"   ❌ {method} decoding failed: {e}")
    
    print(f"\n📋 For Railway, set this environment variable:")
    print(f"GEE_SERVICE_ACCOUNT_JSON={b64_safe[:50]}...{b64_safe[-20:]}")
    print(f"\n(Full base64 string has {len(b64_safe)} characters)")
    
    # Save to file for easy copying
    with open('railway_gee_env.txt', 'w') as f:
        f.write(f"# Copy this to Railway environment variables:\n")
        f.write(f"GEE_SERVICE_ACCOUNT_JSON={b64_safe}\n")
        f.write(f"GEE_PROJECT_ID={data.get('project_id', 'YOUR_PROJECT_ID')}\n")
    print(f"💾 Full configuration saved to: railway_gee_env.txt")
    
    return True

def test_railway_environment():
    """Test current Railway environment configuration"""
    
    print("🚂 Railway Environment Check\n")
    
    # Check for Railway indicators
    is_railway = bool(os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RAILWAY_PROJECT_ID'))
    print(f"Running on Railway: {is_railway}")
    
    # Check key environment variables
    env_vars = {
        'GEE_PROJECT_ID': os.environ.get('GEE_PROJECT_ID'),
        'GOOGLE_EARTH_ENGINE_PROJECT': os.environ.get('GOOGLE_EARTH_ENGINE_PROJECT'),
        'GEE_SERVICE_ACCOUNT_JSON': bool(os.environ.get('GEE_SERVICE_ACCOUNT_JSON')),
        'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'),
        'MAPBOX_ACCESS_TOKEN': bool(os.environ.get('MAPBOX_ACCESS_TOKEN')),
    }
    
    print("\n📊 Environment Variables:")
    for key, value in env_vars.items():
        if isinstance(value, bool):
            status = "✅ Set" if value else "❌ Not set"
        else:
            status = f"✅ {value}" if value else "❌ Not set"
        print(f"  {key}: {status}")
    
    # Test file write permissions
    print("\n📁 File System Permissions:")
    test_paths = ['/tmp/', './', '/app/']
    for path in test_paths:
        try:
            test_file = os.path.join(path, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"  ✅ {path} - writable")
        except:
            print(f"  ❌ {path} - not writable")
    
    # Try to initialize Earth Engine
    print("\n🌍 Earth Engine Initialization:")
    
    # Get service account JSON
    service_json = os.environ.get('GEE_SERVICE_ACCOUNT_JSON')
    if service_json:
        try:
            # Try to decode if base64
            service_json_clean = service_json.strip().replace('\n', '').replace(' ', '')
            decoded = base64.b64decode(service_json_clean).decode()
            service_json = decoded
            print("  ✅ Decoded base64 service account JSON")
        except:
            print("  ℹ️ Using raw service account JSON (not base64)")
        
        try:
            # Parse JSON and get credentials
            data = json.loads(service_json)
            client_email = data.get('client_email')
            
            # Try to write to temp file
            for temp_path in ['/tmp/gee_sa.json', './gee_sa.json', '/app/gee_sa.json']:
                try:
                    with open(temp_path, 'w') as f:
                        f.write(service_json)
                    
                    # Initialize with credentials
                    project_id = os.environ.get('GEE_PROJECT_ID') or os.environ.get('GOOGLE_EARTH_ENGINE_PROJECT')
                    credentials = ee.ServiceAccountCredentials(client_email, temp_path)
                    ee.Initialize(credentials=credentials, project=project_id)
                    print(f"  ✅ Earth Engine initialized successfully!")
                    print(f"     Service account: {client_email}")
                    print(f"     Project: {project_id}")
                    print(f"     Temp file: {temp_path}")
                    
                    # Test a simple EE operation
                    try:
                        point = ee.Geometry.Point(-122.4194, 37.7749)
                        collection = ee.ImageCollection('COPERNICUS/S2').filterBounds(point).limit(1)
                        size = collection.size().getInfo()
                        print(f"  ✅ Earth Engine API test successful (found {size} images)")
                    except Exception as e:
                        print(f"  ⚠️ Earth Engine API test failed: {e}")
                    
                    return True
                    
                except Exception as e:
                    continue
            
            print(f"  ❌ Could not initialize Earth Engine: {e}")
        except Exception as e:
            print(f"  ❌ Failed to parse service account JSON: {e}")
    else:
        print("  ❌ No service account JSON found in environment")
    
    return False

def main():
    """Main entry point"""
    
    print("🔧 Railway Google Earth Engine Configuration Helper\n")
    
    if len(sys.argv) > 1:
        # Test encoding a local file
        json_file = sys.argv[1]
        print(f"Testing with file: {json_file}\n")
        test_base64_encoding(json_file)
    else:
        # Test current environment
        test_railway_environment()
        
        print("\n💡 Tips:")
        print("1. To encode your service account key:")
        print("   python railway_gee_fix.py /path/to/service-account-key.json")
        print("\n2. Make sure to set in Railway:")
        print("   - GEE_SERVICE_ACCOUNT_JSON (base64 encoded)")
        print("   - GEE_PROJECT_ID (your GCP project)")
        print("   - MAPBOX_ACCESS_TOKEN (as fallback)")
        print("\n3. If EE fails, the app will fall back to Mapbox for basic imagery")

if __name__ == "__main__":
    main()