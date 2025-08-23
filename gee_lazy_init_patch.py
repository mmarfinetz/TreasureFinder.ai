#!/usr/bin/env python3
"""
Patch to add lazy Earth Engine initialization to treasure_hunter_module.py
This prevents blocking on startup and provides better error recovery.
"""

import os
import sys
import json
import base64
import tempfile
import threading
from functools import wraps

# Global state for Earth Engine
_ee_initialized = False
_ee_lock = threading.Lock()
_ee_error = None

def lazy_initialize_earth_engine():
    """
    Lazy initialization of Google Earth Engine.
    Call this when actually needed, not at module import.
    Thread-safe and idempotent.
    """
    global _ee_initialized, _ee_error, EE_AVAILABLE
    
    if _ee_initialized:
        return EE_AVAILABLE
    
    with _ee_lock:
        # Double-check after acquiring lock
        if _ee_initialized:
            return EE_AVAILABLE
        
        try:
            import ee
            
            # Get project ID from multiple env vars
            project_id = (
                os.environ.get('GEE_PROJECT_ID') or
                os.environ.get('GOOGLE_EARTH_ENGINE_PROJECT') or
                os.environ.get('GOOGLE_CLOUD_PROJECT')
            )
            
            print(f"🔍 GEE: Project ID = {project_id}")
            
            # Handle Railway's base64-encoded credentials
            credentials = None
            temp_key_path = None
            
            # Check for base64-encoded credentials (Railway pattern)
            b64_creds = (
                os.environ.get('GOOGLE_CREDENTIALS_B64') or
                os.environ.get('GEE_SERVICE_ACCOUNT_JSON_B64') or
                os.environ.get('GEE_SERVICE_ACCOUNT_JSON')
            )
            
            if b64_creds:
                print(f"🔑 GEE: Found credentials in env (length: {len(b64_creds)})")
                try:
                    # Clean the input - remove whitespace that Railway might add
                    b64_clean = b64_creds.strip().replace('\n', '').replace(' ', '')
                    
                    # Try base64 decode first
                    try:
                        decoded = base64.b64decode(b64_clean, validate=True).decode('utf-8')
                        service_json = decoded
                        print("✅ GEE: Decoded base64 credentials")
                    except:
                        # Maybe it's already JSON
                        json.loads(b64_creds)
                        service_json = b64_creds
                        print("✅ GEE: Using raw JSON credentials")
                    
                    # Parse JSON to get service account email and project
                    info = json.loads(service_json)
                    sa_email = info.get('client_email')
                    
                    # Use project from credentials if not set
                    if not project_id and 'project_id' in info:
                        project_id = info['project_id']
                        print(f"📋 GEE: Using project from credentials: {project_id}")
                    
                    # Railway-friendly temp paths
                    temp_paths = ['/tmp/gee_sa.json', '/app/gee_sa.json', './gee_sa.json']
                    
                    # Special handling for Railway
                    if os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RAILWAY_PROJECT_ID'):
                        print("🚂 GEE: Detected Railway environment")
                        temp_paths = ['/app/gee_sa.json', './gee_sa.json', '/tmp/gee_sa.json']
                    
                    for path in temp_paths:
                        try:
                            dir_path = os.path.dirname(path) or '.'
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path, exist_ok=True)
                            if os.access(dir_path, os.W_OK):
                                temp_key_path = path
                                print(f"✅ GEE: Will write credentials to {temp_key_path}")
                                break
                        except Exception as e:
                            print(f"⚠️ GEE: Cannot use {path}: {e}")
                            continue
                    
                    if temp_key_path:
                        with open(temp_key_path, 'w') as f:
                            f.write(service_json)
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_key_path
                        credentials = ee.ServiceAccountCredentials(sa_email, temp_key_path)
                        print(f"✅ GEE: Created service account credentials for {sa_email}")
                    else:
                        print("❌ GEE: No writable path for credentials file")
                    
                except Exception as e:
                    print(f"❌ GEE: Failed to process credentials: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Alternative: Check for file-based credentials
            elif os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                creds_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
                print(f"🔑 GEE: Using credentials file: {creds_path}")
                if os.path.exists(creds_path):
                    try:
                        with open(creds_path) as f:
                            info = json.loads(f.read())
                        sa_email = info.get('client_email')
                        if not project_id and 'project_id' in info:
                            project_id = info['project_id']
                        credentials = ee.ServiceAccountCredentials(sa_email, creds_path)
                        print(f"✅ GEE: Loaded service account from file")
                    except Exception as e:
                        print(f"❌ GEE: Failed to load credentials file: {e}")
                else:
                    print(f"❌ GEE: Credentials file not found: {creds_path}")
            
            # Initialize Earth Engine with multiple fallback strategies
            init_success = False
            init_errors = []
            
            # Strategy 1: Credentials with explicit project
            if credentials and project_id:
                try:
                    print(f"🚀 GEE: Trying init with credentials and project {project_id}")
                    ee.Initialize(credentials=credentials, project=project_id)
                    init_success = True
                    print(f"✅ GEE initialized with credentials and project {project_id}")
                except Exception as e:
                    init_errors.append(f"Credentials+Project: {e}")
                    print(f"⚠️ GEE: Strategy 1 failed: {e}")
            
            # Strategy 2: Credentials without project
            if not init_success and credentials:
                try:
                    print(f"🚀 GEE: Trying init with just credentials")
                    ee.Initialize(credentials=credentials)
                    init_success = True
                    print(f"✅ GEE initialized with credentials (default project)")
                except Exception as e:
                    init_errors.append(f"Credentials only: {e}")
                    print(f"⚠️ GEE: Strategy 2 failed: {e}")
            
            # Strategy 3: ADC with project
            if not init_success and project_id:
                try:
                    print(f"🚀 GEE: Trying init with ADC and project {project_id}")
                    ee.Initialize(project=project_id)
                    init_success = True
                    print(f"✅ GEE initialized with ADC and project {project_id}")
                except Exception as e:
                    init_errors.append(f"ADC+Project: {e}")
                    print(f"⚠️ GEE: Strategy 3 failed: {e}")
            
            # Strategy 4: Default initialization
            if not init_success:
                try:
                    print(f"🚀 GEE: Trying default init")
                    ee.Initialize()
                    init_success = True
                    print(f"✅ GEE initialized with defaults")
                except Exception as e:
                    init_errors.append(f"Default: {e}")
                    print(f"⚠️ GEE: Strategy 4 failed: {e}")
            
            if init_success:
                # Verify it actually works with a simple API call
                try:
                    result = ee.Number(1).getInfo()
                    if result == 1:
                        EE_AVAILABLE = True
                        print("✅ GEE API verified working!")
                    else:
                        print(f"❌ GEE API returned unexpected value: {result}")
                        EE_AVAILABLE = False
                        _ee_error = f"API test returned {result} instead of 1"
                except Exception as e:
                    print(f"❌ GEE API test failed: {e}")
                    EE_AVAILABLE = False
                    _ee_error = str(e)
            else:
                EE_AVAILABLE = False
                _ee_error = f"All initialization strategies failed: {'; '.join(init_errors)}"
                print(f"❌ GEE: {_ee_error}")
                
        except ImportError:
            print("❌ earthengine-api not installed")
            EE_AVAILABLE = False
            _ee_error = "earthengine-api not installed"
        except Exception as e:
            print(f"❌ GEE initialization error: {e}")
            import traceback
            traceback.print_exc()
            EE_AVAILABLE = False
            _ee_error = str(e)
        
        _ee_initialized = True
        return EE_AVAILABLE

def requires_earth_engine(func):
    """Decorator to ensure Earth Engine is initialized before function call."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not lazy_initialize_earth_engine():
            raise RuntimeError(f"Earth Engine not available: {_ee_error}")
        return func(*args, **kwargs)
    return wrapper

# Make EE_AVAILABLE a dynamic property
class _EEAvailable:
    def __bool__(self):
        return lazy_initialize_earth_engine()
    
    def __repr__(self):
        return str(bool(self))

EE_AVAILABLE = _EEAvailable()

if __name__ == "__main__":
    print("Testing lazy Earth Engine initialization...")
    print(f"EE_AVAILABLE: {EE_AVAILABLE}")
    if EE_AVAILABLE:
        print("✅ Earth Engine is working!")
    else:
        print(f"❌ Earth Engine failed: {_ee_error}")