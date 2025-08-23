#!/usr/bin/env python3
"""
Fixed TreasureHunter Module with lazy GEE initialization for Railway deployment.
This prevents blocking on startup and provides better error recovery.
"""

import os
import sys
import json
import base64
import warnings
import tempfile
import threading
from functools import wraps

# Global state for Earth Engine
_ee_initialized = False
_ee_lock = threading.Lock()
_ee_error = None
EE_AVAILABLE = False

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
                try:
                    # Decode base64
                    try:
                        # Try base64 decode first
                        decoded = base64.b64decode(b64_creds.strip()).decode('utf-8')
                        service_json = decoded
                    except:
                        # Maybe it's already JSON
                        json.loads(b64_creds)
                        service_json = b64_creds
                    
                    # Parse JSON to get service account email
                    info = json.loads(service_json)
                    sa_email = info.get('client_email')
                    
                    # Railway-friendly temp path
                    for path in ['/tmp/gee_sa.json', '/app/gee_sa.json', './gee_sa.json']:
                        try:
                            dir_path = os.path.dirname(path) or '.'
                            if os.path.exists(dir_path) and os.access(dir_path, os.W_OK):
                                temp_key_path = path
                                break
                        except:
                            continue
                    
                    if temp_key_path:
                        with open(temp_key_path, 'w') as f:
                            f.write(service_json)
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_key_path
                        credentials = ee.ServiceAccountCredentials(sa_email, temp_key_path)
                        print(f"✅ GEE: Using service account {sa_email}")
                    
                except Exception as e:
                    print(f"⚠️ GEE: Failed to process credentials: {e}")
            
            # Alternative: Check for file-based credentials
            elif os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                creds_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
                if os.path.exists(creds_path):
                    try:
                        with open(creds_path) as f:
                            info = json.loads(f.read())
                        sa_email = info.get('client_email')
                        credentials = ee.ServiceAccountCredentials(sa_email, creds_path)
                        print(f"✅ GEE: Using service account from file")
                    except Exception as e:
                        print(f"⚠️ GEE: Failed to load credentials file: {e}")
            
            # Initialize Earth Engine
            init_success = False
            
            # Try with credentials and project
            if credentials and project_id:
                try:
                    ee.Initialize(credentials=credentials, project=project_id)
                    init_success = True
                    print(f"✅ GEE initialized with project {project_id}")
                except Exception as e:
                    print(f"⚠️ GEE init with credentials failed: {e}")
            
            # Try with just project (ADC)
            if not init_success and project_id:
                try:
                    ee.Initialize(project=project_id)
                    init_success = True
                    print(f"✅ GEE initialized with ADC and project {project_id}")
                except Exception as e:
                    print(f"⚠️ GEE init with ADC failed: {e}")
            
            # Try default init
            if not init_success:
                try:
                    ee.Initialize()
                    init_success = True
                    print(f"✅ GEE initialized with defaults")
                except Exception as e:
                    print(f"⚠️ GEE default init failed: {e}")
            
            if init_success:
                # Verify it actually works
                try:
                    ee.Number(1).getInfo()
                    EE_AVAILABLE = True
                    print("✅ GEE API verified working")
                except Exception as e:
                    print(f"❌ GEE API test failed: {e}")
                    EE_AVAILABLE = False
                    _ee_error = str(e)
            else:
                EE_AVAILABLE = False
                _ee_error = "All initialization methods failed"
                
        except ImportError:
            print("⚠️ earthengine-api not installed")
            EE_AVAILABLE = False
            _ee_error = "earthengine-api not installed"
        except Exception as e:
            print(f"❌ GEE initialization error: {e}")
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

# Export the lazy initializer
__all__ = ['lazy_initialize_earth_engine', 'requires_earth_engine', 'EE_AVAILABLE']