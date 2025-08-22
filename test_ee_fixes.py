#!/usr/bin/env python3
"""
Test script to verify Earth Engine fixes for computePixels and sampling failures.
Tests the problematic coordinates that were previously failing.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Import the module to test
try:
    from treasure_hunter_module import fetch_satellite_image, initialize_earth_engine, setup_auth
except ImportError:
    print("Error: Could not import treasure_hunter_module")
    print("Make sure to run: python convert_notebook.py")
    sys.exit(1)

def test_coordinate(lat, lon, name):
    """Test a single coordinate for Earth Engine data acquisition."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Coordinates: ({lat:.4f}, {lon:.4f})")
    print(f"{'='*60}")
    
    try:
        # Initialize Earth Engine if needed
        try:
            import ee
            if not hasattr(ee, '_initialized') or not ee._initialized:
                setup_auth()
                initialize_earth_engine()
        except Exception as e:
            print(f"Warning: Could not initialize Earth Engine: {e}")
        
        # Test the fetch function
        start_time = datetime.now()
        result = fetch_satellite_image(lat, lon, size=256)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Verify result structure
        if isinstance(result, np.ndarray):
            print(f"✅ SUCCESS: Got image data")
            print(f"   Shape: {result.shape}")
            print(f"   Data type: {result.dtype}")
            print(f"   Value range: [{result.min():.2f}, {result.max():.2f}]")
            print(f"   Time taken: {elapsed:.2f} seconds")
            
            # Verify expected shape
            expected_shape = (6, 256, 256)
            if result.shape == expected_shape:
                print(f"   ✅ Shape matches expected {expected_shape}")
            else:
                print(f"   ⚠️ Shape mismatch: expected {expected_shape}, got {result.shape}")
            
            # Check for valid data (not all zeros or NaN)
            if np.all(result == 0):
                print(f"   ⚠️ Warning: All values are zero")
            elif np.any(np.isnan(result)):
                print(f"   ⚠️ Warning: Contains NaN values")
            else:
                print(f"   ✅ Data values look valid")
                
            return True
        else:
            print(f"❌ FAILED: Expected numpy array, got {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tests on all problematic coordinates."""
    print("Earth Engine Fix Verification Test")
    print("="*60)
    
    # Test coordinates that were previously failing
    test_cases = [
        (-13.1631, -72.5450, "Machu Picchu 1"),
        (-13.1586, -72.5434, "Machu Picchu 2"),
        (-13.1556, -72.5390, "Machu Picchu 3"),
        (-13.1553, -72.5328, "Machu Picchu 4"),
        (-13.1584, -72.5261, "Machu Picchu 5"),
        (37.7749, -122.4194, "San Francisco (Control)"),  # Control test with known good location
    ]
    
    results = []
    
    for lat, lon, name in test_cases:
        success = test_coordinate(lat, lon, name)
        results.append((name, success))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Earth Engine fixes are working correctly.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())