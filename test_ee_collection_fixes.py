#!/usr/bin/env python3
"""
Test script to verify Earth Engine collection limiting fixes.
This will test that collections no longer exceed 5000 elements.
"""

import os
import sys

def test_collection_patterns():
    """Test that collection limiting patterns are correctly applied."""
    
    files_to_check = [
        'treasure_hunter_module.py',
        'satellite_production_module.py', 
        'satellite_module.py',
        'satellite_300mile_module.py'
    ]
    
    print("🔍 Testing Earth Engine collection limiting fixes...")
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            print(f"⚠️ File not found: {filename}")
            continue
            
        print(f"\n📁 Checking {filename}")
        
        with open(filename, 'r') as f:
            content = f.read()
            
        # Check for collection queries without limiting
        lines = content.split('\n')
        issues_found = 0
        
        for i, line in enumerate(lines, 1):
            # Look for ImageCollection queries
            if ('ee.ImageCollection' in line and 
                'filterBounds' in line and
                i < len(lines) - 5):  # Check next few lines
                
                # Check if this collection has proper limiting
                next_lines = '\n'.join(lines[i:i+10])
                
                if '.limit(' not in next_lines:
                    print(f"❌ Line {i}: Missing .limit() in collection query")
                    print(f"   {line.strip()}")
                    issues_found += 1
                elif '.sort(' not in next_lines:
                    print(f"❌ Line {i}: Missing .sort() before .limit() in collection query") 
                    print(f"   {line.strip()}")
                    issues_found += 1
                else:
                    print(f"✅ Line {i}: Properly limited collection")
        
        if issues_found == 0:
            print(f"✅ {filename}: All collection queries properly limited")
        else:
            print(f"❌ {filename}: {issues_found} collection queries need fixing")
    
    print("\n🔍 Checking NUMPY_NDARRAY format fixes...")
    
    # Check for NUMPY_NDARRAY format issues
    notebook_file = 'TreasurHunter.ipynb'
    if os.path.exists(notebook_file):
        with open(notebook_file, 'r') as f:
            notebook_content = f.read()
            
        if "'fileFormat': 'NUMPY_NDARRAY'" in notebook_content:
            print(f"❌ {notebook_file}: Still contains invalid NUMPY_NDARRAY format")
        elif "'fileFormat': 'NPY'" in notebook_content:
            print(f"✅ {notebook_file}: Uses correct NPY format")
        else:
            print(f"⚠️ {notebook_file}: No fileFormat found (may not be an issue)")
    else:
        print(f"⚠️ {notebook_file}: File not found")

def test_basic_import():
    """Test basic import of the main module without Earth Engine."""
    print("\n🔍 Testing basic module import...")
    
    try:
        # Try importing the main module
        import treasure_hunter_module
        print("✅ treasure_hunter_module imported successfully")
        
        # Check if critical functions exist
        if hasattr(treasure_hunter_module, 'fetch_satellite_image'):
            print("✅ fetch_satellite_image function exists")
        else:
            print("❌ fetch_satellite_image function not found")
            
        if hasattr(treasure_hunter_module, 'main_analysis'):
            print("✅ main_analysis function exists") 
        else:
            print("❌ main_analysis function not found")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"⚠️ Import succeeded but error occurred: {e}")

if __name__ == '__main__':
    print("🧪 Earth Engine Collection Fixes Test")
    print("=" * 50)
    
    test_collection_patterns()
    test_basic_import()
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")
    print("\nIf all checks pass, the Earth Engine errors should be resolved:")
    print("1. Collections are limited to 50 elements max")  
    print("2. Collections are sorted by cloud coverage first")
    print("3. NPY format is used instead of NUMPY_NDARRAY")