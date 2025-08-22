#!/usr/bin/env python3
"""
Full pipeline test for Earth Engine fixes with main_analysis.
Tests the complete workflow including anomaly detection.
"""

import sys
import os
import time

# Import the fixed module
from treasure_hunter_module import main_analysis, EE_AVAILABLE

# Test locations
TEST_LOCATIONS = [
    ("Machu Picchu", (-13.1631, -72.5450), 5, 10),  # Known archaeological site
    ("Petra Jordan", (30.3285, 35.4444), 5, 10),    # Known archaeological site
    ("San Francisco", (37.7749, -122.4194), 5, 10), # Urban control
]

def test_full_pipeline():
    """Test the complete analysis pipeline with Earth Engine fixes."""
    
    if not EE_AVAILABLE:
        print("❌ Earth Engine is not available. Please configure credentials.")
        return False
    
    print("🚀 Testing Full Analysis Pipeline with Earth Engine Fixes")
    print("=" * 70)
    
    all_results = []
    failed_locations = []
    
    for location_name, (lat, lon), radius_km, num_points in TEST_LOCATIONS:
        print(f"\n📍 Testing: {location_name}")
        print(f"   Coordinates: ({lat:.4f}, {lon:.4f})")
        print(f"   Radius: {radius_km} km")
        print(f"   Points to analyze: {num_points}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Run main analysis
            results = main_analysis(
                region_name=location_name,
                coordinates=(lat, lon),
                radius_km=radius_km,
                num_points=num_points
            )
            
            elapsed_time = time.time() - start_time
            
            # Check results structure - should be a DataFrame
            if results is not None and hasattr(results, 'shape'):
                print(f"✅ Analysis completed in {elapsed_time:.2f} seconds")
                
                # Display key metrics
                print(f"   Points analyzed: {len(results)}")
                
                if len(results) > 0:
                    # Calculate statistics from DataFrame
                    if 'anomaly_score' in results.columns:
                        avg_score = results['anomaly_score'].mean()
                        max_score = results['anomaly_score'].max()
                        min_score = results['anomaly_score'].min()
                        print(f"   Anomaly scores:")
                        print(f"     Average: {avg_score:.3f}")
                        print(f"     Max: {max_score:.3f}")
                        print(f"     Min: {min_score:.3f}")
                    
                    # Check methods used
                    if 'method' in results.columns:
                        method_counts = results['method'].value_counts()
                        print(f"   Detection methods:")
                        for method, count in method_counts.items():
                            print(f"     {method}: {count}/{len(results)}")
                    
                    # Check confidence levels
                    if 'confidence' in results.columns:
                        avg_confidence = results['confidence'].mean()
                        print(f"   Average confidence: {avg_confidence:.3f}")
                
                all_results.append({
                    'location': location_name,
                    'success': True,
                    'time': elapsed_time,
                    'results': results
                })
                
            else:
                print(f"⚠️ Analysis returned unexpected format")
                failed_locations.append((location_name, "Invalid result format"))
                
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)[:200]}")
            failed_locations.append((location_name, str(e)[:100]))
            all_results.append({
                'location': location_name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 PIPELINE TEST SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in all_results if r.get('success', False))
    print(f"✅ Successful: {successful}/{len(TEST_LOCATIONS)}")
    print(f"❌ Failed: {len(failed_locations)}/{len(TEST_LOCATIONS)}")
    
    if failed_locations:
        print("\n🔍 Failed locations:")
        for name, error in failed_locations:
            print(f"   - {name}: {error}")
    
    # Performance summary
    successful_results = [r for r in all_results if r.get('success', False)]
    if successful_results:
        avg_time = sum(r['time'] for r in successful_results) / len(successful_results)
        print(f"\n⏱️ Average processing time: {avg_time:.2f} seconds")
    
    return len(failed_locations) == 0

def test_edge_cases():
    """Test edge cases and error handling."""
    
    print("\n🧪 Testing Edge Cases")
    print("=" * 70)
    
    # Test 1: Very small radius
    print("\n1. Testing with very small radius (0.5 km)...")
    try:
        results = main_analysis("Test Small", (37.7749, -122.4194), radius_km=0.5, num_points=3)
        print("   ✅ Handled small radius successfully")
    except Exception as e:
        print(f"   ❌ Failed with small radius: {str(e)[:100]}")
    
    # Test 2: Single point
    print("\n2. Testing with single point...")
    try:
        results = main_analysis("Test Single", (37.7749, -122.4194), radius_km=1, num_points=1)
        print("   ✅ Handled single point successfully")
    except Exception as e:
        print(f"   ❌ Failed with single point: {str(e)[:100]}")
    
    # Test 3: Remote location (middle of Pacific Ocean)
    print("\n3. Testing remote location (Pacific Ocean)...")
    try:
        results = main_analysis("Pacific Ocean", (0.0, -160.0), radius_km=10, num_points=5)
        if results and 'points' in results:
            print(f"   ✅ Handled remote location (analyzed {len(results['points'])} points)")
    except Exception as e:
        print(f"   ⚠️ Expected failure for ocean location: {str(e)[:100]}")

if __name__ == "__main__":
    # Run main pipeline test
    pipeline_success = test_full_pipeline()
    
    # Run edge case tests
    test_edge_cases()
    
    print("\n" + "=" * 70)
    if pipeline_success:
        print("✅ ALL PIPELINE TESTS PASSED")
    else:
        print("❌ SOME PIPELINE TESTS FAILED")
    print("=" * 70)
    
    sys.exit(0 if pipeline_success else 1)