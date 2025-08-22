#!/usr/bin/env python3
"""
Quick test to verify Earth Engine fixes work with the problematic coordinates.
"""

import sys
from treasure_hunter_module import analyze_satellite_anomalies, EE_AVAILABLE

# Original problematic coordinates
TEST_COORDS = [
    (-13.1631, -72.5450, "Machu Picchu 1"),
    (-13.1586, -72.5434, "Machu Picchu 2"),
    (-13.1556, -72.5390, "Machu Picchu 3"),
]

def quick_test():
    """Quick test of Earth Engine fixes."""
    
    if not EE_AVAILABLE:
        print("❌ Earth Engine not available")
        return False
    
    print("🚀 Quick Earth Engine Fix Verification")
    print("=" * 50)
    
    success_count = 0
    
    for lat, lon, name in TEST_COORDS:
        print(f"\n📍 Testing: {name} ({lat:.4f}, {lon:.4f})")
        
        try:
            # Test single location analysis
            result = analyze_satellite_anomalies(lat, lon)
            
            if result and 'anomaly_score' in result:
                score = result['anomaly_score']
                method = result.get('method', 'unknown')
                confidence = result.get('confidence', 0)
                
                print(f"   ✅ Success!")
                print(f"      Score: {score:.3f}")
                print(f"      Method: {method}")
                print(f"      Confidence: {confidence:.3f}")
                success_count += 1
            else:
                print(f"   ❌ Failed: Invalid result")
                
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {success_count}/{len(TEST_COORDS)} successful")
    
    return success_count == len(TEST_COORDS)

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)