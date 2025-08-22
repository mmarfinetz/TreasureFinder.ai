#!/usr/bin/env python3
"""
Test script to verify TreasureHunter improvements.
Tests the updated module with specified coordinates.
"""

import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")

# Import the updated module
from treasure_hunter_module import (
    analyze_satellite_anomalies,
    extract_comprehensive_features,
    calculate_confidence,
    cluster_detections,
    validate_data_quality,
    train_scoring_model
)

import pandas as pd
import numpy as np

def test_location(name, lat, lon):
    """Test analysis at a specific location."""
    print(f"\n{'='*60}")
    print(f"📍 Testing: {name}")
    print(f"   Coordinates: ({lat:.4f}, {lon:.4f})")
    print('='*60)
    
    try:
        # Run analysis
        result = analyze_satellite_anomalies(lat, lon)
        
        # Display results
        print(f"\n📊 Analysis Results:")
        print(f"   Anomaly Score: {result['anomaly_score']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Method: {result['method']}")
        print(f"   Description: {result['description']}")
        
        # Display features if available
        if 'features' in result:
            print(f"\n🔬 Extracted Features:")
            features = result['features']
            key_features = ['edge_density', 'ndvi', 'thermal_anomaly', 'spatial_correlation']
            for feat in key_features:
                if feat in features:
                    value = features[feat]
                    if value != -1:  # -1 indicates missing data
                        print(f"   {feat}: {value:.3f}")
                    else:
                        print(f"   {feat}: N/A (missing data)")
        
        # Check for errors
        if 'error' in result:
            print(f"\n⚠️ Error encountered: {result['error']}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_clustering():
    """Test the clustering functionality."""
    print(f"\n{'='*60}")
    print("🔮 Testing Clustering Functionality")
    print('='*60)
    
    # Create sample detection data
    np.random.seed(42)
    n_points = 50
    
    # Create 3 clusters of detections
    cluster1_lat = np.random.normal(29.98, 0.01, 20)
    cluster1_lon = np.random.normal(31.13, 0.01, 20)
    
    cluster2_lat = np.random.normal(44.51, 0.005, 15)
    cluster2_lon = np.random.normal(-64.29, 0.005, 15)
    
    # Add some noise points
    noise_lat = np.random.uniform(20, 50, 15)
    noise_lon = np.random.uniform(-80, 40, 15)
    
    # Combine all points
    all_lats = np.concatenate([cluster1_lat, cluster2_lat, noise_lat])
    all_lons = np.concatenate([cluster1_lon, cluster2_lon, noise_lon])
    all_scores = np.random.uniform(0.3, 0.9, len(all_lats))
    
    # Create dataframe
    detections_df = pd.DataFrame({
        'lat': all_lats,
        'lon': all_lons,
        'score': all_scores
    })
    
    print(f"Created {len(detections_df)} detection points")
    
    # Apply clustering
    clustered_df = cluster_detections(detections_df, eps=0.02, min_samples=5)
    
    # Display results
    n_clusters = len(clustered_df[clustered_df['cluster_id'] != -1]['cluster_id'].unique())
    n_noise = len(clustered_df[clustered_df['cluster_id'] == -1])
    
    print(f"\n📊 Clustering Results:")
    print(f"   Number of clusters found: {n_clusters}")
    print(f"   Noise points: {n_noise}")
    
    if 'cluster_priority' in clustered_df.columns:
        top_clusters = clustered_df[clustered_df['cluster_id'] != -1].drop_duplicates('cluster_id').nlargest(3, 'cluster_priority')
        print(f"\n🎯 Top Priority Clusters:")
        for idx, row in top_clusters.iterrows():
            print(f"   Cluster {row['cluster_id']}: Priority={row['cluster_priority']:.3f}, Size={row['cluster_size']}, Area={row['cluster_area']:.1f} km²")

def test_data_quality():
    """Test data quality validation."""
    print(f"\n{'='*60}")
    print("🔍 Testing Data Quality Validation")
    print('='*60)
    
    # Test with good quality data
    good_data = np.random.rand(6, 256, 256).astype(np.float32)
    print("\n✅ Testing with good quality data (6 bands, no NaN):")
    try:
        result = validate_data_quality(good_data)
        print(f"   Quality Score: {result['quality_score']:.3f}")
        print(f"   Passed: {result['passed']}")
        if result['issues']:
            print(f"   Issues: {', '.join(result['issues'])}")
    except Exception as e:
        print(f"   Validation raised: {e}")
    
    # Test with poor quality data (missing bands)
    poor_data = np.random.rand(2, 256, 256).astype(np.float32)
    print("\n❌ Testing with poor quality data (only 2 bands):")
    try:
        result = validate_data_quality(poor_data)
        print(f"   Quality Score: {result['quality_score']:.3f}")
        print(f"   Passed: {result['passed']}")
    except Exception as e:
        print(f"   Validation raised: {e}")
    
    # Test with NaN values
    nan_data = np.random.rand(6, 256, 256).astype(np.float32)
    nan_data[0, :50, :50] = np.nan
    print("\n⚠️ Testing with data containing NaN values:")
    try:
        result = validate_data_quality(nan_data)
        print(f"   Quality Score: {result['quality_score']:.3f}")
        print(f"   Passed: {result['passed']}")
        if result['issues']:
            print(f"   Issues: {', '.join(result['issues'])}")
    except Exception as e:
        print(f"   Validation raised: {e}")

def main():
    """Main test function."""
    print("🚀 TreasureHunter Improvement Test Suite")
    print("=" * 60)
    
    # Test locations as specified in the prompt
    test_locations = [
        ("Giza Pyramids", 29.9792, 31.1342),  # Should score high
        ("Pacific Ocean", 0, -160),           # Should score low or fail gracefully
        ("Oak Island", 44.5133, -64.2947)     # Should score moderate-high
    ]
    
    # Test each location
    results = []
    for name, lat, lon in test_locations:
        result = test_location(name, lat, lon)
        if result:
            results.append(result)
    
    # Test clustering functionality
    test_clustering()
    
    # Test data quality validation
    test_data_quality()
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 Test Summary")
    print('='*60)
    
    if results:
        scores = [r['anomaly_score'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        print(f"\nAnomaly Scores:")
        for loc, score in zip(test_locations, scores):
            print(f"   {loc[0]:20s}: {score:.3f}")
        
        print(f"\nConfidence Levels:")
        for loc, conf in zip(test_locations, confidences):
            print(f"   {loc[0]:20s}: {conf:.3f}")
        
        # Verify expected behavior
        print(f"\n✅ Verification:")
        giza_score = scores[0] if len(scores) > 0 else 0
        pacific_score = scores[1] if len(scores) > 1 else 0
        oak_score = scores[2] if len(scores) > 2 else 0
        
        if giza_score > 0.6:
            print("   ✅ Giza Pyramids scored high as expected")
        else:
            print("   ⚠️ Giza Pyramids did not score as high as expected")
        
        if pacific_score < 0.3 or results[1].get('method') == 'failed':
            print("   ✅ Pacific Ocean scored low/failed as expected")
        else:
            print("   ⚠️ Pacific Ocean did not behave as expected")
        
        if 0.4 <= oak_score <= 0.8:
            print("   ✅ Oak Island scored moderate-high as expected")
        else:
            print("   ⚠️ Oak Island score outside expected range")
    
    print("\n✅ Test suite completed!")

if __name__ == "__main__":
    main()