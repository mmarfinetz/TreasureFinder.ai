# Geode Detection Integration for satellite_production_production_strict_final.ipynb

Add these cells after Cell 12 (Production Test) in the notebook:

## Cell 13: Geode Detection Integration

```python
# Geode Detection Integration
"""
Add geode detection capabilities alongside archaeological site detection.
This integrates features from satellite_production_modular_unified.ipynb
"""

# Additional imports for geode detection
from typing import Dict, Optional, List
import math
from geopy.distance import geodesic

# Session for API calls
SESSION = requests.Session() if 'SESSION' not in globals() else SESSION

def extract_geode_features(lat: float, lon: float, radius_m: int = 500) -> Dict[str, float]:
    """
    Extract geological features specific to geode detection.
    Uses Earth Engine to compute spectral indices.
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError('Invalid coordinates')
    
    if not EE_AVAILABLE:
        raise RuntimeError("Earth Engine required for geode feature extraction")
    
    pt = ee.Geometry.Point([lon, lat])
    buffer = pt.buffer(radius_m)
    
    # Landsat 8 SR collection - targeting geological features
    landsat = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        .filterBounds(buffer)
        .filterDate('2021-01-01', '2024-12-31')
        .filter(ee.Filter.lt('CLOUD_COVER', 20))
        .median())
    
    if landsat.bandNames().size().getInfo() == 0:
        raise RuntimeError('No satellite imagery available for location')
    
    # Calculate spectral indices for geological analysis
    ndvi = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    ndwi = landsat.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
    red = landsat.select('SR_B4').rename('RED')
    nir = landsat.select('SR_B5').rename('NIR')
    green = landsat.select('SR_B3').rename('GREEN')
    blue = landsat.select('SR_B2').rename('BLUE')
    swir1 = landsat.select('SR_B6').rename('SWIR1')
    swir2 = landsat.select('SR_B7').rename('SWIR2')
    
    # Bare Soil Index - important for exposed rock detection
    bsi = red.add(swir1).subtract(nir.add(green)).divide(
        red.add(swir1).add(nir).add(green)
    ).rename('BSI')
    
    # Iron oxide ratio - proxy for iron-rich minerals
    iron = red.divide(nir).rename('IRON')
    
    # Clay minerals index - SWIR1/SWIR2 ratio
    clay = swir1.divide(swir2).rename('CLAY')
    
    # Terrain features
    elevation = ee.Image('USGS/SRTMGL1_003').rename('ELEV')
    slope = ee.Terrain.slope(elevation).rename('SLOPE')
    aspect = ee.Terrain.aspect(elevation).rename('ASPECT')
    
    # Combine all features
    features_img = ee.Image.cat([ndvi, ndwi, bsi, iron, clay, elevation, slope, aspect])
    
    # Reduce to mean values
    reducer = features_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=30,
        maxPixels=1_000_000
    )
    
    res = reducer.getInfo()
    
    # Validate and return
    features = {}
    for key in ['NDVI', 'NDWI', 'BSI', 'IRON', 'CLAY', 'ELEV', 'SLOPE', 'ASPECT']:
        if key not in res or res[key] is None:
            raise RuntimeError(f'Missing feature {key} at location')
        features[key.lower()] = float(res[key])
    
    # Rename for consistency
    features['iron_oxide_ratio'] = features.pop('iron')
    features['clay_minerals'] = features.pop('clay')
    features['elevation'] = features.pop('elev')
    
    return features

def query_usgs_lithology(lat: float, lon: float, radius_km: float = 10.0) -> Optional[Dict[str, any]]:
    """
    Query USGS for geological lithology data.
    Focus on rock types conducive to geode formation.
    """
    try:
        # Try Macrostrat API (more reliable than USGS direct)
        url = "https://macrostrat.org/api/v2/units"
        params = {
            'lat': lat,
            'lng': lon,
            'format': 'json'
        }
        
        response = SESSION.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        # Parse for geode-relevant lithology
        basalt_presence = False
        limestone_presence = False
        sedimentary_score = 0.0
        lithology_types = []
        
        if 'success' in data and data.get('data'):
            units = data['data'].get('units', [])
            for unit in units:
                lith = unit.get('lith', '').lower() if unit.get('lith') else ''
                if lith:
                    lithology_types.append(lith)
                    
                    # Check for volcanic rocks (common geode hosts)
                    if any(term in lith for term in ['basalt', 'volcanic', 'rhyolite']):
                        basalt_presence = True
                    
                    # Check for limestone (another geode host)
                    if any(term in lith for term in ['limestone', 'carbonate', 'dolomite']):
                        limestone_presence = True
                    
                    # Score sedimentary rocks
                    if any(term in lith for term in ['sedimentary', 'sandstone', 'shale']):
                        sedimentary_score = 0.8
        
        return {
            'basalt_presence': basalt_presence,
            'limestone_presence': limestone_presence,
            'sedimentary_score': sedimentary_score,
            'lithology_types': lithology_types[:5]  # Top 5 types
        }
        
    except Exception as e:
        print(f"⚠️ USGS lithology query failed: {e}")
        return None

def calculate_geode_probability(lat: float, lon: float, features: Dict = None) -> Dict:
    """
    Calculate probability of geode formation at location.
    Uses geological features and heuristic scoring.
    """
    
    # Get features if not provided
    if features is None:
        try:
            features = extract_geode_features(lat, lon)
        except Exception as e:
            print(f"Failed to extract features: {e}")
            return {
                'geode_probability': 0.0,
                'method': 'failed',
                'error': str(e)
            }
    
    # Get lithology data if available
    lithology = query_usgs_lithology(lat, lon)
    
    # Heuristic scoring based on geological indicators
    score = 0.0
    indicators = {}
    
    # Exposed rock (high BSI)
    if 'bsi' in features:
        exposed_rock = max(0.0, min(1.0, (features['bsi'] + 1) / 2))
        score += exposed_rock * 0.25
        indicators['exposed_rock'] = exposed_rock
    
    # Iron content (common in geode-bearing rocks)
    if 'iron_oxide_ratio' in features:
        iron_content = max(0.0, min(1.0, features['iron_oxide_ratio']))
        score += iron_content * 0.20
        indicators['iron_content'] = iron_content
    
    # Clay minerals (weathering indicator)
    if 'clay_minerals' in features:
        clay_index = max(0.0, min(1.0, features['clay_minerals']))
        score += clay_index * 0.15
        indicators['clay_minerals'] = clay_index
    
    # Low vegetation (exposed geology)
    if 'ndvi' in features:
        low_veg = max(0.0, min(1.0, 1 - (features['ndvi'] + 1) / 2))
        score += low_veg * 0.15
        indicators['low_vegetation'] = low_veg
    
    # Terrain complexity (erosion exposes geodes)
    if 'slope' in features:
        terrain = max(0.0, min(1.0, features['slope'] / 45.0))
        score += terrain * 0.10
        indicators['terrain_complexity'] = terrain
    
    # Lithology bonus
    if lithology:
        if lithology.get('basalt_presence'):
            score += 0.10  # Volcanic rocks often host geodes
        if lithology.get('limestone_presence'):
            score += 0.05  # Limestone can host geodes
    
    # Ensure score is 0-1
    score = max(0.0, min(1.0, score))
    
    return {
        'geode_probability': float(score),
        'method': 'heuristic',
        'geological_indicators': indicators,
        'lithology': lithology if lithology else None,
        'coordinates': {'lat': lat, 'lon': lon}
    }

# Known geode locations for reference
KNOWN_GEODE_SITES = {
    'dugway': (39.9, -113.0),  # Dugway Geode Beds, Utah
    'hauser': (32.8, -113.7),  # Hauser Geode Beds, California  
    'keokuk': (40.4, -91.4),   # Keokuk, Iowa
    'woodbury': (36.1, -86.4),  # Woodbury, Tennessee
}

print("✅ Geode detection functions loaded")
print(f"   Known sites: {len(KNOWN_GEODE_SITES)}")
print("   Features: BSI, iron oxide, clay minerals, NDVI, terrain")
print("   Lithology: USGS/Macrostrat integration")
```

## Cell 14: Combined Archaeological + Geological Analysis

```python
# Combined Archaeological + Geological Analysis
"""
Unified analysis that detects both archaeological sites and geological features.
Provides comprehensive assessment of any location.
"""

def combined_analysis(lat: float, lon: float, analysis_type='both'):
    """
    Perform combined archaeological and geological analysis.
    
    Args:
        lat: Latitude
        lon: Longitude
        analysis_type: 'archaeological', 'geological', or 'both'
        
    Returns:
        Dict with comprehensive analysis results
    """
    
    results = {
        'location': {'lat': lat, 'lon': lon},
        'timestamp': datetime.now().isoformat(),
        'analysis_type': analysis_type
    }
    
    # Archaeological analysis (treasure/sites)
    if analysis_type in ['archaeological', 'both']:
        try:
            # Use existing anomaly detection
            arch_result = analyze_satellite_anomalies(lat, lon)
            
            results['archaeological'] = {
                'anomaly_score': arch_result.get('anomaly_score', 0),
                'confidence': arch_result.get('confidence', 0),
                'method': arch_result.get('method', 'unknown'),
                'features': arch_result.get('features', {}),
                'status': arch_result.get('status', 'unknown')
            }
            
            # Add interpretation
            score = arch_result.get('anomaly_score', 0)
            if score > 0.7:
                results['archaeological']['interpretation'] = "High probability archaeological site"
            elif score > 0.4:
                results['archaeological']['interpretation'] = "Possible archaeological interest"
            else:
                results['archaeological']['interpretation'] = "Low archaeological probability"
                
        except Exception as e:
            results['archaeological'] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Geological analysis (geodes/minerals)
    if analysis_type in ['geological', 'both']:
        try:
            # Get geode probability
            geode_result = calculate_geode_probability(lat, lon)
            
            results['geological'] = {
                'geode_probability': geode_result.get('geode_probability', 0),
                'indicators': geode_result.get('geological_indicators', {}),
                'lithology': geode_result.get('lithology', None),
                'method': geode_result.get('method', 'unknown')
            }
            
            # Add interpretation
            prob = geode_result.get('geode_probability', 0)
            if prob > 0.6:
                results['geological']['interpretation'] = "High geode potential"
            elif prob > 0.4:
                results['geological']['interpretation'] = "Moderate geode potential"
            else:
                results['geological']['interpretation'] = "Low geode potential"
                
        except Exception as e:
            results['geological'] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Combined assessment
    if analysis_type == 'both' and 'archaeological' in results and 'geological' in results:
        arch_score = results['archaeological'].get('anomaly_score', 0)
        geode_prob = results['geological'].get('geode_probability', 0)
        
        # Overall interest score
        overall_score = (arch_score * 0.5 + geode_prob * 0.5)
        
        results['combined'] = {
            'overall_score': overall_score,
            'primary_interest': 'archaeological' if arch_score > geode_prob else 'geological',
            'recommendation': get_recommendation(arch_score, geode_prob)
        }
    
    return results

def get_recommendation(arch_score: float, geode_prob: float) -> str:
    """Generate recommendation based on scores."""
    
    if arch_score > 0.7 and geode_prob > 0.6:
        return "🔥 HIGH PRIORITY: Both archaeological and geological interest!"
    elif arch_score > 0.7:
        return "🏛️ Priority for archaeological investigation"
    elif geode_prob > 0.6:
        return "💎 Priority for geological/mineral exploration"
    elif arch_score > 0.4 or geode_prob > 0.4:
        return "🔍 Moderate interest - worth further investigation"
    else:
        return "📍 Low priority site"

# Test function
def test_combined_analysis():
    """Test the combined analysis with known locations."""
    
    test_sites = [
        ("Giza Pyramids", 29.9792, 31.1342),  # Archaeological
        ("Dugway Geode Beds", 39.9, -113.0),  # Geological
        ("Oak Island", 44.5133, -64.2947),    # Mystery site
    ]
    
    for name, lat, lon in test_sites:
        print(f"\n🔬 Analyzing: {name}")
        print("="*50)
        
        result = combined_analysis(lat, lon, 'both')
        
        if 'archaeological' in result:
            arch = result['archaeological']
            print(f"Archaeological Score: {arch.get('anomaly_score', 0):.3f}")
            print(f"  Interpretation: {arch.get('interpretation', 'N/A')}")
        
        if 'geological' in result:
            geo = result['geological']
            print(f"Geode Probability: {geo.get('geode_probability', 0):.3f}")
            print(f"  Interpretation: {geo.get('interpretation', 'N/A')}")
        
        if 'combined' in result:
            comb = result['combined']
            print(f"\nOverall Score: {comb.get('overall_score', 0):.3f}")
            print(f"Recommendation: {comb.get('recommendation', 'N/A')}")

print("✅ Combined analysis functions loaded")
print("Use: combined_analysis(lat, lon, 'both') for full assessment")
print("Run: test_combined_analysis() to test with known sites")
```

## Cell 15: Region-Wide Scanning with Both Detection Types

```python
# Region-Wide Scanning for Archaeological Sites AND Geodes
"""
Scan an entire region for both treasure sites and geological features.
Creates comprehensive maps showing different types of interest points.
"""

def scan_region_comprehensive(
    center_lat: float, 
    center_lon: float, 
    radius_km: float = 20,
    grid_points: int = 25
):
    """
    Comprehensive regional scan for all types of anomalies.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_km: Search radius in kilometers
        grid_points: Number of points to analyze
        
    Returns:
        DataFrame with all analysis results
    """
    
    print(f"🌍 Comprehensive Regional Scan")
    print(f"   Center: ({center_lat:.4f}, {center_lon:.4f})")
    print(f"   Radius: {radius_km} km")
    print(f"   Points: {grid_points}")
    print("="*50)
    
    results = []
    
    # Generate grid of points
    angles = np.linspace(0, 2 * np.pi, grid_points)
    distances = np.random.uniform(0, radius_km, grid_points)
    
    for i, (angle, dist) in enumerate(zip(angles, distances)):
        # Calculate point coordinates
        lat_offset = (dist / 111.0) * np.cos(angle)
        lon_offset = (dist / (111.0 * np.cos(np.radians(center_lat)))) * np.sin(angle)
        
        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        
        print(f"\r📍 Analyzing point {i+1}/{grid_points}...", end='')
        
        # Perform combined analysis
        try:
            analysis = combined_analysis(lat, lon, 'both')
            
            # Extract key metrics
            result = {
                'lat': lat,
                'lon': lon,
                'arch_score': analysis.get('archaeological', {}).get('anomaly_score', 0),
                'geode_prob': analysis.get('geological', {}).get('geode_probability', 0),
                'overall_score': analysis.get('combined', {}).get('overall_score', 0),
                'primary_interest': analysis.get('combined', {}).get('primary_interest', 'unknown'),
                'recommendation': analysis.get('combined', {}).get('recommendation', '')
            }
            
            # Add specific features if available
            if 'archaeological' in analysis and 'features' in analysis['archaeological']:
                features = analysis['archaeological']['features']
                result['edge_density'] = features.get('edge_density', 0)
                result['vegetation_index'] = features.get('vegetation_index', 0)
            
            if 'geological' in analysis and 'indicators' in analysis['geological']:
                indicators = analysis['geological']['indicators']
                result['exposed_rock'] = indicators.get('exposed_rock', 0)
                result['iron_content'] = indicators.get('iron_content', 0)
            
            results.append(result)
            
        except Exception as e:
            print(f"\n  ⚠️ Failed at ({lat:.4f}, {lon:.4f}): {e}")
            continue
    
    print("\n✅ Scan complete!")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        # Sort by overall score
        df = df.sort_values('overall_score', ascending=False)
        
        # Print summary
        print("\n📊 ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total sites analyzed: {len(df)}")
        print(f"High archaeological interest (>0.7): {len(df[df['arch_score'] > 0.7])}")
        print(f"High geode potential (>0.6): {len(df[df['geode_prob'] > 0.6])}")
        print(f"Dual interest sites: {len(df[(df['arch_score'] > 0.5) & (df['geode_prob'] > 0.5)])}")
        
        print("\n🏆 Top 5 Overall Sites:")
        for idx, row in df.head(5).iterrows():
            print(f"  ({row['lat']:.4f}, {row['lon']:.4f})")
            print(f"    Archaeological: {row['arch_score']:.3f} | Geode: {row['geode_prob']:.3f}")
            print(f"    {row['recommendation']}")
    
    return df

# Example usage
print("✅ Regional scanning with dual detection loaded")
print("\nExample usage:")
print("  # Scan Utah region (has both archaeological sites and geode beds)")
print("  df = scan_region_comprehensive(39.5, -111.5, radius_km=50, grid_points=20)")
print("\n  # Analyze specific known sites")
print("  result = combined_analysis(39.9, -113.0, 'both')  # Dugway Geode Beds")
```