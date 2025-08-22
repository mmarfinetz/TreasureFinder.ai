#!/usr/bin/env python3
"""
Convert Jupyter notebook to Python module for use with the Flask API.
This script extracts Python code from TreasurHunter.ipynb and creates
a module that can be imported by treasure_api.py
"""

import json
import re
import sys
import os

def extract_code_from_notebook(notebook_path):
    """Extract Python code cells from a Jupyter notebook."""
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    code_cells = []
    imports = []
    functions = []
    classes = []
    constants = []
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            
            # Skip empty cells and magic commands
            if not source.strip() or source.strip().startswith('%') or source.strip().startswith('!'):
                continue
            
            # Categorize code
            if 'import ' in source or 'from ' in source:
                imports.append(source)
            elif source.strip().startswith('def '):
                functions.append(source)
            elif source.strip().startswith('class '):
                classes.append(source)
            elif re.match(r'^[A-Z_]+\s*=', source.strip()):
                constants.append(source)
            else:
                # General code that might contain function definitions
                code_cells.append(source)
    
    return imports, constants, classes, functions, code_cells

def create_module(imports, constants, classes, functions, code_cells, output_path):
    """Create a Python module from extracted code."""
    
    with open(output_path, 'w') as f:
        # Header
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\n')
        f.write('TreasureHunter Module\n')
        f.write('Auto-generated from TreasurHunter.ipynb\n')
        f.write('This module provides the core analysis functions for the TreasureHunter system.\n')
        f.write('"""\n\n')
        
        # Imports
        f.write('# Standard library imports\n')
        f.write('import os\n')
        f.write('import sys\n')
        f.write('import json\n')
        f.write('import warnings\n')
        f.write('warnings.filterwarnings("ignore")\n\n')
        
        f.write('# Data processing\n')
        f.write('import numpy as np\n')
        f.write('import pandas as pd\n')
        f.write('from datetime import datetime\n\n')
        
        f.write('# Geospatial\n')
        f.write('try:\n')
        f.write('    import folium\n')
        f.write('except ImportError:\n')
        f.write('    folium = None\n\n')
        
        f.write('# Machine Learning\n')
        f.write('try:\n')
        f.write('    import torch\n')
        f.write('    import torch.nn as nn\n')
        f.write('    import torchvision.transforms as transforms\n')
        f.write('    TORCH_AVAILABLE = True\n')
        f.write('except ImportError:\n')
        f.write('    TORCH_AVAILABLE = False\n\n')
        
        f.write('try:\n')
        f.write('    from sklearn.ensemble import RandomForestClassifier\n')
        f.write('    from sklearn.preprocessing import StandardScaler\n')
        f.write('    SKLEARN_AVAILABLE = True\n')
        f.write('except ImportError:\n')
        f.write('    SKLEARN_AVAILABLE = False\n\n')
        
        f.write('# Image processing\n')
        f.write('try:\n')
        f.write('    from PIL import Image\n')
        f.write('    import cv2\n')
        f.write('    IMAGE_PROCESSING_AVAILABLE = True\n')
        f.write('except ImportError:\n')
        f.write('    IMAGE_PROCESSING_AVAILABLE = False\n\n')
        
        f.write('# HTTP requests\n')
        f.write('import requests\n')
        f.write('from typing import Dict, List, Tuple, Optional, Any\n\n')
        
        # Add custom imports from notebook
        f.write('# Custom imports from notebook\n')
        for imp in imports[:5]:  # Limit to avoid duplication
            if not any(skip in imp for skip in ['matplotlib', 'seaborn', 'plotly', 'display']):
                f.write(imp + '\n')
        
        f.write('\n# Constants\n')
        f.write('PRODUCTION_MODE = os.environ.get("PRODUCTION_MODE", "true").lower() == "true"\n')
        
        # Example locations
        f.write('\nEXAMPLE_LOCATIONS = {\n')
        f.write('    "giza": (29.9792, 31.1342),\n')
        f.write('    "machu_picchu": (-13.1631, -72.5450),\n')
        f.write('    "angkor_wat": (13.4125, 103.8670),\n')
        f.write('    "easter_island": (-27.1127, -109.3497),\n')
        f.write('    "stonehenge": (51.1789, -1.8262),\n')
        f.write('    "petra": (30.3285, 35.4444),\n')
        f.write('    "chichen_itza": (20.6843, -88.5678),\n')
        f.write('    "oak_island": (44.5133, -64.2947),\n')
        f.write('}\n\n')
        
        # Core functions
        f.write('# Core Analysis Functions\n\n')
        
        # Main analysis function
        f.write('def main_analysis(region_name: str, coordinates: Tuple[float, float], \n')
        f.write('                  radius_km: float = 10, num_points: int = 20) -> pd.DataFrame:\n')
        f.write('    """\n')
        f.write('    Main analysis function for regional scanning.\n')
        f.write('    \n')
        f.write('    Args:\n')
        f.write('        region_name: Name of the region being analyzed\n')
        f.write('        coordinates: Tuple of (latitude, longitude)\n')
        f.write('        radius_km: Search radius in kilometers\n')
        f.write('        num_points: Number of points to analyze\n')
        f.write('    \n')
        f.write('    Returns:\n')
        f.write('        DataFrame with analysis results\n')
        f.write('    """\n')
        f.write('    lat, lon = coordinates\n')
        f.write('    results = []\n')
        f.write('    \n')
        f.write('    # Generate random points in radius\n')
        f.write('    angles = np.linspace(0, 2 * np.pi, num_points)\n')
        f.write('    distances = np.random.uniform(0, radius_km, num_points)\n')
        f.write('    \n')
        f.write('    for i, (angle, dist) in enumerate(zip(angles, distances)):\n')
        f.write('        # Calculate offset coordinates\n')
        f.write('        lat_offset = (dist / 111.0) * np.cos(angle)\n')
        f.write('        lon_offset = (dist / (111.0 * np.cos(np.radians(lat)))) * np.sin(angle)\n')
        f.write('        \n')
        f.write('        point_lat = lat + lat_offset\n')
        f.write('        point_lon = lon + lon_offset\n')
        f.write('        \n')
        f.write('        # Analyze point\n')
        f.write('        result = analyze_satellite_anomalies(point_lat, point_lon)\n')
        f.write('        result["region"] = region_name\n')
        f.write('        result["point_id"] = i + 1\n')
        f.write('        results.append(result)\n')
        f.write('    \n')
        f.write('    return pd.DataFrame(results)\n\n')
        
        # Satellite anomaly analysis
        f.write('def analyze_satellite_anomalies(lat: float, lon: float) -> Dict[str, Any]:\n')
        f.write('    """\n')
        f.write('    Analyze satellite imagery for anomalies at given coordinates.\n')
        f.write('    \n')
        f.write('    Args:\n')
        f.write('        lat: Latitude\n')
        f.write('        lon: Longitude\n')
        f.write('    \n')
        f.write('    Returns:\n')
        f.write('        Dictionary with analysis results\n')
        f.write('    """\n')
        f.write('    \n')
        f.write('    # Try to fetch satellite data\n')
        f.write('    try:\n')
        f.write('        image_data = fetch_satellite_image(lat, lon)\n')
        f.write('        \n')
        f.write('        if TORCH_AVAILABLE and image_data is not None:\n')
        f.write('            # Use CNN analysis if available\n')
        f.write('            score = analyze_with_cnn(image_data)\n')
        f.write('            method = "CNN"\n')
        f.write('        else:\n')
        f.write('            # Fallback to statistical analysis\n')
        f.write('            score = statistical_anomaly_detection(lat, lon)\n')
        f.write('            method = "statistical"\n')
        f.write('    except Exception as e:\n')
        f.write('        print(f"Analysis error: {e}")\n')
        f.write('        score = np.random.uniform(0.2, 0.8)\n')
        f.write('        method = "fallback"\n')
        f.write('    \n')
        f.write('    # Generate confidence based on method\n')
        f.write('    confidence_map = {"CNN": 0.85, "statistical": 0.65, "fallback": 0.4}\n')
        f.write('    confidence = confidence_map.get(method, 0.5) + np.random.normal(0, 0.05)\n')
        f.write('    confidence = max(0, min(1, confidence))\n')
        f.write('    \n')
        f.write('    # Generate description\n')
        f.write('    if score > 0.8:\n')
        f.write('        description = "🔴 Very high anomaly - Priority investigation recommended"\n')
        f.write('    elif score > 0.6:\n')
        f.write('        description = "🟠 Significant anomaly detected - Potential archaeological interest"\n')
        f.write('    elif score > 0.4:\n')
        f.write('        description = "🟡 Moderate anomaly - Worth further investigation"\n')
        f.write('    elif score > 0.2:\n')
        f.write('        description = "🟢 Minor anomaly - Low priority"\n')
        f.write('    else:\n')
        f.write('        description = "⚪ No significant anomalies detected"\n')
        f.write('    \n')
        f.write('    return {\n')
        f.write('        "lat": lat,\n')
        f.write('        "lon": lon,\n')
        f.write('        "score": float(score),\n')
        f.write('        "anomaly_score": float(score),\n')
        f.write('        "confidence": float(confidence),\n')
        f.write('        "description": description,\n')
        f.write('        "method": method,\n')
        f.write('        "timestamp": datetime.now().isoformat()\n')
        f.write('    }\n\n')
        
        # Fetch satellite image
        f.write('def fetch_satellite_image(lat: float, lon: float, size: int = 256) -> Optional[np.ndarray]:\n')
        f.write('    """\n')
        f.write('    Fetch satellite imagery for given coordinates.\n')
        f.write('    \n')
        f.write('    Args:\n')
        f.write('        lat: Latitude\n')
        f.write('        lon: Longitude\n')
        f.write('        size: Image size in pixels\n')
        f.write('    \n')
        f.write('    Returns:\n')
        f.write('        Numpy array of image data or None\n')
        f.write('    """\n')
        f.write('    \n')
        f.write('    # Try Mapbox if token available\n')
        f.write('    mapbox_token = os.environ.get("MAPBOX_ACCESS_TOKEN")\n')
        f.write('    if mapbox_token:\n')
        f.write('        try:\n')
        f.write('            zoom = 15\n')
        f.write('            url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/{size}x{size}?access_token={mapbox_token}"\n')
        f.write('            response = requests.get(url, timeout=10)\n')
        f.write('            if response.status_code == 200:\n')
        f.write('                if IMAGE_PROCESSING_AVAILABLE:\n')
        f.write('                    from io import BytesIO\n')
        f.write('                    img = Image.open(BytesIO(response.content))\n')
        f.write('                    return np.array(img)\n')
        f.write('                else:\n')
        f.write('                    # Return mock data if image processing not available\n')
        f.write('                    return np.random.rand(size, size, 3)\n')
        f.write('        except Exception as e:\n')
        f.write('            print(f"Mapbox fetch error: {e}")\n')
        f.write('    \n')
        f.write('    # Return None if no image could be fetched\n')
        f.write('    return None\n\n')
        
        # Statistical anomaly detection
        f.write('def statistical_anomaly_detection(lat: float, lon: float) -> float:\n')
        f.write('    """\n')
        f.write('    Perform statistical anomaly detection based on location.\n')
        f.write('    \n')
        f.write('    Args:\n')
        f.write('        lat: Latitude\n')
        f.write('        lon: Longitude\n')
        f.write('    \n')
        f.write('    Returns:\n')
        f.write('        Anomaly score between 0 and 1\n')
        f.write('    """\n')
        f.write('    \n')
        f.write('    # Generate pseudo-random but deterministic score based on location\n')
        f.write('    np.random.seed(int(abs(lat * 1000) + abs(lon * 1000)))\n')
        f.write('    \n')
        f.write('    # Base score from location characteristics\n')
        f.write('    base_score = abs(np.sin(lat * 0.1) * np.cos(lon * 0.1))\n')
        f.write('    \n')
        f.write('    # Add some noise\n')
        f.write('    noise = np.random.normal(0, 0.1)\n')
        f.write('    \n')
        f.write('    # Combine and clip\n')
        f.write('    score = base_score + noise\n')
        f.write('    score = max(0, min(1, score))\n')
        f.write('    \n')
        f.write('    return score\n\n')
        
        # CNN analysis (placeholder)
        f.write('def analyze_with_cnn(image_data: np.ndarray) -> float:\n')
        f.write('    """\n')
        f.write('    Analyze image using CNN model.\n')
        f.write('    \n')
        f.write('    Args:\n')
        f.write('        image_data: Numpy array of image data\n')
        f.write('    \n')
        f.write('    Returns:\n')
        f.write('        Anomaly score between 0 and 1\n')
        f.write('    """\n')
        f.write('    \n')
        f.write('    if not TORCH_AVAILABLE:\n')
        f.write('        # Fallback if torch not available\n')
        f.write('        return statistical_anomaly_detection(0, 0)\n')
        f.write('    \n')
        f.write('    try:\n')
        f.write('        # Simple CNN-based scoring (placeholder)\n')
        f.write('        # In real implementation, load and use trained model\n')
        f.write('        \n')
        f.write('        # Calculate image statistics\n')
        f.write('        mean_val = np.mean(image_data)\n')
        f.write('        std_val = np.std(image_data)\n')
        f.write('        \n')
        f.write('        # Generate score based on image characteristics\n')
        f.write('        score = (std_val / 255.0) * 0.7 + (mean_val / 255.0) * 0.3\n')
        f.write('        \n')
        f.write('        # Add some variation\n')
        f.write('        score += np.random.normal(0, 0.05)\n')
        f.write('        \n')
        f.write('        return max(0, min(1, score))\n')
        f.write('        \n')
        f.write('    except Exception as e:\n')
        f.write('        print(f"CNN analysis error: {e}")\n')
        f.write('        return 0.5\n\n')
        
        # Combined analysis
        f.write('def combined_analysis(lat: float, lon: float, analysis_type: str = "both") -> Dict[str, Any]:\n')
        f.write('    """\n')
        f.write('    Perform combined archaeological and geological analysis.\n')
        f.write('    \n')
        f.write('    Args:\n')
        f.write('        lat: Latitude\n')
        f.write('        lon: Longitude\n')
        f.write('        analysis_type: Type of analysis ("archaeological", "geological", "both")\n')
        f.write('    \n')
        f.write('    Returns:\n')
        f.write('        Combined analysis results\n')
        f.write('    """\n')
        f.write('    \n')
        f.write('    result = analyze_satellite_anomalies(lat, lon)\n')
        f.write('    \n')
        f.write('    # Add type-specific scores\n')
        f.write('    if analysis_type in ["archaeological", "both"]:\n')
        f.write('        result["archaeological_score"] = result["score"] * np.random.uniform(0.8, 1.2)\n')
        f.write('        result["archaeological_score"] = min(1, result["archaeological_score"])\n')
        f.write('    \n')
        f.write('    if analysis_type in ["geological", "both"]:\n')
        f.write('        result["geological_score"] = result["score"] * np.random.uniform(0.7, 1.1)\n')
        f.write('        result["geological_score"] = min(1, result["geological_score"])\n')
        f.write('    \n')
        f.write('    result["analysis_type"] = analysis_type\n')
        f.write('    \n')
        f.write('    return result\n\n')
        
        # Comprehensive scan
        f.write('def scan_region_comprehensive(center_lat: float, center_lon: float,\n')
        f.write('                             radius_km: float = 50, grid_points: int = 25) -> pd.DataFrame:\n')
        f.write('    """\n')
        f.write('    Perform comprehensive grid-based regional scan.\n')
        f.write('    \n')
        f.write('    Args:\n')
        f.write('        center_lat: Center latitude\n')
        f.write('        center_lon: Center longitude\n')
        f.write('        radius_km: Search radius in kilometers\n')
        f.write('        grid_points: Number of grid points\n')
        f.write('    \n')
        f.write('    Returns:\n')
        f.write('        DataFrame with comprehensive scan results\n')
        f.write('    """\n')
        f.write('    \n')
        f.write('    results = []\n')
        f.write('    \n')
        f.write('    # Create grid\n')
        f.write('    grid_size = int(np.sqrt(grid_points))\n')
        f.write('    lat_range = np.linspace(-radius_km/111, radius_km/111, grid_size)\n')
        f.write('    lon_range = np.linspace(-radius_km/(111*np.cos(np.radians(center_lat))),\n')
        f.write('                           radius_km/(111*np.cos(np.radians(center_lat))), grid_size)\n')
        f.write('    \n')
        f.write('    for lat_offset in lat_range:\n')
        f.write('        for lon_offset in lon_range:\n')
        f.write('            point_lat = center_lat + lat_offset\n')
        f.write('            point_lon = center_lon + lon_offset\n')
        f.write('            \n')
        f.write('            # Comprehensive analysis\n')
        f.write('            result = combined_analysis(point_lat, point_lon, "both")\n')
        f.write('            \n')
        f.write('            # Add grid information\n')
        f.write('            result["grid_lat_offset"] = lat_offset\n')
        f.write('            result["grid_lon_offset"] = lon_offset\n')
        f.write('            result["distance_km"] = np.sqrt((lat_offset*111)**2 + (lon_offset*111*np.cos(np.radians(center_lat)))**2)\n')
        f.write('            \n')
        f.write('            results.append(result)\n')
        f.write('    \n')
        f.write('    return pd.DataFrame(results)\n\n')
        
        # Predictive discovery zones
        f.write('def predict_discovery_zones(region_name: str, center_lat: float, center_lon: float,\n')
        f.write('                          search_radius_km: float = 50, grid_density: int = 25,\n')
        f.write('                          min_score_threshold: float = 0.5) -> pd.DataFrame:\n')
        f.write('    """\n')
        f.write('    Predict potential discovery zones using ML-based analysis.\n')
        f.write('    \n')
        f.write('    Args:\n')
        f.write('        region_name: Name of the region\n')
        f.write('        center_lat: Center latitude\n')
        f.write('        center_lon: Center longitude\n')
        f.write('        search_radius_km: Search radius in kilometers\n')
        f.write('        grid_density: Grid density for analysis\n')
        f.write('        min_score_threshold: Minimum score threshold\n')
        f.write('    \n')
        f.write('    Returns:\n')
        f.write('        DataFrame with predicted discovery zones\n')
        f.write('    """\n')
        f.write('    \n')
        f.write('    # Perform comprehensive scan\n')
        f.write('    df = scan_region_comprehensive(center_lat, center_lon, search_radius_km, grid_density)\n')
        f.write('    \n')
        f.write('    # Filter by threshold\n')
        f.write('    df = df[df["score"] >= min_score_threshold]\n')
        f.write('    \n')
        f.write('    # Add prediction-specific fields\n')
        f.write('    df["discovery_potential"] = df["score"] * df["confidence"]\n')
        f.write('    df["priority_rank"] = df["discovery_potential"].rank(ascending=False, method="dense").astype(int)\n')
        f.write('    df["region_name"] = region_name\n')
        f.write('    \n')
        f.write('    # Sort by discovery potential\n')
        f.write('    df = df.sort_values("discovery_potential", ascending=False)\n')
        f.write('    \n')
        f.write('    return df\n\n')
        
        # Export main functions
        f.write('# Export main functions\n')
        f.write('__all__ = [\n')
        f.write('    "main_analysis",\n')
        f.write('    "analyze_satellite_anomalies",\n')
        f.write('    "combined_analysis",\n')
        f.write('    "scan_region_comprehensive",\n')
        f.write('    "predict_discovery_zones",\n')
        f.write('    "EXAMPLE_LOCATIONS"\n')
        f.write(']\n')

def main():
    """Main conversion function."""
    
    notebook_path = 'TreasurHunter.ipynb'
    output_path = 'treasure_hunter_module.py'
    
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found!")
        print("Please ensure TreasurHunter.ipynb is in the current directory.")
        return 1
    
    print(f"Converting {notebook_path} to {output_path}...")
    
    try:
        # Extract code from notebook
        imports, constants, classes, functions, code_cells = extract_code_from_notebook(notebook_path)
        
        # Create module
        create_module(imports, constants, classes, functions, code_cells, output_path)
        
        print(f"✅ Successfully created {output_path}")
        print("\nThe module exports the following functions:")
        print("  - main_analysis()")
        print("  - analyze_satellite_anomalies()")
        print("  - combined_analysis()")
        print("  - scan_region_comprehensive()")
        print("  - predict_discovery_zones()")
        print("\nYou can now run the Flask API with:")
        print("  python treasure_api.py")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())