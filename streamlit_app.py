import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from treasure_hunter_module import (
    analyze_satellite_anomalies,
    scan_region_comprehensive as scan_region,
    cluster_detections,
    EXAMPLE_LOCATIONS
)
import os

st.set_page_config(page_title="TreasureHunter", layout="wide")

st.title("🗺️ TreasureHunter - Satellite Anomaly Detection")

# Sidebar for API keys
with st.sidebar:
    st.header("Configuration")
    mapbox_token = st.text_input("Mapbox Token", type="password")
    if mapbox_token:
        os.environ['MAPBOX_ACCESS_TOKEN'] = mapbox_token
        st.success("✅ Mapbox configured")

# Main interface
tab1, tab2, tab3 = st.tabs(["Single Location", "Region Scan", "Examples"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=29.9792, min_value=-90.0, max_value=90.0)
        lon = st.number_input("Longitude", value=31.1342, min_value=-180.0, max_value=180.0)
        
    if st.button("Analyze Location"):
        with st.spinner("Analyzing satellite data..."):
            result = analyze_satellite_anomalies(lat, lon)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Anomaly Score", f"{result['anomaly_score']:.3f}")
            col2.metric("Confidence", f"{result['confidence']:.3f}")
            col3.metric("Method", result['method'])
            
            st.info(result['description'])
            
            # Show features if available
            if 'features' in result:
                st.subheader("Extracted Features")
                features_df = pd.DataFrame([result['features']])
                st.dataframe(features_df.T)

with tab2:
    center_lat = st.number_input("Center Latitude", value=44.5133)
    center_lon = st.number_input("Center Longitude", value=-64.2947)
    radius = st.slider("Radius (km)", 5, 100, 20)
    
    if st.button("Scan Region"):
        with st.spinner(f"Scanning {radius}km radius..."):
            # scan_region is an alias of scan_region_comprehensive(center_lat, center_lon, radius_km, grid_points)
            results_df = scan_region(center_lat, center_lon, radius, grid_points=20)
            
            # Apply clustering
            clustered = cluster_detections(results_df)
            
            # Display map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            for _, row in clustered.iterrows():
                color = 'red' if row['score'] > 0.7 else 'orange' if row['score'] > 0.4 else 'green'
                folium.CircleMarker(
                    [row['lat'], row['lon']],
                    radius=row['score'] * 10,
                    color=color,
                    fill=True,
                    popup=f"Score: {row['score']:.3f}"
                ).add_to(m)
            
            st_folium(m, height=500)
            
            # Show top detections
            st.subheader("Top Anomalies")
            top_5 = clustered.nlargest(5, 'score')[['lat', 'lon', 'score', 'confidence']]
            st.dataframe(top_5)

with tab3:
    st.subheader("Example Locations")
    for name, coords in EXAMPLE_LOCATIONS.items():
        if st.button(f"Analyze {name}"):
            result = analyze_satellite_anomalies(coords[0], coords[1])
            st.success(f"{name}: Score={result['score']:.3f}, {result['description']}")