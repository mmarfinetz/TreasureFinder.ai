#!/usr/bin/env python3
"""
Streamlit App for TreasureHunter Satellite Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# Import our module
try:
    from treasure_hunter_module import (
        main_analysis, 
        EXAMPLE_LOCATIONS,
        analyze_satellite_anomalies,
        generate_map,
        create_simple_map_html
    )
    MODULE_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import treasure_hunter_module: {e}")
    MODULE_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="🏴‍☠️ TreasureHunter",
    page_icon="🏴‍☠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏴‍☠️ TreasureHunter Satellite Analysis")
st.subtitle("Advanced Archaeological Site Detection Using Satellite Imagery")

if not MODULE_AVAILABLE:
    st.error("TreasureHunter module not available. Please check the installation.")
    st.stop()

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# Analysis mode
analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    ["Quick Analysis", "Detailed Analysis", "Example Location"]
)

# Location input
if analysis_mode == "Example Location":
    location_name = st.sidebar.selectbox(
        "Select Example Location",
        list(EXAMPLE_LOCATIONS.keys())
    )
    coordinates = EXAMPLE_LOCATIONS[location_name]
    st.sidebar.info(f"Coordinates: {coordinates[0]:.4f}, {coordinates[1]:.4f}")
else:
    location_name = st.sidebar.text_input("Location Name", "My Location")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=44.5133, format="%.4f")
    with col2:
        lon = st.number_input("Longitude", value=-64.2947, format="%.4f")
    coordinates = (lat, lon)

# Analysis parameters
st.sidebar.subheader("Analysis Parameters")
radius_km = st.sidebar.slider("Search Radius (km)", 1, 50, 10)

if analysis_mode == "Quick Analysis":
    num_points = st.sidebar.slider("Number of Points", 3, 10, 5)
elif analysis_mode == "Detailed Analysis":
    num_points = st.sidebar.slider("Number of Points", 10, 50, 20)
else:
    num_points = st.sidebar.slider("Number of Points", 3, 20, 10)

# Run analysis button
if st.sidebar.button("🔍 Run Analysis", type="primary"):
    st.session_state['run_analysis'] = True
    st.session_state['location_name'] = location_name
    st.session_state['coordinates'] = coordinates
    st.session_state['radius_km'] = radius_km
    st.session_state['num_points'] = num_points

# Main content
if st.session_state.get('run_analysis', False):
    st.header(f"📊 Analysis Results for {st.session_state['location_name']}")
    
    # Show parameters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latitude", f"{st.session_state['coordinates'][0]:.4f}")
    with col2:
        st.metric("Longitude", f"{st.session_state['coordinates'][1]:.4f}")
    with col3:
        st.metric("Radius", f"{st.session_state['radius_km']} km")
    with col4:
        st.metric("Points", st.session_state['num_points'])
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🛰️ Initializing satellite analysis...")
        progress_bar.progress(10)
        
        # Run analysis
        status_text.text("🔍 Analyzing satellite imagery...")
        progress_bar.progress(30)
        
        results_df = main_analysis(
            region_name=st.session_state['location_name'],
            coordinates=st.session_state['coordinates'],
            radius_km=st.session_state['radius_km'],
            num_points=st.session_state['num_points']
        )
        
        progress_bar.progress(80)
        status_text.text("📊 Processing results...")
        
        if results_df is not None and len(results_df) > 0:
            # Store results in session state
            st.session_state['results_df'] = results_df
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Summary metrics
            st.subheader("📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sites = len(results_df)
                st.metric("Total Sites Analyzed", total_sites)
            
            with col2:
                high_priority = len(results_df[results_df['score'] > 0.7])
                st.metric("High Priority Sites", high_priority)
            
            with col3:
                avg_score = results_df['score'].mean()
                st.metric("Average Score", f"{avg_score:.3f}")
            
            with col4:
                max_score = results_df['score'].max()
                st.metric("Highest Score", f"{max_score:.3f}")
            
            # Results table
            st.subheader("🎯 Top Results")
            
            # Sort by score
            sorted_df = results_df.sort_values('score', ascending=False)
            
            # Display top 10 results
            display_df = sorted_df.head(10)[['lat', 'lon', 'score', 'confidence', 'description', 'method']].copy()
            display_df['score'] = display_df['score'].round(3)
            display_df['confidence'] = display_df['confidence'].round(3)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "lat": "Latitude",
                    "lon": "Longitude", 
                    "score": st.column_config.NumberColumn("Score", format="%.3f"),
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
                    "description": "Description",
                    "method": "Method"
                }
            )
            
            # Map generation
            st.subheader("🗺️ Interactive Map")
            
            try:
                map_placeholder = st.empty()
                
                # Try to generate interactive map
                map_obj = generate_map(
                    results_df, 
                    center=[st.session_state['coordinates'][0], st.session_state['coordinates'][1]],
                    output_file='treasure_map.html'
                )
                
                if map_obj is not None:
                    # Display the map file
                    with open('treasure_map.html', 'r', encoding='utf-8') as f:
                        map_html = f.read()
                    st.components.v1.html(map_html, height=600)
                else:
                    # Fallback to simple HTML table
                    create_simple_map_html(results_df, 'simple_treasure_map.html')
                    st.info("Interactive map not available - using simple table view")
                    
                    with open('simple_treasure_map.html', 'r', encoding='utf-8') as f:
                        simple_html = f.read()
                    st.components.v1.html(simple_html, height=600)
                
            except Exception as e:
                st.error(f"Map generation failed: {e}")
                st.info("Showing data table instead")
                st.dataframe(results_df)
            
            # Download options
            st.subheader("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"treasure_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = results_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"treasure_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
        else:
            progress_bar.progress(100)
            status_text.text("❌ No results found")
            st.error("No analysis results were generated. Please check your parameters and try again.")
    
    except Exception as e:
        progress_bar.progress(100)
        status_text.text("❌ Analysis failed")
        st.error(f"Analysis failed: {str(e)}")
        
        # Show detailed error in expander
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

else:
    # Show welcome message
    st.header("🗺️ Welcome to TreasureHunter")
    
    st.markdown("""
    **TreasureHunter** uses advanced satellite imagery analysis to detect potential archaeological sites and anomalies.
    
    ### 🚀 How it works:
    1. **Select a location** - Choose from example sites or enter custom coordinates
    2. **Configure analysis** - Set search radius and number of analysis points
    3. **Run analysis** - Our AI examines satellite imagery for anomalies
    4. **Review results** - Get scored locations with interactive maps
    
    ### 📡 Available Analysis Methods:
    - **Spectral Analysis** - NDVI, NDWI, and other vegetation indices
    - **Edge Detection** - Geometric pattern recognition
    - **Thermal Analysis** - Temperature anomaly detection
    - **Machine Learning** - CNN-based anomaly scoring
    
    ### 🏴‍☠️ Example Locations:
    """)
    
    # Show example locations in a nice grid
    locations_data = []
    for name, coords in EXAMPLE_LOCATIONS.items():
        locations_data.append({
            "Location": name.replace('_', ' ').title(),
            "Latitude": f"{coords[0]:.4f}",
            "Longitude": f"{coords[1]:.4f}"
        })
    
    locations_df = pd.DataFrame(locations_data)
    st.dataframe(locations_df, use_container_width=True)
    
    st.info("👈 Use the sidebar to configure and start your analysis!")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🏴‍☠️ TreasureHunter v1.0**")
st.sidebar.markdown("Advanced Archaeological Site Detection")
st.sidebar.markdown(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")