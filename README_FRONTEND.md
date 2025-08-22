# TreasureHunter Frontend System

A modern web-based frontend for the TreasureHunter satellite analysis system, providing an intuitive interface for archaeological and geological anomaly detection.

## 🚀 Quick Start

### 1. Convert the Jupyter Notebook to Python Module
```bash
python convert_notebook.py
```

### 2. Start the Flask API Server
```bash
python treasure_api.py
```

### 3. Open the Frontend
Navigate to http://localhost:5000 in your web browser

## 📁 Project Structure

```
Compare_Satellite_scripts/
├── frontend/
│   ├── index.html          # Main HTML interface
│   ├── styles.css          # Custom styling
│   └── app.js             # Frontend JavaScript logic
├── treasure_api.py         # Flask backend API
├── convert_notebook.py     # Notebook converter script
├── treasure_hunter_module.py  # Generated Python module (after conversion)
└── TreasurHunter.ipynb    # Original Jupyter notebook
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js (optional, for development)

### Backend Dependencies
```bash
pip install flask flask-cors pandas numpy requests
```

### Optional Dependencies (for full functionality)
```bash
# For machine learning features
pip install torch torchvision scikit-learn

# For image processing
pip install pillow opencv-python

# For geospatial analysis
pip install folium geopandas
```

## 🔧 Configuration

### Environment Variables
Set these in your terminal or create a `.env` file:

```bash
# Required for satellite imagery
export MAPBOX_ACCESS_TOKEN="your_mapbox_token_here"

# Optional: Enable debug mode
export FLASK_DEBUG=true

# Optional: Production mode (default: true)
export PRODUCTION_MODE=true
```

### Getting API Keys

1. **Mapbox Token** (Required for satellite imagery):
   - Sign up at https://www.mapbox.com/
   - Go to Account → Tokens
   - Copy your default public token

2. **Google Earth Engine** (Optional, for advanced features):
   - Sign up at https://earthengine.google.com/
   - Create a service account
   - Download credentials JSON

## 🎯 Features

### Analysis Types
- **Single Point Analysis**: Analyze a specific coordinate
- **Regional Analysis**: Scan multiple points in a radius
- **Predictive Discovery**: ML-based prediction of discovery zones
- **Comprehensive Scan**: Grid-based detailed analysis

### Frontend Features
- 🗺️ Interactive Leaflet map with markers
- 📍 Click-to-set coordinates
- 🔍 Location search and geocoding
- 📊 Real-time analysis results
- 🎨 Dark theme optimized UI
- 📱 Responsive design
- 🔔 Toast notifications
- 📈 Progress tracking

### Analysis Parameters
- **Location Input**: Coordinates or place names
- **Search Radius**: 1-100 km
- **Analysis Points**: 5-50 points
- **Score Threshold**: 0.0-1.0
- **Analysis Types**: Archaeological, Geological, Combined

## 💻 Usage

### Basic Workflow

1. **Enter Location**:
   - Use coordinates tab for precise input
   - Search by location name
   - Select from example locations
   - Click on map to set coordinates

2. **Configure Parameters**:
   - Choose analysis type
   - Set search radius
   - Adjust number of points
   - Set minimum score threshold

3. **Run Analysis**:
   - **Analyze Single Point**: Quick analysis of one location
   - **Analyze Region**: Multiple points in circular area
   - **Predict Discovery**: ML-based zone prediction

4. **Review Results**:
   - View markers on map (color-coded by score)
   - Check summary statistics
   - Browse detailed results table
   - Sort and filter results
   - Click markers for details

### API Endpoints

The Flask backend provides these REST endpoints:

```
GET  /                              # Serve frontend
GET  /api/status                    # API health check
GET  /api/example-locations         # Get example locations
POST /api/analyze/single            # Single point analysis
POST /api/analyze/region            # Regional analysis
POST /api/predict/discovery         # Predictive analysis
GET  /api/geocode?location=name     # Geocode location name
```

### Example API Calls

```javascript
// Single point analysis
fetch('/api/analyze/single', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        lat: 29.9792,
        lon: 31.1342,
        analysis_type: 'archaeological'
    })
})

// Regional analysis
fetch('/api/analyze/region', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        lat: 29.9792,
        lon: 31.1342,
        radius_km: 10,
        num_points: 20,
        analysis_type: 'both',
        region_name: 'Giza Plateau'
    })
})
```

## 🔍 Troubleshooting

### Common Issues

1. **"Cannot connect to backend API"**
   - Ensure Flask server is running: `python treasure_api.py`
   - Check if port 5000 is available
   - Verify firewall settings

2. **"Real analysis functions not available"**
   - Run `python convert_notebook.py` to generate the module
   - Check for import errors in console

3. **No satellite imagery**
   - Set MAPBOX_ACCESS_TOKEN environment variable
   - Verify token is valid

4. **Module import errors**
   - Install missing dependencies
   - Check Python version (3.8+ required)

### Running in Production

```bash
# Set production environment
export PRODUCTION_MODE=true
export FLASK_DEBUG=false

# Use production server (gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 treasure_api:app
```

### Using Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python convert_notebook.py
EXPOSE 5000
CMD ["python", "treasure_api.py"]
```

Build and run:
```bash
docker build -t treasurehunter .
docker run -p 5000:5000 -e MAPBOX_ACCESS_TOKEN=your_token treasurehunter
```

## 🎨 Customization

### Modify Analysis Parameters
Edit `treasure_api.py`:
```python
MAX_ANALYSIS_POINTS = 100  # Maximum points per analysis
MAX_RADIUS_KM = 500        # Maximum search radius
```

### Customize UI Theme
Edit `frontend/styles.css`:
```css
:root {
    --primary-color: #D4AF37;  /* Gold theme */
    --secondary-color: #8B4513; /* Brown accent */
    /* ... other color variables */
}
```

### Add New Analysis Types
Extend `treasure_hunter_module.py`:
```python
def custom_analysis(lat, lon):
    # Your custom analysis logic
    return results
```

## 📊 Performance Tips

1. **Optimize for Large Regions**:
   - Reduce number of analysis points
   - Increase score threshold
   - Use predictive mode for initial scan

2. **Improve Response Time**:
   - Enable caching in Flask
   - Use CDN for static assets
   - Implement pagination for results

3. **Handle Rate Limits**:
   - Implement request throttling
   - Cache satellite imagery
   - Use batch processing

## 🤝 Contributing

To contribute to the frontend system:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is provided as-is for educational and research purposes.

## 🆘 Support

For issues or questions:
- Check the troubleshooting section
- Review API documentation at `/api/docs`
- Open an issue on GitHub

---

**Note**: This system is designed for archaeological and geological research. Always respect local laws and obtain proper permissions before conducting field investigations based on analysis results.