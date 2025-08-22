// TreasureHunter Frontend JavaScript

// Use port 5000 for the API server in local dev; on Vercel use relative /api via proxy
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000/api'
    : window.location.origin + '/api';

// Global state
let map = null;
let markers = [];
let currentResults = [];
let apiStatus = 'connecting';

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeMap();
    initializeEventListeners();
    checkAPIStatus();
    loadExampleLocations();
});

// Initialize Leaflet map
function initializeMap() {
    // Create map centered on default location (Giza)
    map = L.map('map').setView([29.9792, 31.1342], 10);
    
    // Add OpenStreetMap tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);
    
    // Add click handler for map
    map.on('click', (e) => {
        const { lat, lng } = e.latlng;
        document.getElementById('latitude').value = lat.toFixed(4);
        document.getElementById('longitude').value = lng.toFixed(4);
        showToast('Coordinates updated from map click', 'info');
    });
}

// Initialize event listeners
function initializeEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => switchTab(button));
    });
    
    // Parameter sliders
    document.getElementById('radius').addEventListener('input', (e) => {
        document.getElementById('radius-display').textContent = `${e.target.value} km`;
    });
    
    document.getElementById('num-points').addEventListener('input', (e) => {
        document.getElementById('points-display').textContent = `${e.target.value} points`;
    });
    
    document.getElementById('min-threshold').addEventListener('input', (e) => {
        document.getElementById('threshold-display').textContent = e.target.value;
    });
    
    // Analysis buttons
    document.getElementById('analyze-single-btn').addEventListener('click', analyzeSingleLocation);
    document.getElementById('analyze-region-btn').addEventListener('click', analyzeRegion);
    document.getElementById('predict-discovery-btn').addEventListener('click', predictDiscovery);
    document.getElementById('train-model-btn').addEventListener('click', openTrainingModal);
    document.getElementById('clear-results-btn').addEventListener('click', clearResults);
    
    // Geocoding
    document.getElementById('geocode-btn').addEventListener('click', geocodeLocation);
    document.getElementById('location-search').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') geocodeLocation();
    });
    
    // Map controls
    document.getElementById('toggle-satellite').addEventListener('click', toggleSatelliteView);
    document.getElementById('center-map').addEventListener('click', centerMapOnLocation);
    document.getElementById('fullscreen-map').addEventListener('click', toggleFullscreen);
    
    // Results controls
    document.getElementById('sort-by').addEventListener('change', sortResults);
    document.getElementById('filter-by').addEventListener('change', filterResults);
    
    // Modal close buttons
    document.querySelectorAll('.modal-close').forEach(button => {
        button.addEventListener('click', () => closeModal(button.closest('.modal')));
    });
    
    // Footer links
    document.getElementById('about-btn').addEventListener('click', (e) => {
        e.preventDefault();
        showAboutModal();
    });
    
    document.getElementById('help-btn').addEventListener('click', (e) => {
        e.preventDefault();
        showHelpModal();
    });
    
    document.getElementById('api-docs-btn').addEventListener('click', (e) => {
        e.preventDefault();
        window.open('/api/docs', '_blank');
    });
}

// Check API status
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/status`);
        const data = await response.json();
        
        if (data.status === 'online') {
            updateAPIStatus('online');
            
            // Show warning if notebook functions not available
            if (!data.notebook_functions_available) {
                showToast('Running in demo mode - real analysis functions not available', 'warning');
            }
        }
    } catch (error) {
        updateAPIStatus('offline');
        showToast('Cannot connect to backend API', 'error');
    }
}

// Update API status indicator
function updateAPIStatus(status) {
    apiStatus = status;
    const statusElement = document.getElementById('api-status');
    
    statusElement.className = `status ${status}`;
    statusElement.querySelector('span').textContent = 
        status === 'online' ? 'Connected' : 
        status === 'offline' ? 'Offline' : 
        'Connecting...';
}

// Load example locations
async function loadExampleLocations() {
    try {
        const response = await fetch(`${API_BASE_URL}/example-locations`);
        const data = await response.json();
        
        const container = document.getElementById('example-locations');
        container.innerHTML = '';
        
        Object.entries(data.locations).forEach(([key, coords]) => {
            const button = document.createElement('button');
            button.className = 'example-location';
            button.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            button.onclick = () => {
                document.getElementById('latitude').value = coords[0];
                document.getElementById('longitude').value = coords[1];
                switchToTab('coordinates');
                centerMapOnLocation();
            };
            container.appendChild(button);
        });
    } catch (error) {
        console.error('Failed to load example locations:', error);
    }
}

// Tab switching
function switchTab(button) {
    const tabName = button.dataset.tab;
    switchToTab(tabName);
}

function switchToTab(tabName) {
    // Update buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-tab`);
    });
}

// Geocode location
async function geocodeLocation() {
    const locationName = document.getElementById('location-search').value.trim();
    
    if (!locationName) {
        showToast('Please enter a location name', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/geocode?location=${encodeURIComponent(locationName)}`);
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('latitude').value = data.coordinates.lat;
            document.getElementById('longitude').value = data.coordinates.lon;
            switchToTab('coordinates');
            centerMapOnLocation();
            showToast(`Found coordinates for ${data.location}`, 'success');
        } else {
            showToast(data.error || 'Location not found', 'error');
        }
    } catch (error) {
        showToast('Geocoding failed', 'error');
    }
}

// Analyze single location
async function analyzeSingleLocation() {
    const lat = parseFloat(document.getElementById('latitude').value);
    const lon = parseFloat(document.getElementById('longitude').value);
    const analysisType = document.getElementById('analysis-type').value;
    
    if (isNaN(lat) || isNaN(lon)) {
        showToast('Please enter valid coordinates', 'error');
        return;
    }
    
    showLoading(true);
    clearResults();
    
    try {
        const response = await fetch(`${API_BASE_URL}/analyze/single`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lon, analysis_type: analysisType })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displaySingleResult(data.data);
            showToast('Analysis complete', 'success');
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        showToast(`Analysis failed: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Analyze region
async function analyzeRegion() {
    const lat = parseFloat(document.getElementById('latitude').value);
    const lon = parseFloat(document.getElementById('longitude').value);
    const radius = parseFloat(document.getElementById('radius').value);
    const numPoints = parseInt(document.getElementById('num-points').value);
    const analysisType = document.getElementById('analysis-type').value;
    const regionName = document.getElementById('region-name').value || 'Analysis Region';
    
    if (isNaN(lat) || isNaN(lon)) {
        showToast('Please enter valid coordinates', 'error');
        return;
    }
    
    showLoading(true, 'Analyzing region...');
    clearResults();
    
    try {
        const response = await fetch(`${API_BASE_URL}/analyze/region`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                lat, lon,
                radius_km: radius,
                num_points: numPoints,
                analysis_type: analysisType,
                region_name: regionName
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayRegionalResults(data.data);
            showToast(`Analyzed ${data.data.results.length} locations`, 'success');
        } else {
            throw new Error(data.error || 'Regional analysis failed');
        }
    } catch (error) {
        showToast(`Regional analysis failed: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Predict discovery zones
async function predictDiscovery() {
    const lat = parseFloat(document.getElementById('latitude').value);
    const lon = parseFloat(document.getElementById('longitude').value);
    const radius = parseFloat(document.getElementById('radius').value);
    const numPoints = parseInt(document.getElementById('num-points').value);
    const threshold = parseFloat(document.getElementById('min-threshold').value);
    const regionName = document.getElementById('region-name').value || 'Discovery Zone';
    
    if (isNaN(lat) || isNaN(lon)) {
        showToast('Please enter valid coordinates', 'error');
        return;
    }
    
    showLoading(true, 'Running predictive analysis...');
    clearResults();
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict/discovery`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                lat, lon,
                region_name: regionName,
                search_radius_km: radius,
                grid_density: numPoints,
                min_score_threshold: threshold
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayPredictionResults(data.data);
            showToast(`Found ${data.data.total_predictions} potential discovery zones`, 'success');
        } else {
            throw new Error(data.error || 'Prediction failed');
        }
    } catch (error) {
        showToast(`Predictive analysis failed: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Display single result
function displaySingleResult(result) {
    currentResults = [result];
    
    // Add marker to map
    clearMarkers();
    addMarker(result);
    
    // Center map on result
    map.setView([result.lat, result.lon], 12);
    
    // Update summary
    updateSummary([result]);
    
    // Show detailed results
    showDetailedResults([result]);
}

// Display regional results
function displayRegionalResults(data) {
    currentResults = data.results;
    
    // Add markers to map
    clearMarkers();
    data.results.forEach(result => addMarker(result));
    
    // Fit map to bounds
    if (data.results.length > 0) {
        const bounds = L.latLngBounds(data.results.map(r => [r.lat, r.lon]));
        map.fitBounds(bounds, { padding: [50, 50] });
    }
    
    // Update summary
    updateSummary(data.results, data.summary);
    
    // Show detailed results
    showDetailedResults(data.results);
}

// Display prediction results
function displayPredictionResults(data) {
    currentResults = data.predictions;
    
    // Add markers to map with special styling
    clearMarkers();
    data.predictions.forEach((result, index) => {
        result.isPrediction = true;
        result.rank = index + 1;
        addMarker(result);
    });
    
    // Fit map to bounds
    if (data.predictions.length > 0) {
        const bounds = L.latLngBounds(data.predictions.map(r => [r.lat, r.lon]));
        map.fitBounds(bounds, { padding: [50, 50] });
    }
    
    // Update summary
    const summary = {
        total_sites: data.total_predictions,
        high_priority: data.predictions.filter(r => r.score > 0.7).length,
        medium_priority: data.predictions.filter(r => r.score >= 0.4 && r.score <= 0.7).length,
        low_priority: data.predictions.filter(r => r.score < 0.4).length
    };
    updateSummary(data.predictions, summary);
    
    // Show detailed results
    showDetailedResults(data.predictions);
}

// Add marker to map
function addMarker(result) {
    const score = result.score || result.anomaly_score || 0;
    
    // Determine marker color based on score
    let color = '#28a745'; // Green (low)
    if (score > 0.7) color = '#dc3545'; // Red (high)
    else if (score > 0.4) color = '#ffc107'; // Yellow (medium)
    
    // Create custom icon
    const icon = L.divIcon({
        className: 'custom-marker',
        html: `<div style="background-color: ${color}; width: 20px; height: 20px; border-radius: 50%; border: 2px solid white; ${result.isPrediction ? 'animation: markerPulse 2s infinite;' : ''}"></div>`,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
    });
    
    // Create marker
    const marker = L.marker([result.lat, result.lon], { icon }).addTo(map);
    
    // Create popup content
    const popupContent = `
        <div class="marker-popup">
            <h4>${result.isPrediction ? `Prediction #${result.rank}` : 'Analysis Result'}</h4>
            <p class="score">Score: ${(score * 100).toFixed(1)}%</p>
            <p>Confidence: ${((result.confidence || 0.5) * 100).toFixed(1)}%</p>
            <p>Coordinates: ${result.lat.toFixed(4)}, ${result.lon.toFixed(4)}</p>
            ${result.description ? `<p>${result.description}</p>` : ''}
        </div>
    `;
    
    marker.bindPopup(popupContent);
    markers.push(marker);
    
    // Add click handler
    marker.on('click', () => {
        showLocationDetails(result);
    });
}

// Clear all markers
function clearMarkers() {
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
}

// Update summary statistics
function updateSummary(results, summary = null) {
    if (!summary) {
        // Calculate summary from results
        const scores = results.map(r => r.score || r.anomaly_score || 0);
        summary = {
            total_sites: results.length,
            high_priority: scores.filter(s => s > 0.7).length,
            medium_priority: scores.filter(s => s >= 0.4 && s <= 0.7).length,
            low_priority: scores.filter(s => s < 0.4).length
        };
    }
    
    document.getElementById('total-sites').textContent = summary.total_sites;
    document.getElementById('high-priority').textContent = summary.high_priority;
    document.getElementById('medium-priority').textContent = summary.medium_priority;
    document.getElementById('low-priority').textContent = summary.low_priority;
    
    document.getElementById('results-summary').classList.remove('hidden');
    document.getElementById('no-results').style.display = 'none';
}

// Show detailed results table
function showDetailedResults(results) {
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = '';
    
    results.forEach((result, index) => {
        const score = result.score || result.anomaly_score || 0;
        const confidence = result.confidence || 0.5;
        
        // Determine score class
        let scoreClass = 'score-low';
        if (score > 0.7) scoreClass = 'score-high';
        else if (score > 0.4) scoreClass = 'score-medium';
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${result.lat.toFixed(4)}, ${result.lon.toFixed(4)}</td>
            <td><span class="score-badge ${scoreClass}">${(score * 100).toFixed(1)}%</span></td>
            <td>${(confidence * 100).toFixed(1)}%</td>
            <td>${result.description || 'No description available'}</td>
            <td>
                <button class="action-btn" onclick="viewOnMap(${result.lat}, ${result.lon})">
                    <i class="fas fa-map-marker-alt"></i> View
                </button>
                <button class="action-btn" onclick="showLocationDetails(${JSON.stringify(result).replace(/"/g, '&quot;')})">
                    <i class="fas fa-info-circle"></i> Details
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
    
    document.getElementById('detailed-results').classList.remove('hidden');
}

// View location on map
function viewOnMap(lat, lon) {
    map.setView([lat, lon], 14);
    
    // Find and open marker popup
    const marker = markers.find(m => {
        const latlng = m.getLatLng();
        return Math.abs(latlng.lat - lat) < 0.0001 && Math.abs(latlng.lng - lon) < 0.0001;
    });
    
    if (marker) {
        marker.openPopup();
    }
}

// Show location details modal
function showLocationDetails(result) {
    const modal = document.getElementById('location-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    
    modalTitle.textContent = `Location Details - ${result.lat.toFixed(4)}, ${result.lon.toFixed(4)}`;
    
    // Build detailed content
    let content = `
        <div class="location-details">
            <h4>Analysis Scores</h4>
            <p><strong>Anomaly Score:</strong> ${((result.score || 0) * 100).toFixed(1)}%</p>
            <p><strong>Confidence:</strong> ${((result.confidence || 0.5) * 100).toFixed(1)}%</p>
            <p><strong>Method:</strong> ${result.method || 'Unknown'}</p>
    `;
    
    if (result.features) {
        content += `
            <h4>Feature Analysis</h4>
            <ul>
        `;
        for (const [key, value] of Object.entries(result.features)) {
            content += `<li><strong>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</strong> ${(value * 100).toFixed(1)}%</li>`;
        }
        content += `</ul>`;
    }
    
    if (result.description) {
        content += `
            <h4>Description</h4>
            <p>${result.description}</p>
        `;
    }
    
    content += `</div>`;
    modalBody.innerHTML = content;
    
    // Set up analyze button
    document.getElementById('analyze-this-location').onclick = () => {
        document.getElementById('latitude').value = result.lat;
        document.getElementById('longitude').value = result.lon;
        closeModal(modal);
        analyzeSingleLocation();
    };
    
    showModal(modal);
}

// Sort results
function sortResults() {
    const sortBy = document.getElementById('sort-by').value;
    
    let sorted = [...currentResults];
    
    switch (sortBy) {
        case 'score':
            sorted.sort((a, b) => (b.score || b.anomaly_score || 0) - (a.score || a.anomaly_score || 0));
            break;
        case 'confidence':
            sorted.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
            break;
        case 'distance':
            const centerLat = parseFloat(document.getElementById('latitude').value);
            const centerLon = parseFloat(document.getElementById('longitude').value);
            sorted.sort((a, b) => {
                const distA = getDistance(centerLat, centerLon, a.lat, a.lon);
                const distB = getDistance(centerLat, centerLon, b.lat, b.lon);
                return distA - distB;
            });
            break;
    }
    
    showDetailedResults(sorted);
}

// Filter results
function filterResults() {
    const filterBy = document.getElementById('filter-by').value;
    
    let filtered = [...currentResults];
    
    switch (filterBy) {
        case 'high':
            filtered = filtered.filter(r => (r.score || r.anomaly_score || 0) > 0.7);
            break;
        case 'medium':
            filtered = filtered.filter(r => {
                const score = r.score || r.anomaly_score || 0;
                return score >= 0.4 && score <= 0.7;
            });
            break;
        case 'low':
            filtered = filtered.filter(r => (r.score || r.anomaly_score || 0) < 0.4);
            break;
    }
    
    showDetailedResults(filtered);
}

// Clear results
function clearResults() {
    clearMarkers();
    currentResults = [];
    
    document.getElementById('results-summary').classList.add('hidden');
    document.getElementById('detailed-results').classList.add('hidden');
    document.getElementById('no-results').style.display = 'block';
}

// Toggle satellite view
let satelliteLayer = null;
function toggleSatelliteView() {
    if (satelliteLayer) {
        map.removeLayer(satelliteLayer);
        satelliteLayer = null;
    } else {
        satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri',
            maxZoom: 19
        }).addTo(map);
    }
}

// Center map on current location
function centerMapOnLocation() {
    const lat = parseFloat(document.getElementById('latitude').value);
    const lon = parseFloat(document.getElementById('longitude').value);
    
    if (!isNaN(lat) && !isNaN(lon)) {
        map.setView([lat, lon], 12);
    }
}

// Toggle fullscreen
function toggleFullscreen() {
    const mapContainer = document.getElementById('map');
    
    if (!document.fullscreenElement) {
        mapContainer.requestFullscreen().catch(err => {
            showToast('Could not enter fullscreen mode', 'error');
        });
    } else {
        document.exitFullscreen();
    }
}

// Show loading indicator
function showLoading(show, message = 'Analyzing satellite data...') {
    const loading = document.getElementById('loading-indicator');
    
    if (show) {
        loading.classList.remove('hidden');
        loading.querySelector('p').textContent = message;
        
        // Simulate progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 95) progress = 95;
            updateProgress(progress);
            
            if (!loading.classList.contains('hidden')) {
                clearInterval(interval);
            }
        }, 500);
    } else {
        loading.classList.add('hidden');
        updateProgress(0);
    }
}

// Update progress bar
function updateProgress(percent) {
    document.getElementById('progress-fill').style.width = `${percent}%`;
    document.getElementById('progress-text').textContent = `${Math.round(percent)}%`;
}

// Show modal
function showModal(modal) {
    modal.classList.add('show');
}

// Close modal
function closeModal(modal) {
    modal.classList.remove('show');
}

// Show about modal
function showAboutModal() {
    const modal = document.getElementById('location-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    
    modalTitle.textContent = 'About TreasureHunter';
    modalBody.innerHTML = `
        <div class="about-content">
            <h4>TreasureHunter Satellite Analysis System</h4>
            <p>Version 1.0</p>
            <br>
            <p>TreasureHunter is an advanced satellite imagery analysis system that uses machine learning and computer vision to identify potential archaeological sites, geological anomalies, and other points of interest.</p>
            <br>
            <h5>Features:</h5>
            <ul>
                <li>Real-time satellite imagery analysis</li>
                <li>Machine learning-based anomaly detection</li>
                <li>Regional scanning capabilities</li>
                <li>Predictive discovery zone identification</li>
                <li>Interactive mapping and visualization</li>
            </ul>
            <br>
            <h5>Technologies:</h5>
            <ul>
                <li>Python backend with Flask API</li>
                <li>TensorFlow/PyTorch for ML models</li>
                <li>Satellite imagery from multiple providers</li>
                <li>Leaflet.js for interactive mapping</li>
            </ul>
        </div>
    `;
    
    document.getElementById('analyze-this-location').style.display = 'none';
    showModal(modal);
}

// Show help modal
function showHelpModal() {
    const modal = document.getElementById('location-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    
    modalTitle.textContent = 'Help & Instructions';
    modalBody.innerHTML = `
        <div class="help-content">
            <h4>Getting Started</h4>
            <ol>
                <li><strong>Enter Location:</strong> Use coordinates, search for a place name, or select from examples</li>
                <li><strong>Configure Parameters:</strong> Adjust analysis type, search radius, and other settings</li>
                <li><strong>Run Analysis:</strong> Choose single point, regional, or predictive analysis</li>
                <li><strong>Review Results:</strong> Explore the map markers and detailed results table</li>
            </ol>
            <br>
            <h4>Analysis Types</h4>
            <ul>
                <li><strong>Archaeological Sites:</strong> Searches for potential archaeological features</li>
                <li><strong>Geological Features:</strong> Identifies geological anomalies and formations</li>
                <li><strong>Combined Analysis:</strong> Runs both archaeological and geological analysis</li>
                <li><strong>Comprehensive Scan:</strong> Deep analysis with extended feature extraction</li>
            </ul>
            <br>
            <h4>Tips</h4>
            <ul>
                <li>Click on the map to set coordinates quickly</li>
                <li>Use higher point counts for more detailed regional analysis</li>
                <li>Adjust the score threshold to filter results</li>
                <li>Click on map markers for detailed information</li>
            </ul>
        </div>
    `;
    
    document.getElementById('analyze-this-location').style.display = 'none';
    showModal(modal);
}

// Show toast notification
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-circle',
        warning: 'fas fa-exclamation-triangle',
        info: 'fas fa-info-circle'
    }[type];
    
    toast.innerHTML = `
        <i class="${icon}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// Calculate distance between two coordinates (Haversine formula)
function getDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
}

// Add animation for slide out
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// CNN Training Functions
let trainingChart = null;
let trainingInterval = null;
let currentModelInfo = {
    name: null,
    accuracy: null,
    trainedDate: null
};

// Open training modal
function openTrainingModal() {
    const modal = document.getElementById('training-modal');
    
    // Initialize training configuration listeners
    initializeTrainingListeners();
    
    // Check current model status
    checkModelStatus();
    
    // Load saved models list
    loadSavedModels();
    
    showModal(modal);
}

// Initialize training configuration listeners
function initializeTrainingListeners() {
    // Training mode selector
    const trainingMode = document.getElementById('training-mode');
    const customConfig = document.getElementById('custom-config');
    
    trainingMode.addEventListener('change', (e) => {
        if (e.target.value === 'custom') {
            customConfig.style.display = 'block';
        } else {
            customConfig.style.display = 'none';
        }
    });
    
    // Start training button
    document.getElementById('start-training-btn').addEventListener('click', startTraining);
    document.getElementById('stop-training-btn').addEventListener('click', stopTraining);
    
    // Model management buttons
    document.getElementById('save-model-btn').addEventListener('click', saveModel);
    document.getElementById('load-model-btn').addEventListener('click', loadModel);
    document.getElementById('test-model-btn').addEventListener('click', testModel);
}

// Check current model status
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/model/status`);
        const data = await response.json();
        
        if (data.success && data.model_loaded) {
            document.getElementById('current-model-name').textContent = data.model_name || 'Custom Model';
            document.getElementById('current-model-accuracy').textContent = `${(data.accuracy * 100).toFixed(1)}%`;
            document.getElementById('current-model-date').textContent = new Date(data.trained_date).toLocaleDateString();
            
            currentModelInfo = {
                name: data.model_name,
                accuracy: data.accuracy,
                trainedDate: data.trained_date
            };
            
            // Enable test button if model is loaded
            document.getElementById('test-model-btn').disabled = false;
        } else {
            document.getElementById('current-model-name').textContent = 'No model loaded';
            document.getElementById('current-model-accuracy').textContent = '--';
            document.getElementById('current-model-date').textContent = 'Never';
        }
    } catch (error) {
        console.error('Failed to check model status:', error);
    }
}

// Load saved models list
async function loadSavedModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/model/list`);
        const data = await response.json();
        
        const modelsList = document.getElementById('models-list');
        modelsList.innerHTML = '';
        
        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const modelItem = document.createElement('div');
                modelItem.className = 'model-item';
                modelItem.innerHTML = `
                    <div>
                        <div class="model-name">${model.name}</div>
                        <div class="model-stats">
                            <span>Accuracy: ${(model.accuracy * 100).toFixed(1)}%</span>
                            <span>Date: ${new Date(model.date).toLocaleDateString()}</span>
                        </div>
                    </div>
                    <div class="model-actions">
                        <button class="btn btn-sm" onclick="loadSpecificModel('${model.filename}')">
                            <i class="fas fa-upload"></i> Load
                        </button>
                        <button class="btn btn-sm" onclick="deleteModel('${model.filename}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                `;
                modelsList.appendChild(modelItem);
            });
        } else {
            modelsList.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No saved models</p>';
        }
    } catch (error) {
        console.error('Failed to load saved models:', error);
    }
}

// Start training
async function startTraining() {
    const mode = document.getElementById('training-mode').value;
    let config = {};
    
    // Get configuration based on mode
    if (mode === 'custom') {
        config = {
            num_samples: parseInt(document.getElementById('num-samples').value),
            epochs: parseInt(document.getElementById('num-epochs').value),
            learning_rate: parseFloat(document.getElementById('learning-rate').value),
            use_gpu: document.getElementById('use-gpu').checked
        };
    } else {
        const presets = {
            quick: { num_samples: 100, epochs: 10, learning_rate: 0.001 },
            balanced: { num_samples: 500, epochs: 30, learning_rate: 0.001 },
            comprehensive: { num_samples: 1000, epochs: 50, learning_rate: 0.0005 }
        };
        config = presets[mode];
        config.use_gpu = document.getElementById('use-gpu').checked;
    }
    
    // Show progress section
    document.getElementById('training-progress-section').style.display = 'block';
    document.getElementById('start-training-btn').style.display = 'none';
    document.getElementById('stop-training-btn').style.display = 'inline-flex';
    
    // Initialize progress chart
    initializeTrainingChart();
    
    // Update total epochs display
    document.getElementById('total-epochs').textContent = config.epochs;
    
    try {
        // Start training via API
        const response = await fetch(`${API_BASE_URL}/model/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('Training started successfully', 'success');
            
            // Start polling for progress
            trainingInterval = setInterval(() => updateTrainingProgress(data.training_id), 2000);
        } else {
            throw new Error(data.error || 'Failed to start training');
        }
    } catch (error) {
        showToast(`Training failed: ${error.message}`, 'error');
        resetTrainingUI();
    }
}

// Update training progress
async function updateTrainingProgress(trainingId) {
    try {
        const response = await fetch(`${API_BASE_URL}/model/training-progress/${trainingId}`);
        const data = await response.json();
        
        if (data.status === 'completed') {
            // Training completed
            clearInterval(trainingInterval);
            showToast('Training completed successfully!', 'success');
            
            // Update model info
            currentModelInfo = {
                name: 'Newly Trained Model',
                accuracy: data.final_accuracy,
                trainedDate: new Date().toISOString()
            };
            
            // Enable save button
            document.getElementById('save-model-btn').disabled = false;
            document.getElementById('test-model-btn').disabled = false;
            
            // Update displays
            document.getElementById('current-model-accuracy').textContent = `${(data.final_accuracy * 100).toFixed(1)}%`;
            
            resetTrainingUI();
            checkModelStatus();
            
        } else if (data.status === 'training') {
            // Update progress displays
            document.getElementById('current-epoch').textContent = data.current_epoch;
            document.getElementById('training-loss').textContent = data.training_loss.toFixed(4);
            document.getElementById('val-accuracy').textContent = `${(data.val_accuracy * 100).toFixed(1)}%`;
            
            // Update progress bar
            const progress = (data.current_epoch / data.total_epochs) * 100;
            document.getElementById('training-progress-bar').style.width = `${progress}%`;
            document.getElementById('training-progress-text').textContent = `${Math.round(progress)}%`;
            
            // Update chart
            updateTrainingChart(data.current_epoch, data.training_loss, data.val_loss);
            
        } else if (data.status === 'failed') {
            clearInterval(trainingInterval);
            showToast(`Training failed: ${data.error}`, 'error');
            resetTrainingUI();
        }
    } catch (error) {
        console.error('Failed to get training progress:', error);
    }
}

// Initialize training chart
function initializeTrainingChart() {
    const ctx = document.getElementById('loss-chart').getContext('2d');
    
    if (trainingChart) {
        trainingChart.destroy();
    }
    
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: 'rgb(212, 175, 55)',
                backgroundColor: 'rgba(212, 175, 55, 0.1)',
                tension: 0.1
            }, {
                label: 'Validation Loss',
                data: [],
                borderColor: 'rgb(23, 162, 184)',
                backgroundColor: 'rgba(23, 162, 184, 0.1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#b0b0b0'
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Epoch',
                        color: '#b0b0b0'
                    },
                    ticks: {
                        color: '#b0b0b0'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Loss',
                        color: '#b0b0b0'
                    },
                    ticks: {
                        color: '#b0b0b0'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });
}

// Update training chart
function updateTrainingChart(epoch, trainLoss, valLoss) {
    if (!trainingChart) return;
    
    trainingChart.data.labels.push(epoch);
    trainingChart.data.datasets[0].data.push(trainLoss);
    trainingChart.data.datasets[1].data.push(valLoss);
    trainingChart.update();
}

// Stop training
function stopTraining() {
    if (trainingInterval) {
        clearInterval(trainingInterval);
    }
    
    // Call API to stop training
    fetch(`${API_BASE_URL}/model/stop-training`, { method: 'POST' });
    
    showToast('Training stopped', 'warning');
    resetTrainingUI();
}

// Reset training UI
function resetTrainingUI() {
    document.getElementById('training-progress-section').style.display = 'none';
    document.getElementById('start-training-btn').style.display = 'inline-flex';
    document.getElementById('stop-training-btn').style.display = 'none';
    
    // Reset progress displays
    document.getElementById('current-epoch').textContent = '0';
    document.getElementById('training-loss').textContent = '--';
    document.getElementById('val-accuracy').textContent = '--';
    document.getElementById('training-progress-bar').style.width = '0%';
    document.getElementById('training-progress-text').textContent = '0%';
    
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
}

// Save model
async function saveModel() {
    const modelName = prompt('Enter a name for this model:');
    
    if (!modelName) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/model/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: modelName,
                accuracy: currentModelInfo.accuracy,
                trained_date: currentModelInfo.trainedDate
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('Model saved successfully', 'success');
            loadSavedModels();
        } else {
            throw new Error(data.error || 'Failed to save model');
        }
    } catch (error) {
        showToast(`Failed to save model: ${error.message}`, 'error');
    }
}

// Load model
function loadModel() {
    // This opens a file picker or loads from saved models
    loadSavedModels();
}

// Load specific model
async function loadSpecificModel(filename) {
    try {
        const response = await fetch(`${API_BASE_URL}/model/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('Model loaded successfully', 'success');
            checkModelStatus();
            
            // Close modal
            closeModal(document.getElementById('training-modal'));
        } else {
            throw new Error(data.error || 'Failed to load model');
        }
    } catch (error) {
        showToast(`Failed to load model: ${error.message}`, 'error');
    }
}

// Delete model
async function deleteModel(filename) {
    if (!confirm('Are you sure you want to delete this model?')) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/model/delete`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('Model deleted successfully', 'success');
            loadSavedModels();
        } else {
            throw new Error(data.error || 'Failed to delete model');
        }
    } catch (error) {
        showToast(`Failed to delete model: ${error.message}`, 'error');
    }
}

// Test model
async function testModel() {
    showToast('Testing model on known locations...', 'info');
    
    try {
        const response = await fetch(`${API_BASE_URL}/model/test`);
        const data = await response.json();
        
        if (data.success) {
            // Display test results
            const resultsHtml = `
                <h4>Model Test Results</h4>
                <p><strong>Overall Accuracy:</strong> ${(data.accuracy * 100).toFixed(1)}%</p>
                <h5>Test Locations:</h5>
                <ul>
                    ${data.test_results.map(r => `
                        <li>${r.location}: ${(r.score * 100).toFixed(1)}% confidence</li>
                    `).join('')}
                </ul>
            `;
            
            // Show results in modal
            const modal = document.getElementById('location-modal');
            document.getElementById('modal-title').textContent = 'Model Test Results';
            document.getElementById('modal-body').innerHTML = resultsHtml;
            document.getElementById('analyze-this-location').style.display = 'none';
            showModal(modal);
        } else {
            throw new Error(data.error || 'Failed to test model');
        }
    } catch (error) {
        showToast(`Failed to test model: ${error.message}`, 'error');
    }
}