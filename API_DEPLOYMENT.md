# API Deployment Guide

## Quick Local Deployment (5 minutes)

### Option 1: Direct Python (Simplest)

```bash
# 1. Install dependencies
pip install flask flask-cors pandas numpy folium matplotlib requests

# 2. Convert notebooks to modules (REQUIRED!)
python convert_notebook.py

# 3. Set environment variables (optional but recommended)
export MAPBOX_ACCESS_TOKEN="your_token_here"  # Get free at mapbox.com

# 4. Start the API server
python treasure_api.py

# API is now running at http://localhost:5000
```

### Option 2: Docker (Recommended for Production)

```bash
# 1. Build and run with Docker Compose
docker-compose up -d

# 2. Check logs
docker-compose logs -f

# 3. Test the API
curl http://localhost:5000/health

# API is running at http://localhost:5000
```

## Cloud Deployment Options

### Deploy to Render (Free Tier Available)

1. **Fork/Clone the repo to your GitHub**

2. **Create a Render account** at https://render.com

3. **Create New Web Service:**
   - Connect your GitHub repo
   - Build Command: `pip install -r requirements.txt && python convert_notebook.py`
   - Start Command: `python treasure_api.py`
   - Environment: Python 3
   - Add environment variables:
     ```
     MAPBOX_ACCESS_TOKEN=your_token
     PORT=5000
     ```

4. **Deploy** - Render will automatically deploy on push to main

### Deploy to Railway (Simple & Fast)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and initialize
railway login
railway init

# Add environment variables
railway variables set MAPBOX_ACCESS_TOKEN="your_token"

# Deploy
railway up

# Get your URL
railway open
```

### Deploy to Google Cloud Run

```bash
# 1. Make sure deploy script is executable
chmod +x deploy_gcp.sh

# 2. Set your project ID
export GCP_PROJECT_ID="your-project-id"

# 3. Deploy
./deploy_gcp.sh cloudrun

# Your API will be at: https://treasurefinder-xxxxx-uc.a.run.app
```

### Deploy to AWS (EC2 or Lambda)

```bash
# 1. Make deploy script executable  
chmod +x deploy_aws.sh

# 2. Configure AWS credentials
aws configure

# 3. Deploy to EC2
./deploy_aws.sh launch

# Or deploy to Lambda (serverless)
./deploy_aws.sh lambda
```

### Deploy to Heroku

1. **Create `Procfile`:**
```bash
echo "web: python treasure_api.py" > Procfile
```

2. **Create `runtime.txt`:**
```bash
echo "python-3.11.0" > runtime.txt
```

3. **Deploy:**
```bash
# Install Heroku CLI
# Create app
heroku create treasurefinder-api

# Set environment variables
heroku config:set MAPBOX_ACCESS_TOKEN="your_token"

# Deploy
git push heroku main

# Open app
heroku open
```

## Production Setup with HTTPS

### Using Nginx Reverse Proxy

1. **Install Nginx:**
```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx
```

2. **Configure Nginx:**
```bash
sudo nano /etc/nginx/sites-available/treasurefinder
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

3. **Enable site and get SSL:**
```bash
sudo ln -s /etc/nginx/sites-available/treasurefinder /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl reload nginx

# Get free SSL certificate
sudo certbot --nginx -d your-domain.com
```

## API Endpoints

Once deployed, your API provides these endpoints:

### Health Check
```bash
curl https://your-api.com/health
```

### Analyze Single Location
```bash
curl -X POST https://your-api.com/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 37.7749,
    "lon": -122.4194,
    "analysis_type": "both"
  }'
```

### Scan Region
```bash
curl -X POST https://your-api.com/scan_region \
  -H "Content-Type: application/json" \
  -d '{
    "center_lat": 43.0,
    "center_lon": -111.0,
    "radius_miles": 50,
    "num_points": 100
  }'
```

### Get Geode Probability
```bash
curl -X POST https://your-api.com/geode_probability \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 43.0,
    "lon": -111.0
  }'
```

## Environment Variables

### Required for Basic Operation
```bash
# None - will work with statistical fallback
```

### Recommended for Full Features
```bash
MAPBOX_ACCESS_TOKEN=pk.xxx...           # Satellite imagery (easy to get)
GEE_PROJECT_ID=your-project             # Google Earth Engine
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json  # GEE service account
```

### Optional Enhancements
```bash
MINDAT_API_KEY=xxx                      # Mineral occurrence data
USGS_API_KEY=xxx                        # Geological surveys
SENTINEL_HUB_CLIENT_ID=xxx              # Sentinel satellite data
SENTINEL_HUB_CLIENT_SECRET=xxx
PLANET_API_KEY=xxx                      # Planet Labs imagery
```

## Testing Your Deployment

### 1. Basic Health Check
```python
import requests

# Replace with your deployed URL
API_URL = "http://localhost:5000"  # or "https://your-api.com"

# Test health endpoint
response = requests.get(f"{API_URL}/health")
print(response.json())
```

### 2. Test Analysis
```python
import requests
import json

# Test location analysis
data = {
    "lat": 37.7749,
    "lon": -122.4194,
    "analysis_type": "both"
}

response = requests.post(
    f"{API_URL}/analyze",
    json=data,
    headers={"Content-Type": "application/json"}
)

result = response.json()
print(json.dumps(result, indent=2))
```

### 3. Load Test (Optional)
```bash
# Install Apache Bench
sudo apt install apache2-utils

# Test with 100 requests, 10 concurrent
ab -n 100 -c 10 -p test.json -T application/json http://localhost:5000/analyze
```

## Monitoring & Logging

### Add Basic Monitoring
```python
# In treasure_api.py, add:
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.before_request
def log_request():
    logging.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
```

### Using PM2 for Process Management
```bash
# Install PM2
npm install -g pm2

# Start API with PM2
pm2 start treasure_api.py --interpreter python3 --name treasurefinder-api

# Monitor
pm2 monit

# Auto-restart on crash
pm2 startup
pm2 save
```

## Troubleshooting

### API Won't Start
```bash
# Check if port is in use
lsof -i :5000

# Kill process using port
kill -9 $(lsof -t -i:5000)

# Try different port
export PORT=8080
python treasure_api.py
```

### Module Import Errors
```bash
# Always run conversion first!
python convert_notebook.py

# Verify modules exist
ls -la treasure_hunter_module.py
```

### Memory Issues
```bash
# Limit worker memory (in treasure_api.py)
import resource
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, -1))  # 512MB limit
```

### Slow Response Times
- Add Redis caching for frequently accessed locations
- Use connection pooling for database connections
- Implement request queuing for heavy computations

## Security Best Practices

### 1. Add API Key Authentication
```python
# In treasure_api.py
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/analyze', methods=['POST'])
@require_api_key
def analyze():
    # ... existing code
```

### 2. Add Rate Limiting
```bash
pip install flask-limiter
```

```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per hour", "10 per minute"]
)

@app.route('/analyze', methods=['POST'])
@limiter.limit("5 per minute")
def analyze():
    # ... existing code
```

### 3. Enable CORS Properly
```python
from flask_cors import CORS

# Configure CORS for production
CORS(app, origins=['https://your-frontend.com'])
```

## Next Steps

1. **Deploy locally first** to test everything works
2. **Add environment variables** for API keys
3. **Choose a cloud provider** based on your needs:
   - Render/Railway: Easiest, free tier available
   - Google Cloud Run: Scalable, pay-per-use
   - AWS: Most options, enterprise-ready
4. **Add monitoring** to track usage and errors
5. **Implement caching** for frequently accessed locations