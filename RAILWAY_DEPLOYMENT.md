# Railway Deployment Guide

## Required Environment Variables

Set these in your Railway project settings under Variables:

### Essential for Google Earth Engine

1. **GEE_SERVICE_ACCOUNT_JSON** (Base64 encoded)
   ```bash
   # Convert your service account JSON to base64:
   base64 -i service-account-key.json | tr -d '\n'
   ```
   Then paste the output as the value for this variable.

2. **GEE_PROJECT_ID**
   - Your Google Cloud Project ID (e.g., "my-project-123456")
   - Must have Earth Engine API enabled

### Optional but Recommended

3. **MAPBOX_ACCESS_TOKEN**
   - Fallback satellite provider when GEE is unavailable
   - Get from: https://account.mapbox.com/access-tokens/

4. **WORKERS** (default: 1)
   - Number of Gunicorn workers
   - Keep at 1 for Railway free tier

5. **THREADS** (default: 2)
   - Threads per worker
   - Increase if handling multiple concurrent requests

6. **PORT**
   - Automatically provided by Railway
   - DO NOT set manually

## Setup Steps

1. **Create Google Cloud Service Account**
   - Go to: https://console.cloud.google.com/iam-admin/serviceaccounts
   - Create new service account
   - Add role: "Earth Engine Resource Viewer"
   - Create JSON key
   - Enable Earth Engine API in your project

2. **Prepare Service Account for Railway**
   ```bash
   # Base64 encode the JSON file
   base64 -i your-service-account.json | tr -d '\n' > encoded.txt
   ```

3. **Configure Railway Variables**
   - Go to your Railway project
   - Navigate to Variables tab
   - Add each variable listed above
   - Deploy will automatically restart

4. **Verify Deployment**
   - Check logs: `railway logs`
   - Test endpoint: `https://your-app.railway.app/api/status`
   - Should return: `{"status": "healthy", "providers": [...]}`

## Troubleshooting

### 502 Bad Gateway
- Check Railway logs for Earth Engine authentication errors
- Verify GEE_SERVICE_ACCOUNT_JSON is properly base64 encoded
- Ensure GEE_PROJECT_ID matches your Google Cloud project

### "No satellite providers available"
- Earth Engine authentication failed
- Add MAPBOX_ACCESS_TOKEN as fallback
- Check service account has proper permissions

### Container keeps restarting
- Health check failing
- Check /api/status endpoint is accessible
- Review memory usage (may need to upgrade from free tier)

### Slow response times
- Earth Engine operations can take 10-30 seconds
- Consider increasing WORKERS/THREADS
- May need to upgrade Railway plan for better CPU

## Environment Variable Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| GEE_SERVICE_ACCOUNT_JSON | Yes* | Base64 encoded service account JSON | eyJ0eXBlIjoi... |
| GEE_PROJECT_ID | Yes* | Google Cloud Project ID | my-project-123 |
| MAPBOX_ACCESS_TOKEN | No | Mapbox API token (fallback provider) | pk.eyJ1I... |
| WORKERS | No | Gunicorn workers (default: 1) | 1 |
| THREADS | No | Threads per worker (default: 2) | 2 |
| PORT | Auto | Railway provides this | 5000 |

*Required only if using Google Earth Engine. If you only have MAPBOX_ACCESS_TOKEN, the app will work with limited functionality.