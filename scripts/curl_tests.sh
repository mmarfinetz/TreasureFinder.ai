#!/bin/bash

# TreasureHunter API cURL Test Commands
# Tests both payload variants to identify backend version

BASE_URL="${1:-https://treasurefinderai-production.up.railway.app/api}"

echo "Testing API at: $BASE_URL"
echo "================================"

# Test status endpoint
echo -e "\n1. GET /api/status"
curl -s -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n" \
  "$BASE_URL/status" | jq '.' 2>/dev/null || cat

# Test single analysis with lat/lon (treasure_api.py format)
echo -e "\n2. POST /api/analyze/single (lat/lon format)"
curl -s -X POST -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n" \
  -H "Content-Type: application/json" \
  -d '{"lat": 37.7749, "lon": -122.4194, "analysis_type": "treasure"}' \
  "$BASE_URL/analyze/single" | jq '.' 2>/dev/null || cat

# Test single analysis with latitude/longitude (treasure_api_fixed.py format)
echo -e "\n3. POST /api/analyze/single (latitude/longitude format)"
curl -s -X POST -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n" \
  -H "Content-Type: application/json" \
  -d '{"latitude": 37.7749, "longitude": -122.4194}' \
  "$BASE_URL/analyze/single" | jq '.' 2>/dev/null || cat

# Test region analysis with lat/lon
echo -e "\n4. POST /api/analyze/region (lat/lon format)"
curl -s -X POST -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n" \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 37.7749,
    "lon": -122.4194,
    "radius_km": 10,
    "num_points": 5,
    "analysis_type": "treasure",
    "region_name": "San Francisco Test"
  }' \
  "$BASE_URL/analyze/region" | jq '.' 2>/dev/null || cat

# Test region analysis with latitude/longitude
echo -e "\n5. POST /api/analyze/region (latitude/longitude format)"
curl -s -X POST -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 37.7749,
    "longitude": -122.4194,
    "radius_km": 10,
    "num_points": 5,
    "region_name": "San Francisco Test"
  }' \
  "$BASE_URL/analyze/region" | jq '.' 2>/dev/null || cat

# Test predict discovery (only in treasure_api.py)
echo -e "\n6. POST /api/predict/discovery (lat/lon format)"
curl -s -X POST -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n" \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 37.7749,
    "lon": -122.4194,
    "region_name": "Discovery Test",
    "search_radius_km": 10,
    "grid_density": 5,
    "min_score_threshold": 0.5
  }' \
  "$BASE_URL/predict/discovery" | jq '.' 2>/dev/null || cat

# Test model status (only in treasure_api.py)
echo -e "\n7. GET /api/model/status"
curl -s -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n" \
  "$BASE_URL/model/status" | jq '.' 2>/dev/null || cat

echo -e "\n================================"
echo "Test complete. Check status codes:"
echo "  200-299: Success"
echo "  400-499: Client error (likely wrong payload format)"
echo "  500-599: Server error"
echo "  404: Endpoint not found (likely wrong backend version)"