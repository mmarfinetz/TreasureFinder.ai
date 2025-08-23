#!/usr/bin/env python3
"""Quick API test to diagnose hanging issue"""

import requests
import time

base_url = "https://treasurefinderai-production.up.railway.app/api"

tests = [
    ("GET", "/status", None),
    ("POST", "/analyze/single", {"lat": 37.7749, "lon": -122.4194}),
    ("POST", "/analyze/single", {"latitude": 37.7749, "longitude": -122.4194}),
]

print("Testing Railway API...")
print(f"Base URL: {base_url}\n")

for method, endpoint, data in tests:
    url = f"{base_url}{endpoint}"
    print(f"{method} {endpoint}...")
    
    try:
        start = time.time()
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=data, timeout=10)
        
        elapsed = time.time() - start
        
        print(f"  Status: {response.status_code}")
        print(f"  Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            if 'status' in data:
                print(f"  API Status: {data.get('status')}")
            if 'notebook_functions_available' in data:
                print(f"  Functions Available: {data.get('notebook_functions_available')}")
            elif 'notebook_functions' in data:
                print(f"  Functions Available: {data.get('notebook_functions')}")
        else:
            print(f"  Response: {response.text[:200]}")
            
    except requests.Timeout:
        print(f"  ❌ TIMEOUT after 10s")
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)}")
    
    print()

print("\nRecommendation:")
print("If all endpoints timeout, the Railway deployment might be stuck during startup.")
print("Check Railway dashboard for deployment status and logs.")