#!/usr/bin/env python3
"""
TreasureHunter API Probe Script
Tests all endpoints with both payload variants to diagnose 500 errors
"""

import asyncio
import aiohttp
import json
import argparse
import sys
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Test coordinates (San Francisco)
TEST_LAT = 37.7749
TEST_LON = -122.4194

# ANSI colors for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class APIProbe:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.results = []
        
    async def probe_endpoint(self, session: aiohttp.ClientSession, 
                            method: str, path: str, 
                            payload: Optional[Dict] = None,
                            description: str = "") -> Dict:
        """Probe a single endpoint and capture response details"""
        url = f"{self.base_url}{path}"
        start_time = time.time()
        
        try:
            kwargs = {'timeout': aiohttp.ClientTimeout(total=30)}
            if payload:
                kwargs['json'] = payload
                kwargs['headers'] = {'Content-Type': 'application/json'}
            
            async with session.request(method, url, **kwargs) as response:
                elapsed_ms = (time.time() - start_time) * 1000
                
                # Read response body
                try:
                    body = await response.text()
                    try:
                        json_body = json.loads(body)
                        body_preview = json.dumps(json_body, indent=2)[:300]
                    except:
                        body_preview = body[:300]
                except:
                    body_preview = "<unable to read body>"
                
                return {
                    'method': method,
                    'path': path,
                    'description': description,
                    'status': response.status,
                    'elapsed_ms': elapsed_ms,
                    'headers': dict(response.headers),
                    'body_preview': body_preview,
                    'payload': payload,
                    'success': 200 <= response.status < 300
                }
                
        except asyncio.TimeoutError:
            return {
                'method': method,
                'path': path,
                'description': description,
                'status': 'TIMEOUT',
                'elapsed_ms': 30000,
                'error': 'Request timed out after 30s',
                'payload': payload,
                'success': False
            }
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return {
                'method': method,
                'path': path,
                'description': description,
                'status': 'ERROR',
                'elapsed_ms': elapsed_ms,
                'error': str(e),
                'payload': payload,
                'success': False
            }
    
    async def run_probes(self, repeat: int = 1, interval: float = 0):
        """Run all probes with optional repetition"""
        
        for iteration in range(repeat):
            if iteration > 0 and interval > 0:
                print(f"\n{BLUE}Waiting {interval}s before iteration {iteration + 1}/{repeat}...{RESET}")
                await asyncio.sleep(interval)
            
            print(f"\n{BOLD}=== Probe Iteration {iteration + 1}/{repeat} ==={RESET}")
            print(f"Target: {self.base_url}")
            print(f"Time: {datetime.now().isoformat()}\n")
            
            async with aiohttp.ClientSession() as session:
                # Phase 1: GET endpoints
                print(f"{BOLD}Phase 1: Testing GET endpoints{RESET}")
                get_probes = [
                    self.probe_endpoint(session, 'GET', '/status', 
                                      description="API Status"),
                    self.probe_endpoint(session, 'GET', '/example-locations',
                                      description="Example Locations"),
                ]
                
                get_results = await asyncio.gather(*get_probes)
                for result in get_results:
                    self.print_result(result)
                    self.results.append(result)
                
                # Phase 2: POST endpoints with lat/lon keys (treasure_api.py format)
                print(f"\n{BOLD}Phase 2: Testing POST endpoints with lat/lon keys{RESET}")
                post_lat_lon_probes = [
                    self.probe_endpoint(
                        session, 'POST', '/analyze/single',
                        payload={'lat': TEST_LAT, 'lon': TEST_LON, 'analysis_type': 'treasure'},
                        description="Single Analysis (lat/lon)"
                    ),
                    self.probe_endpoint(
                        session, 'POST', '/analyze/region',
                        payload={
                            'lat': TEST_LAT, 'lon': TEST_LON,
                            'radius_km': 10, 'num_points': 5,
                            'analysis_type': 'treasure',
                            'region_name': 'Test Region'
                        },
                        description="Region Analysis (lat/lon)"
                    ),
                    self.probe_endpoint(
                        session, 'POST', '/predict/discovery',
                        payload={
                            'lat': TEST_LAT, 'lon': TEST_LON,
                            'region_name': 'Discovery Test',
                            'search_radius_km': 10,
                            'grid_density': 5,
                            'min_score_threshold': 0.5
                        },
                        description="Predict Discovery (lat/lon)"
                    ),
                ]
                
                lat_lon_results = await asyncio.gather(*post_lat_lon_probes)
                for result in lat_lon_results:
                    self.print_result(result)
                    self.results.append(result)
                
                # Phase 3: POST endpoints with latitude/longitude keys (treasure_api_fixed.py format)
                print(f"\n{BOLD}Phase 3: Testing POST endpoints with latitude/longitude keys{RESET}")
                post_latitude_longitude_probes = [
                    self.probe_endpoint(
                        session, 'POST', '/analyze/single',
                        payload={'latitude': TEST_LAT, 'longitude': TEST_LON},
                        description="Single Analysis (latitude/longitude)"
                    ),
                    self.probe_endpoint(
                        session, 'POST', '/analyze/region',
                        payload={
                            'latitude': TEST_LAT, 'longitude': TEST_LON,
                            'radius_km': 10, 'num_points': 5,
                            'region_name': 'Test Region'
                        },
                        description="Region Analysis (latitude/longitude)"
                    ),
                ]
                
                latitude_longitude_results = await asyncio.gather(*post_latitude_longitude_probes)
                for result in latitude_longitude_results:
                    self.print_result(result)
                    self.results.append(result)
                
                # Phase 4: Model endpoints (only in treasure_api.py)
                print(f"\n{BOLD}Phase 4: Testing Model endpoints (treasure_api.py only){RESET}")
                model_probes = [
                    self.probe_endpoint(session, 'GET', '/model/status',
                                      description="Model Status"),
                    self.probe_endpoint(session, 'GET', '/model/list',
                                      description="List Models"),
                ]
                
                model_results = await asyncio.gather(*model_probes)
                for result in model_results:
                    self.print_result(result)
                    self.results.append(result)
    
    def print_result(self, result: Dict):
        """Print a single probe result with formatting"""
        status = result['status']
        elapsed = result['elapsed_ms']
        
        # Color code based on status
        if status == 'ERROR' or status == 'TIMEOUT':
            status_color = RED
            status_text = f"{status}"
        elif isinstance(status, int):
            if status >= 500:
                status_color = RED
                status_text = f"{status} ❌"
            elif status >= 400:
                status_color = YELLOW
                status_text = f"{status} ⚠️"
            elif status >= 200 and status < 300:
                status_color = GREEN
                status_text = f"{status} ✅"
            else:
                status_color = BLUE
                status_text = f"{status}"
        else:
            status_color = YELLOW
            status_text = str(status)
        
        # Format output
        print(f"{BOLD}{result['method']:6} {result['path']:30}{RESET} - {result['description']:35}")
        print(f"  Status: {status_color}{status_text}{RESET} | Time: {elapsed:.1f}ms")
        
        if result.get('payload'):
            print(f"  Payload keys: {list(result['payload'].keys())}")
        
        if result.get('error'):
            print(f"  {RED}Error: {result['error']}{RESET}")
        else:
            body = result.get('body_preview', '')
            if body:
                # Indent body preview
                body_lines = body.split('\n')
                print(f"  Response preview:")
                for line in body_lines[:5]:  # Show first 5 lines
                    print(f"    {line}")
        
        # Show key response headers
        if result.get('headers'):
            headers = result['headers']
            important_headers = ['content-type', 'x-powered-by', 'server']
            header_info = []
            for h in important_headers:
                if h in headers:
                    header_info.append(f"{h}={headers[h]}")
            if header_info:
                print(f"  Headers: {', '.join(header_info)}")
        
        print()  # Empty line for readability
    
    def analyze_results(self) -> int:
        """Analyze results and return exit code"""
        print(f"\n{BOLD}=== Analysis Summary ==={RESET}")
        
        total = len(self.results)
        errors_5xx = [r for r in self.results if isinstance(r.get('status'), int) and r['status'] >= 500]
        errors_4xx = [r for r in self.results if isinstance(r.get('status'), int) and 400 <= r['status'] < 500]
        timeouts = [r for r in self.results if r.get('status') == 'TIMEOUT']
        connection_errors = [r for r in self.results if r.get('status') == 'ERROR']
        successes = [r for r in self.results if r.get('success')]
        
        print(f"Total probes: {total}")
        print(f"  {GREEN}✅ Successful (2xx):{RESET} {len(successes)}")
        print(f"  {YELLOW}⚠️  Client errors (4xx):{RESET} {len(errors_4xx)}")
        print(f"  {RED}❌ Server errors (5xx):{RESET} {len(errors_5xx)}")
        print(f"  {RED}⏱️  Timeouts:{RESET} {len(timeouts)}")
        print(f"  {RED}🔌 Connection errors:{RESET} {len(connection_errors)}")
        
        # Payload variant analysis
        lat_lon_success = [r for r in self.results 
                          if r.get('payload') and 'lat' in r['payload'] and r.get('success')]
        latitude_longitude_success = [r for r in self.results 
                                     if r.get('payload') and 'latitude' in r['payload'] and r.get('success')]
        
        print(f"\n{BOLD}Payload Variant Analysis:{RESET}")
        print(f"  lat/lon format: {len(lat_lon_success)} successful")
        print(f"  latitude/longitude format: {len(latitude_longitude_success)} successful")
        
        if lat_lon_success and not latitude_longitude_success:
            print(f"  {YELLOW}⚠️  Backend appears to be treasure_api.py (expects lat/lon){RESET}")
        elif latitude_longitude_success and not lat_lon_success:
            print(f"  {YELLOW}⚠️  Backend appears to be treasure_api_fixed.py (expects latitude/longitude){RESET}")
        elif lat_lon_success and latitude_longitude_success:
            print(f"  {GREEN}✅ Backend accepts both payload formats{RESET}")
        
        # List 5xx errors in detail
        if errors_5xx:
            print(f"\n{RED}{BOLD}Server Errors (5xx) Detail:{RESET}")
            for err in errors_5xx:
                print(f"  {err['method']} {err['path']} - {err['description']}")
                if err.get('payload'):
                    print(f"    Payload: {json.dumps(err['payload'], separators=(',', ':'))}")
                if err.get('body_preview'):
                    print(f"    Response: {err['body_preview'][:100]}")
        
        # Performance analysis
        if successes:
            avg_time = sum(r['elapsed_ms'] for r in successes) / len(successes)
            max_time = max(r['elapsed_ms'] for r in successes)
            min_time = min(r['elapsed_ms'] for r in successes)
            print(f"\n{BOLD}Performance (successful requests):{RESET}")
            print(f"  Average: {avg_time:.1f}ms")
            print(f"  Min: {min_time:.1f}ms")
            print(f"  Max: {max_time:.1f}ms")
        
        # Return non-zero if any 5xx errors
        return 1 if errors_5xx else 0


async def main():
    parser = argparse.ArgumentParser(description='Probe TreasureHunter API endpoints')
    parser.add_argument('--base-url', 
                       default='https://treasurefinderai-production.up.railway.app/api',
                       help='Base API URL (default: production Railway URL)')
    parser.add_argument('--repeat', type=int, default=1,
                       help='Number of times to repeat probes')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Seconds between probe iterations')
    
    args = parser.parse_args()
    
    print(f"{BOLD}TreasureHunter API Probe{RESET}")
    print(f"{'=' * 50}")
    
    probe = APIProbe(args.base_url)
    await probe.run_probes(repeat=args.repeat, interval=args.interval)
    
    exit_code = probe.analyze_results()
    
    if exit_code != 0:
        print(f"\n{RED}❌ Probe detected server errors (5xx){RESET}")
    else:
        print(f"\n{GREEN}✅ No server errors detected{RESET}")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    asyncio.run(main())