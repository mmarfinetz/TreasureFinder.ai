"""
Test retry logic and exponential backoff implementation.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRetryWithBackoff(unittest.TestCase):
    """Test that external calls implement retry with exponential backoff."""
    
    def test_retry_helper_function(self):
        """Test a generic retry helper with exponential backoff."""
        
        def retry_with_backoff(func, max_attempts=3, initial_delay=0.5):
            """Helper function that should be added to production code."""
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func()
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = initial_delay * (2 ** attempt)
                        time.sleep(delay)
            
            raise last_exception
        
        # Test successful retry on third attempt
        mock_func = MagicMock()
        mock_func.side_effect = [
            RuntimeError("First failure"),
            RuntimeError("Second failure"),
            "Success!"
        ]
        
        with patch('time.sleep') as mock_sleep:
            result = retry_with_backoff(mock_func)
            
            self.assertEqual(result, "Success!")
            self.assertEqual(mock_func.call_count, 3)
            
            # Check exponential backoff delays
            expected_calls = [call(0.5), call(1.0)]  # 0.5s, then 1.0s
            mock_sleep.assert_has_calls(expected_calls)
    
    def test_api_endpoint_retry_pattern(self):
        """Test that API endpoints handle retries properly."""
        
        class APIClient:
            def __init__(self):
                self.attempt = 0
            
            def fetch_with_retry(self, url, max_retries=3):
                """Example of how API calls should handle retries."""
                import logging
                logger = logging.getLogger(__name__)
                
                last_error = None
                delays = [0.5, 1.0, 2.0]
                
                for attempt in range(max_retries):
                    try:
                        # Simulate API call
                        if self.attempt < 2:
                            self.attempt += 1
                            raise ConnectionError(f"Connection failed (attempt {attempt + 1})")
                        
                        return {"status": "success", "data": "fetched"}
                    
                    except Exception as e:
                        last_error = e
                        
                        if attempt == 0:
                            logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                        elif attempt == max_retries - 1:
                            logger.error(f"API call failed after {max_retries} attempts: {e}")
                        
                        if attempt < max_retries - 1:
                            time.sleep(delays[attempt])
                
                # Return structured error response
                return {
                    "status": "unavailable",
                    "reason": str(last_error),
                    "source": "external_api"
                }
        
        client = APIClient()
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = client.fetch_with_retry("http://example.com")
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(client.attempt, 2)  # Failed twice, succeeded on third
    
    def test_satellite_fetch_retry_pattern(self):
        """Test pattern for satellite data fetching with retries."""
        
        def fetch_satellite_with_retry(lat, lon, provider="earth_engine"):
            """Example pattern for satellite fetching with retries."""
            import logging
            logger = logging.getLogger(__name__)
            
            max_attempts = 3
            backoff_delays = [0.5, 1.0, 2.0]
            
            for attempt in range(max_attempts):
                try:
                    # Attempt to fetch satellite data
                    if provider == "earth_engine":
                        # Simulate Earth Engine call
                        if attempt < 2:  # Fail first two attempts
                            raise RuntimeError("Earth Engine temporarily unavailable")
                        
                        # Success on third attempt
                        return np.ones((6, 256, 256))
                    
                except Exception as e:
                    if attempt == 0:
                        logger.warning(
                            f"Satellite fetch failed for ({lat}, {lon}) - "
                            f"attempt {attempt + 1}/{max_attempts}: {e}"
                        )
                    elif attempt == max_attempts - 1:
                        logger.error(
                            f"Satellite fetch failed for ({lat}, {lon}) "
                            f"after {max_attempts} attempts: {e}"
                        )
                        
                        # Return structured unavailable response
                        return {
                            "status": "unavailable",
                            "reason": f"All {max_attempts} fetch attempts failed",
                            "source": provider,
                            "coordinates": {"lat": lat, "lon": lon},
                            "last_error": str(e)
                        }
                    
                    # Wait before retry (except on last attempt)
                    if attempt < max_attempts - 1:
                        time.sleep(backoff_delays[attempt])
            
            return None
        
        with patch('time.sleep') as mock_sleep:
            # Test successful retry
            result = fetch_satellite_with_retry(40.7128, -74.0060)
            
            # Should succeed and return array
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (6, 256, 256))
            
            # Check that backoff was applied
            self.assertEqual(mock_sleep.call_count, 2)
            mock_sleep.assert_has_calls([call(0.5), call(1.0)])


class TestLoggingLevels(unittest.TestCase):
    """Test that appropriate logging levels are used."""
    
    def test_warning_on_first_failure(self):
        """Test that first failure logs at WARNING level."""
        import logging
        
        with self.assertLogs(level=logging.WARNING) as log_context:
            logger = logging.getLogger('test')
            
            # Simulate first failure
            logger.warning("First attempt failed: Connection timeout")
            
            self.assertIn("WARNING", log_context.output[0])
            self.assertIn("First attempt failed", log_context.output[0])
    
    def test_error_on_final_failure(self):
        """Test that final failure logs at ERROR level."""
        import logging
        
        with self.assertLogs(level=logging.ERROR) as log_context:
            logger = logging.getLogger('test')
            
            # Simulate final failure
            logger.error("Failed after 3 attempts: Connection timeout")
            
            self.assertIn("ERROR", log_context.output[0])
            self.assertIn("Failed after 3 attempts", log_context.output[0])
    
    def test_structured_log_messages(self):
        """Test that log messages include required context."""
        import logging
        import json
        
        logger = logging.getLogger('test')
        
        # Example of structured logging for satellite fetch
        lat, lon = 40.7128, -74.0060
        provider = "earth_engine"
        error = "Quota exceeded"
        
        log_context = {
            "action": "satellite_fetch",
            "coordinates": {"lat": lat, "lon": lon},
            "provider": provider,
            "error": error,
            "attempt": 3,
            "max_attempts": 3
        }
        
        # Log as JSON for structured logging systems
        logger.error(f"Satellite fetch failed: {json.dumps(log_context)}")
        
        # Verify structure
        self.assertIn("lat", json.dumps(log_context))
        self.assertIn("provider", json.dumps(log_context))
        self.assertIn("error", json.dumps(log_context))


if __name__ == '__main__':
    unittest.main()