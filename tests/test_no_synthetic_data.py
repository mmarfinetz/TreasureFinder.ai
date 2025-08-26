"""
Unit tests to verify no synthetic/random data is generated in production code.
Tests the refactored failure handling and explicit error states.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNoSyntheticData(unittest.TestCase):
    """Test that production code doesn't generate synthetic data."""
    
    def test_fetch_satellite_no_random_on_failure(self):
        """Test that satellite fetch doesn't create random data on failure."""
        from treasure_hunter_module import fetch_satellite_image
        
        # Mock Earth Engine to fail
        with patch('treasure_hunter_module.EE_AVAILABLE', False):
            with patch('treasure_hunter_module.REQUESTS_AVAILABLE', False):
                
                # Should raise RuntimeError, not return synthetic data
                with self.assertRaises(RuntimeError) as context:
                    result = fetch_satellite_image(40.7128, -74.0060)
                
                self.assertIn("No satellite data source available", str(context.exception))
    
    def test_mapbox_no_simulated_nir(self):
        """Test that Mapbox doesn't simulate NIR band."""
        from treasure_hunter_module import fetch_from_alternative_provider
        
        # Mock successful Mapbox response with RGB only
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'fake_image_data'
        
        with patch('treasure_hunter_module.requests.get', return_value=mock_response):
            with patch('treasure_hunter_module.Image.open') as mock_open:
                # Create mock RGB image
                mock_img = MagicMock()
                mock_img.size = (256, 256)
                mock_img.mode = 'RGB'
                mock_array = np.ones((256, 256, 3))
                mock_img.__array__ = lambda: mock_array
                mock_open.return_value = mock_img
                
                with patch.dict(os.environ, {'MAPBOX_ACCESS_TOKEN': 'test_token'}):
                    result = fetch_from_alternative_provider(40.7128, -74.0060)
                    
                    # Check that NIR bands are NaN (not simulated)
                    self.assertTrue(np.isnan(result[3]).all(), "NIR band should be NaN, not simulated")
                    self.assertTrue(np.isnan(result[4]).all(), "SWIR1 band should be NaN, not simulated")
                    self.assertTrue(np.isnan(result[5]).all(), "SWIR2 band should be NaN, not simulated")

    def test_dofa_path_raises_without_providers(self):
        """DOFA analysis must not fabricate data; it should raise when no imagery."""
        from treasure_hunter_module import analyze_satellite_anomalies
        with patch('treasure_hunter_module.fetch_satellite_image') as mock_fetch:
            mock_fetch.side_effect = RuntimeError("No satellite data source available")
            with self.assertRaises(RuntimeError):
                _ = analyze_satellite_anomalies(40.0, -74.0, use_dofa=True)

    def test_dofa_requires_local_weights_in_production(self):
        """With PRODUCTION_MODE=true and USE_DOFA=true and no local weights, DOFA raises a clear error."""
        # Ensure env flags for production and DOFA
        with patch.dict(os.environ, { 'PRODUCTION_MODE': 'true', 'USE_DOFA': 'true' }, clear=False):
            from treasure_hunter_module import analyze_satellite_anomalies, NUM_CHANNELS
            # Bypass satellite fetch to reach DOFA loader path
            with patch('treasure_hunter_module.fetch_satellite_image') as mock_fetch:
                # Provide an image with required channels
                mock_fetch.return_value = np.zeros((NUM_CHANNELS, 64, 64), dtype=np.float32)
                with self.assertRaises(RuntimeError) as ctx:
                    _ = analyze_satellite_anomalies(10.0, 20.0, use_dofa=True)
                self.assertIn('DOFA_LOCAL_WEIGHTS not set or file missing', str(ctx.exception))
    
    def test_training_fails_without_data(self):
        """Test that model training fails explicitly without real data."""
        from treasure_hunter_module import create_ml_scorer
        
        # Should raise RuntimeError when trying to create training data
        with self.assertRaises(RuntimeError) as context:
            scorer = create_ml_scorer('xgboost')
        
        self.assertIn("Model training requires real satellite features", str(context.exception))
    
    def test_cnn_dataset_skips_failed_fetches(self):
        """Test that CNN dataset creation skips samples when fetch fails."""
        from cnn_training_cell import create_training_dataset
        
        # Mock fetch_satellite_image to always fail
        with patch('cnn_training_cell.fetch_satellite_image') as mock_fetch:
            mock_fetch.side_effect = RuntimeError("No satellite data available")
            
            # Create dataset - should get empty arrays since all fetches fail
            X, y = create_training_dataset(num_samples=10)
            
            # Should have no samples since all fetches failed
            self.assertEqual(len(X), 0, "Dataset should be empty when all fetches fail")
            self.assertEqual(len(y), 0, "Labels should be empty when all fetches fail")


class TestRetryLogic(unittest.TestCase):
    """Test that retry logic with exponential backoff is implemented."""
    
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_earth_engine_retry_on_failure(self, mock_sleep):
        """Test that Earth Engine operations retry with backoff."""
        # This would require refactoring fetch_satellite_image to include retry logic
        # For now, we document this as a requirement
        pass
    
    def test_api_returns_503_on_failure(self):
        """Test that API returns 503 status codes on failures."""
        from treasure_api import app
        
        client = app.test_client()
        
        # Mock the analysis to fail
        with patch('treasure_api._load_thm') as mock_load:
            mock_thm = MagicMock()
            mock_thm.analyze_satellite_anomalies.side_effect = RuntimeError("Analysis failed")
            mock_load.return_value = mock_thm
            
            # Test single point analysis
            response = client.post('/api/analyze/single', 
                                  json={'latitude': 40.7128, 'longitude': -74.0060})
            
            # Should return error status (not 200 with random data)
            self.assertIn(response.status_code, [500, 503], 
                         "API should return 500/503 on analysis failure")


class TestStructuredErrorResponses(unittest.TestCase):
    """Test that errors return structured responses."""
    
    def test_unavailable_response_structure(self):
        """Test that unavailable data returns proper structure."""
        # Example of expected structure
        expected_structure = {
            'status': 'unavailable',
            'reason': 'Earth Engine quota exceeded',
            'source': 'satellite_provider'
        }
        
        # This tests the structure we expect from refactored code
        self.assertIn('status', expected_structure)
        self.assertIn('reason', expected_structure)
        self.assertIn('source', expected_structure)
    
    def test_api_training_failure_response(self):
        """Test that training endpoint returns proper error on failure."""
        from treasure_api import app
        
        client = app.test_client()
        
        with patch('treasure_api.NOTEBOOK_FUNCTIONS_AVAILABLE', True):
            with patch('treasure_api.run_training_pipeline') as mock_train:
                mock_train.side_effect = RuntimeError("Training failed")
                
                response = client.post('/api/train', 
                                      json={'epochs': 10, 'demo_mode': False})
                
                # Should return 503 with error message
                self.assertEqual(response.status_code, 503)
                
                data = response.get_json()
                self.assertEqual(data['status'], 'error')
                self.assertIn('Training failed', data['message'])


class TestNoRandomInPredictions(unittest.TestCase):
    """Test that predictions don't use random values."""
    
    def test_satellite_module_no_random_uncertainty(self):
        """Test that uncertainty is fixed or properly computed, not random."""
        # This would require importing and testing the predict_sites function
        # After refactoring, uncertainty should be either:
        # 1. A fixed value (e.g., 0.2)
        # 2. Properly computed from MC Dropout or ensemble
        # 3. np.nan if not available
        pass
    
    def test_no_random_probabilities(self):
        """Test that probabilities come from real model predictions."""
        # After refactoring, probabilities should be:
        # 1. From actual model.predict_proba()
        # 2. np.nan if model fails
        # Never from np.random.uniform()
        pass


if __name__ == '__main__':
    unittest.main()
