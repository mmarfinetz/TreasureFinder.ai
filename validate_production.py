#!/usr/bin/env python3
"""
Production Validation Script
Validates that the system is configured for production use with real data only.
"""

import os
import sys
import csv
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Constants
REQUIRED_ENV_VARS = ['PRODUCTION_MODE']
FORBIDDEN_ENV_VARS = ['MOCK_DATA', 'ALLOW_TEST_MODE', 'DEBUG']
SATELLITE_PROVIDERS = {
    'google_earth_engine': ['GEE_PROJECT_ID', 'GOOGLE_EARTH_ENGINE_PROJECT'],
    'mapbox': ['MAPBOX_ACCESS_TOKEN'],
    'sentinel_hub': ['SENTINELHUB_CLIENT_ID', 'SENTINELHUB_CLIENT_SECRET'],
    'planet': ['PLANET_API_KEY']
}

class ProductionValidator:
    """Validates production configuration and data integrity."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.successes: List[str] = []
    
    def log(self, message: str, level: str = 'info'):
        """Log a message with appropriate formatting."""
        if not self.verbose and level == 'info':
            return
        
        symbols = {
            'error': '❌',
            'warning': '⚠️',
            'success': '✅',
            'info': 'ℹ️'
        }
        print(f"{symbols.get(level, '')} {message}")
    
    def validate_environment(self) -> bool:
        """Validate environment variables for production mode."""
        self.log("Validating environment configuration...", 'info')
        
        # Check required vars
        for var in REQUIRED_ENV_VARS:
            value = os.environ.get(var, '').lower()
            if value != 'true':
                self.errors.append(f"Environment variable {var} must be 'true' (current: '{value}')")
                self.log(f"{var} not set to 'true'", 'error')
            else:
                self.successes.append(f"{var} correctly set to 'true'")
                self.log(f"{var} = true", 'success')
        
        # Check forbidden vars
        for var in FORBIDDEN_ENV_VARS:
            value = os.environ.get(var, '').lower()
            if value in ['true', '1', 'yes']:
                self.errors.append(f"Environment variable {var} must not be set in production (current: '{value}')")
                self.log(f"{var} is set (forbidden in production)", 'error')
            else:
                self.successes.append(f"{var} not set (correct)")
        
        return len(self.errors) == 0
    
    def validate_satellite_providers(self) -> Dict[str, bool]:
        """Check which satellite providers are configured."""
        self.log("Checking satellite provider configuration...", 'info')
        
        configured = {}
        for provider, env_vars in SATELLITE_PROVIDERS.items():
            # Provider is configured if ANY of its env vars are set
            is_configured = any(os.environ.get(var) for var in env_vars)
            configured[provider] = is_configured
            
            if is_configured:
                self.successes.append(f"{provider} is configured")
                self.log(f"{provider}: configured", 'success')
            else:
                self.log(f"{provider}: not configured", 'warning')
        
        # At least one provider must be configured
        if not any(configured.values()):
            self.errors.append("No satellite providers configured! At least one provider is required.")
            self.log("No providers configured", 'error')
        
        return configured
    
    def validate_training_data(self, data_root: str = "frontend/training_data",
                              sample_size: int = 50) -> Dict[str, Dict]:
        """Validate training data integrity."""
        self.log(f"Validating training data in {data_root}...", 'info')
        
        results = {}
        data_root_path = Path(data_root)
        
        if not data_root_path.exists():
            self.errors.append(f"Data root directory not found: {data_root}")
            self.log(f"Data root not found: {data_root}", 'error')
            return results
        
        # Check each dataset directory
        datasets = ['DTM', 'Hillshade', 'Local_dominance', 'Open_Positive', 
                   'Sky_View_Factor', 'Slope']
        
        for dataset in datasets:
            dataset_path = data_root_path / dataset
            if not dataset_path.exists():
                self.warnings.append(f"Dataset directory not found: {dataset}")
                continue
            
            result = {'train': {}, 'valid': {}}
            
            for split in ['train', 'valid']:
                csv_file = f"{split}_annotations.csv"
                csv_path = dataset_path / csv_file
                
                if not csv_path.exists():
                    self.warnings.append(f"CSV not found: {csv_path}")
                    result[split] = {'total': 0, 'missing': 0, 'sample_missing': 0}
                    continue
                
                # Read CSV and validate files
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                
                total_rows = len(rows)
                
                # Check random sample
                sample = random.sample(rows, min(sample_size, total_rows)) if rows else []
                missing_in_sample = 0
                
                for row in sample:
                    if len(row) < 1:
                        continue
                    img_path = dataset_path / row[0]
                    if not img_path.exists():
                        missing_in_sample += 1
                
                result[split] = {
                    'total': total_rows,
                    'sample_size': len(sample),
                    'sample_missing': missing_in_sample,
                    'sample_valid_pct': 100 * (1 - missing_in_sample/max(len(sample), 1))
                }
                
                if missing_in_sample > 0:
                    self.errors.append(
                        f"{dataset}/{split}: {missing_in_sample}/{len(sample)} "
                        f"files missing in random sample"
                    )
                    self.log(
                        f"{dataset}/{split}: {missing_in_sample} missing files", 
                        'error'
                    )
                else:
                    self.successes.append(f"{dataset}/{split}: All sampled files exist")
                    self.log(
                        f"{dataset}/{split}: {len(sample)} files validated", 
                        'success'
                    )
            
            results[dataset] = result
        
        return results
    
    def validate_api_module(self) -> bool:
        """Check if the API module is properly configured."""
        self.log("Checking API module configuration...", 'info')
        
        # Check if treasure_hunter_module exists
        module_path = Path("treasure_hunter_module.py")
        if not module_path.exists():
            self.errors.append(
                "treasure_hunter_module.py not found. "
                "Run: python convert_notebook.py"
            )
            self.log("API module not found", 'error')
            return False
        
        # Check if it imports successfully
        try:
            import treasure_hunter_module
            self.successes.append("API module loads successfully")
            self.log("API module ready", 'success')
            return True
        except ImportError as e:
            self.errors.append(f"Failed to import API module: {e}")
            self.log(f"API module import failed: {e}", 'error')
            return False
    
    def run_full_validation(self) -> bool:
        """Run complete production validation."""
        print("=" * 60)
        print("PRODUCTION VALIDATION")
        print("=" * 60)
        
        # 1. Environment validation
        env_valid = self.validate_environment()
        
        # 2. Satellite provider validation
        providers = self.validate_satellite_providers()
        
        # 3. Training data validation
        training_data = self.validate_training_data()
        
        # 4. API module validation
        api_valid = self.validate_api_module()
        
        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"\n✅ Successes: {len(self.successes)}")
        for success in self.successes[:5]:  # Show first 5
            print(f"  - {success}")
        if len(self.successes) > 5:
            print(f"  ... and {len(self.successes) - 5} more")
        
        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings[:5]:
                print(f"  - {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")
        
        if self.errors:
            print(f"\n❌ Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")
        
        # Overall status
        is_valid = len(self.errors) == 0
        print("\n" + "=" * 60)
        if is_valid:
            print("✅ PRODUCTION VALIDATION PASSED")
            print("System is configured for production use with real data only.")
        else:
            print("❌ PRODUCTION VALIDATION FAILED")
            print("Please fix the errors above before running in production.")
        print("=" * 60)
        
        return is_valid

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate production configuration and data integrity"
    )
    parser.add_argument(
        "--data-root",
        default="frontend/training_data",
        help="Root directory for training data"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of files to sample for validation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Set production mode for validation
    os.environ['PRODUCTION_MODE'] = 'true'
    
    validator = ProductionValidator(verbose=not args.quiet)
    
    # Override data root if specified
    if args.data_root != "frontend/training_data":
        validator.validate_training_data(args.data_root, args.sample_size)
    else:
        is_valid = validator.run_full_validation()
        sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()