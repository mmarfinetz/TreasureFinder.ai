#!/usr/bin/env python3
"""
Convert Jupyter notebooks to Python modules for import.
This is required to use the notebook code as a module in the API/web apps.
"""

import json
import sys
import re
from pathlib import Path

def convert_notebook_to_module(notebook_path, output_path=None):
    """
    Convert a Jupyter notebook to a Python module.
    
    Args:
        notebook_path: Path to the .ipynb file
        output_path: Optional output path for .py file (defaults to same name)
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    if output_path is None:
        output_path = notebook_path.with_suffix('.py')
    else:
        output_path = Path(output_path)
    
    print(f"Converting {notebook_path} to {output_path}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    code_cells = []
    
    # Add header
    code_cells.append(f'"""')
    code_cells.append(f'Converted from {notebook_path.name}')
    code_cells.append(f'This module contains all code from the Jupyter notebook.')
    code_cells.append(f'"""')
    code_cells.append('')
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Get the source code
            source = ''.join(cell['source'])
            
            # Skip empty cells
            if not source.strip():
                continue
            
            # Skip cells that are just pip installs or shell commands
            if source.strip().startswith('!') or source.strip().startswith('%'):
                # Convert important pip installs to comments
                if 'pip install' in source:
                    code_cells.append(f"# Required: {source.strip()}")
                continue
            
            # Remove IPython magic commands
            source = re.sub(r'^%.*$', '', source, flags=re.MULTILINE)
            source = re.sub(r'^!.*$', '', source, flags=re.MULTILINE)
            
            # Handle display/print statements that might be Jupyter-specific
            source = source.replace('display(', 'print(')
            
            # Add the code
            code_cells.append(source)
            
            # Add spacing between cells
            if not source.endswith('\n'):
                code_cells.append('')
    
    # Join all code cells
    module_code = '\n'.join(code_cells)
    
    # Clean up multiple blank lines
    module_code = re.sub(r'\n{3,}', '\n\n', module_code)
    
    # Write the module
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(module_code)
    
    print(f"✅ Successfully converted to {output_path}")
    return output_path

def main():
    """Main entry point for command-line usage."""
    
    # Default conversion: TreasurHunter.ipynb to treasure_hunter_module.py
    primary_conversions = [
        ('TreasurHunter.ipynb', 'treasure_hunter_module.py'),
        ('satellite.ipynb', 'satellite_module.py'),
        ('satellite_300mile.ipynb', 'satellite_300mile_module.py'),
        ('satellite_production_modular_unified.ipynb', 'satellite_production_module.py'),
    ]
    
    if len(sys.argv) > 1:
        # Convert specific notebook provided as argument
        notebook_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        try:
            convert_notebook_to_module(notebook_path, output_path)
        except Exception as e:
            print(f"❌ Error converting {notebook_path}: {e}")
            sys.exit(1)
    else:
        # Convert all primary notebooks
        print("Converting primary notebooks to Python modules...")
        print("=" * 60)
        
        converted = []
        failed = []
        
        for notebook, module in primary_conversions:
            if Path(notebook).exists():
                try:
                    convert_notebook_to_module(notebook, module)
                    converted.append(module)
                except Exception as e:
                    print(f"❌ Error converting {notebook}: {e}")
                    failed.append(notebook)
            else:
                print(f"⚠️  Skipping {notebook} (not found)")
        
        print("=" * 60)
        if converted:
            print(f"✅ Successfully converted {len(converted)} notebooks:")
            for module in converted:
                print(f"   - {module}")
        
        if failed:
            print(f"❌ Failed to convert {len(failed)} notebooks:")
            for notebook in failed:
                print(f"   - {notebook}")
            sys.exit(1)
        
        if not converted and not failed:
            print("⚠️  No notebooks found to convert!")
            print("\nUsage:")
            print("  python convert_notebook.py                    # Convert all primary notebooks")
            print("  python convert_notebook.py notebook.ipynb     # Convert specific notebook")
            print("  python convert_notebook.py in.ipynb out.py    # Convert with custom output name")

if __name__ == "__main__":
    main()