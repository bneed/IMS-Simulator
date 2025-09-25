#!/usr/bin/env python3
"""
Installation script for IMS Physics Pro
This script installs the package in development mode for deployment
"""

import sys
import subprocess
import os
from pathlib import Path

def install_package():
    """Install the package in development mode"""
    try:
        # Get the current directory
        current_dir = Path(__file__).parent
        
        # Install the package in development mode
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], cwd=current_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Package installed successfully!")
            print("Output:", result.stdout)
            return True
        else:
            print("❌ Package installation failed!")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

if __name__ == "__main__":
    print("Installing IMS Physics Pro package...")
    success = install_package()
    if success:
        print("Installation completed successfully!")
    else:
        print("Installation failed!")
        sys.exit(1)
