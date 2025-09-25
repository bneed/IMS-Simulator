"""
IMS Physics Pro - Streamlit Apps
"""

import sys
from pathlib import Path

# Add the parent directory to Python path to find imsim module
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
