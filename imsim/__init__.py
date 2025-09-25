"""
IMS Physics Pro - Core Package
Free and Pro tiers for ion mobility spectrometry simulation and analysis
"""

__version__ = "0.1.0"

# Core modules
from . import physics
from . import sim
from . import calibrate
from . import library
from . import ml
from . import viz
from . import licensing
from . import utils
from . import schemas

__all__ = [
    "physics", "sim", "calibrate", 
    "library", "ml", "viz", "licensing", "utils", "schemas"
]