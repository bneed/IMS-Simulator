"""
Import helper for IMS Physics Pro apps
Handles module path setup and imports
"""

import sys
import os
from pathlib import Path

def setup_imports():
    """Setup the Python path to find the imsim module"""
    # Get the current directory (apps/)
    current_dir = Path(__file__).parent
    
    # Get the parent directory (project root)
    parent_dir = current_dir.parent
    
    # Add parent directory to Python path if not already there
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # Also add the current directory to path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

def safe_import_imsim():
    """Safely import imsim modules with fallback"""
    try:
        # Try to import the main modules
        from imsim.physics import Gas, Ion, Tube, E_over_N
        from imsim.sim import simulate_multi_ion, simulate_trajectories, generate_peak_table
        from imsim.library import LibraryManager
        from imsim.ml import MLManager
        from imsim.viz import create_2d_schematic, create_3d_tube, plot_spectrum, plot_trajectories, create_ml_performance_plot
        from imsim.licensing import is_pro, get_cached_info
        from imsim.utils import safe_tab, format_time_ms, format_mobility, format_ccs
        from imsim.schemas import LibraryCompound, MLFeatures, MLPrediction, Peak
        
        return {
            'physics': {'Gas': Gas, 'Ion': Ion, 'Tube': Tube, 'E_over_N': E_over_N},
            'sim': {'simulate_multi_ion': simulate_multi_ion, 'simulate_trajectories': simulate_trajectories, 'generate_peak_table': generate_peak_table},
            'library': {'LibraryManager': LibraryManager},
            'ml': {'MLManager': MLManager},
            'viz': {'create_2d_schematic': create_2d_schematic, 'create_3d_tube': create_3d_tube, 'plot_spectrum': plot_spectrum, 'plot_trajectories': plot_trajectories, 'create_ml_performance_plot': create_ml_performance_plot},
            'licensing': {'is_pro': is_pro, 'get_cached_info': get_cached_info},
            'utils': {'safe_tab': safe_tab, 'format_time_ms': format_time_ms, 'format_mobility': format_mobility, 'format_ccs': format_ccs},
            'schemas': {'LibraryCompound': LibraryCompound, 'MLFeatures': MLFeatures, 'MLPrediction': MLPrediction, 'Peak': Peak}
        }
    except ImportError as e:
        print(f"Warning: Could not import imsim modules: {e}")
        return None

# Setup imports when this module is imported
setup_imports()
