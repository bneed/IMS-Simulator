"""
Utility functions for IMS Physics Pro
"""

import traceback
import streamlit as st
from typing import Callable, Any, Dict
from pathlib import Path
import numpy as np

def safe_tab(title: str, fn: Callable, **kwargs) -> Any:
    """
    Safe tab wrapper that catches exceptions and shows them in UI.
    
    Args:
        title: Tab title for error display
        fn: Function to execute
        **kwargs: Arguments to pass to function
    
    Returns:
        Function result or None if error
    """
    try:
        return fn(**kwargs)
    except Exception as e:
        st.error(f"Error in {title}")
        with st.expander("Error details", expanded=False):
            st.code(traceback.format_exc())
        return None

def get_data_dir() -> Path:
    """Get data directory path."""
    return Path("data")

def get_models_dir() -> Path:
    """Get models directory path."""
    return Path("models")

def get_library_path() -> Path:
    """Get library database path."""
    # Always resolve relative to the project root, not the current working directory
    # This ensures the library works regardless of where the app is launched from
    current_dir = Path(__file__).parent.parent  # Go up from imsim/ to project root
    return current_dir / "db" / "imspro.sqlite"

def normalize_gas_symbol(symbol: str) -> str:
    """Normalize gas symbol to standard form."""
    return symbol.strip().upper()

def safe_filename(name: str) -> str:
    """Convert string to safe filename."""
    import re
    # Remove/replace unsafe characters
    safe = re.sub(r'[^\w\-_\.]', '_', name)
    return safe[:100]  # Limit length

def format_time_ms(time_ms: float) -> str:
    """Format time in milliseconds."""
    if time_ms < 1:
        return f"{time_ms*1000:.1f} μs"
    elif time_ms < 1000:
        return f"{time_ms:.2f} ms"
    else:
        return f"{time_ms/1000:.2f} s"

def format_mobility(K0_cm2_Vs: float) -> str:
    """Format mobility value."""
    if K0_cm2_Vs < 1:
        return f"{K0_cm2_Vs:.3f} cm²/V·s"
    else:
        return f"{K0_cm2_Vs:.2f} cm²/V·s"

def format_ccs(ccs_A2: float) -> str:
    """Format CCS value."""
    if ccs_A2 < 100:
        return f"{ccs_A2:.1f} Å²"
    else:
        return f"{ccs_A2:.0f} Å²"

# Constants
BOLTZMANN_J_K = 1.380649e-23  # J/K
ELEMENTARY_CHARGE_C = 1.602176634e-19  # C
AVOGADRO = 6.02214076e23  # mol⁻¹
STD_PRESSURE_PA = 101325  # Pa
STD_TEMPERATURE_K = 273.15  # K