"""
Calibration module for t0, C fit, and K/K0 conversions
"""

import numpy as np
from typing import List, Tuple, Dict
from .schemas import Peak, Tube

def K_from_td(td_ms: float, tube: Tube) -> float:
    """
    Calculate mobility K from drift time.
    
    Args:
        td_ms: Drift time in ms
        tube: Tube configuration
    
    Returns:
        Mobility K in cm²/V·s
    """
    # Electric field
    E_V_m = tube.voltage_V / tube.length_m
    
    # Drift velocity
    v_d = tube.length_m / (td_ms / 1000)  # m/s
    
    # Mobility
    K = v_d / E_V_m  # m²/V·s
    
    return K * 10000  # Convert to cm²/V·s

def K0_from_K(K_cm2_Vs: float, tube: Tube) -> float:
    """
    Calculate reduced mobility K0 from mobility K.
    
    Args:
        K_cm2_Vs: Mobility in cm²/V·s
        tube: Tube configuration
    
    Returns:
        Reduced mobility K0 in cm²/V·s
    """
    # Convert to SI
    K_m2_Vs = K_cm2_Vs / 10000
    
    # Standard conditions
    T_std = 273.15  # K
    P_std = 101325  # Pa
    
    # Reduced mobility
    K0 = K_m2_Vs * (tube.pressure_kPa * 1000 / P_std) * (T_std / tube.temperature_K)
    
    return K0 * 10000  # Convert back to cm²/V·s

def fit_t0_C_from_refs(reference_peaks: List[Dict], tube: Tube) -> Tuple[float, float]:
    """
    Fit t0 and C parameters from reference compounds.
    
    Args:
        reference_peaks: List of dicts with 'td_ms' and 'K0_known' keys
        tube: Tube configuration
    
    Returns:
        Tuple of (t0_ms, C_factor)
    """
    if len(reference_peaks) < 2:
        raise ValueError("Need at least 2 reference peaks for calibration")
    
    # Extract data
    td_measured = np.array([p['td_ms'] for p in reference_peaks])
    K0_known = np.array([p['K0_known'] for p in reference_peaks])
    
    # Linear fit: 1/K0 = (t_d - t0) / C
    # Rearranged: t_d = t0 + C/K0
    A = np.column_stack([np.ones(len(K0_known)), 1/K0_known])
    params, residuals, rank, s = np.linalg.lstsq(A, td_measured, rcond=None)
    
    t0_ms = params[0]
    C_factor = params[1]
    
    return t0_ms, C_factor

def K0_from_td_with_fit(td_ms: float, t0_ms: float, C_factor: float) -> float:
    """
    Calculate K0 from drift time using fitted parameters.
    
    Args:
        td_ms: Drift time in ms
        t0_ms: Fitted t0 in ms
        C_factor: Fitted C factor
    
    Returns:
        K0 in cm²/V·s
    """
    if td_ms <= t0_ms:
        return 0.0
    
    return C_factor / (td_ms - t0_ms)

def expected_C_physical(tube: Tube) -> float:
    """
    Calculate expected C factor from physical parameters.
    
    Args:
        tube: Tube configuration
    
    Returns:
        Expected C factor
    """
    # C = L² / V * (T/T_std) * (P_std/P)
    T_std = 273.15  # K
    P_std = 101325  # Pa
    
    C = (tube.length_m ** 2) / tube.voltage_V * \
        (tube.temperature_K / T_std) * \
        (P_std / (tube.pressure_kPa * 1000))
    
    return C

def calibrate_spectrum(peaks: List[Peak], reference_peaks: List[Dict], 
                      tube: Tube) -> List[Peak]:
    """
    Calibrate spectrum peaks using reference compounds.
    
    Args:
        peaks: List of detected peaks
        reference_peaks: List of reference peaks with known K0
        tube: Tube configuration
    
    Returns:
        List of calibrated peaks
    """
    if not reference_peaks:
        # No calibration - use basic conversion
        calibrated = []
        for peak in peaks:
            K = K_from_td(peak.time_ms, tube)
            K0 = K0_from_K(K, tube)
            
            calibrated_peak = Peak(
                time_ms=peak.time_ms,
                intensity=peak.intensity,
                K0_cm2_Vs=K0,
                fwhm_ms=peak.fwhm_ms,
                prominence=peak.prominence
            )
            calibrated.append(calibrated_peak)
        
        return calibrated
    
    # Fit calibration parameters
    t0_ms, C_factor = fit_t0_C_from_refs(reference_peaks, tube)
    
    # Apply calibration
    calibrated = []
    for peak in peaks:
        K0 = K0_from_td_with_fit(peak.time_ms, t0_ms, C_factor)
        
        calibrated_peak = Peak(
            time_ms=peak.time_ms,
            intensity=peak.intensity,
            K0_cm2_Vs=K0,
            fwhm_ms=peak.fwhm_ms,
            prominence=peak.prominence
        )
        calibrated.append(calibrated_peak)
    
    return calibrated