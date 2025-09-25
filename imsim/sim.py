"""
Simulation module for generating IMS spectra and trajectories
"""

import numpy as np
from typing import List, Tuple
from .schemas import Ion, Tube, Spectrum, Peak
from .physics import drift_time_from_mobility, diffusion_broadening, E_over_N

def simulate_multi_ion(ions: List[Ion], tube: Tube, 
                      time_window_ms: float = 80.0,
                      n_points: int = 3000,
                      noise_level: float = 0.02) -> Tuple[List[Peak], Spectrum]:
    """
    Simulate IMS spectrum for multiple ions.
    
    Args:
        ions: List of ions to simulate
        tube: Drift tube configuration
        time_window_ms: Time window for simulation in ms
        n_points: Number of points in spectrum
        noise_level: Gaussian noise level (fraction of max intensity)
    
    Returns:
        Tuple of (peaks, spectrum)
    """
    # Time axis
    t = np.linspace(0, time_window_ms, n_points)
    
    # Initialize spectrum
    spectrum_intensity = np.zeros_like(t)
    peaks = []
    
    # Gas properties
    gas_mass = tube.gas.mass_amu
    
    for ion in ions:
        # Calculate drift time
        K = tube.gas.mass_amu / (ion.mass_amu + tube.gas.mass_amu) * \
            np.sqrt(ion.mass_amu / tube.gas.mass_amu) * \
            2.0 / (ion.ccs_A2 ** 0.5) * \
            (tube.temperature_K / 300.0) ** 0.5 * \
            (101.325 / tube.pressure_kPa)
        
        K *= 1e-4  # Convert to cm²/V·s
        
        td = drift_time_from_mobility(K, tube.length_m, tube.voltage_V)
        
        # Calculate FWHM from diffusion
        fwhm = diffusion_broadening(
            ion.ccs_A2, ion.mass_amu, ion.charge,
            gas_mass, tube.temperature_K, tube.pressure_kPa * 1000,
            tube.length_m, tube.voltage_V
        )
        
        # Create Gaussian peak
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        peak_intensity = ion.intensity * np.exp(-0.5 * ((t - td) / sigma) ** 2)
        
        # Add to spectrum
        spectrum_intensity += peak_intensity
        
        # Store peak info
        peak = Peak(
            time_ms=td,
            intensity=ion.intensity,
            K0_cm2_Vs=K,
            fwhm_ms=fwhm,
            snr=ion.intensity  # Use snr instead of prominence
        )
        peaks.append(peak)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * np.max(spectrum_intensity), 
                               len(spectrum_intensity))
        spectrum_intensity += noise
    
    # Create spectrum object
    spectrum = Spectrum(
        time_ms=t,
        intensity=spectrum_intensity,
        metadata={
            "tube": tube,
            "ions": ions,
            "noise_level": noise_level
        }
    )
    
    return peaks, spectrum

def simulate_trajectories(ions: List[Ion], tube: Tube,
                         n_segments: int = 3,
                         n_frames: int = 100) -> List[np.ndarray]:
    """
    Simulate ion trajectories through segmented tube.
    
    Args:
        ions: List of ions to simulate
        tube: Drift tube configuration
        n_segments: Number of tube segments
        n_frames: Number of animation frames
    
    Returns:
        List of trajectory arrays (one per ion)
    """
    trajectories = []
    
    # Gas properties
    gas_mass = tube.gas.mass_amu
    
    for ion in ions:
        # Calculate mobility (simplified)
        K = tube.gas.mass_amu / (ion.mass_amu + tube.gas.mass_amu) * \
            np.sqrt(ion.mass_amu / tube.gas.mass_amu) * \
            2.0 / (ion.ccs_A2 ** 0.5) * \
            (tube.temperature_K / 300.0) ** 0.5 * \
            (101.325 / tube.pressure_kPa)
        
        K *= 1e-4  # Convert to cm²/V·s
        
        # Electric field
        E = tube.voltage_V / tube.length_m
        
        # Drift velocity
        v_d = K * E * 1e-4  # m/s
        
        # Time to traverse tube
        t_total = tube.length_m / v_d
        
        # Time points
        t = np.linspace(0, t_total, n_frames)
        
        # Position along tube
        z = v_d * t
        
        # Add some diffusion spread (simplified)
        sigma = 0.001 * np.sqrt(t)  # 1mm spread at 1s
        z_spread = np.random.normal(0, sigma, len(t))
        
        # Ensure z + z_spread doesn't exceed tube length
        z_final = np.clip(z + z_spread, 0, tube.length_m)
        
        trajectory = np.column_stack([t, z_final])
        trajectories.append(trajectory)
    
    return trajectories

def generate_peak_table(peaks: List[Peak], tube: Tube) -> List[dict]:
    """
    Generate peak table with calculated properties.
    
    Args:
        peaks: List of peaks
        tube: Tube configuration
    
    Returns:
        List of peak dictionaries
    """
    table = []
    
    for i, peak in enumerate(peaks):
        row = {
            "Peak": i + 1,
            "Drift Time (ms)": f"{peak.time_ms:.2f}",
            "K0 (cm²/V·s)": f"{peak.K0_cm2_Vs:.3f}",
            "FWHM (ms)": f"{peak.fwhm_ms:.2f}" if peak.fwhm_ms else "N/A",
            "Intensity": f"{peak.intensity:.2f}",
            "SNR": f"{peak.snr:.2f}" if peak.snr else "N/A"
        }
        table.append(row)
    
    return table
