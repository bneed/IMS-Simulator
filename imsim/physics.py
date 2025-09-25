"""
Core physics calculations for IMS
Mason-Schamp equation, mobility, E/N calculations
"""

import numpy as np
from .schemas import Gas, Ion, Tube
from .utils import BOLTZMANN_J_K, ELEMENTARY_CHARGE_C

def mobility_from_ccs(ccs_A2: float, mass_amu: float, charge: int, 
                     gas_mass_amu: float, temperature_K: float, 
                     pressure_Pa: float) -> float:
    """
    Calculate mobility from CCS using Mason-Schamp equation.
    
    Args:
        ccs_A2: Collision cross section in Å²
        mass_amu: Ion mass in amu
        charge: Ion charge
        gas_mass_amu: Gas mass in amu
        temperature_K: Temperature in K
        pressure_Pa: Pressure in Pa
    
    Returns:
        Mobility in cm²/V·s
    """
    # Convert to SI units
    ccs_m2 = ccs_A2 * 1e-20  # Å² to m²
    mass_kg = mass_amu * 1.66053906660e-27  # amu to kg
    gas_mass_kg = gas_mass_amu * 1.66053906660e-27  # amu to kg
    
    # Reduced mass
    mu = (mass_kg * gas_mass_kg) / (mass_kg + gas_mass_kg)
    
    # Number density
    n = pressure_Pa / (BOLTZMANN_J_K * temperature_K)
    
    # Average thermal velocity
    v_avg = np.sqrt(8 * BOLTZMANN_J_K * temperature_K / (np.pi * mu))
    
    # Mobility (cm²/V·s)
    K = (3 * ELEMENTARY_CHARGE_C * charge) / (16 * n * ccs_m2 * np.sqrt(2 * mu / (np.pi * BOLTZMANN_J_K * temperature_K)))
    
    return K * 10000  # m²/V·s to cm²/V·s

def drift_time_from_mobility(K_cm2_Vs: float, length_m: float, 
                           voltage_V: float) -> float:
    """
    Calculate drift time from mobility.
    
    Args:
        K_cm2_Vs: Mobility in cm²/V·s
        length_m: Drift tube length in m
        voltage_V: Drift voltage in V
    
    Returns:
        Drift time in ms
    """
    # Electric field
    E_V_m = voltage_V / length_m
    
    # Mobility in m²/V·s
    K_m2_Vs = K_cm2_Vs / 10000
    
    # Drift velocity
    v_d = K_m2_Vs * E_V_m
    
    # Drift time
    t_d_s = length_m / v_d
    
    return t_d_s * 1000  # s to ms

def K0_from_td(td_ms: float, length_m: float, voltage_V: float,
               temperature_K: float, pressure_Pa: float) -> float:
    """
    Calculate reduced mobility K0 from drift time.
    
    Args:
        td_ms: Drift time in ms
        length_m: Drift tube length in m
        voltage_V: Drift voltage in V
        temperature_K: Temperature in K
        pressure_Pa: Pressure in Pa
    
    Returns:
        Reduced mobility K0 in cm²/V·s
    """
    # Electric field
    E_V_m = voltage_V / length_m
    
    # Drift velocity
    v_d = length_m / (td_ms / 1000)  # m/s
    
    # Mobility
    K = v_d / E_V_m  # m²/V·s
    
    # Reduced mobility (correct to STP)
    K0 = K * (pressure_Pa / STD_PRESSURE_PA) * (STD_TEMPERATURE_K / temperature_K)
    
    return K0 * 10000  # m²/V·s to cm²/V·s

def E_over_N(tube: Tube) -> float:
    """
    Calculate E/N (reduced electric field) in Townsend.
    
    Args:
        tube: Tube configuration
    
    Returns:
        E/N in Townsend (Td)
    """
    # Electric field
    E_V_m = tube.voltage_V / tube.length_m
    
    # Number density
    n_m3 = (tube.pressure_kPa * 1000) / (BOLTZMANN_J_K * tube.temperature_K)
    
    # E/N in Townsend (1 Td = 10^-21 V·m²)
    E_over_N_Td = E_V_m / (n_m3 * 1e-21)
    
    return E_over_N_Td

def diffusion_broadening(ccs_A2: float, mass_amu: float, charge: int,
                        gas_mass_amu: float, temperature_K: float,
                        pressure_Pa: float, length_m: float,
                        voltage_V: float) -> float:
    """
    Calculate diffusion broadening (FWHM) in ms.
    
    Args:
        ccs_A2: Collision cross section in Å²
        mass_amu: Ion mass in amu
        charge: Ion charge
        gas_mass_amu: Gas mass in amu
        temperature_K: Temperature in K
        pressure_Pa: Pressure in Pa
        length_m: Drift tube length in m
        voltage_V: Drift voltage in V
    
    Returns:
        FWHM in ms
    """
    # Get mobility and drift time
    K = mobility_from_ccs(ccs_A2, mass_amu, charge, gas_mass_amu, 
                         temperature_K, pressure_Pa)
    td = drift_time_from_mobility(K, length_m, voltage_V)
    
    # Diffusion coefficient
    D = (BOLTZMANN_J_K * temperature_K) / (ELEMENTARY_CHARGE_C * charge) * K / 10000
    
    # FWHM from diffusion
    fwhm_s = 2.355 * np.sqrt(2 * D * (td / 1000)) / length_m * (td / 1000)
    
    return fwhm_s * 1000  # s to ms

# Constants
STD_PRESSURE_PA = 101325  # Pa
STD_TEMPERATURE_K = 273.15  # K





