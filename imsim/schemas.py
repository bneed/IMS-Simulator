"""Data schemas for IMS Physics Pro."""

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

@dataclass
class Gas:
    """Gas properties for IMS calculations."""
    name: str
    mass_amu: float
    density_kg_m3: float = 1.225  # Air density at STP
    viscosity_Pa_s: float = 1.81e-5  # Air viscosity at STP
    
    @classmethod
    def from_name(cls, gas_name: str):
        """Create Gas object from gas name."""
        gas_properties = {
            "air": {"mass_amu": 28.97, "density_kg_m3": 1.225, "viscosity_Pa_s": 1.81e-5},
            "nitrogen": {"mass_amu": 28.01, "density_kg_m3": 1.250, "viscosity_Pa_s": 1.78e-5},
            "helium": {"mass_amu": 4.00, "density_kg_m3": 0.178, "viscosity_Pa_s": 1.97e-5},
            "argon": {"mass_amu": 39.95, "density_kg_m3": 1.784, "viscosity_Pa_s": 2.24e-5},
            "carbon_dioxide": {"mass_amu": 44.01, "density_kg_m3": 1.977, "viscosity_Pa_s": 1.49e-5},
        }
        
        if gas_name.lower() in gas_properties:
            props = gas_properties[gas_name.lower()]
            return cls(
                name=gas_name,
                mass_amu=props["mass_amu"],
                density_kg_m3=props["density_kg_m3"],
                viscosity_Pa_s=props["viscosity_Pa_s"]
            )
        else:
            # Default to air properties
            return cls(
                name=gas_name,
                mass_amu=28.97,
                density_kg_m3=1.225,
                viscosity_Pa_s=1.81e-5
            )

@dataclass
class Ion:
    """Ion properties for IMS calculations."""
    name: str
    mass_amu: float
    charge: int
    ccs_A2: float
    intensity: float = 1.0

@dataclass
class Tube:
    """Drift tube properties."""
    length_m: float
    voltage_V: float
    gas: Gas
    temperature_K: float = 298.15
    pressure_kPa: float = 101.325

@dataclass
class Segment:
    """Tube segment for multi-segment simulations."""
    length_m: float
    voltage_V: float
    gas: Gas
    temperature_K: float = 298.15
    pressure_kPa: float = 101.325

@dataclass
class Spectrum:
    """IMS spectrum data."""
    time_ms: np.ndarray
    intensity: np.ndarray
    metadata: Optional[dict] = None

@dataclass
class Peak:
    """Peak in IMS spectrum."""
    time_ms: float
    intensity: float
    K0_cm2_Vs: float
    fwhm_ms: float = 0.0
    snr: float = 0.0

@dataclass
class LibraryCompound:
    """Library compound for database storage."""
    name: str
    family: str
    func_group: str
    gas: str
    adduct: str
    z: int
    mz: float
    mono_mass: float
    K0: float
    K0_sd: float = 0.0
    CCS: float = 0.0
    CCS_sd: float = 0.0
    notes: str = ""
    T_K: float = 298.15
    P_kPa: float = 101.325
    E_over_N_Td: float = 0.0
    L_m: float = 0.0
    V_V: float = 0.0
    source: str = ""
    doi: str = ""
    created_utc: int = 0

@dataclass
class MLFeatures:
    """Features for ML models."""
    mass_amu: float
    z: int
    ccs_A2: float
    gas_mass_amu: float
    T_K: float
    P_Pa: float
    E_over_N_Td: float

@dataclass
class MLPrediction:
    """ML model prediction result."""
    K0_pred: float = 0.0
    family_pred: str = ""
    K0_uncertainty: float = 0.0
    family_confidence: float = 0.0
    K0_lower: float = 0.0
    K0_upper: float = 0.0
    top_predictions: List[Dict] = None
    model_name: str = ""
    features_used: List[str] = None
    
    def __post_init__(self):
        if self.features_used is None:
            self.features_used = []
        if self.top_predictions is None:
            self.top_predictions = []
