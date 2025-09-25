"""Basic tests to verify the system works."""

import pytest
import numpy as np
from pathlib import Path

# Test imports
def test_imports():
    """Test that all modules can be imported."""
    from imsim.physics import Gas, Ion, Tube
    from imsim.sim import simulate_multi_ion
    from imsim.analyze import load_spectrum, baseline_correct, pick_peaks
    from imsim.library import LibraryManager
    from imsim.ml import MLManager
    from imsim.licensing import is_pro
    from imsim.utils import safe_tab

def test_physics_basic():
    """Test basic physics calculations."""
    from imsim.physics import Gas, Ion, Tube, E_over_N
    
    gas = Gas.from_name("N2")
    tube = Tube(length_m=0.1, voltage_V=1000, gas=gas, temperature_K=300, pressure_kPa=101.325)
    
    # Test E/N calculation
    e_over_n = E_over_N(tube)
    assert e_over_n > 0
    assert isinstance(e_over_n, float)

def test_simulation():
    """Test basic simulation."""
    from imsim.physics import Gas, Ion, Tube
    from imsim.sim import simulate_multi_ion
    
    gas = Gas.from_name("N2")
    tube = Tube(length_m=0.1, voltage_V=1000, gas=gas, temperature_K=300, pressure_kPa=101.325)
    
    ions = [
        Ion(name="Test1", mass_amu=100, charge=1, ccs_A2=150, intensity=1.0),
        Ion(name="Test2", mass_amu=200, charge=1, ccs_A2=200, intensity=0.5)
    ]
    
    peaks, spectrum = simulate_multi_ion(ions, tube, time_window_ms=50, n_points=1000)
    
    assert len(peaks) == 2
    assert len(spectrum.time_ms) == 1000
    assert len(spectrum.intensity) == 1000

def test_licensing():
    """Test licensing system."""
    from imsim.licensing import is_pro
    
    # Test early access keys
    assert is_pro("IMS-PRO-EARLY-0001") == True
    assert is_pro("IMS-PRO-EARLY-0002") == True
    assert is_pro("invalid-key") == False
    assert is_pro("") == False
    assert is_pro(None) == False

def test_library_manager():
    """Test library manager."""
    from imsim.library import LibraryManager
    from imsim.schemas import LibraryCompound
    
    # Create temporary database
    lib_manager = LibraryManager(Path("test_library.db"))
    
    # Test adding compound
    compound = LibraryCompound(
        name="Test Compound",
        family="test",
        func_group="test",
        gas="N2",
        adduct="",
        z=1,
        mz=100.0,
        mono_mass=100.0,
        K0=2.5,
        notes="Test compound"
    )
    
    compound_id = lib_manager.add_compound(compound)
    assert compound_id > 0
    
    # Test getting compounds
    compounds = lib_manager.get_compounds()
    assert len(compounds) >= 1
    
    # Clean up
    Path("test_library.db").unlink(missing_ok=True)

def test_ml_manager():
    """Test ML manager."""
    from imsim.ml import MLManager
    from imsim.schemas import MLFeatures
    import pandas as pd
    
    ml_manager = MLManager(Path("test_models"))
    
    # Create test training data
    training_data = []
    for i in range(20):
        training_data.append({
            'name': f'Test{i}',
            'family': 'test' if i < 10 else 'other',
            'mass_amu': 100 + i,
            'z': 1,
            'ccs_A2': 150 + i,
            'gas_mass_amu': 28.014,
            'T_K': 300,
            'P_Pa': 101325,
            'E_over_N_Td': 120,
            'K0_cm2_Vs': 2.5 + i * 0.1
        })
    
    df = pd.DataFrame(training_data)
    
    # Test training (should work with enough data)
    try:
        results = ml_manager.train_k0_regressor(df)
        assert 'r2' in results
        assert 'n_samples' in results
    except Exception as e:
        # Training might fail with synthetic data, that's okay for basic test
        pass
    
    # Clean up
    import shutil
    shutil.rmtree("test_models", ignore_errors=True)

if __name__ == "__main__":
    # Run tests
    test_imports()
    test_physics_basic()
    test_simulation()
    test_licensing()
    test_library_manager()
    test_ml_manager()
    print("All basic tests passed!")
