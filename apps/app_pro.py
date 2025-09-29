"""
IMS Physics Pro - Pro Version
All features including Library, ML, Sim Lab, Visualization, Trajectories
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
from pathlib import Path

# Import helper for robust module loading
from import_helper import safe_import_imsim

# Get imsim modules
imsim_modules = safe_import_imsim()

if imsim_modules is None:
    st.error("❌ Failed to load IMS Physics modules. Please check the installation.")
    st.stop()

# Extract modules
Gas = imsim_modules['physics']['Gas']
Ion = imsim_modules['physics']['Ion']
Tube = imsim_modules['physics']['Tube']
E_over_N = imsim_modules['physics']['E_over_N']
simulate_multi_ion = imsim_modules['sim']['simulate_multi_ion']
simulate_trajectories = imsim_modules['sim']['simulate_trajectories']
generate_peak_table = imsim_modules['sim']['generate_peak_table']
# Analysis functions - temporarily disabled
def load_spectrum(file_path):
    """Temporary placeholder - load spectrum from file."""
    import pandas as pd
    import numpy as np
    
    # Try to read the file directly and split by tabs
    try:
        if hasattr(file_path, 'read'):
            file_path.seek(0)
            content = file_path.read().decode('utf-8')
        else:
            with open(file_path, 'r') as f:
                content = f.read()
        
        # Split into lines and then split each line by tabs
        lines = content.strip().split('\n')
        
        # Parse each line as tab-separated values
        data = []
        for line in lines:
            if line.strip():  # Skip empty lines
                values = line.strip().split('\t')
                # Convert to float, taking first two values
                if len(values) >= 2:
                    data.append([float(values[0]), float(values[1])])
        
        if not data:
            raise ValueError("No valid data found in file")
        
        # Convert to numpy arrays
        data_array = np.array(data)
        time_ms = data_array[:, 0]
        intensity = data_array[:, 1]
        
        return time_ms, intensity
        
    except Exception:
        # Fall back to pandas CSV reading
        try:
            if hasattr(file_path, 'read'):
                file_path.seek(0)
                df = pd.read_csv(file_path)
            else:
                df = pd.read_csv(file_path)
            
            time_col = df.columns[0]
            intensity_col = df.columns[1]
            return df[time_col].values, df[intensity_col].values
            
        except Exception as e:
            raise ValueError(f"Could not parse file. Error: {e}")

def load_ims_stream(file_path, aggregate="mean"):
    """Load IMS stream with multiple spectra."""
    import pandas as pd
    import numpy as np
    
    # Read the file directly to parse multiple spectra
    try:
        if hasattr(file_path, 'read'):
            file_path.seek(0)
            content = file_path.read().decode('utf-8')
        else:
            with open(file_path, 'r') as f:
                content = f.read()
        
        # Split into lines
        lines = content.strip().split('\n')
        
        if not lines:
            raise ValueError("No data found in file")
        
        # Skip first line (description), second line contains drift times (time axis)
        time_line = lines[1].strip().split('\t')
        time_ms = np.array([float(x) for x in time_line])  # Already in milliseconds
        
        # Remaining lines contain intensity data for each spectrum
        raw_matrix = []
        for line in lines[2:]:  # Start from line 3 (index 2)
            if line.strip():  # Skip empty lines
                values = line.strip().split('\t')
                # First value is time (seconds), rest are intensities (volts)
                if len(values) >= 2:
                    intensities = [float(x) for x in values[1:]]  # Skip time column
                    raw_matrix.append(intensities)
        
        if not raw_matrix:
            raise ValueError("No intensity data found")
        
        # Convert to numpy array
        raw_matrix = np.array(raw_matrix)
        
        # Calculate aggregated intensity based on requested method
        if aggregate == "mean":
            intensity = np.mean(raw_matrix, axis=0)
        elif aggregate == "max":
            intensity = np.max(raw_matrix, axis=0)
        elif aggregate == "min":
            intensity = np.min(raw_matrix, axis=0)
        elif aggregate == "sum":
            intensity = np.sum(raw_matrix, axis=0)
        else:
            intensity = np.mean(raw_matrix, axis=0)
        
        return time_ms, intensity, raw_matrix
        
    except Exception as e:
        # Fall back to simple spectrum loading
        time_ms, intensity = load_spectrum(file_path)
        return time_ms, intensity, np.array([intensity])

def baseline_correct(intensity, smooth_frac=0.05, base_frac=0.2):
    """Temporary placeholder - baseline correction."""
    return intensity

def pick_peaks(time_ms, intensity, prom_frac=0.02, min_height=None):
    """Temporary placeholder - peak detection."""
    import numpy as np
    from scipy import signal
    peak_indices, properties = signal.find_peaks(intensity, prominence=np.max(intensity) * prom_frac)
    return peak_indices, properties

def filter_by_drift_time(time_ms, intensity, min_time, max_time):
    """Filter by drift time window."""
    import numpy as np
    
    # Handle case where time and intensity arrays have different lengths
    # This happens when time has 6001 points but intensity has 6000 points
    min_len = min(len(time_ms), len(intensity))
    time_ms = time_ms[:min_len]
    intensity = intensity[:min_len]
    
    mask = (time_ms >= min_time) & (time_ms <= max_time)
    return time_ms[mask], intensity[mask]

def calculate_peak_stats(peaks):
    """Temporary placeholder - calculate peak statistics."""
    return {"count": len(peaks), "time_range_ms": 0, "avg_intensity": 0}

def export_peaks_csv(peaks, output_path):
    """Temporary placeholder - export peaks."""
    import pandas as pd
    df = pd.DataFrame([{"time": p.time_ms, "intensity": p.intensity} for p in peaks])
    df.to_csv(output_path, index=False)
    return len(peaks)
# Extract remaining modules
LibraryManager = imsim_modules['library']['LibraryManager']
MLManager = imsim_modules['ml']['MLManager']
create_2d_schematic = imsim_modules['viz']['create_2d_schematic']
create_3d_tube = imsim_modules['viz']['create_3d_tube']
plot_spectrum = imsim_modules['viz']['plot_spectrum']
plot_trajectories = imsim_modules['viz']['plot_trajectories']
create_ml_performance_plot = imsim_modules['viz']['create_ml_performance_plot']
is_pro = imsim_modules['licensing']['is_pro']
get_cached_info = imsim_modules['licensing']['get_cached_info']
safe_tab = imsim_modules['utils']['safe_tab']
format_time_ms = imsim_modules['utils']['format_time_ms']
format_mobility = imsim_modules['utils']['format_mobility']
format_ccs = imsim_modules['utils']['format_ccs']
LibraryCompound = imsim_modules['schemas']['LibraryCompound']
MLFeatures = imsim_modules['schemas']['MLFeatures']
MLPrediction = imsim_modules['schemas']['MLPrediction']

st.set_page_config(
    page_title="IMS Physics Pro - Pro Version",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ IMS Physics Pro - Pro Version")
st.caption("Complete IMS simulation, analysis, and machine learning platform")

# Sidebar - Pro Key and Instrument Controls
with st.sidebar:
    st.header("🔐 Pro License")
    
    # Pro key input
    pro_key = st.text_input(
        "Pro License Key", 
        type="password",
        value=os.getenv("IMS_PRO_KEY", ""),
        help="Enter your Pro license key to unlock all features"
    )
    
    PRO = is_pro(pro_key)
    
    if PRO:
        st.success("✅ Pro License Active")
        cached_info = get_cached_info()
        if cached_info.get("email"):
            st.caption(f"Licensed to: {cached_info['email']}")
    else:
        st.warning("🔒 Free Mode - Limited Features")
        st.caption("Enter a Pro key to unlock all features")
    
    st.markdown("---")
    st.header("🔧 Instrument Settings")
    
    # Tube parameters
    L_m = st.number_input("Tube Length (m)", 0.02, 2.0, 0.10, 0.001)
    V_V = st.number_input("Drift Voltage (V)", 10.0, 80000.0, 1000.0, 10.0)
    gas_name = st.selectbox("Drift Gas", ["N2", "He", "Air"], index=0)
    T_K = st.number_input("Temperature (K)", 150.0, 1000.0, 300.0, 1.0)
    P_kPa = st.number_input("Pressure (kPa)", 20.0, 300.0, 101.325, 0.1)
    
    # Create gas and tube objects
    gas = Gas.from_name(gas_name)
    tube = Tube(length_m=L_m, voltage_V=V_V, gas=gas, temperature_K=T_K, pressure_kPa=P_kPa)
    
    # Display KPIs
    try:
        E = V_V / max(1e-9, L_m)
        N = (P_kPa * 1000.0) / (1.380649e-23 * T_K)
        E_over_N_Td = E / max(1e-9, N) / 1e-21
        
        st.markdown("### 📊 Key Parameters")
        cols = st.columns(2)
        cols[0].metric("E", f"{E:,.0f}", "V/m")
        cols[1].metric("E/N", f"{E_over_N_Td:,.1f}", "Td")
        cols[0].metric("Gas", gas_name)
        cols[1].metric("P", f"{P_kPa:.1f}", "kPa")
    except Exception as e:
        st.error(f"Error calculating parameters: {e}")

# Pro gating helper
def pro_only():
    """Gate Pro-only features."""
    if not PRO:
        st.warning("🔒 This is a Pro feature. Enter your Pro license key in the sidebar to unlock.")
        st.stop()

# Main tabs
tabs = st.tabs([
    "🎯 Simulate", 
    "📊 Analyze", 
    "📚 Library", 
    "🤖 ML Training", 
    "🔬 Sim Lab", 
    "🧬 Molecular Structure",
    "📐 Visualization", 
    "🚀 Trajectories"
])

def render_simulate():
    """Render simulation tab."""
    st.header("🎯 Ion Mobility Simulation")
    
    # Ion input section
    st.subheader("Ion Configuration")
    max_ions = 12 if PRO else 3
    n_ions = st.number_input("Number of Ions", 1, max_ions, 2, 1, 
                            help=f"Pro version supports up to {max_ions} ions" if PRO else "Free version limited to 3 ions")
    
    if not PRO and n_ions > 3:
        st.warning("Free version limited to 3 ions. Upgrade to Pro for more.")
        n_ions = 3
    
    ions = []
    cols = st.columns(5)
    cols[0].markdown("**Name**")
    cols[1].markdown("**Mass (amu)**")
    cols[2].markdown("**Charge**")
    cols[3].markdown("**CCS (Å²)**")
    cols[4].markdown("**Intensity**")
    
    for i in range(int(n_ions)):
        c = st.columns(5)
        name = c[0].text_input("", f"Ion{i+1}", key=f"ion_name_{i}")
        mass = c[1].number_input("", 10.0, 5000.0, 145.0, 1.0, key=f"ion_mass_{i}")
        charge = c[2].number_input("", 1, 5, 1, 1, key=f"ion_charge_{i}")
        ccs = c[3].number_input("", 40.0, 4000.0, 150.0, 1.0, key=f"ion_ccs_{i}")
        intensity = c[4].slider("", 0.0, 5.0, 1.0, 0.1, key=f"ion_intensity_{i}")
        
        ion = Ion(name=name, mass_amu=mass, charge=charge, ccs_A2=ccs, intensity=intensity)
        ions.append(ion)
    
    # Simulation parameters
    st.subheader("Simulation Parameters")
    col1, col2 = st.columns(2)
    time_window = col1.number_input("Time Window (ms)", 1.0, 500.0, 80.0, 1.0)
    n_points = col2.slider("Data Points", 500, 20000, 3000, 500)
    noise_level = st.slider("Noise Level", 0.0, 0.1, 0.02, 0.001)
    
    # Run simulation
    if st.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                peaks, spectrum = simulate_multi_ion(
                    ions=ions,
                    tube=tube,
                    time_window_ms=time_window,
                    n_points=int(n_points),
                    noise_level=noise_level
                )
                
                st.success("Simulation completed!")
                
                # Display results
                st.subheader("📈 Results")
                
                # Plot spectrum
                fig = plot_spectrum(spectrum, peaks, "Simulated IMS Spectrum")
                st.plotly_chart(fig, width='stretch')
                
                # Peak table
                st.subheader("📋 Peak Table")
                peak_data = generate_peak_table(peaks, tube)
                df_peaks = pd.DataFrame(peak_data)
                st.dataframe(df_peaks, width='stretch')
                
                # Export options
                st.subheader("💾 Export")
                csv_buffer = io.StringIO()
                df_peaks.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Peaks CSV",
                    data=csv_buffer.getvalue(),
                    file_name="simulated_peaks.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Simulation failed: {e}")

def render_analyze():
    """Render analysis tab."""
    st.header("📊 Spectrum Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload IMS Spectrum", 
        type=['csv', 'ims', 'txt'],
        help="Upload CSV (2 columns: time, intensity) or IMS format file"
    )
    
    if uploaded_file is not None:
        try:
            # Load spectrum - try to get raw matrix if IMS format
            try:
                # Try IMS format first to get raw matrix
                t_ms, intensity, raw_matrix = load_ims_stream(uploaded_file, aggregate="mean")
                has_multiple_scans = True
            except:
                # Fall back to simple CSV format
                t_ms, intensity = load_spectrum(uploaded_file)
                raw_matrix = None
                has_multiple_scans = False
            
            st.success(f"Loaded spectrum with {len(t_ms)} data points")
            
            # Analysis parameters
            st.subheader("🔧 Analysis Parameters")
            
            col1, col2, col3 = st.columns(3)
            smooth_pct = col1.slider("Smoothing (%)", 1, 20, 5, 1)
            base_pct = col2.slider("Baseline Window (%)", 5, 60, 20, 5)
            prom_frac = col3.slider("Peak Prominence", 0.001, 0.5, 0.02, 0.001)
            
            # Spectrum type selection
            if has_multiple_scans:
                st.subheader("📊 Spectrum Type")
                spectrum_type = st.selectbox(
                    "Select spectrum to display:",
                    ["Mean", "Max", "Min", "Sum", "Raw"],
                    index=0,
                    help="Choose how to aggregate multiple scans"
                )
                
                # Update intensity based on selection
                if spectrum_type == "Mean":
                    intensity = raw_matrix.mean(axis=0)
                elif spectrum_type == "Max":
                    intensity = raw_matrix.max(axis=0)
                elif spectrum_type == "Min":
                    intensity = raw_matrix.min(axis=0)
                elif spectrum_type == "Sum":
                    intensity = raw_matrix.sum(axis=0)
                else:  # Raw - show first scan
                    intensity = raw_matrix[0] if len(raw_matrix) > 0 else intensity
                
                # Ensure t_ms and intensity have matching lengths
                min_len = min(len(t_ms), len(intensity))
                t_ms = t_ms[:min_len]
                intensity = intensity[:min_len]
                
                st.info(f"Showing {spectrum_type.lower()} spectrum from {len(raw_matrix)} scans")
            
            # Drift time window
            st.subheader("⏱️ Drift Time Window")
            use_window = st.checkbox("Apply drift time window", value=False)
            
            if use_window:
                col1, col2 = st.columns(2)
                min_td = col1.number_input("Min Drift Time (ms)", 0.0, float(np.max(t_ms)), 0.0, 0.1)
                max_td = col2.number_input("Max Drift Time (ms)", 0.0, float(np.max(t_ms)), float(np.max(t_ms)), 0.1)
                
                if min_td >= max_td:
                    st.warning("Min drift time must be less than max drift time")
                else:
                    t_filtered, intensity_filtered = filter_by_drift_time(t_ms, intensity, min_td, max_td)
            else:
                t_filtered, intensity_filtered = t_ms, intensity
                min_td, max_td = 0, float(np.max(t_ms))
            
            # Baseline correction
            intensity_corrected = baseline_correct(
                intensity_filtered, 
                smooth_frac=smooth_pct/100, 
                base_frac=base_pct/100
            )
            
            # Peak detection
            peak_indices, peak_props = pick_peaks(
                t_filtered, 
                intensity_corrected, 
                prom_frac=prom_frac
            )
            
            # Convert to peaks
            peaks = []
            for idx in peak_indices:
                peak_time = t_filtered[idx]
                peak_intensity = intensity_corrected[idx]
                
                # Calculate K0 using proper physics
                from imsim.physics import K0_from_td
                K0 = K0_from_td(
                    td_ms=peak_time,
                    length_m=tube.length_m,
                    voltage_V=tube.voltage_V,
                    temperature_K=tube.temperature_K,
                    pressure_Pa=tube.pressure_kPa * 1000
                )
                
                from imsim.schemas import Peak
                peak = Peak(
                    time_ms=peak_time,
                    intensity=peak_intensity,
                    K0_cm2_Vs=K0
                )
                peaks.append(peak)
            
            # Visualization
            st.subheader("📈 Analysis Results")
            
            fig = go.Figure()
            
            # Add statistics overlays if multiple scans available
            if has_multiple_scans and raw_matrix is not None:
                # Filter raw matrix to drift time window
                if use_window:
                    mask = (t_ms >= min_td) & (t_ms <= max_td)
                    t_stats = t_ms[mask]
                    raw_matrix_filtered = raw_matrix[:, mask]
                else:
                    t_stats = t_ms
                    raw_matrix_filtered = raw_matrix
                
                # Calculate statistics
                mean_spectrum = raw_matrix_filtered.mean(axis=0)
                std_spectrum = raw_matrix_filtered.std(axis=0)
                min_spectrum = raw_matrix_filtered.min(axis=0)
                max_spectrum = raw_matrix_filtered.max(axis=0)
                
                # Add mean ± std fill
                fig.add_trace(go.Scatter(
                    x=np.concatenate([t_stats, t_stats[::-1]]),
                    y=np.concatenate([mean_spectrum + std_spectrum, (mean_spectrum - std_spectrum)[::-1]]),
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Mean ± Std',
                    hoverinfo="skip"
                ))
                
                # Add min/max lines
                fig.add_trace(go.Scatter(
                    x=t_stats,
                    y=min_spectrum,
                    mode='lines',
                    name='Min',
                    line=dict(color='lightblue', width=1, dash='dot'),
                    opacity=0.7
                ))
                
                fig.add_trace(go.Scatter(
                    x=t_stats,
                    y=max_spectrum,
                    mode='lines',
                    name='Max',
                    line=dict(color='lightcoral', width=1, dash='dot'),
                    opacity=0.7
                ))
                
                # Add mean line
                fig.add_trace(go.Scatter(
                    x=t_stats,
                    y=mean_spectrum,
                    mode='lines',
                    name='Mean',
                    line=dict(color='green', width=1),
                    opacity=0.8
                ))
            
            # Raw spectrum (current selection)
            fig.add_trace(go.Scatter(
                x=t_filtered,
                y=intensity_filtered,
                mode='lines',
                name='Selected Spectrum',
                line=dict(color='gray', width=2, dash='dot'),
                opacity=0.9
            ))
            
            # Corrected spectrum
            fig.add_trace(go.Scatter(
                x=t_filtered,
                y=intensity_corrected,
                mode='lines',
                name='Baseline Corrected',
                line=dict(color='blue', width=2)
            ))
            
            # Peaks
            if peaks:
                peak_times = [p.time_ms for p in peaks]
                peak_intensities = [p.intensity for p in peaks]
                
                fig.add_trace(go.Scatter(
                    x=peak_times,
                    y=peak_intensities,
                    mode='markers',
                    name='Detected Peaks',
                    marker=dict(color='red', size=10, symbol='diamond'),
                    hovertemplate="<b>Peak</b><br>" +
                                 "Time: %{x:.2f} ms<br>" +
                                 "Intensity: %{y:.2f}<br>" +
                                 "<extra></extra>"
                ))
            
            # Add drift time window shading
            if use_window:
                fig.add_vrect(
                    x0=min_td, x1=max_td,
                    fillcolor="yellow", opacity=0.2,
                    annotation_text="Analysis Window",
                    annotation_position="top"
                )
            
            fig.update_layout(
                title="IMS Spectrum Analysis",
                xaxis_title="Drift Time (ms)",
                yaxis_title="Intensity (V)",
                height=500,
                template="simple_white"
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Peak statistics
            if peaks:
                st.subheader("📊 Peak Statistics")
                stats = calculate_peak_stats(peaks)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Peaks Found", stats["count"])
                col2.metric("Time Range", f"{stats['time_range_ms']:.1f} ms")
                col3.metric("Avg Intensity", f"{stats['avg_intensity']:.2f}")
                
                # Peak table
                st.subheader("📋 Detected Peaks")
                peak_data = []
                for i, peak in enumerate(peaks):
                    peak_data.append({
                        "Peak": i + 1,
                        "Drift Time (ms)": f"{peak.time_ms:.2f}",
                        "Intensity": f"{peak.intensity:.2f}",
                        "K0 (cm²/V·s)": f"{peak.K0_cm2_Vs:.3f}"
                    })
                
                df_peaks = pd.DataFrame(peak_data)
                st.dataframe(df_peaks, width='stretch')
                
                # Peak Labeling for Training Data
                st.subheader("🏷️ Label Peaks for Training Data")
                st.caption("Label your known peaks to add them to the library for ML training")
                
                # Initialize library manager
                lib_manager = LibraryManager()
                
                # Create expandable sections for each peak
                for i, peak in enumerate(peaks):
                    with st.expander(f"Peak {i+1}: {peak.time_ms:.2f} ms (K0: {peak.K0_cm2_Vs:.3f} cm²/V·s)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            compound_name = st.text_input(
                                "Compound Name", 
                                key=f"peak_name_{i}",
                                placeholder="e.g., Acetone, Benzene, etc."
                            )
                            family = st.selectbox(
                                "Chemical Family",
                                ["", "alkane", "alkene", "alkyne", "aromatic", "alcohol", "ketone", 
                                 "aldehyde", "carboxylic_acid", "ester", "ether", "amine", "amide", 
                                 "halide", "nitrile", "thiol", "other"],
                                key=f"peak_family_{i}"
                            )
                        
                        with col2:
                            mz = st.number_input(
                                "m/z", 
                                10.0, 5000.0, 100.0, 1.0,
                                key=f"peak_mz_{i}",
                                help="Mass-to-charge ratio"
                            )
                            charge = st.number_input(
                                "Charge", 
                                1, 5, 1, 1,
                                key=f"peak_charge_{i}"
                            )
                        
                        notes = st.text_area(
                            "Notes",
                            key=f"peak_notes_{i}",
                            placeholder="Additional information about this compound"
                        )
                        
                        # Add to library button
                        if st.button(f"Add Peak {i+1} to Library", key=f"add_peak_{i}"):
                            if compound_name and family:
                                try:
                                    compound = LibraryCompound(
                                        name=compound_name,
                                        family=family,
                                        func_group="",  # Could be added later
                                        gas=gas.name,
                                        adduct="",
                                        z=charge,
                                        mz=mz,
                                        mono_mass=mz * charge,  # Approximate
                                        K0=peak.K0_cm2_Vs,
                                        K0_sd=0.0,  # No uncertainty estimate yet
                                        notes=f"From analysis: {peak.time_ms:.2f}ms peak. {notes}"
                                    )
                                    
                                    lib_manager.add_compound(compound)
                                    st.success(f"✅ Added {compound_name} to library!")
                                    
                                except Exception as e:
                                    st.error(f"Failed to add compound: {e}")
                            else:
                                st.warning("Please provide compound name and family")
                
                # Export
                st.subheader("💾 Export")
                csv_buffer = io.StringIO()
                df_peaks.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Peaks CSV",
                    data=csv_buffer.getvalue(),
                    file_name="analyzed_peaks.csv",
                    mime="text/csv"
                )
                
                # Store peaks in session for Sim Lab
                st.session_state['analyzed_peaks'] = peaks
                st.session_state['analyzed_spectrum'] = {'time': t_filtered, 'intensity': intensity_corrected}
            else:
                st.warning("No peaks detected. Try adjusting the analysis parameters.")
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")

def render_library():
    """Render library management tab."""
    pro_only()
    
    st.header("📚 Reference Library")
    
    # Initialize library manager
    lib_manager = LibraryManager()
    
    # Filter controls
    st.subheader("🔍 Filter Library")
    col1, col2, col3 = st.columns(3)
    family_filter = col1.text_input("Family contains", "")
    gas_filter = col2.selectbox("Gas", ["", "N2", "He", "Air"], index=0)
    name_filter = col3.text_input("Name contains", "")
    
    # Get filtered compounds
    compounds = lib_manager.get_compounds(
        family=family_filter if family_filter else None,
        gas=gas_filter if gas_filter else None,
        name_filter=name_filter if name_filter else None
    )
    
    # Display compounds
    st.subheader(f"📋 Library Compounds ({len(compounds)} found)")
    
    if compounds:
        df = pd.DataFrame(compounds)
        # Select relevant columns for display
        display_cols = ['id', 'name', 'family', 'gas', 'z', 'mz', 'K0', 'notes']
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols], width='stretch')
    else:
        st.info("No compounds found. Add some compounds or adjust filters.")
    
    # Add new compound
    st.subheader("➕ Add New Compound")
    
    with st.form("add_compound"):
        col1, col2 = st.columns(2)
        name = col1.text_input("Compound Name")
        family = col2.text_input("Family")
        
        col1, col2 = st.columns(2)
        gas = col1.selectbox("Gas", ["N2", "He", "Air"], index=0)
        z = col2.number_input("Charge", 1, 5, 1)
        
        col1, col2 = st.columns(2)
        mz = col1.number_input("m/z", 10.0, 5000.0, 100.0)
        K0 = col2.number_input("K0 (cm²/V·s)", 0.1, 10.0, 2.0, 0.001)
        
        notes = st.text_area("Notes")
        
        if st.form_submit_button("Add Compound"):
            if name:
                compound = LibraryCompound(
                    name=name,
                    family=family,
                    gas=gas,
                    z=z,
                    mz=mz,
                    K0=K0,
                    notes=notes
                )
                lib_manager.add_compound(compound)
                st.success(f"Added {name} to library")
                st.rerun()
            else:
                st.error("Compound name is required")
    
    # Import/Export
    st.subheader("📁 Import/Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Import CSV**")
        uploaded_csv = st.file_uploader("Upload CSV", type=['csv'], key="lib_import")
        if uploaded_csv:
            try:
                # Save uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
                    temp_file.write(uploaded_csv.getbuffer())
                    temp_path = Path(temp_file.name)
                
                try:
                    imported = lib_manager.import_csv(temp_path)
                finally:
                    # Clean up temporary file
                    if temp_path.exists():
                        temp_path.unlink()
                
                st.success(f"Imported {imported} compounds")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")
    
    with col2:
        st.markdown("**Export CSV**")
        if st.button("Export Library"):
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                    export_path = Path(temp_file.name)
                
                try:
                    exported = lib_manager.export_csv(export_path)
                    
                    if exported > 0:
                        with open(export_path, "r") as f:
                            csv_data = f.read()
                        
                        st.download_button(
                            label=f"📥 Download {exported} compounds",
                            data=csv_data,
                            file_name="library_export.csv",
                            mime="text/csv"
                        )
                finally:
                    # Clean up temporary file
                    if export_path.exists():
                        export_path.unlink()
                
                if exported == 0:
                    st.warning("No compounds to export")
            except Exception as e:
                st.error(f"Export failed: {e}")

def render_ml_training():
    """Render ML training tab."""
    pro_only()
    
    st.header("🤖 Machine Learning Training")
    
    # Initialize ML manager
    ml_manager = MLManager()
    
    # Load existing models
    k0_loaded, family_loaded = ml_manager.load_models()
    
    st.subheader("📊 Model Status")
    col1, col2 = st.columns(2)
    col1.metric("K0 Regressor", "✅ Loaded" if k0_loaded else "❌ Not Available")
    col2.metric("Family Classifier", "✅ Loaded" if family_loaded else "❌ Not Available")
    
    # Model performance summary
    if k0_loaded or family_loaded:
        if st.button("📈 Show Model Performance Summary"):
            try:
                summary = ml_manager.get_model_performance_summary()
                
                st.subheader("🔍 Model Performance Summary")
                
                # Feature information
                st.write(f"**Total Features**: {summary['feature_count']}")
                if summary['feature_names']:
                    st.write("**Feature Names**:")
                    st.code(", ".join(summary['feature_names']))
                
                # K0 model details
                if "k0_model" in summary:
                    st.subheader("K0 Regressor Details")
                    k0_info = summary["k0_model"]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Estimators", k0_info["n_estimators"])
                    col2.metric("Max Depth", k0_info["max_depth"] or "Unlimited")
                    
                    # Feature importance
                    if k0_info["feature_importance"]:
                        st.write("**Top 5 Most Important Features:**")
                        sorted_features = sorted(k0_info["feature_importance"].items(), 
                                               key=lambda x: x[1], reverse=True)[:5]
                        for feature, importance in sorted_features:
                            st.write(f"- {feature}: {importance:.3f}")
                
                # Family model details
                if "family_model" in summary:
                    st.subheader("Family Classifier Details")
                    family_info = summary["family_model"]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Estimators", family_info["n_estimators"])
                    col2.metric("Max Depth", family_info["max_depth"] or "Unlimited")
                    col3.metric("Classes", family_info["n_classes"])
                    
                    # Classes
                    if family_info["classes"]:
                        st.write("**Available Classes:**")
                        st.code(", ".join(family_info["classes"]))
                    
                    # Feature importance
                    if family_info["feature_importance"]:
                        st.write("**Top 5 Most Important Features:**")
                        sorted_features = sorted(family_info["feature_importance"].items(), 
                                               key=lambda x: x[1], reverse=True)[:5]
                        for feature, importance in sorted_features:
                            st.write(f"- {feature}: {importance:.3f}")
                
            except Exception as e:
                st.error(f"Failed to get model summary: {e}")
    
    # Training data status
    lib_manager = LibraryManager()
    compounds = lib_manager.get_compounds()
    st.info(f"📚 **Training Data**: {len(compounds)} compounds in library. Add more compounds in the Library tab or label peaks in the Analyze tab.")
    
    # Training data source
    st.subheader("📚 Training Data")
    
    # Get library data for training
    lib_manager = LibraryManager()
    compounds = lib_manager.get_compounds()
    
    if not compounds:
        st.warning("No library compounds available for training. Add compounds to the library first.")
        return
    
    # Convert to training dataframe
    training_data = []
    for comp in compounds:
        if all(comp.get(col) for col in ['name', 'K0', 'mz', 'z']):
            training_data.append({
                'name': comp['name'],
                'family': comp.get('family', 'unknown'),
                'mass_amu': comp['mz'] * comp['z'],  # Approximate mass from m/z
                'z': comp['z'],
                'ccs_A2': 150.0,  # Default CCS
                'gas_mass_amu': gas.mass_amu,
                'T_K': tube.temperature_K,
                'P_Pa': tube.pressure_kPa * 1000,
                'E_over_N_Td': E_over_N(tube),
                'K0_cm2_Vs': comp['K0']
            })
    
    if not training_data:
        st.warning("No valid training data found. Compounds need name, K0, mz, and z fields.")
        return
    
    df_training = pd.DataFrame(training_data)
    st.info(f"Training dataset: {len(df_training)} samples")
    
    # Show training data preview
    st.subheader("📋 Training Data Preview")
    st.dataframe(df_training[['name', 'family', 'mass_amu', 'z', 'K0_cm2_Vs']], width='stretch')
    
    # Training controls
    st.subheader("🚀 Train Models")
    
    col1, col2 = st.columns(2)
    
    # Training options
    st.subheader("⚙️ Training Options")
    col1, col2, col3 = st.columns(3)
    
    use_cv = col1.checkbox("Use Cross-Validation", value=True, help="Enable 5-fold cross-validation for better model evaluation")
    optimize_hyperparams = col2.checkbox("Optimize Hyperparameters", value=True, help="Automatically find best model parameters")
    train_both = col3.checkbox("Train Both Models", value=True, help="Train K0 regressor and family classifier together")
    
    # Training buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Train K0 Regressor", type="primary"):
            try:
                with st.spinner("Training K0 regressor with enhanced features..."):
                    results = ml_manager.train_k0_regressor(df_training, use_cv=use_cv, optimize_hyperparams=optimize_hyperparams)
                    
                st.success("K0 regressor trained successfully!")
                
                # Display enhanced metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R² Score", f"{results['r2']:.3f}")
                col2.metric("CV R² Mean", f"{results.get('cv_r2_mean', 0):.3f}")
                col3.metric("MAE", f"{results.get('mae', 0):.3f}")
                col4.metric("Features", results.get('n_features', 0))
                
                if results.get('cv_r2_std'):
                    st.info(f"Cross-validation R² std: ±{results['cv_r2_std']:.3f}")
                
                if results.get('best_params'):
                    st.json(results['best_params'])
                
            except Exception as e:
                st.error(f"Training failed: {e}")
    
    with col2:
        if st.button("🎯 Train Family Classifier"):
            try:
                with st.spinner("Training family classifier with enhanced features..."):
                    results = ml_manager.train_family_classifier(df_training, use_cv=use_cv, optimize_hyperparams=optimize_hyperparams)
                
                st.success("Family classifier trained successfully!")
                
                # Display enhanced metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{results['accuracy']:.3f}")
                col2.metric("F1 Score", f"{results['f1_macro']:.3f}")
                col3.metric("CV F1 Mean", f"{results.get('cv_f1_mean', 0):.3f}")
                col4.metric("Classes", results['n_classes'])
                
                if results.get('cv_f1_std'):
                    st.info(f"Cross-validation F1 std: ±{results['cv_f1_std']:.3f}")
                
                if results.get('best_params'):
                    st.json(results['best_params'])
                
            except Exception as e:
                st.error(f"Training failed: {e}")
    
    with col3:
        if st.button("🎯 Train Both Models", type="secondary"):
            try:
                with st.spinner("Training both models with enhanced features..."):
                    results = ml_manager.train_models_easy(df_training)
                
                st.success("Both models trained successfully!")
                
                # Display results for both models
                if "k0" in results and "error" not in results["k0"]:
                    st.subheader("K0 Regressor Results")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R² Score", f"{results['k0']['r2']:.3f}")
                    col2.metric("CV R²", f"{results['k0'].get('cv_r2_mean', 0):.3f}")
                    col3.metric("Features", results['k0'].get('n_features', 0))
                
                if "family" in results and "error" not in results["family"]:
                    st.subheader("Family Classifier Results")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{results['family']['accuracy']:.3f}")
                    col2.metric("F1 Score", f"{results['family']['f1_macro']:.3f}")
                    col3.metric("Classes", results['family']['n_classes'])
                
                # Show any errors
                for model_name, result in results.items():
                    if "error" in result:
                        st.error(f"{model_name.title()} training failed: {result['error']}")
                
            except Exception as e:
                st.error(f"Training failed: {e}")
    
    # Model predictions
    if k0_loaded or family_loaded:
        st.subheader("🔮 Make Predictions")
        
        with st.form("ml_predictions"):
            col1, col2 = st.columns(2)
            mass = col1.number_input("Mass (amu)", 10.0, 5000.0, 100.0)
            charge = col2.number_input("Charge", 1, 5, 1)
            
            col1, col2 = st.columns(2)
            ccs = col1.number_input("CCS (Å²)", 50.0, 1000.0, 150.0)
            
            if st.form_submit_button("Predict"):
                try:
                    features = MLFeatures(
                        mass_amu=mass,
                        z=charge,
                        ccs_A2=ccs,
                        gas_mass_amu=gas.mass_amu,
                        T_K=tube.temperature_K,
                        P_Pa=tube.pressure_kPa * 1000,
                        E_over_N_Td=E_over_N(tube)
                    )
                    
                    prediction = ml_manager.predict_all(features)
                    
                    st.success("Prediction completed!")
                    
                    # Enhanced prediction display
                    col1, col2 = st.columns(2)
                    if prediction.K0_pred > 0:
                        col1.metric("Predicted K0", f"{prediction.K0_pred:.3f} cm²/V·s")
                        if prediction.K0_uncertainty:
                            col1.metric("Uncertainty", f"±{prediction.K0_uncertainty:.3f} cm²/V·s")
                    
                    if prediction.family_pred:
                        col2.metric("Predicted Family", prediction.family_pred)
                        if prediction.family_confidence:
                            col2.metric("Confidence", f"{prediction.family_confidence:.2f}")
                    
                    # Show prediction intervals
                    if hasattr(prediction, 'K0_lower') and hasattr(prediction, 'K0_upper'):
                        st.info(f"95% Prediction Interval: [{prediction.K0_lower:.3f}, {prediction.K0_upper:.3f}] cm²/V·s")
                    
                    # Show top predictions for family
                    if hasattr(prediction, 'top_predictions') and prediction.top_predictions:
                        st.subheader("Top Family Predictions")
                        for i, pred in enumerate(prediction.top_predictions[:3]):
                            st.write(f"{i+1}. {pred['class']}: {pred['confidence']:.3f}")
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

def render_molecular_structure():
    """Render molecular structure tab."""
    pro_only()
    
    st.header("🧬 Molecular Structure & Reactions")
    
    # Check if RDKit is available
    try:
        import rdkit
        from rdkit import Chem
        from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
        from rdkit.Chem.Draw import IPythonConsole
        rdkit_available = True
    except ImportError:
        rdkit_available = False
        st.warning("⚠️ RDKit not available. Install with: `pip install rdkit` for full molecular structure features.")
    
    # Molecular input section
    st.subheader("🔬 Molecular Input")
    
    input_method = st.radio(
        "Choose input method:",
        ["SMILES String", "Chemical Formula", "Manual Entry"],
        horizontal=True
    )
    
    molecule_data = None
    
    if input_method == "SMILES String":
        smiles_input = st.text_input(
            "Enter SMILES string:",
            placeholder="e.g., CCO (ethanol), C1=CC=CC=C1 (benzene)",
            help="Enter a valid SMILES string for molecular structure"
        )
        
        if smiles_input and rdkit_available:
            try:
                mol = Chem.MolFromSmiles(smiles_input)
                if mol is not None:
                    molecule_data = {
                        'mol': mol,
                        'smiles': smiles_input,
                        'formula': rdMolDescriptors.CalcMolFormula(mol),
                        'mw': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'tpsa': Descriptors.TPSA(mol)
                    }
                    st.success("✅ Valid SMILES string!")
                else:
                    st.error("❌ Invalid SMILES string")
            except Exception as e:
                st.error(f"❌ Error parsing SMILES: {e}")
    
    elif input_method == "Chemical Formula":
        formula_input = st.text_input(
            "Enter chemical formula:",
            placeholder="e.g., C2H6O, C6H6",
            help="Enter a chemical formula (limited functionality without SMILES)"
        )
        
        if formula_input:
            # Basic formula parsing (simplified)
            st.info("ℹ️ Chemical formula input provides basic information. Use SMILES for full structure analysis.")
            molecule_data = {
                'formula': formula_input,
                'smiles': None,
                'mol': None
            }
    
    else:  # Manual Entry
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Molecule Name", "Unknown")
            formula = st.text_input("Chemical Formula", "")
            mw = st.number_input("Molecular Weight (g/mol)", 0.0, 5000.0, 100.0)
        
        with col2:
            logp = st.number_input("LogP", -10.0, 10.0, 0.0)
            hbd = st.number_input("H-bond Donors", 0, 20, 0)
            hba = st.number_input("H-bond Acceptors", 0, 20, 0)
        
        if name and formula:
            molecule_data = {
                'name': name,
                'formula': formula,
                'mw': mw,
                'logp': logp,
                'hbd': hbd,
                'hba': hba,
                'smiles': None,
                'mol': None
            }
    
    # Display molecular information
    if molecule_data:
        st.subheader("📊 Molecular Properties")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if 'formula' in molecule_data:
            col1.metric("Formula", molecule_data['formula'])
        if 'mw' in molecule_data:
            col2.metric("MW (g/mol)", f"{molecule_data['mw']:.2f}")
        if 'logp' in molecule_data:
            col3.metric("LogP", f"{molecule_data['logp']:.2f}")
        if 'tpsa' in molecule_data:
            col4.metric("TPSA (Å²)", f"{molecule_data['tpsa']:.2f}")
        
        # Additional properties
        if 'hbd' in molecule_data and 'hba' in molecule_data:
            col1, col2 = st.columns(2)
            col1.metric("H-bond Donors", molecule_data['hbd'])
            col2.metric("H-bond Acceptors", molecule_data['hba'])
    
    # 3D Molecular Structure Visualization
    if molecule_data and molecule_data.get('mol') and rdkit_available:
        st.subheader("🔬 3D Molecular Structure")
        
        try:
            # Generate 3D coordinates
            from rdkit.Chem import rdMolDescriptors
            mol_3d = Chem.AddHs(molecule_data['mol'])
            from rdkit.Chem import AllChem
            AllChem.EmbedMolecule(mol_3d)
            AllChem.MMFFOptimizeMolecule(mol_3d)
            
            # Create 3D visualization using plotly
            conf = mol_3d.GetConformer()
            xyz = conf.GetPositions()
            atoms = [mol_3d.GetAtomWithIdx(i).GetSymbol() for i in range(mol_3d.GetNumAtoms())]
            
            # Create 3D scatter plot
            fig_3d = go.Figure()
            
            # Color mapping for atoms
            atom_colors = {
                'C': 'black', 'H': 'white', 'O': 'red', 'N': 'blue',
                'S': 'yellow', 'P': 'orange', 'F': 'green', 'Cl': 'green',
                'Br': 'brown', 'I': 'purple'
            }
            
            for atom_type in set(atoms):
                mask = [i for i, a in enumerate(atoms) if a == atom_type]
                if mask:
                    fig_3d.add_trace(go.Scatter3d(
                        x=[xyz[i][0] for i in mask],
                        y=[xyz[i][1] for i in mask],
                        z=[xyz[i][2] for i in mask],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=atom_colors.get(atom_type, 'gray'),
                            symbol='circle'
                        ),
                        name=atom_type,
                        text=[f"{atom_type}{i}" for i in mask],
                        hovertemplate=f"<b>{atom_type}</b><br>Index: %{{text}}<extra></extra>"
                    ))
            
            # Add bonds
            for bond in mol_3d.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                fig_3d.add_trace(go.Scatter3d(
                    x=[xyz[i][0], xyz[j][0]],
                    y=[xyz[i][1], xyz[j][1]],
                    z=[xyz[i][2], xyz[j][2]],
                    mode='lines',
                    line=dict(color='gray', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            fig_3d.update_layout(
                title="3D Molecular Structure",
                scene=dict(
                    xaxis_title="X (Å)",
                    yaxis_title="Y (Å)",
                    zaxis_title="Z (Å)",
                    aspectmode='data'
                ),
                width=800,
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating 3D structure: {e}")
    
    # 2D Structure (if available)
    if molecule_data and molecule_data.get('mol') and rdkit_available:
        st.subheader("📐 2D Molecular Structure")
        
        try:
            # Generate 2D structure
            img = Draw.MolToImage(molecule_data['mol'], size=(400, 300))
            st.image(img, caption="2D Molecular Structure")
        except Exception as e:
            st.error(f"Error generating 2D structure: {e}")
    
    # Energy Diagrams and Transition States
    st.subheader("⚡ Energy Diagrams & Transition States")
    
    # Reaction type selection
    reaction_type = st.selectbox(
        "Select reaction type:",
        ["Ion-Molecule Reaction", "Fragmentation", "Adduct Formation", "Charge Transfer", "Custom"]
    )
    
    if reaction_type != "Custom":
        # Predefined reaction templates
        if reaction_type == "Ion-Molecule Reaction":
            st.info("🔄 Ion-molecule reactions in IMS involve collision between ions and neutral molecules")
            
            # Energy profile
            fig_energy = go.Figure()
            
            # Sample energy profile data
            x_vals = [0, 1, 2, 3, 4, 5]
            y_vals = [0, 15, 25, 20, 10, 5]  # Energy in kcal/mol
            labels = ["Reactants", "TS1", "Intermediate", "TS2", "Products", "Final"]
            
            fig_energy.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=10, color='red'),
                name="Energy Profile"
            ))
            
            # Add labels
            for i, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
                fig_energy.add_annotation(
                    x=x, y=y, text=label,
                    showarrow=True, arrowhead=2, arrowcolor="black",
                    ax=0, ay=-30
                )
            
            fig_energy.update_layout(
                title="Ion-Molecule Reaction Energy Profile",
                xaxis_title="Reaction Coordinate",
                yaxis_title="Energy (kcal/mol)",
                height=400
            )
            
            st.plotly_chart(fig_energy, use_container_width=True)
        
        elif reaction_type == "Fragmentation":
            st.info("💥 Fragmentation reactions involve breaking of chemical bonds")
            
            # Fragmentation energy diagram
            fig_frag = go.Figure()
            
            x_vals = [0, 1, 2, 3, 4]
            y_vals = [0, 20, 35, 25, 15]
            labels = ["Parent Ion", "TS", "Fragment 1", "Fragment 2", "Products"]
            
            fig_frag.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=10, color='orange'),
                name="Fragmentation"
            ))
            
            for i, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
                fig_frag.add_annotation(
                    x=x, y=y, text=label,
                    showarrow=True, arrowhead=2, arrowcolor="black",
                    ax=0, ay=-30
                )
            
            fig_frag.update_layout(
                title="Fragmentation Energy Profile",
                xaxis_title="Reaction Coordinate",
                yaxis_title="Energy (kcal/mol)",
                height=400
            )
            
            st.plotly_chart(fig_frag, use_container_width=True)
    
    # Transition State Analysis
    st.subheader("🔬 Transition State Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        activation_energy = st.number_input(
            "Activation Energy (kcal/mol)", 
            0.0, 100.0, 15.0, 1.0,
            help="Energy barrier for the reaction"
        )
        
        reaction_enthalpy = st.number_input(
            "Reaction Enthalpy (kcal/mol)", 
            -50.0, 50.0, -5.0, 1.0,
            help="Overall energy change"
        )
    
    with col2:
        temperature = st.number_input(
            "Temperature (K)", 
            200.0, 1000.0, 298.15, 10.0,
            help="Reaction temperature"
        )
        
        pressure = st.number_input(
            "Pressure (atm)", 
            0.1, 10.0, 1.0, 0.1,
            help="Reaction pressure"
        )
    
    # Calculate reaction kinetics
    if st.button("Calculate Reaction Kinetics"):
        # Arrhenius equation: k = A * exp(-Ea/RT)
        R = 0.001987  # kcal/(mol·K)
        A = 1e13  # Pre-exponential factor (s⁻¹)
        
        k = A * np.exp(-activation_energy / (R * temperature))
        half_life = np.log(2) / k if k > 0 else float('inf')
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rate Constant (s⁻¹)", f"{k:.2e}")
        col2.metric("Half-life (s)", f"{half_life:.2e}")
        col3.metric("Arrhenius Factor", f"{A:.2e}")
    
    # IMS Integration
    if molecule_data:
        st.subheader("🔗 IMS Integration")
        
        if st.button("Calculate IMS Properties"):
            try:
                # Estimate CCS from molecular properties
                if molecule_data.get('mw'):
                    # Simple CCS estimation (more sophisticated methods available)
                    estimated_ccs = 0.5 * (molecule_data['mw'] ** 0.67)  # Rough approximation
                    
                    st.success("✅ IMS Properties Calculated")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Estimated CCS (Å²)", f"{estimated_ccs:.1f}")
                    
                    if molecule_data.get('mw'):
                        col2.metric("Molecular Weight", f"{molecule_data['mw']:.1f} g/mol")
                    
                    if molecule_data.get('tpsa'):
                        col3.metric("TPSA", f"{molecule_data['tpsa']:.1f} Å²")
                    
                    # Add to IMS simulation
                    if st.button("Add to IMS Simulation"):
                        st.session_state['molecular_ion'] = {
                            'name': molecule_data.get('name', 'Molecule'),
                            'mass_amu': molecule_data.get('mw', 100.0),
                            'ccs_A2': estimated_ccs,
                            'charge': 1,
                            'smiles': molecule_data.get('smiles', ''),
                            'formula': molecule_data.get('formula', '')
                        }
                        st.success("✅ Added to IMS simulation parameters")
                
            except Exception as e:
                st.error(f"Error calculating IMS properties: {e}")

def render_sim_lab():
    """Render simulation lab tab."""
    pro_only()
    
    st.header("🔬 Simulation Laboratory")
    
    # Check for analyzed peaks from Analyze tab
    analyzed_peaks = st.session_state.get('analyzed_peaks', [])
    
    if not analyzed_peaks:
        st.info("No analyzed peaks available. Go to the Analyze tab first to detect peaks.")
        return
    
    st.success(f"Found {len(analyzed_peaks)} peaks from analysis")
    
    # Initialize managers
    ml_manager = MLManager()
    lib_manager = LibraryManager()
    
    # Load models
    k0_loaded, family_loaded = ml_manager.load_models()
    
    if not k0_loaded:
        st.warning("K0 regressor not available. Train models in ML Training tab first.")
        return
    
    # Unknown characterization
    st.subheader("🔍 Unknown Characterization")
    
    # Get library compounds for comparison
    library_compounds = lib_manager.get_compounds()
    
    results = []
    
    for i, peak in enumerate(analyzed_peaks):
        st.write(f"**Peak {i+1}**: {peak.time_ms:.2f} ms, K0 = {peak.K0_cm2_Vs:.3f} cm²/V·s")
        
        # Find closest library matches
        if library_compounds:
            # Calculate K0 differences
            k0_diffs = []
            for comp in library_compounds:
                if comp.get('K0'):
                    diff = abs(comp['K0'] - peak.K0_cm2_Vs)
                    k0_diffs.append((comp, diff))
            
            # Sort by difference
            k0_diffs.sort(key=lambda x: x[1])
            
            # Show top 5 matches
            st.write("**Top Library Matches:**")
            for j, (comp, diff) in enumerate(k0_diffs[:5]):
                st.write(f"{j+1}. {comp['name']} (ΔK0 = {diff:.3f}, Family: {comp.get('family', 'Unknown')})")
        
        # ML prediction
        try:
            features = MLFeatures(
                mass_amu=100.0,  # Default mass
                z=1,
                ccs_A2=150.0,  # Default CCS
                gas_mass_amu=gas.mass_amu,
                T_K=tube.temperature_K,
                P_Pa=tube.pressure_kPa * 1000,
                E_over_N_Td=E_over_N(tube)
            )
            
            prediction = ml_manager.predict_all(features)
            
            if prediction.family_pred:
                st.write(f"**ML Prediction**: Family = {prediction.family_pred}")
                if prediction.family_confidence:
                    st.write(f"Confidence: {prediction.family_confidence:.2f}")
        
        except Exception as e:
            st.write(f"ML prediction failed: {e}")
        
        st.markdown("---")
    
    # Add unknowns to staging
    st.subheader("📝 Add to Staging")
    
    if st.button("Add All Peaks to Staging"):
        added = 0
        for peak in analyzed_peaks:
            try:
                lib_manager.add_staging_unknown(
                    peak_time_ms=peak.time_ms,
                    K0=peak.K0_cm2_Vs,
                    gas=gas.name,
                    suggested_name=f"Unknown_{peak.time_ms:.1f}ms",
                    confidence=0.5
                )
                added += 1
            except Exception as e:
                st.error(f"Failed to add peak {peak.time_ms:.2f}ms: {e}")
        
        if added > 0:
            st.success(f"Added {added} peaks to staging area")

def render_visualization():
    """Render visualization tab."""
    pro_only()
    
    st.header("📐 IMS Visualization")
    
    # Import new visualization functions
    from imsim.viz_new import (create_temperature_heatmap, create_electric_field_lines, 
                              create_trajectory_animation, create_combined_visualization)
    
    # Electrode Configuration
    st.subheader("⚡ Electrode Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Number of electrodes
        n_electrodes = st.slider("Number of Electrodes", 3, 10, 5, 1, key="n_electrodes")
        
        # Electrode voltage controls
        electrode_voltages = []
        for i in range(n_electrodes):
            voltage = st.number_input(
                f"Electrode {i+1} Voltage (V)", 
                value=float(tube.voltage_V * i / max(1, n_electrodes - 1)),
                key=f"electrode_voltage_{i}"
            )
            electrode_voltages.append(voltage)
    
    with col2:
        st.markdown("**Tube Parameters**")
        st.metric("Length", f"{tube.length_m:.3f} m")
        st.metric("Total Voltage", f"{tube.voltage_V:.0f} V")
        st.metric("Gas", tube.gas.name)
        st.metric("Temperature", f"{tube.temperature_K:.1f} K")
        st.metric("Pressure", f"{tube.pressure_kPa:.1f} kPa")
    
    # Visualization Options
    st.subheader("🎨 Visualization Options")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        show_heatmap = st.checkbox("Show Temperature Heatmap", value=True)
        show_field_lines = st.checkbox("Show Electric Field Lines", value=True)
        show_trajectories = st.checkbox("Show Ion Trajectories", value=False)
    
    with viz_col2:
        if show_trajectories:
            # Get trajectories from session state if available
            if 'trajectories' in st.session_state and 'ions' in st.session_state:
                trajectories = st.session_state.trajectories
                ions = st.session_state.ions
            else:
                trajectories = None
                ions = None
                st.info("Run trajectory simulation first to show ion paths")
        else:
            trajectories = None
            ions = None
    
    # Combined Visualization
    st.subheader("🔬 Combined Visualization")
    
    if trajectories and ions and show_trajectories:
        fig_combined = create_combined_visualization(
            tube, trajectories, ions, electrode_voltages, 
            show_heatmap, show_field_lines
        )
        st.plotly_chart(fig_combined, width='stretch')
    else:
        # Show basic combined view without trajectories
        fig_combined = create_combined_visualization(
            tube, None, None, electrode_voltages, 
            show_heatmap, show_field_lines
        )
        st.plotly_chart(fig_combined, width='stretch')
    
    # Individual Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Temperature Heatmap")
        fig_heatmap = create_temperature_heatmap(tube, electrode_voltages)
        st.plotly_chart(fig_heatmap, width='stretch')
    
    with col2:
        st.subheader("⚡ Electric Field Lines")
        fig_field_lines = create_electric_field_lines(tube, electrode_voltages)
        st.plotly_chart(fig_field_lines, width='stretch')
    
    # Trajectory Animation (if available)
    if trajectories and ions:
        st.subheader("🚀 Ion Trajectories in Drift Tube")
        fig_trajectory = create_trajectory_animation(tube, trajectories, ions, electrode_voltages)
        st.plotly_chart(fig_trajectory, width='stretch')
    
    # 3D Visualization
    st.subheader("🌐 3D Tube Model")
    
    fig_3d = create_3d_tube(tube)
    st.plotly_chart(fig_3d, width='stretch')
    
    # Export options
    st.subheader("💾 Export Visualizations")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📥 Export Combined View"):
            html_combined = fig_combined.to_html(include_plotlyjs=True)
            st.download_button(
                label="Download Combined HTML",
                data=html_combined,
                file_name="ims_combined_visualization.html",
                mime="text/html"
            )
    
    with export_col2:
        if st.button("📥 Export Temperature Heatmap"):
            html_heatmap = fig_heatmap.to_html(include_plotlyjs=True)
            st.download_button(
                label="Download Temperature Heatmap HTML",
                data=html_heatmap,
                file_name="ims_temperature_heatmap.html",
                mime="text/html"
            )
    
    with export_col3:
        if st.button("📥 Export 3D Model"):
            html_3d = fig_3d.to_html(include_plotlyjs=True)
            st.download_button(
                label="Download 3D HTML",
                data=html_3d,
                file_name="ims_3d_model.html",
                mime="text/html"
            )

def render_trajectories():
    """Render trajectories tab."""
    pro_only()
    
    st.header("🚀 Ion Trajectories")
    
    # Ion configuration
    st.subheader("Ion Configuration")
    
    n_ions = st.number_input("Number of Ions", 1, 5, 2)
    
    ions = []
    for i in range(int(n_ions)):
        with st.expander(f"Ion {i+1}"):
            col1, col2 = st.columns(2)
            name = col1.text_input("Name", f"Ion{i+1}", key=f"traj_ion_name_{i}")
            mass = col1.number_input("Mass (amu)", 10.0, 5000.0, 100.0, key=f"traj_ion_mass_{i}")
            charge = col2.number_input("Charge", 1, 5, 1, key=f"traj_ion_charge_{i}")
            ccs = col2.number_input("CCS (Å²)", 50.0, 1000.0, 150.0, key=f"traj_ion_ccs_{i}")
            
            ion = Ion(name=name, mass_amu=mass, charge=charge, ccs_A2=ccs)
            ions.append(ion)
    
    # Trajectory parameters
    st.subheader("Trajectory Parameters")
    
    col1, col2 = st.columns(2)
    n_segments = col1.number_input("Tube Segments", 1, 6, 3)
    n_frames = col2.slider("Animation Frames", 50, 500, 100)
    
    # Run trajectory simulation
    if st.button("🚀 Simulate Trajectories", type="primary"):
        with st.spinner("Simulating trajectories..."):
            try:
                trajectories = simulate_trajectories(
                    ions=ions,
                    tube=tube,
                    n_segments=int(n_segments),
                    n_frames=int(n_frames)
                )
                
                # Store in session state for use in visualization tab
                st.session_state.trajectories = trajectories
                st.session_state.ions = ions
                
                st.success("Trajectory simulation completed!")
                
                # Plot trajectories
                st.subheader("📈 Trajectory Plot")
                
                fig = plot_trajectories(trajectories, ions, tube, "Ion Trajectories")
                st.plotly_chart(fig, width='stretch')
                
                # Animation controls
                st.subheader("🎬 Animation")
                
                frame_slider = st.slider("Frame", 0, int(n_frames)-1, 0)
                
                # Show current frame
                fig_frame = go.Figure()
                
                for i, (traj, ion) in enumerate(zip(trajectories, ions)):
                    t, z = traj[:, 0], traj[:, 1]
                    current_pos = z[frame_slider] if frame_slider < len(z) else z[-1]
                    current_time = t[frame_slider] if frame_slider < len(t) else t[-1]
                    
                    fig_frame.add_trace(go.Scatter(
                        x=[current_time * 1000],  # Convert to ms
                        y=[current_pos * 1000],   # Convert to mm
                        mode='markers',
                        name=f"{ion.name}",
                        marker=dict(size=15, symbol='circle'),
                        hovertemplate=f"<b>{ion.name}</b><br>" +
                                     f"Time: {current_time*1000:.1f} ms<br>" +
                                     f"Position: {current_pos*1000:.1f} mm<br>" +
                                     "<extra></extra>"
                    ))
                
                # Calculate max time safely
                max_times = [t[-1] for t in trajectories if len(t) > 0]
                max_time = max(max_times) if max_times else 1.0
                
                fig_frame.update_layout(
                    title=f"Trajectory Frame {frame_slider}",
                    xaxis_title="Time (ms)",
                    yaxis_title="Position (mm)",
                    xaxis=dict(range=[0, max_time * 1000]),
                    yaxis=dict(range=[0, tube.length_m * 1000]),
                    height=400,
                    template="simple_white"
                )
                
                st.plotly_chart(fig_frame, width='stretch')
                
                # Note about enhanced visualization
                st.info("💡 **Enhanced Visualization Available!** Go to the Visualization tab to see ion trajectories moving through the drift tube with electric field visualizations.")
                
                # Export trajectory data
                st.subheader("💾 Export Trajectories")
                
                if st.button("📥 Download Trajectory Data"):
                    # Create CSV with trajectory data
                    trajectory_data = []
                    for i, (traj, ion) in enumerate(zip(trajectories, ions)):
                        for t, z in traj:
                            trajectory_data.append({
                                'ion_name': ion.name,
                                'time_s': t,
                                'position_m': z,
                                'time_ms': t * 1000,
                                'position_mm': z * 1000
                            })
                    
                    df_traj = pd.DataFrame(trajectory_data)
                    csv_buffer = io.StringIO()
                    df_traj.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="📥 Download Trajectory CSV",
                        data=csv_buffer.getvalue(),
                        file_name="ion_trajectories.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"Trajectory simulation failed: {e}")

# Render all tabs
with tabs[0]:
    render_simulate()

with tabs[1]:
    render_analyze()

with tabs[2]:
    render_library()

with tabs[3]:
    render_ml_training()

with tabs[4]:
    render_sim_lab()

with tabs[5]:
    render_molecular_structure()

with tabs[6]:
    render_visualization()

with tabs[7]:
    render_trajectories()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>IMS Physics Pro - Pro Version | 
    <a href='#' style='color: gray;'>Documentation</a> | 
    <a href='#' style='color: gray;'>Support</a></p>
</div>
""", unsafe_allow_html=True)
