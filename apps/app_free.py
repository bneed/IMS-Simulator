"""
IMS Physics Pro - Free Version
Simulate and Analyze tabs only
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
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
generate_peak_table = imsim_modules['sim']['generate_peak_table']
safe_tab = imsim_modules['utils']['safe_tab']
format_time_ms = imsim_modules['utils']['format_time_ms']
format_mobility = imsim_modules['utils']['format_mobility']
format_ccs = imsim_modules['utils']['format_ccs']

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

st.set_page_config(
    page_title="IMS Physics Pro - Free",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ IMS Physics Pro - Free Version")
st.caption("Simulate and analyze ion mobility spectrometry data")

# Sidebar - Instrument Controls
with st.sidebar:
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
        
        st.markdown("---")
        st.markdown("### 📊 Key Parameters")
        cols = st.columns(2)
        cols[0].metric("E", f"{E:,.0f}", "V/m")
        cols[1].metric("E/N", f"{E_over_N_Td:,.1f}", "Td")
        cols[0].metric("Gas", gas_name)
        cols[1].metric("P", f"{P_kPa:.1f}", "kPa")
    except Exception as e:
        st.error(f"Error calculating parameters: {e}")

# Main tabs
tabs = st.tabs(["🎯 Simulate", "📊 Analyze"])

def render_simulate():
    """Render simulation tab."""
    st.header("🎯 Ion Mobility Simulation")
    
    # Ion input section
    st.subheader("Ion Configuration")
    n_ions = st.number_input("Number of Ions", 1, 3, 2, 1, 
                            help="Free version limited to 3 ions")
    
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
    n_points = col2.slider("Data Points", 500, 10000, 3000, 500)
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
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=spectrum.time_ms,
                    y=spectrum.intensity,
                    mode='lines',
                    name='Total Spectrum',
                    line=dict(color='blue', width=2)
                ))
                
                # Add individual peaks
                colors = ['red', 'green', 'orange', 'purple', 'brown']
                for i, (peak, ion) in enumerate(zip(peaks, ions)):
                    fig.add_trace(go.Scatter(
                        x=[peak.time_ms],
                        y=[peak.intensity],
                        mode='markers',
                        name=f'{ion.name} Peak',
                        marker=dict(color=colors[i % len(colors)], size=10, symbol='diamond')
                    ))
                
                fig.update_layout(
                    title="Simulated IMS Spectrum",
                    xaxis_title="Drift Time (ms)",
                    yaxis_title="Intensity",
                    height=500,
                    template="simple_white"
                )
                
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
            
            # Spectrum type selection (basic for Free version)
            if has_multiple_scans:
                st.subheader("📊 Spectrum Type")
                spectrum_type = st.selectbox(
                    "Select spectrum to display:",
                    ["Mean", "Max", "Min"],
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
                
                st.info(f"Showing {spectrum_type.lower()} spectrum from {len(raw_matrix)} scans")
            
            # Drift time window
            st.subheader("⏱️ Drift Time Window")
            col1, col2 = st.columns(2)
            use_window = st.checkbox("Apply drift time window", value=False)
            
            if use_window:
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
            
            # Raw spectrum
            fig.add_trace(go.Scatter(
                x=t_filtered,
                y=intensity_filtered,
                mode='lines',
                name='Raw Spectrum',
                line=dict(color='gray', width=1, dash='dot'),
                opacity=0.7
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
                yaxis_title="Intensity",
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
                
                # Note about Pro version for labeling
                st.info("💡 **Upgrade to Pro** to label these peaks and add them to your training library for ML models!")
                
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
            else:
                st.warning("No peaks detected. Try adjusting the analysis parameters.")
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")

# Render tabs
with tabs[0]:
    render_simulate()

with tabs[1]:
    render_analyze()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>IMS Physics Pro - Free Version | 
    <a href='#' style='color: gray;'>Upgrade to Pro</a> for Library, ML, and advanced features</p>
</div>
""", unsafe_allow_html=True)
