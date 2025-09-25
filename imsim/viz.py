"""Visualization module for IMS Physics Pro."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
from .schemas import Spectrum, Peak, Tube, Ion

def create_2d_schematic(tube: Tube, show_field: bool = True) -> go.Figure:
    """Create 2D schematic of drift tube."""
    fig = go.Figure()
    
    # Tube outline
    tube_length_mm = tube.length_m * 1000
    tube_width_mm = 20
    
    # Electrodes
    n_electrodes = 5
    electrode_positions = np.linspace(0, tube_length_mm, n_electrodes)
    
    # Add tube walls
    fig.add_shape(
        type="rect",
        x0=0, y0=-tube_width_mm/2,
        x1=tube_length_mm, y1=tube_width_mm/2,
        fillcolor="lightgray",
        opacity=0.3,
        line=dict(color="black", width=2)
    )
    
    # Add electrodes
    for i, pos in enumerate(electrode_positions):
        voltage = tube.voltage_V * (i / (n_electrodes - 1))
        color = f"rgb({int(255 * (1 - voltage/tube.voltage_V))}, 0, {int(255 * voltage/tube.voltage_V)})"
        
        fig.add_shape(
            type="rect",
            x0=pos-2, y0=-tube_width_mm/2-5,
            x1=pos+2, y1=tube_width_mm/2+5,
            fillcolor=color,
            opacity=0.7,
            line=dict(color="black", width=1)
        )
        
        # Voltage label
        fig.add_annotation(
            x=pos, y=-tube_width_mm/2-10,
            text=f"{voltage:.0f}V",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Add field lines if requested
    if show_field:
        field_lines_x = []
        field_lines_y = []
        for y in np.linspace(-tube_width_mm/2+2, tube_width_mm/2-2, 5):
            for x in np.linspace(0, tube_length_mm, 20):
                field_lines_x.append(x)
                field_lines_y.append(y)
        
        fig.add_trace(go.Scatter(
            x=field_lines_x, y=field_lines_y,
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.3),
            name='Field Lines'
        ))
    
    fig.update_layout(
        title=f"IMS Drift Tube Schematic<br>L={tube.length_m:.3f}m, V={tube.voltage_V:.0f}V",
        xaxis_title="Position (mm)",
        yaxis_title="Position (mm)",
        showlegend=False,
        width=800,
        height=400
    )
    
    return fig

def create_3d_tube(tube: Tube) -> go.Figure:
    """Create 3D visualization of drift tube."""
    fig = go.Figure()
    
    # Tube dimensions
    tube_length_mm = tube.length_m * 1000
    tube_radius_mm = 10
    
    # Create cylinder
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0, tube_length_mm, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = tube_radius_mm * np.cos(theta_grid)
    y = tube_radius_mm * np.sin(theta_grid)
    
    fig.add_trace(go.Surface(
        x=x, y=y, z=z_grid,
        colorscale='Viridis',
        opacity=0.3,
        name='Drift Tube'
    ))
    
    # Add electrodes as rings
    n_electrodes = 5
    for i, pos in enumerate(np.linspace(0, tube_length_mm, n_electrodes)):
        voltage = tube.voltage_V * (i / (n_electrodes - 1))
        color = f"rgb({int(255 * (1 - voltage/tube.voltage_V))}, 0, {int(255 * voltage/tube.voltage_V)})"
        
        electrode_theta = np.linspace(0, 2*np.pi, 100)
        electrode_x = tube_radius_mm * np.cos(electrode_theta)
        electrode_y = tube_radius_mm * np.sin(electrode_theta)
        electrode_z = np.full_like(electrode_theta, pos)
        
        fig.add_trace(go.Scatter3d(
            x=electrode_x, y=electrode_y, z=electrode_z,
            mode='lines',
            line=dict(color=color, width=5),
            name=f'Electrode {i+1}'
        ))
    
    fig.update_layout(
        title=f"3D IMS Drift Tube<br>L={tube.length_m:.3f}m, V={tube.voltage_V:.0f}V",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600
    )
    
    return fig

def plot_spectrum(spectrum: Spectrum, peaks: Optional[List[Peak]] = None,
                 title: str = "IMS Spectrum", show_baseline: bool = False) -> go.Figure:
    """Plot IMS spectrum with optional peak markers."""
    fig = go.Figure()
    
    # Main spectrum
    fig.add_trace(go.Scatter(
        x=spectrum.time_ms,
        y=spectrum.intensity,
        mode='lines',
        name='Spectrum',
        line=dict(color='blue', width=2)
    ))
    
    # Add peaks if provided
    if peaks:
        peak_times = [p.time_ms for p in peaks]
        peak_intensities = [p.intensity for p in peaks]
        
        fig.add_trace(go.Scatter(
            x=peak_times,
            y=peak_intensities,
            mode='markers',
            name='Peaks',
            marker=dict(color='red', size=8, symbol='diamond'),
            text=[f"K0: {p.K0_cm2_Vs:.3f}" for p in peaks],
            hovertemplate="<b>Peak</b><br>" +
                         "Time: %{x:.2f} ms<br>" +
                         "Intensity: %{y:.2f}<br>" +
                         "%{text} cm²/V·s<br>" +
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Drift Time (ms)",
        yaxis_title="Intensity",
        width=800,
        height=500,
        template="simple_white"
    )
    
    return fig

def plot_trajectories(trajectories: List[np.ndarray], ions: List[Ion],
                     tube: Tube, title: str = "Ion Trajectories") -> go.Figure:
    """Plot ion trajectories through drift tube."""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, (traj, ion) in enumerate(zip(trajectories, ions)):
        t, z = traj[:, 0], traj[:, 1]
        
        fig.add_trace(go.Scatter(
            x=t * 1000,  # Convert to ms
            y=z * 1000,  # Convert to mm
            mode='lines+markers',
            name=f"{ion.name} (m/z={ion.mass_amu/ion.charge:.1f})",
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (ms)",
        yaxis_title="Position (mm)",
        width=800,
        height=500,
        template="simple_white"
    )
    
    return fig

def create_ml_performance_plot(y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = "ML Performance") -> go.Figure:
    """Create scatter plot for ML model performance."""
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', opacity=0.6),
        hovertemplate="<b>Prediction</b><br>" +
                     "True: %{x:.3f}<br>" +
                     "Pred: %{y:.3f}<br>" +
                     "<extra></extra>"
    ))
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash'),
        showlegend=True
    ))
    
    # Calculate R²
    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
    
    fig.update_layout(
        title=f"{title} (R² = {r2:.3f})",
        xaxis_title="True Value",
        yaxis_title="Predicted Value",
        width=600,
        height=500,
        template="simple_white"
    )
    
    return fig

def create_confusion_matrix_plot(cm: np.ndarray, labels: List[str],
                                title: str = "Confusion Matrix") -> go.Figure:
    """Create confusion matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="<b>Confusion Matrix</b><br>" +
                     "True: %{y}<br>" +
                     "Predicted: %{x}<br>" +
                     "Count: %{z}<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        width=600,
        height=500,
        template="simple_white"
    )
    
    return fig

