"""Additional visualization functions for IMS Physics Pro."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional
from .schemas import Tube, Ion

def create_temperature_heatmap(tube: Tube, electrode_voltages: Optional[List[float]] = None) -> go.Figure:
    """Create temperature heatmap visualization showing thermal distribution in drift tube."""
    fig = go.Figure()
    
    # Tube dimensions
    tube_length_mm = tube.length_m * 1000
    tube_width_mm = 20
    
    # Create grid for temperature calculation
    x = np.linspace(0, tube_length_mm, 50)
    y = np.linspace(-tube_width_mm/2, tube_width_mm/2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Calculate temperature distribution
    base_temp = tube.temperature_K
    
    if electrode_voltages:
        # Temperature variation based on electrode heating and gas flow
        n_electrodes = len(electrode_voltages)
        electrode_positions = np.linspace(0, tube_length_mm, n_electrodes)
        temperature = np.full_like(X, base_temp)
        
        for i, (pos, voltage) in enumerate(zip(electrode_positions, electrode_voltages)):
            # Electrode heating effect (higher voltage = more heating)
            heating_factor = (voltage / max(electrode_voltages)) * 50  # Up to 50K heating
            
            # Distance from electrode
            distance = np.sqrt((X - pos)**2 + Y**2)
            
            # Gaussian temperature profile around electrode
            electrode_temp = base_temp + heating_factor * np.exp(-(distance**2) / (2 * (tube_width_mm/4)**2))
            
            # Add to temperature field
            temperature = np.maximum(temperature, electrode_temp)
            
        # Add gas flow cooling effect (cooler at inlet, warmer at outlet)
        flow_cooling = np.linspace(0, 10, X.shape[1])  # 10K cooling gradient
        temperature -= flow_cooling[np.newaxis, :]
        
    else:
        # Uniform temperature with slight gradient
        temperature = np.full_like(X, base_temp)
        # Add slight thermal gradient along tube length
        temp_gradient = np.linspace(0, 5, X.shape[1])  # 5K gradient
        temperature += temp_gradient[np.newaxis, :]
    
    # Create heatmap
    fig.add_trace(go.Heatmap(
        x=x,
        y=y,
        z=temperature,
        colorscale='RdYlBu_r',  # Red-hot to blue-cool
        colorbar=dict(title="Temperature (K)"),
        hovertemplate="<b>Temperature</b><br>" +
                     "X: %{x:.1f} mm<br>" +
                     "Y: %{y:.1f} mm<br>" +
                     "Temp: %{z:.1f} K<br>" +
                     "<extra></extra>"
    ))
    
    # Add tube outline
    fig.add_shape(
        type="rect",
        x0=0, y0=-tube_width_mm/2,
        x1=tube_length_mm, y1=tube_width_mm/2,
        fillcolor="rgba(0,0,0,0)",
        line=dict(color="black", width=2)
    )
    
    # Add temperature contour lines
    fig.add_trace(go.Contour(
        x=x,
        y=y,
        z=temperature,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color="white"),
            start=np.min(temperature),
            end=np.max(temperature),
            size=(np.max(temperature) - np.min(temperature)) / 10
        ),
        line=dict(width=1, color="white"),
        showscale=False,
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f"Temperature Distribution<br>L={tube.length_m:.3f}m, Base Temp={tube.temperature_K:.1f}K",
        xaxis_title="Position (mm)",
        yaxis_title="Position (mm)",
        width=800,
        height=400
    )
    
    return fig

def create_electric_field_lines(tube: Tube, electrode_voltages: Optional[List[float]] = None) -> go.Figure:
    """Create electric field lines visualization."""
    fig = go.Figure()
    
    # Tube dimensions
    tube_length_mm = tube.length_m * 1000
    tube_width_mm = 20
    
    # Add tube outline
    fig.add_shape(
        type="rect",
        x0=0, y0=-tube_width_mm/2,
        x1=tube_length_mm, y1=tube_width_mm/2,
        fillcolor="lightgray",
        opacity=0.3,
        line=dict(color="black", width=2)
    )
    
    # Create field lines
    n_lines = 10
    start_y = np.linspace(-tube_width_mm/2+2, tube_width_mm/2-2, n_lines)
    
    for y_start in start_y:
        # Simple field line calculation (equipotential lines)
        x_line = np.linspace(0, tube_length_mm, 100)
        y_line = np.full_like(x_line, y_start)
        
        # Add some curvature based on field strength
        if electrode_voltages:
            for i, pos in enumerate(np.linspace(0, tube_length_mm, len(electrode_voltages))):
                if i < len(electrode_voltages) - 1:
                    mask = (x_line >= pos) & (x_line <= np.linspace(0, tube_length_mm, len(electrode_voltages))[i+1])
                    # Simple field line bending
                    field_strength = abs(electrode_voltages[i+1] - electrode_voltages[i]) / tube.length_m
                    y_line[mask] += 0.5 * np.sin(2 * np.pi * (x_line[mask] - pos) / tube_length_mm) * (field_strength / 1000)
        
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color='blue', width=1),
            showlegend=False,
            hovertemplate="<b>Field Line</b><br>" +
                         "X: %{x:.1f} mm<br>" +
                         "Y: %{y:.1f} mm<br>" +
                         "<extra></extra>"
        ))
    
    # Add electrodes
    if electrode_voltages:
        n_electrodes = len(electrode_voltages)
        electrode_positions = np.linspace(0, tube_length_mm, n_electrodes)
        
        for i, (pos, voltage) in enumerate(zip(electrode_positions, electrode_voltages)):
            color_intensity = voltage / max(electrode_voltages) if max(electrode_voltages) > 0 else 0
            color = f"rgb({int(255 * (1 - color_intensity))}, 0, {int(255 * color_intensity)})"
            
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
    
    fig.update_layout(
        title=f"Electric Field Lines<br>L={tube.length_m:.3f}m",
        xaxis_title="Position (mm)",
        yaxis_title="Position (mm)",
        width=800,
        height=400
    )
    
    return fig

def create_trajectory_animation(tube: Tube, trajectories: List[np.ndarray], ions: List[Ion], 
                               electrode_voltages: Optional[List[float]] = None) -> go.Figure:
    """Create animated visualization of ion trajectories through drift tube."""
    fig = go.Figure()
    
    # Tube dimensions
    tube_length_mm = tube.length_m * 1000
    tube_width_mm = 20
    
    # Add tube outline
    fig.add_shape(
        type="rect",
        x0=0, y0=-tube_width_mm/2,
        x1=tube_length_mm, y1=tube_width_mm/2,
        fillcolor="lightgray",
        opacity=0.2,
        line=dict(color="black", width=2)
    )
    
    # Add electrodes
    if electrode_voltages:
        n_electrodes = len(electrode_voltages)
        electrode_positions = np.linspace(0, tube_length_mm, n_electrodes)
        
        for i, (pos, voltage) in enumerate(zip(electrode_positions, electrode_voltages)):
            color_intensity = voltage / max(electrode_voltages) if max(electrode_voltages) > 0 else 0
            color = f"rgb({int(255 * (1 - color_intensity))}, 0, {int(255 * color_intensity)})"
            
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
    
    colors = px.colors.qualitative.Set1
    
    # Add trajectory traces
    for i, (traj, ion) in enumerate(zip(trajectories, ions)):
        t, z = traj[:, 0], traj[:, 1]
        
        # Convert to mm and add some random y-position for 3D effect
        x_pos = z * 1000  # Convert to mm
        y_pos = np.random.normal(0, 2, len(t))  # Random y-position within tube
        
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='lines+markers',
            name=f"{ion.name}",
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=6),
            hovertemplate=f"<b>{ion.name}</b><br>" +
                         "Position: %{x:.1f} mm<br>" +
                         "Time: %{text:.2f} ms<br>" +
                         "<extra></extra>",
            text=t * 1000  # Time in ms for hover
        ))
        
        # Add starting position marker
        fig.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            name=f"{ion.name} Start",
            marker=dict(color=colors[i % len(colors)], size=12, symbol='circle'),
            showlegend=False
        ))
        
        # Add ending position marker
        if len(x_pos) > 0:
            fig.add_trace(go.Scatter(
                x=[x_pos[-1]],
                y=[y_pos[-1]],
                mode='markers',
                name=f"{ion.name} End",
                marker=dict(color=colors[i % len(colors)], size=12, symbol='diamond'),
                showlegend=False
            ))
    
    fig.update_layout(
        title=f"Ion Trajectories in Drift Tube<br>L={tube.length_m:.3f}m, V={tube.voltage_V:.0f}V",
        xaxis_title="Position (mm)",
        yaxis_title="Position (mm)",
        width=800,
        height=500,
        template="simple_white"
    )
    
    return fig

def create_combined_visualization(tube: Tube, trajectories: Optional[List[np.ndarray]] = None, 
                                 ions: Optional[List[Ion]] = None,
                                 electrode_voltages: Optional[List[float]] = None,
                                 show_heatmap: bool = True, show_field_lines: bool = True) -> go.Figure:
    """Create combined visualization with heatmap, field lines, and trajectories."""
    fig = go.Figure()
    
    # Tube dimensions
    tube_length_mm = tube.length_m * 1000
    tube_width_mm = 20
    
    # Add temperature heatmap
    if show_heatmap:
        x = np.linspace(0, tube_length_mm, 50)
        y = np.linspace(-tube_width_mm/2, tube_width_mm/2, 20)
        X, Y = np.meshgrid(x, y)
        
        # Calculate temperature distribution
        base_temp = tube.temperature_K
        
        if electrode_voltages:
            # Temperature variation based on electrode heating and gas flow
            n_electrodes = len(electrode_voltages)
            electrode_positions = np.linspace(0, tube_length_mm, n_electrodes)
            temperature = np.full_like(X, base_temp)
            
            for i, (pos, voltage) in enumerate(zip(electrode_positions, electrode_voltages)):
                # Electrode heating effect (higher voltage = more heating)
                heating_factor = (voltage / max(electrode_voltages)) * 50  # Up to 50K heating
                
                # Distance from electrode
                distance = np.sqrt((X - pos)**2 + Y**2)
                
                # Gaussian temperature profile around electrode
                electrode_temp = base_temp + heating_factor * np.exp(-(distance**2) / (2 * (tube_width_mm/4)**2))
                
                # Add to temperature field
                temperature = np.maximum(temperature, electrode_temp)
                
            # Add gas flow cooling effect (cooler at inlet, warmer at outlet)
            flow_cooling = np.linspace(0, 10, X.shape[1])  # 10K cooling gradient
            temperature -= flow_cooling[np.newaxis, :]
            
        else:
            # Uniform temperature with slight gradient
            temperature = np.full_like(X, base_temp)
            # Add slight thermal gradient along tube length
            temp_gradient = np.linspace(0, 5, X.shape[1])  # 5K gradient
            temperature += temp_gradient[np.newaxis, :]
        
        fig.add_trace(go.Heatmap(
            x=x, y=y, z=temperature,
            colorscale='RdYlBu_r',  # Red-hot to blue-cool
            opacity=0.3,
            colorbar=dict(title="Temperature (K)"),
            showscale=False
        ))
    
    # Add field lines
    if show_field_lines:
        n_lines = 8
        start_y = np.linspace(-tube_width_mm/2+2, tube_width_mm/2-2, n_lines)
        
        for y_start in start_y:
            x_line = np.linspace(0, tube_length_mm, 100)
            y_line = np.full_like(x_line, y_start)
            
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color='blue', width=1, dash='dot'),
                showlegend=False,
                opacity=0.6
            ))
    
    # Add tube outline
    fig.add_shape(
        type="rect",
        x0=0, y0=-tube_width_mm/2,
        x1=tube_length_mm, y1=tube_width_mm/2,
        fillcolor="rgba(0,0,0,0)",
        line=dict(color="black", width=2)
    )
    
    # Add electrodes
    if electrode_voltages:
        n_electrodes = len(electrode_voltages)
        electrode_positions = np.linspace(0, tube_length_mm, n_electrodes)
        
        for i, (pos, voltage) in enumerate(zip(electrode_positions, electrode_voltages)):
            color_intensity = voltage / max(electrode_voltages) if max(electrode_voltages) > 0 else 0
            color = f"rgb({int(255 * (1 - color_intensity))}, 0, {int(255 * color_intensity)})"
            
            fig.add_shape(
                type="rect",
                x0=pos-2, y0=-tube_width_mm/2-5,
                x1=pos+2, y1=tube_width_mm/2+5,
                fillcolor=color,
                opacity=0.7,
                line=dict(color="black", width=1)
            )
            
            fig.add_annotation(
                x=pos, y=-tube_width_mm/2-10,
                text=f"{voltage:.0f}V",
                showarrow=False,
                font=dict(size=10)
            )
    
    # Add trajectories
    if trajectories and ions:
        colors = px.colors.qualitative.Set1
        
        for i, (traj, ion) in enumerate(zip(trajectories, ions)):
            t, z = traj[:, 0], traj[:, 1]
            x_pos = z * 1000
            y_pos = np.random.normal(0, 2, len(t))
            
            fig.add_trace(go.Scatter(
                x=x_pos, y=y_pos,
                mode='lines+markers',
                name=f"{ion.name}",
                line=dict(color=colors[i % len(colors)], width=4),
                marker=dict(size=8),
                hovertemplate=f"<b>{ion.name}</b><br>" +
                             "Position: %{x:.1f} mm<br>" +
                             "Time: %{text:.2f} ms<br>" +
                             "<extra></extra>",
                text=t * 1000
            ))
    
    fig.update_layout(
        title=f"Combined IMS Visualization<br>L={tube.length_m:.3f}m, V={tube.voltage_V:.0f}V",
        xaxis_title="Position (mm)",
        yaxis_title="Position (mm)",
        width=900,
        height=600,
        template="simple_white"
    )
    
    return fig

