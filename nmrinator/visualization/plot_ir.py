"""Interactive Plotly charts for Infrared (IR) vibrational spectra."""

import plotly.graph_objects as go
from typing import Optional, List
import numpy as np

from nmrinator.models import IRSpectrum, IRMode


def create_ir_plot(
    spectrum: IRSpectrum,
    title: Optional[str] = None,
    plot_type: str = "transmittance",  # "transmittance" (%T) or "absorbance"
    show_band_labels: bool = True
) -> go.Figure:
    """Generates an interactive IR spectrum with standard inverted wavenumber axis (4000 to 400 cm^-1)."""
    fig = go.Figure()
    
    is_transmittance = plot_type.lower() == "transmittance"
    default_title = f"Spettro Infrarosso (IR) - {'Trasmittanza %T' if is_transmittance else 'Assorbanza'}"
    plot_title = title or default_title
    
    if spectrum.wavenumber_axis:
        x_vals = spectrum.wavenumber_axis
        y_vals = spectrum.transmittance_axis if is_transmittance else spectrum.absorbance_axis
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name="% Trasmittanza" if is_transmittance else "Assorbanza",
            line=dict(color="#d62728" if is_transmittance else "#1f77b4", width=1.8),
            hoverinfo="x+y"
        ))
        
    # Annotate significant functional group bands (intensity > 15 km/mol)
    if show_band_labels and spectrum.modes:
        strong_modes = [m for m in spectrum.modes if m.intensity > 15.0 and not m.is_imaginary]
        # Keep top 8 strongest modes to avoid visual clutter
        strong_modes.sort(key=lambda m: m.intensity, reverse=True)
        top_modes = strong_modes[:8]
        
        for m in top_modes:
            freq = m.frequency_scaled
            if is_transmittance:
                # Find corresponding %T near this frequency
                y_val = 30.0
                if spectrum.wavenumber_axis and spectrum.transmittance_axis:
                    idx = int(np.argmin(np.abs(np.array(spectrum.wavenumber_axis) - freq)))
                    y_val = spectrum.transmittance_axis[idx]
                
                fig.add_annotation(
                    x=freq,
                    y=y_val,
                    text=f"<b>{freq:.0f} cm⁻¹</b><br>{m.assignment or ''}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="#555555",
                    ax=0,
                    ay=35,
                    font=dict(size=9, color="#222222"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="#cccccc",
                    borderwidth=1
                )
            else:
                y_val = 0.5
                if spectrum.wavenumber_axis and spectrum.absorbance_axis:
                    idx = int(np.argmin(np.abs(np.array(spectrum.wavenumber_axis) - freq)))
                    y_val = spectrum.absorbance_axis[idx]
                    
                fig.add_annotation(
                    x=freq,
                    y=y_val,
                    text=f"<b>{freq:.0f} cm⁻¹</b><br>{m.assignment or ''}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="#555555",
                    ax=0,
                    ay=-35,
                    font=dict(size=9, color="#222222"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="#cccccc",
                    borderwidth=1
                )
                
    # Standard IR layout: Inverted x-axis (4000 on left, 400 on right)
    y_axis_config = dict(
        title="% Trasmittanza (%T)" if is_transmittance else "Assorbanza (U.A.)",
        showgrid=True,
        gridcolor="#f0f0f0"
    )
    if is_transmittance:
        y_axis_config["range"] = [0, 105]  # Standard %T view from 0% to 105%
        
    fig.update_layout(
        title=dict(text=plot_title, font=dict(size=16, family="Arial, sans-serif")),
        xaxis=dict(
            title="Numero d'onda (cm⁻¹)",
            autorange="reversed",  # IR convention: 4000 cm^-1 on left, 400 cm^-1 on right
            range=[4000, 400],
            showgrid=True,
            gridcolor="#e5e5e5",
            zeroline=False
        ),
        yaxis=y_axis_config,
        template="plotly_white",
        margin=dict(l=50, r=40, t=70, b=40),
        hovermode="closest"
    )
    
    return fig

