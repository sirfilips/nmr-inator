"""Interactive Plotly charts for 1H and 13C NMR spectra."""

import plotly.graph_objects as go
from typing import Optional, List
import numpy as np

from nmrinator.models import NMRSpectrum, NMRSignal


def create_nmr_plot(
    spectrum: NMRSpectrum,
    title: Optional[str] = None,
    show_integral: bool = True,
    show_sticks: bool = True
) -> go.Figure:
    """Generates an interactive publication-quality NMR plot with inverted ppm axis."""
    is_1h = spectrum.nucleus.upper() in ["1H", "H"]
    fig = go.Figure()
    
    default_title = f"Spettro 1D NMR ({spectrum.nucleus}) @ {spectrum.spectrometer_mhz:.0f} MHz"
    plot_title = title or default_title
    
    # 1. Plot continuous simulated Lorentzian spectrum
    if spectrum.ppm_axis and spectrum.intensity_axis:
        fig.add_trace(go.Scatter(
            x=spectrum.ppm_axis,
            y=spectrum.intensity_axis,
            mode="lines",
            name="Spettro Simulato",
            line=dict(color="#1f77b4" if is_1h else "#2ca02c", width=1.8),
            hoverinfo="x+y"
        ))
        
    # 2. Add Integral Trace (for 1H NMR)
    if is_1h and show_integral and spectrum.ppm_axis and spectrum.integral_axis:
        # Scale integral line so it fits elegantly on the top half of the plot
        max_intens = max(spectrum.intensity_axis) if spectrum.intensity_axis else 1.0
        max_integral = max(spectrum.integral_axis) if spectrum.integral_axis else 1.0
        scaled_integral = [y * (max_intens * 0.45 / max_integral) + (max_intens * 0.55) for y in spectrum.integral_axis]
        
        fig.add_trace(go.Scatter(
            x=spectrum.ppm_axis,
            y=scaled_integral,
            mode="lines",
            name="Curva di Integrazione",
            line=dict(color="#d62728", width=1.2, dash="dot"),
            hoverinfo="x+y"
        ))
        
    # 3. Add discrete stick peaks and labels
    if show_sticks and spectrum.signals:
        max_y = max(spectrum.intensity_axis) if (spectrum.intensity_axis and len(spectrum.intensity_axis) > 0) else 10.0
        stick_x = []
        stick_y = []
        hover_texts = []
        
        for sig in spectrum.signals:
            stick_x.extend([sig.delta, sig.delta, None])
            peak_height = max_y * 0.85 * (sig.integral / max(s.integral for s in spectrum.signals))
            stick_y.extend([0, peak_height, None])
            
            j_str = ", ".join(f"{j:.1f}" for j in sig.coupling_j) if sig.coupling_j else "N/A"
            txt = (
                f"<b>δ:</b> {sig.delta:.2f} ppm<br>"
                f"<b>Nucleo:</b> {sig.nucleus}<br>"
                f"<b>Molteplicità:</b> {sig.multiplicity}<br>"
                f"<b>J (Hz):</b> {j_str}<br>"
                f"<b>Integrazione:</b> {sig.integral}H<br>"
                f"<b>Assegnazione:</b> {sig.label}"
            )
            hover_texts.extend([txt, txt, None])
            
            # Add text annotation above peak
            fig.add_annotation(
                x=sig.delta,
                y=peak_height * 1.05,
                text=f"{sig.delta:.2f}",
                showarrow=False,
                font=dict(size=10, color="#333333"),
                yshift=5
            )
            
        fig.add_trace(go.Scatter(
            x=stick_x,
            y=stick_y,
            mode="lines",
            name="Picchi (Stick)",
            line=dict(color="#7f7f7f", width=1, dash="dash"),
            hoverinfo="text",
            hovertext=hover_texts
        ))
        
    # Range limits
    if is_1h:
        x_min = -0.5
        x_max = 13.0
    else:
        x_min = -5.0
        x_max = 215.0
        
    # Standard NMR layout: Inverted x-axis (ppm runs from right to left!)
    fig.update_layout(
        title=dict(text=plot_title, font=dict(size=16, family="Arial, sans-serif")),
        xaxis=dict(
            title="Chemical Shift δ (ppm)",
            autorange="reversed",  # NMR convention: 0 on right, higher ppm on left
            range=[x_max, x_min],
            showgrid=True,
            gridcolor="#e5e5e5",
            zeroline=True,
            zerolinecolor="#999999"
        ),
        yaxis=dict(
            title="Intensità (U.A.)",
            showgrid=True,
            gridcolor="#f0f0f0",
            showticklabels=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        margin=dict(l=40, r=40, t=70, b=40),
        hovermode="closest"
    )
    
    return fig

