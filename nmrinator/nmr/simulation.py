"""1D NMR line shape simulation, multiplet splitting, and integral trace calculation."""

import numpy as np
from typing import List, Tuple, Optional

from nmrinator.models import NMRSignal, NMRSpectrum


def get_multiplet_components(
    delta: float,
    multiplicity: str,
    coupling_j: List[float],
    integral: int,
    spectrometer_mhz: float
) -> List[Tuple[float, float]]:
    """Splits a signal into individual multiplet resonance lines: list of (ppm_position, line_intensity).
    
    Coupling J is in Hz; converted to ppm delta via: delta_ppm = J_hz / spectrometer_mhz.
    """
    m = multiplicity.lower().strip()
    j1 = coupling_j[0] if coupling_j else 7.0
    j2 = coupling_j[1] if len(coupling_j) > 1 else (coupling_j[0] if coupling_j else 7.0)
    
    # Conversion from Hz to ppm
    shift_j1 = j1 / spectrometer_mhz
    shift_j2 = j2 / spectrometer_mhz
    
    if m == "s" or not coupling_j or m == "":
        return [(delta, float(integral))]
    elif m == "d":
        return [
            (delta - shift_j1 / 2.0, integral * 0.5),
            (delta + shift_j1 / 2.0, integral * 0.5)
        ]
    elif m == "t":
        return [
            (delta - shift_j1, integral * 0.25),
            (delta, integral * 0.5),
            (delta + shift_j1, integral * 0.25)
        ]
    elif m == "q":
        return [
            (delta - 1.5 * shift_j1, integral * 0.125),
            (delta - 0.5 * shift_j1, integral * 0.375),
            (delta + 0.5 * shift_j1, integral * 0.375),
            (delta + 1.5 * shift_j1, integral * 0.125)
        ]
    elif m in ["dd", "double doublet"]:
        # 4 lines of equal intensity
        return [
            (delta - (shift_j1 + shift_j2) / 2.0, integral * 0.25),
            (delta - (shift_j1 - shift_j2) / 2.0, integral * 0.25),
            (delta + (shift_j1 - shift_j2) / 2.0, integral * 0.25),
            (delta + (shift_j1 + shift_j2) / 2.0, integral * 0.25)
        ]
    elif m in ["quin", "quintet"]:
        # 1:4:6:4:1
        weights = np.array([1, 4, 6, 4, 1], dtype=np.float64)
        weights = weights / np.sum(weights) * integral
        offsets = np.array([-2, -1, 0, 1, 2]) * shift_j1
        return [(delta + off, w) for off, w in zip(offsets, weights)]
    elif m in ["sex", "sextet"]:
        # 1:5:10:10:5:1
        weights = np.array([1, 5, 10, 10, 5, 1], dtype=np.float64)
        weights = weights / np.sum(weights) * integral
        offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * shift_j1
        return [(delta + off, w) for off, w in zip(offsets, weights)]
    elif m in ["sept", "septet"]:
        # 1:6:15:20:15:6:1
        weights = np.array([1, 6, 15, 20, 15, 6, 1], dtype=np.float64)
        weights = weights / np.sum(weights) * integral
        offsets = np.array([-3, -2, -1, 0, 1, 2, 3]) * shift_j1
        return [(delta + off, w) for off, w in zip(offsets, weights)]
    else:
        # Generic complex multiplet (m) modeled as a 5-line envelope
        weights = np.array([1, 3, 4, 3, 1], dtype=np.float64)
        weights = weights / np.sum(weights) * integral
        offsets = np.array([-2, -1, 0, 1, 2]) * shift_j1
        return [(delta + off, w) for off, w in zip(offsets, weights)]


def simulate_nmr_spectrum(
    spectrum: NMRSpectrum,
    ppm_min: Optional[float] = None,
    ppm_max: Optional[float] = None,
    step: Optional[float] = None,
    linewidth_hz: float = 1.0
) -> NMRSpectrum:
    """Computes continuous Lorentzian lineshape and integral curve for an NMR spectrum."""
    is_1h = spectrum.nucleus.upper() in ["1H", "H"]
    
    # Set default ppm grid if not specified
    if ppm_min is None:
        ppm_min = -0.5 if is_1h else -5.0
    if ppm_max is None:
        ppm_max = 14.0 if is_1h else 220.0
    if step is None:
        step = 0.005 if is_1h else 0.05
        
    ppm_axis = np.arange(ppm_min, ppm_max + step, step, dtype=np.float64)
    intensity_axis = np.zeros_like(ppm_axis)
    
    # Line width FWHM in ppm
    gamma = (linewidth_hz / spectrum.spectrometer_mhz) / 2.0
    half_gamma_sq = gamma ** 2
    
    for sig in spectrum.signals:
        lines = get_multiplet_components(
            delta=sig.delta,
            multiplicity=sig.multiplicity,
            coupling_j=sig.coupling_j,
            integral=sig.integral,
            spectrometer_mhz=spectrum.spectrometer_mhz
        )
        for pos_ppm, line_area in lines:
            # Normalized Lorentzian peak: area * (1/pi) * gamma / ((x - x0)^2 + gamma^2)
            intensity_axis += line_area * (1.0 / np.pi) * (gamma / ((ppm_axis - pos_ppm) ** 2 + half_gamma_sq))
            
    # Calculate integral trace (integrated from high field / right to low field / left)
    # Total proton count
    total_integral = sum(s.integral for s in spectrum.signals)
    integral_curve = np.cumsum(intensity_axis) * step
    max_integral = integral_curve[-1] if len(integral_curve) > 0 and integral_curve[-1] > 0 else 1.0
    integral_axis = (integral_curve / max_integral) * total_integral
    
    spectrum.ppm_axis = ppm_axis.tolist()
    spectrum.intensity_axis = intensity_axis.tolist()
    spectrum.integral_axis = integral_axis.tolist()
    spectrum.linewidth_hz = linewidth_hz
    
    return spectrum

