"""Line shape broadening and IR spectra convolution (Absorbance and % Transmittance)."""

import numpy as np
from typing import List, Tuple, Optional

from nmrinator.models import IRMode, IRSpectrum
from nmrinator.ir.bands import annotate_modes


def lorentzian(x: np.ndarray, x0: float, gamma: float) -> np.ndarray:
    """Calculates a normalized Lorentzian peak function: L(x; x0, gamma) = (1/pi) * (gamma/2) / ((x - x0)^2 + (gamma/2)^2)."""
    half_gamma = gamma / 2.0
    return (1.0 / np.pi) * (half_gamma / ((x - x0) ** 2 + half_gamma ** 2))


def simulate_ir_spectrum(
    modes: List[IRMode],
    scaling_factor: float = 0.965,
    fwhm: float = 12.0,
    wavenumber_min: float = 400.0,
    wavenumber_max: float = 4000.0,
    step: float = 2.0
) -> IRSpectrum:
    """Convolves discrete vibrational modes into continuous Absorbance and % Transmittance IR curves."""
    annotated = annotate_modes(modes)
    
    # Generate wavenumber grid (400 to 4000 cm^-1)
    wavenumbers = np.arange(wavenumber_min, wavenumber_max + step, step, dtype=np.float64)
    absorbance = np.zeros_like(wavenumbers)
    
    for m in annotated:
        if m.is_imaginary:
            continue
        freq = m.frequency_scaled
        intens = m.intensity
        if wavenumber_min - 50 <= freq <= wavenumber_max + 50 and intens > 0.01:
            absorbance += intens * lorentzian(wavenumbers, freq, fwhm)
            
    # Normalize Absorbance and calculate % Transmittance via Beer-Lambert
    max_abs = np.max(absorbance) if len(absorbance) > 0 else 0.0
    if max_abs > 1e-6:
        # Scale so strongest peak produces ~15% Transmittance (typical clean IR spectrum)
        # T = 100 * 10^(-k * A) -> log10(0.15) = -k * max_abs -> k = 0.824 / max_abs
        k_factor = 0.824 / max_abs
        transmittance = 100.0 * (10.0 ** (-k_factor * absorbance))
    else:
        transmittance = np.full_like(wavenumbers, 100.0)
        
    return IRSpectrum(
        modes=annotated,
        scaling_factor=scaling_factor,
        fwhm=fwhm,
        wavenumber_axis=wavenumbers.tolist(),
        absorbance_axis=absorbance.tolist(),
        transmittance_axis=transmittance.tolist()
    )

