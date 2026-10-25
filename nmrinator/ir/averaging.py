"""Boltzmann ensemble averaging for Infrared (IR) spectra."""

from typing import List, Optional
import numpy as np

from nmrinator.models import ConformerSpectralData, IRSpectrum, IRMode
from nmrinator.ir.convolution import simulate_ir_spectrum


def compute_ensemble_ir(
    conformers: List[ConformerSpectralData],
    scaling_factor: float = 0.965,
    fwhm: float = 12.0
) -> Optional[IRSpectrum]:
    """Computes the Boltzmann-weighted ensemble IR spectrum as a linear superposition of conformer spectra."""
    valid_confs = [c for c in conformers if c.ir and c.ir.absorbance_axis and len(c.ir.absorbance_axis) > 0]
    if not valid_confs:
        return None
        
    total_pop = sum(c.boltzmann_pop for c in valid_confs)
    if total_pop <= 1e-6:
        total_pop = 100.0
        for c in valid_confs:
            c.boltzmann_pop = 100.0 / len(valid_confs)
            
    # Reference wavenumber axis from first valid conformer
    ref_wavenumbers = np.array(valid_confs[0].ir.wavenumber_axis, dtype=np.float64)
    ensemble_absorbance = np.zeros_like(ref_wavenumbers)
    
    all_weighted_modes: List[IRMode] = []
    
    for c in valid_confs:
        weight = (c.boltzmann_pop / total_pop)
        c_abs = np.array(c.ir.absorbance_axis, dtype=np.float64)
        if len(c_abs) == len(ensemble_absorbance):
            ensemble_absorbance += weight * c_abs
            
        for m in c.ir.modes:
            # Scale mode intensity by Boltzmann population
            weighted_mode = IRMode(
                mode_index=m.mode_index,
                frequency_calc=m.frequency_calc,
                frequency_scaled=m.frequency_scaled,
                intensity=round(m.intensity * weight, 3),
                is_imaginary=m.is_imaginary,
                symmetry=m.symmetry,
                assignment=m.assignment
            )
            all_weighted_modes.append(weighted_mode)
            
    # Compute ensemble % Transmittance
    max_abs = np.max(ensemble_absorbance) if len(ensemble_absorbance) > 0 else 0.0
    if max_abs > 1e-6:
        k_factor = 0.824 / max_abs
        transmittance = 100.0 * (10.0 ** (-k_factor * ensemble_absorbance))
    else:
        transmittance = np.full_like(ref_wavenumbers, 100.0)
        
    # Sort modes by frequency descending
    all_weighted_modes.sort(key=lambda m: m.frequency_scaled, reverse=True)
    
    return IRSpectrum(
        modes=all_weighted_modes,
        scaling_factor=scaling_factor,
        fwhm=fwhm,
        wavenumber_axis=ref_wavenumbers.tolist(),
        absorbance_axis=ensemble_absorbance.tolist(),
        transmittance_axis=transmittance.tolist()
    )

