"""Tests for IR scaling factors, band assignments, and line shape convolution."""

import pytest
import numpy as np
from nmrinator.ir.scaling import get_scaling_factor
from nmrinator.ir.bands import assign_ir_band
from nmrinator.ir.convolution import simulate_ir_spectrum
from nmrinator.models import IRMode


def test_scaling_factors():
    # B3LYP 6-31G(d) scaling factor
    s1 = get_scaling_factor("b3lyp", "6-31g(d)")
    assert 0.95 <= s1 <= 0.97
    
    # HF scaling factor
    s_hf = get_scaling_factor("hf", "6-31g(d)")
    assert 0.88 <= s_hf <= 0.91


def test_band_assignments():
    desc_co = assign_ir_band(1715.0, 150.0)
    assert "C=O" in desc_co

    desc_oh = assign_ir_band(3600.0, 80.0)
    assert "O-H" in desc_oh


def test_simulate_ir_spectrum():
    mode = IRMode(
        mode_index=1,
        frequency_calc=1750.0,
        frequency_scaled=1700.0,
        intensity=100.0
    )
    spec = simulate_ir_spectrum([mode], wavenumber_min=400.0, wavenumber_max=4000.0, step=2.0)
    assert spec.wavenumber_axis is not None
    assert spec.transmittance_axis is not None
    assert len(spec.wavenumber_axis) == len(spec.transmittance_axis)
    # The transmittance minimum (%T dip) should occur near 1700 cm^-1
    min_idx = np.argmin(spec.transmittance_axis)
    min_freq = spec.wavenumber_axis[min_idx]
    assert abs(min_freq - 1700.0) < 5.0
    # Minimum transmittance should dip down significantly
    assert spec.transmittance_axis[min_idx] < 30.0

