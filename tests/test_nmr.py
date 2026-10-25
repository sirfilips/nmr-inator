"""Tests for NMR simulation, multiplet splitting, and line shapes."""

import pytest
import numpy as np
from nmrinator.models import NMRSignal, NMRSpectrum
from nmrinator.nmr.simulation import get_multiplet_components, simulate_nmr_spectrum


def test_multiplet_components():
    # Singlet
    lines_s = get_multiplet_components(delta=2.0, multiplicity="s", coupling_j=[], integral=1, spectrometer_mhz=400.0)
    assert len(lines_s) == 1
    assert lines_s[0][0] == 2.0
    assert lines_s[0][1] == 1.0

    # Doublet (J = 8 Hz @ 400 MHz -> shift = 0.02 ppm -> lines at 1.99 and 2.01 ppm)
    lines_d = get_multiplet_components(delta=2.0, multiplicity="d", coupling_j=[8.0], integral=2, spectrometer_mhz=400.0)
    assert len(lines_d) == 2
    assert abs(lines_d[0][0] - 1.99) < 1e-4
    assert abs(lines_d[1][0] - 2.01) < 1e-4
    assert lines_d[0][1] == 1.0
    assert lines_d[1][1] == 1.0

    # Triplet
    lines_t = get_multiplet_components(delta=3.0, multiplicity="t", coupling_j=[8.0], integral=3, spectrometer_mhz=400.0)
    assert len(lines_t) == 3


def test_simulate_nmr_spectrum():
    sig = NMRSignal(nucleus="1H", delta=1.5, multiplicity="d", coupling_j=[7.0], integral=2)
    spec = NMRSpectrum(nucleus="1H", spectrometer_mhz=400.0, signals=[sig])
    sim = simulate_nmr_spectrum(spec, ppm_min=0.0, ppm_max=4.0, step=0.01)
    
    assert sim.ppm_axis is not None
    assert sim.intensity_axis is not None
    assert len(sim.ppm_axis) == len(sim.intensity_axis)
    # Peak intensity maximum should be near 1.5 ppm
    max_idx = np.argmax(sim.intensity_axis)
    max_ppm = sim.ppm_axis[max_idx]
    assert abs(max_ppm - 1.5) < 0.05

