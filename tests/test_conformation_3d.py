"""Tests for 3D dihedral calculations, Karplus equations, and ring current shielding."""

import numpy as np
import pytest
from nmrinator.conformation_3d import (
    compute_dihedral_angle,
    karplus_coupling
)


def test_dihedral_calculation():
    # Planar trans (anti-periplanar): dihedral = 180 deg
    p1 = np.array([1.0, 1.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 0.0, 0.0])
    p4 = np.array([-1.0, 0.0, 0.0])
    angle = compute_dihedral_angle(p1, p2, p3, p4)
    assert abs(abs(angle) - 180.0) < 1e-4

    # Cis (syn-periplanar): dihedral = 0 deg
    p4_cis = np.array([1.0, 0.0, 0.0])
    angle_cis = compute_dihedral_angle(p1, p2, p3, p4_cis)
    assert abs(angle_cis) < 1e-4


def test_karplus_coupling():
    # Anti (180 deg) gives largest coupling (~9-11 Hz)
    j_anti = karplus_coupling(180.0)
    assert 9.0 <= j_anti <= 12.0
    
    # Orthogonal (90 deg) gives small coupling (~1.0 - 2.5 Hz)
    j_ortho = karplus_coupling(90.0)
    assert 1.0 <= j_ortho <= 2.5
    
    # Cis (0 deg) gives intermediate coupling (~7.0 - 9.0 Hz)
    j_cis = karplus_coupling(0.0)
    assert 7.0 <= j_cis <= 9.5

