"""Tests for I/O loaders, XYZ converters, and JCAMP-DX export."""

import pytest
import os
from nmrinator.io import load_molecule, mol_to_xyz, parse_xyz, export_to_jcamp_dx


def test_load_molecule_smiles():
    mol, xyz_str, smiles = load_molecule(smiles="CC")
    assert mol.GetNumAtoms() == 8  # 2 C + 6 H
    assert smiles == "CC"
    symbols, coords = parse_xyz(xyz_str)
    assert len(symbols) == 8
    assert symbols.count("C") == 2
    assert symbols.count("H") == 6


def test_export_to_jcamp_dx():
    x = [1.0, 2.0, 3.0]
    y = [10.0, 50.0, 5.0]
    jdx = export_to_jcamp_dx(
        spectrum_type="NMR",
        title="Test NMR",
        x_axis=x,
        y_axis=y,
        nucleus="1H",
        frequency_mhz=400.0
    )
    assert "##TITLE= Test NMR" in jdx
    assert "##JCAMP-DX= 5.01" in jdx
    assert "##.OBSERVE NUCLEUS= 1H" in jdx
    assert "##NPOINTS= 3" in jdx
    assert "##END=" in jdx


def test_session_discovery_and_restoration(tmp_path):
    from gui.app_gui import get_available_sessions
    from nmrinator.models import SpectraResult
    import json
    
    out_base = str(tmp_path / "output")
    sess_dir = os.path.join(out_base, "session_20261025_120000_C2H6O")
    os.makedirs(sess_dir, exist_ok=True)
    
    # Create fake nmrinator_result.json
    res_data = {
        "smiles": "CCO",
        "mode": "full",
        "spectrometer_mhz": 400.0,
        "psi4_method": "b3lyp",
        "psi4_basis": "6-31G(d)",
        "solvent": "gas",
        "elapsed_time": 3.2,
        "conformers": []
    }
    with open(os.path.join(sess_dir, "nmrinator_result.json"), "w", encoding="utf-8") as f:
        json.dump(res_data, f)
        
    sessions = get_available_sessions(out_base)
    assert len(sessions) == 1
    assert sessions[0]["dir_name"] == "session_20261025_120000_C2H6O"
    assert "CCO" in sessions[0]["label"]
    
    # Test reconstruction
    loaded = SpectraResult.from_json(os.path.join(sess_dir, "nmrinator_result.json"))
    assert loaded.smiles == "CCO"
    assert loaded.spectrometer_mhz == 400.0

