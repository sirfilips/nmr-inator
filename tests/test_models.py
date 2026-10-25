"""Tests for spectroscopy data models and JSON serialization."""

import pytest
from nmrinator.models import (
    NMRSignal,
    NMRSpectrum,
    IRMode,
    IRSpectrum,
    ConformerSpectralData,
    SpectraResult
)


def test_nmr_signal_and_spectrum():
    sig = NMRSignal(
        nucleus="1H",
        delta=1.23,
        multiplicity="t",
        coupling_j=[7.1],
        integral=3,
        atom_indices=[1, 2, 3],
        label="Methyl -CH3"
    )
    assert sig.delta == 1.23
    assert sig.integral == 3
    assert sig.to_dict()["multiplicity"] == "t"
    
    spec = NMRSpectrum(nucleus="1H", spectrometer_mhz=400.0, signals=[sig])
    d = spec.to_dict()
    assert d["nucleus"] == "1H"
    assert len(d["signals"]) == 1


def test_ir_mode_and_spectrum():
    mode = IRMode(
        mode_index=1,
        frequency_calc=1750.0,
        frequency_scaled=1690.0,
        intensity=120.5,
        is_imaginary=False,
        assignment="C=O stretch"
    )
    spec = IRSpectrum(modes=[mode], scaling_factor=0.965)
    assert len(spec.modes) == 1
    assert spec.modes[0].assignment == "C=O stretch"


def test_spectra_result_serialization(tmp_path):
    sig = NMRSignal(nucleus="1H", delta=2.1, multiplicity="s", integral=3)
    spec = NMRSpectrum(nucleus="1H", signals=[sig])
    conf = ConformerSpectralData(conf_id=1, energy_kcalmol=0.5, boltzmann_pop=100.0, nmr_1h=spec)
    
    res = SpectraResult(
        smiles="CC(=O)C",
        mode="single",
        conformers=[conf],
        ensemble_nmr_1h=spec
    )
    
    json_file = str(tmp_path / "test_result.json")
    res.save_json(json_file)
    
    # Reload
    reloaded = SpectraResult.from_dict(res.to_dict())
    assert reloaded.smiles == "CC(=O)C"
    assert len(reloaded.conformers) == 1
    assert reloaded.ensemble_nmr_1h.signals[0].delta == 2.1

