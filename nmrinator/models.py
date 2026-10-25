"""Data models and serialization structures for NMR and IR spectroscopy."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json
import numpy as np


@dataclass
class NMRSignal:
    """Represents a discrete NMR peak or multiplet signal."""
    nucleus: str  # "1H", "13C", etc.
    delta: float  # Chemical shift in ppm
    multiplicity: str = "s"  # "s", "d", "t", "q", "m", "dd", etc.
    coupling_j: List[float] = field(default_factory=list)  # J couplings in Hz
    integral: int = 1  # Number of equivalent protons/carbons
    atom_indices: List[int] = field(default_factory=list)  # RDKit/XYZ atom indices
    label: str = ""  # Structural assignment label, e.g. "CH3", "C=O"
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NMRSpectrum:
    """Represents a complete 1D NMR spectrum (discrete signals and continuous simulated line)."""
    nucleus: str  # "1H" or "13C"
    spectrometer_mhz: float = 400.0  # Spectrometer Larmor frequency in MHz
    signals: List[NMRSignal] = field(default_factory=list)
    
    # Continuous simulated spectrum arrays (populated by simulation module)
    ppm_axis: Optional[List[float]] = None
    intensity_axis: Optional[List[float]] = None
    integral_axis: Optional[List[float]] = None
    linewidth_hz: float = 1.0  # Full width at half maximum (FWHM) in Hz

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IRMode:
    """Represents a single vibrational normal mode."""
    mode_index: int
    frequency_calc: float  # Unscaled harmonic frequency in cm^-1
    frequency_scaled: float  # Anharmonic scaled frequency in cm^-1
    intensity: float  # IR intensity (km/mol)
    is_imaginary: bool = False  # True if transition state (imaginary frequency)
    symmetry: Optional[str] = None
    assignment: Optional[str] = None  # e.g. "C=O stretch (Carbonyl)"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IRSpectrum:
    """Represents a complete infrared vibrational spectrum."""
    modes: List[IRMode] = field(default_factory=list)
    scaling_factor: float = 0.965  # Scaling factor applied to frequencies
    fwhm: float = 12.0  # Lorentzian FWHM in cm^-1
    
    # Continuous simulated curves (e.g. 400 to 4000 cm^-1)
    wavenumber_axis: Optional[List[float]] = None
    absorbance_axis: Optional[List[float]] = None
    transmittance_axis: Optional[List[float]] = None  # %T (0 to 100%)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConformerSpectralData:
    """Stores individual spectroscopic predictions for a single conformer."""
    conf_id: int
    energy_kcalmol: float = 0.0
    boltzmann_pop: float = 100.0  # in %
    xyz: str = ""
    nmr_1h: Optional[NMRSpectrum] = None
    nmr_13c: Optional[NMRSpectrum] = None
    ir: Optional[IRSpectrum] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        return res


@dataclass
class SpectraResult:
    """Comprehensive summary of calculated NMR and IR spectra."""
    smiles: Optional[str] = None
    mode: str = "full"  # "full" (ensemble Boltzmann) or "single" (single conformer)
    temperature_k: float = 298.15
    spectrometer_mhz: float = 400.0
    psi4_method: str = "b3lyp"
    psi4_basis: str = "6-31G(d)"
    solvent: Optional[str] = None
    ir_scaling_factor: float = 0.965
    
    # Individual conformers data
    conformers: List[ConformerSpectralData] = field(default_factory=list)
    
    # Boltzmann ensemble-averaged spectra
    ensemble_nmr_1h: Optional[NMRSpectrum] = None
    ensemble_nmr_13c: Optional[NMRSpectrum] = None
    ensemble_ir: Optional[IRSpectrum] = None
    
    elapsed_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, filepath: str) -> None:
        """Saves result as JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpectraResult":
        """Reconstructs SpectraResult from dictionary."""
        d = dict(data)
        confs_raw = d.pop("conformers", [])
        
        conformers = []
        for c in confs_raw:
            c_copy = dict(c)
            nmr_1h_data = c_copy.pop("nmr_1h", None)
            nmr_13c_data = c_copy.pop("nmr_13c", None)
            ir_data = c_copy.pop("ir", None)
            
            nmr_1h = None
            if nmr_1h_data:
                signals = [NMRSignal(**s) for s in nmr_1h_data.pop("signals", [])]
                nmr_1h = NMRSpectrum(signals=signals, **nmr_1h_data)
                
            nmr_13c = None
            if nmr_13c_data:
                signals = [NMRSignal(**s) for s in nmr_13c_data.pop("signals", [])]
                nmr_13c = NMRSpectrum(signals=signals, **nmr_13c_data)
                
            ir = None
            if ir_data:
                modes = [IRMode(**m) for m in ir_data.pop("modes", [])]
                ir = IRSpectrum(modes=modes, **ir_data)
                
            conformers.append(ConformerSpectralData(
                nmr_1h=nmr_1h,
                nmr_13c=nmr_13c,
                ir=ir,
                **c_copy
            ))
            
        ens_1h_data = d.pop("ensemble_nmr_1h", None)
        ensemble_nmr_1h = None
        if ens_1h_data:
            sig = [NMRSignal(**s) for s in ens_1h_data.pop("signals", [])]
            ensemble_nmr_1h = NMRSpectrum(signals=sig, **ens_1h_data)
            
        ens_13c_data = d.pop("ensemble_nmr_13c", None)
        ensemble_nmr_13c = None
        if ens_13c_data:
            sig = [NMRSignal(**s) for s in ens_13c_data.pop("signals", [])]
            ensemble_nmr_13c = NMRSpectrum(signals=sig, **ens_13c_data)
            
        ens_ir_data = d.pop("ensemble_ir", None)
        ensemble_ir = None
        if ens_ir_data:
            mod = [IRMode(**m) for m in ens_ir_data.pop("modes", [])]
            ensemble_ir = IRSpectrum(modes=mod, **ens_ir_data)

        return cls(
            conformers=conformers,
            ensemble_nmr_1h=ensemble_nmr_1h,
            ensemble_nmr_13c=ensemble_nmr_13c,
            ensemble_ir=ensemble_ir,
            **d
        )

    @classmethod
    def from_json(cls, filepath: str) -> "SpectraResult":
        """Loads and reconstructs a SpectraResult from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
