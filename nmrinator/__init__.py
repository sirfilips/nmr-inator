"""nmr-inator: Quantum and Ensemble-Averaged NMR and IR Spectroscopy Suite."""

from nmrinator.config import SpectraConfig
from nmrinator.models import (
    NMRSignal,
    NMRSpectrum,
    IRMode,
    IRSpectrum,
    ConformerSpectralData,
    SpectraResult
)
from nmrinator.pipeline import run_nmrinator
from nmrinator import io, symmetry, conformation_3d, quantum, nmr, ir, visualization

__version__ = "0.1.0"

__all__ = [
    "SpectraConfig",
    "NMRSignal",
    "NMRSpectrum",
    "IRMode",
    "IRSpectrum",
    "ConformerSpectralData",
    "SpectraResult",
    "run_nmrinator",
    "io",
    "symmetry",
    "conformation_3d",
    "quantum",
    "nmr",
    "ir",
    "visualization"
]

