"""Configuration settings for NMR and IR spectroscopic calculation pipeline."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SpectraConfig:
    """Configuration class for nmr-inator calculation pipeline."""
    # Pipeline execution mode: "full" (ensemble Boltzmann) or "single" (single conformer)
    mode: str = "full"
    
    # Input options (at least one must be provided)
    smiles: Optional[str] = None
    xyz_path: Optional[str] = None
    mol_path: Optional[str] = None
    mol_block: Optional[str] = None
    
    # Spectroscopic modules activation
    calculate_1h_nmr: bool = True
    calculate_13c_nmr: bool = True
    calculate_ir: bool = True
    
    # NMR parameters
    spectrometer_mhz: float = 400.0  # 300, 400, 500, 600, etc.
    nmr_linewidth_hz: float = 1.2    # Lorentzian line width FWHM for simulated peak
    apply_3d_corrections: bool = True  # Karplus dihedral J-coupling and ring current anisotropy
    
    # IR parameters
    psi4_method: str = "b3lyp"
    psi4_basis: str = "6-31G(d)"
    solvent: Optional[str] = None  # None/gas, water, chloroform, dmso, methanol, etc.
    ir_scaling_factor: Optional[float] = None  # None = auto-lookup according to method/basis
    ir_fwhm: float = 12.0  # FWHM for IR Lorentzian broadening in cm^-1
    
    # Conformer Search settings (used when mode == "full")
    n_conformers: int = 30
    rmsd_threshold: float = 0.2
    force_field: str = "mmff94"
    use_xtb: bool = False
    top_n_confs: int = 5  # Top N conformers to evaluate quantum IR/DFT properties
    min_boltzmann_pop: float = 1.0  # Minimum % population to include in IR computation
    temperature_k: float = 298.15
    
    # Single conformer optimization option
    optimize_single_conformer: bool = True  # If True, optimizes input geometry with Psi4 or FF before spectra
    
    # Computational resources
    psi4_memory: str = "8 GB"
    psi4_threads: int = 8
    
    # Output and logging
    output_dir: Optional[str] = None
    verbose: bool = True
