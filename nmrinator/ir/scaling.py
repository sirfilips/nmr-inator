"""Harmonic vibrational frequency scaling factors from NIST Computational Chemistry Comparison Benchmark (CCCBDB)."""

from typing import Dict, Tuple

# Database of harmonic vibrational frequency scaling factors (NIST CCCBDB & Kesharwani et al.)
SCALING_FACTORS: Dict[Tuple[str, str], float] = {
    # B3LYP & variants
    ("b3lyp", "6-31g(d)"): 0.9614,
    ("b3lyp-d3bj", "6-31g(d)"): 0.9614,
    ("b3lyp", "6-311g(d,p)"): 0.9668,
    ("b3lyp-d3bj", "6-311g(d,p)"): 0.9668,
    ("b3lyp", "def2-svp"): 0.9630,
    ("b3lyp-d3bj", "def2-svp"): 0.9630,
    ("b3lyp", "def2-tzvp"): 0.9680,
    ("b3lyp-d3bj", "def2-tzvp"): 0.9680,
    
    # wB97X-D
    ("wb97x-d", "6-31g(d)"): 0.9570,
    ("wb97x-d", "def2-svp"): 0.9580,
    ("wb97x-d", "def2-tzvp"): 0.9620,
    
    # PBE0
    ("pbe0", "6-31g(d)"): 0.9590,
    ("pbe0", "def2-svp"): 0.9600,
    
    # Hartree-Fock (HF) - overestimates frequencies considerably (~10%)
    ("hf", "sto-3g"): 0.8950,
    ("hf", "3-21g"): 0.9085,
    ("hf", "6-31g(d)"): 0.8992,
}


def get_scaling_factor(method: str, basis: str) -> float:
    """Returns the empirical harmonic frequency scaling factor for the given method and basis set.
    
    Defaults to 0.965 for typical DFT methods, or 0.900 for HF if method/basis combination is not in lookup table.
    """
    m = method.lower().strip()
    b = basis.lower().strip()
    
    # Direct lookup
    if (m, b) in SCALING_FACTORS:
        return SCALING_FACTORS[(m, b)]
        
    # Generalized heuristics
    if "hf" in m:
        return 0.899
    elif "b3lyp" in m:
        return 0.964
    elif "wb97" in m:
        return 0.958
    elif "pbe0" in m:
        return 0.960
    elif "mp2" in m:
        return 0.943
        
    return 0.965  # General DFT consensus default

