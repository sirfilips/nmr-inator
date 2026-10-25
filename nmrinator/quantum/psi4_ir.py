"""Psi4 Quantum Engine integration for harmonic frequencies and IR intensities."""

import os
import sys
import logging
import tempfile
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

import psi4

from nmrinator.models import IRMode
from nmrinator.ir.scaling import get_scaling_factor

logger = logging.getLogger("nmrinator")

# PCM Solvents supported by Psi4
PSI4_SOLVENTS = {
    "water": "Water",
    "h2o": "Water",
    "chloroform": "Chloroform",
    "dmso": "Dimethylsulfoxide",
    "methanol": "Methanol",
    "toluene": "Toluene",
    "acetonitrile": "Acetonitrile",
    "dcm": "Dichloromethane",
    "dichloromethane": "Dichloromethane",
    "thf": "Tetrahydrofuran",
    "ethanol": "Ethanol"
}


def setup_psi4(
    memory: str = "8 GB",
    threads: int = 8,
    log_file: Optional[str] = None
):
    """Sets up Psi4 memory, threads, and safe output streams for cross-platform stability."""
    try:
        psi4.set_memory(memory)
    except Exception:
        pass
    try:
        psi4.set_num_threads(threads)
    except Exception:
        pass

    if log_file:
        psi4.core.set_output_file(log_file, False)
    else:
        # Avoid Windows /dev/null crash by using 'nul'
        devnull = "nul" if os.name == "nt" else "/dev/null"
        try:
            psi4.core.set_output_file(devnull, False)
        except Exception:
            pass


def calculate_psi4_ir(
    xyz_str: str,
    method: str = "b3lyp-d3bj",
    basis: str = "6-31G(d)",
    solvent: Optional[str] = None,
    memory: str = "8 GB",
    threads: int = 8,
    scaling_factor: Optional[float] = None
) -> Tuple[List[IRMode], float]:
    """Calculates harmonic vibrational frequencies and IR intensities with Psi4.
    
    Returns:
        (modes_list, zpve_kcalmol)
    """
    setup_psi4(memory=memory, threads=threads)
    
    # Configure PCM solvation if requested
    if solvent and solvent.lower() not in ["gas", "none"]:
        solv_key = solvent.lower().strip()
        pcm_solv = PSI4_SOLVENTS.get(solv_key, solvent)
        psi4.set_options({
            "pcm": True,
            "pcm_scf_type": "total"
        })
        # Define pcm block in molecule or options
        pcm_block = f"""
        Units = Angstrom
        Medium {{
            SolverType = IEFPCM
            Solvent = {pcm_solv}
        }}
        Cavity {{
            Type = GePol
            Area = 0.3
            Scaling = False
            RadiiSet = UFF
        }}
        """
        try:
            psi4.pcm_helper(pcm_block)
        except Exception as e:
            logger.debug(f"Nota PCM setup: {e}")
            
    # Format molecule string ensuring C1 symmetry to calculate all vibrational modes
    geom_lines = []
    for line in xyz_str.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                float(parts[1])
                float(parts[2])
                float(parts[3])
                geom_lines.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")
            except ValueError:
                continue
                
    geom_str = "0 1\n" + "\n".join(geom_lines) + "\nsymmetry c1\nno_reorient\nno_com"
    mol = psi4.geometry(geom_str)
    
    # Resolve frequency scaling factor
    s_factor = scaling_factor or get_scaling_factor(method, basis)
    
    calc_call = f"{method}/{basis}"
    logger.info(f"Esecuzione calcolo frequenze Psi4: {calc_call}")
    
    # Run frequency calculation
    energy, wfn = psi4.frequency(calc_call, molecule=mol, return_wfn=True)
    
    vib_data = getattr(wfn, "frequency_analysis", None)
    if not vib_data or "omega" not in vib_data:
        raise RuntimeError("Psi4 non ha prodotto dati di analisi frequenziale.")
        
    omega_raw = vib_data["omega"].data
    intensity_raw = vib_data.get("IR_intensity", None)
    if intensity_raw is not None:
        intensity_data = intensity_raw.data
    else:
        intensity_data = np.zeros(len(omega_raw))
        
    trv_raw = vib_data.get("TRV", None)
    trv_data = trv_raw.data if trv_raw is not None else ["V"] * len(omega_raw)
    
    modes: List[IRMode] = []
    mode_idx = 1
    
    for i in range(len(omega_raw)):
        mode_type = str(trv_data[i]).upper()
        # Filter only genuine vibrational normal modes ('V'), skipping translational/rotational ('TR')
        if "V" not in mode_type:
            continue
            
        freq_c = omega_raw[i]
        # Frequency can be complex (imaginary for saddle points / transition states)
        is_imag = False
        if np.iscomplexobj(freq_c) or isinstance(freq_c, complex):
            if abs(freq_c.imag) > 1e-4:
                is_imag = True
                freq_val = float(abs(freq_c.imag))
            else:
                freq_val = float(freq_c.real)
        else:
            freq_val = float(freq_c)
            
        # Ignore negligible near-zero acoustic modes
        if freq_val < 10.0 and not is_imag:
            continue
            
        intensity_val = float(max(0.0, intensity_data[i]))
        scaled_freq = float(freq_val * s_factor) if not is_imag else freq_val
        
        mode = IRMode(
            mode_index=mode_idx,
            frequency_calc=round(freq_val, 2),
            frequency_scaled=round(scaled_freq, 2),
            intensity=round(intensity_val, 3),
            is_imaginary=is_imag
        )
        modes.append(mode)
        mode_idx += 1
        
    zpve_kcalmol = 0.0
    try:
        zpve_hartree = psi4.core.get_variable("ZPVE - ZERO POINT VIBRATIONAL ENERGY")
        zpve_kcalmol = zpve_hartree * 627.509474
    except Exception:
        pass
        
    return modes, zpve_kcalmol
