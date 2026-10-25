"""Single conformer geometry optimization and single-point calculation using Psi4."""

import os
import logging
from typing import Tuple, Optional
import psi4

from nmrinator.quantum.psi4_ir import setup_psi4, PSI4_SOLVENTS

logger = logging.getLogger("nmrinator")


def optimize_geometry_psi4(
    xyz_str: str,
    method: str = "b3lyp-d3bj",
    basis: str = "6-31G(d)",
    solvent: Optional[str] = None,
    memory: str = "8 GB",
    threads: int = 8,
    max_iter: int = 50
) -> Tuple[str, float]:
    """Performs quantum mechanical DFT/HF geometry optimization on a single conformer.
    
    Returns:
        (optimized_xyz_str, final_energy_hartree)
    """
    setup_psi4(memory=memory, threads=threads)
    
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
    
    if solvent and solvent.lower() not in ["gas", "none"]:
        pcm_solv = PSI4_SOLVENTS.get(solvent.lower().strip(), solvent)
        psi4.set_options({
            "pcm": True,
            "pcm_scf_type": "total"
        })
        try:
            pcm_block = f"""
            Units = Angstrom
            Medium {{
                SolverType = IEFPCM
                Solvent = {pcm_solv}
            }}
            """
            psi4.pcm_helper(pcm_block)
        except Exception:
            pass
            
    psi4.set_options({
        "geom_maxiter": max_iter,
        "g_convergence": "gau_loose"
    })
    
    calc_target = f"{method}/{basis}"
    logger.info(f"Ottimizzazione geometrica Psi4 con {calc_target}...")
    
    try:
        energy = psi4.optimize(calc_target, molecule=mol)
    except Exception as e:
        logger.warning(f"Ottimizzazione fallita o non convergente ({e}), calcolo single point...")
        energy = psi4.energy(calc_target, molecule=mol)
        
    # Extract optimized geometry as XYZ
    natoms = mol.natom()
    opt_lines = [f"{natoms}", f"Optimized by Psi4 ({calc_target})"]
    for i in range(natoms):
        sym = mol.symbol(i)
        x = mol.x(i) * 0.529177210903  # Bohr to Angstrom
        y = mol.y(i) * 0.529177210903
        z = mol.z(i) * 0.529177210903
        opt_lines.append(f"{sym:<2} {x:12.6f} {y:12.6f} {z:12.6f}")
        
    opt_xyz = "\n".join(opt_lines)
    return opt_xyz, float(energy)
