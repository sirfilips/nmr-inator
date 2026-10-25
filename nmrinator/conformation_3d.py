"""3D Conformation-dependent spectroscopic analysis: Karplus couplings, ring current anisotropy, and H-bonds."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem

from nmrinator.io import parse_xyz


def compute_dihedral_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Calculates the dihedral angle between 4 points in space (in degrees, -180 to 180)."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normal vector to planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    norm_b2 = np.linalg.norm(b2)

    if norm_n1 < 1e-7 or norm_n2 < 1e-7 or norm_b2 < 1e-7:
        return 0.0

    n1 /= norm_n1
    n2 /= norm_n2
    u_b2 = b2 / norm_b2

    # Vector in plane 1 perpendicular to b2
    m1 = np.cross(n1, u_b2)

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    rad = np.arctan2(y, x)
    return float(np.degrees(rad))


def karplus_coupling(dihedral_deg: float) -> float:
    """Calculates 3J(H,H) vicinal scalar coupling (in Hz) using Bothner-By / Karplus parameterization."""
    phi = np.radians(dihedral_deg)
    # Generalized Karplus: J = A cos^2(phi) - B cos(phi) + C
    A = 7.76
    B = 1.10
    C = 1.40
    j_val = A * (np.cos(phi) ** 2) - B * np.cos(phi) + C
    return max(0.5, float(j_val))


def find_vicinal_proton_pairs(mol: Chem.Mol) -> List[Tuple[int, int, int, int]]:
    """Identifies all vicinal H-C-C-H proton pairs in the molecule.
    Returns list of tuples: (h1_idx, c1_idx, c2_idx, h2_idx).
    """
    mol_with_h = Chem.AddHs(mol) if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() else mol
    pairs = []
    
    for bond in mol_with_h.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        # Look for C-C single bonds
        if a1.GetSymbol() == "C" and a2.GetSymbol() == "C":
            h_on_a1 = [n.GetIdx() for n in a1.GetNeighbors() if n.GetSymbol() == "H"]
            h_on_a2 = [n.GetIdx() for n in a2.GetNeighbors() if n.GetSymbol() == "H"]
            
            for h1 in h_on_a1:
                for h2 in h_on_a2:
                    pairs.append((h1, a1.GetIdx(), a2.GetIdx(), h2))
                    
    return pairs


def calculate_3d_karplus_couplings(mol: Chem.Mol, coords: np.ndarray) -> Dict[Tuple[int, int], float]:
    """Calculates 3D Karplus 3J(H,H) scalar coupling constants for all vicinal pairs from 3D coordinates."""
    vicinal_pairs = find_vicinal_proton_pairs(mol)
    couplings: Dict[Tuple[int, int], float] = {}
    
    for h1, c1, c2, h2 in vicinal_pairs:
        if max(h1, c1, c2, h2) < len(coords):
            p1 = coords[h1]
            p2 = coords[c1]
            p3 = coords[c2]
            p4 = coords[h2]
            phi = compute_dihedral_angle(p1, p2, p3, p4)
            j = karplus_coupling(phi)
            key = (min(h1, h2), max(h1, h2))
            couplings[key] = j
            
    return couplings


def find_aromatic_rings(mol: Chem.Mol) -> List[List[int]]:
    """Identifies indices of atoms in planar aromatic rings (e.g. 5- and 6-membered aromatic rings)."""
    mol_clean = Chem.RemoveHs(mol)
    rings = mol_clean.GetRingInfo().AtomRings()
    aromatic_rings = []
    for ring in rings:
        if 5 <= len(ring) <= 6:
            if all(mol_clean.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                aromatic_rings.append(list(ring))
    return aromatic_rings


def calculate_ring_current_shielding(
    mol: Chem.Mol,
    coords: np.ndarray,
    target_protons: List[int]
) -> Dict[int, float]:
    """Calculates McConnell dipolar magnetic anisotropy (delta shift perturbation) from aromatic rings.
    
    Protons positioned above the ring plane experience upfield shielding (delta < 0),
    while in-plane protons experience downfield deshielding (delta > 0).
    """
    aromatic_rings = find_aromatic_rings(mol)
    deltas: Dict[int, float] = {h: 0.0 for h in target_protons}
    if not aromatic_rings:
        return deltas
        
    for ring in aromatic_rings:
        ring_coords = coords[ring]
        center = np.mean(ring_coords, axis=0)
        
        # Approximate ring normal using SVD on ring coordinates
        centered = ring_coords - center
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]  # third eigenvector is normal to the ring plane
        normal /= np.linalg.norm(normal)
        
        # Classical ring current susceptibility constant for benzene-like rings (ppm * Angstrom^3)
        # Delta chi ~ -60 * 10^-30 cm^3/molecule -> factor ~ 25.0 in ppm * A^3
        CHI_FACTOR = 28.0
        
        for h_idx in target_protons:
            if h_idx < len(coords):
                h_pos = coords[h_idx]
                r_vec = h_pos - center
                r = np.linalg.norm(r_vec)
                if r < 1.0:
                    continue  # avoid singularity
                    
                cos_theta = np.dot(r_vec, normal) / r
                # McConnell equation: delta_shift = (CHI_FACTOR / (3 * r^3)) * (1 - 3 * cos^2(theta))
                # Note: Over the ring (cos_theta ~ 1), 1 - 3*1 = -2 -> negative delta (shielding, upfield)
                delta_aniso = (CHI_FACTOR / (3.0 * (r ** 3))) * (1.0 - 3.0 * (cos_theta ** 2))
                deltas[h_idx] += float(delta_aniso)
                
    return deltas


def detect_intramolecular_h_bonds(mol: Chem.Mol, coords: np.ndarray) -> Dict[int, float]:
    """Detects intramolecular hydrogen bonds and returns downfield shift perturbations for donor protons."""
    mol_with_h = Chem.AddHs(mol) if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() else mol
    perturbations: Dict[int, float] = {}
    
    # Identify donor hydrogens (on O or N)
    donors = []
    for atom in mol_with_h.GetAtoms():
        if atom.GetSymbol() == "H":
            nbrs = atom.GetNeighbors()
            if nbrs and nbrs[0].GetSymbol() in ["O", "N"]:
                donors.append((atom.GetIdx(), nbrs[0].GetIdx()))
                
    # Identify potential acceptors (O, N, F, Cl with lone pairs)
    acceptors = [
        atom.GetIdx() for atom in mol_with_h.GetAtoms()
        if atom.GetSymbol() in ["O", "N", "F"]
    ]
    
    for h_idx, donor_parent_idx in donors:
        h_pos = coords[h_idx]
        d_pos = coords[donor_parent_idx]
        
        for acc_idx in acceptors:
            if acc_idx == donor_parent_idx:
                continue
            # Avoid immediately adjacent atoms
            bond = mol_with_h.GetBondBetweenAtoms(donor_parent_idx, acc_idx)
            if bond is not None:
                continue
                
            acc_pos = coords[acc_idx]
            dist_h_acc = np.linalg.norm(h_pos - acc_pos)
            
            # Typical H-bond distance threshold 1.6 to 2.4 Angstroms
            if 1.5 <= dist_h_acc <= 2.4:
                # Check D-H...A angle (must be > 110 degrees for genuine H-bond)
                v_dh = d_pos - h_pos
                v_ha = acc_pos - h_pos
                cos_ang = np.dot(v_dh, v_ha) / (np.linalg.norm(v_dh) * np.linalg.norm(v_ha))
                ang_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
                
                if ang_deg >= 110.0:
                    # Intramolecular H-bonds shift OH/NH dramatically downfield (+2.0 to +4.5 ppm)
                    shift = 2.5 * (1.0 - (dist_h_acc - 1.5) / 0.9)
                    perturbations[h_idx] = float(np.clip(shift, 1.0, 4.5))
                    break
                    
    return perturbations

