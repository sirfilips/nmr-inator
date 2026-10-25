"""NMR chemical shift predictor with online HOSE database, robust offline fallback, and 3D geometric corrections."""

import logging
from typing import List, Dict, Tuple, Optional, Any
import requests
import numpy as np
from rdkit import Chem

from nmrinator.models import NMRSignal, NMRSpectrum
from nmrinator.symmetry import (
    get_atom_equivalence_groups,
    identify_proton_environments,
    identify_carbon_environments
)
from nmrinator.conformation_3d import (
    calculate_3d_karplus_couplings,
    calculate_ring_current_shielding,
    detect_intramolecular_h_bonds
)
from nmrinator.io import parse_xyz

logger = logging.getLogger("nmrinator")

ZAKODIUM_PROTON_API = "https://nmr-prediction.service.zakodium.com/v1/predict/proton"
ZAKODIUM_CARBON_API = "https://nmr-prediction.service.zakodium.com/v1/predict/carbon"


def predict_nmrdb_online(smiles: str, nucleus: str = "1H") -> Optional[List[Dict[str, Any]]]:
    """Queries online Zakodium/NMRDB predictor service for high-accuracy HOSE-code NMR predictions."""
    url = ZAKODIUM_PROTON_API if nucleus.upper() in ["1H", "H"] else ZAKODIUM_CARBON_API
    try:
        response = requests.get(url, params={"smiles": smiles}, timeout=6.0)
        if response.status_code == 200:
            payload = response.json()
            data = payload.get("data", {})
            signals = data.get("joinedSignals", []) or data.get("signals", [])
            return signals
    except Exception as e:
        logger.debug(f"Online NMRDB query skipped or timed out: {e}")
    return None


def _has_double_bond_to(atom: Chem.Atom, target_sym: str = "O") -> bool:
    """Helper to check if atom has a double bond to an atom of target_sym."""
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetSymbol() == target_sym and bond.GetBondType() == Chem.BondType.DOUBLE:
            return True
    return False


def _has_single_bond_to(atom: Chem.Atom, target_sym: str = "O") -> bool:
    """Helper to check if atom has a single bond to an atom of target_sym."""
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetSymbol() == target_sym and bond.GetBondType() == Chem.BondType.SINGLE:
            return True
    return False


def _has_triple_bond_to(atom: Chem.Atom, target_sym: str = "N") -> bool:
    """Helper to check if atom has a triple bond to an atom of target_sym."""
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetSymbol() == target_sym and bond.GetBondType() == Chem.BondType.TRIPLE:
            return True
    return False


def predict_1h_empirical_offline(mol: Chem.Mol) -> Dict[int, float]:
    """Robust offline rule-based predictor for 1H chemical shifts based on functional groups and connectivity."""
    mol_with_h = Chem.AddHs(mol) if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() else mol
    shifts: Dict[int, float] = {}
    
    for atom in mol_with_h.GetAtoms():
        if atom.GetSymbol() == "H":
            idx = atom.GetIdx()
            nbrs = atom.GetNeighbors()
            if not nbrs:
                shifts[idx] = 1.0
                continue
            parent = nbrs[0]
            p_sym = parent.GetSymbol()
            
            if p_sym == "O":
                # Check if carboxylic acid (-COOH)
                is_acid = any(
                    n.GetSymbol() == "C" and _has_double_bond_to(n, "O")
                    for n in parent.GetNeighbors()
                )
                shifts[idx] = 11.5 if is_acid else 2.5  # Alcohol / acid
            elif p_sym == "N":
                shifts[idx] = 3.0
            elif p_sym == "C":
                if parent.GetIsAromatic():
                    shifts[idx] = 7.26
                elif parent.GetHybridization() == Chem.HybridizationType.SP:
                    shifts[idx] = 2.0  # Alkyne
                elif parent.GetHybridization() == Chem.HybridizationType.SP2:
                    is_ald = _has_double_bond_to(parent, "O")
                    shifts[idx] = 9.8 if is_ald else 5.3  # Aldehyde vs Alkene
                else:
                    # sp3 aliphatic carbon
                    h_count = sum(1 for n in parent.GetNeighbors() if n.GetSymbol() == "H")
                    base = 0.90 if h_count >= 3 else (1.25 if h_count == 2 else 1.50)
                    
                    # Additive substituent increments
                    for n in parent.GetNeighbors():
                        if n.GetIdx() == idx:
                            continue
                        nsym = n.GetSymbol()
                        if nsym == "O":
                            base += 2.40  # ether / alcohol alpha proton (~3.4-3.7 ppm)
                        elif nsym == "N":
                            base += 1.50
                        elif nsym in ["Cl", "Br"]:
                            base += 2.20
                        elif nsym == "F":
                            base += 3.10
                        elif n.GetIsAromatic():
                            base += 1.30  # benzylic (~2.3-2.6 ppm)
                        elif nsym == "C":
                            if _has_double_bond_to(n, "O"):
                                base += 1.10  # alpha to carbonyl (~2.1-2.4 ppm)
                            elif _has_triple_bond_to(n, "N"):
                                base += 1.15  # alpha to nitrile (~2.3-2.5 ppm)
                            elif _has_triple_bond_to(n, "C"):
                                base += 0.80  # propargyl (~2.0 ppm)
                                
                    shifts[idx] = round(base, 2)
            else:
                shifts[idx] = 2.0
                
    return shifts


def predict_13c_empirical_offline(mol: Chem.Mol) -> Dict[int, float]:
    """Robust offline rule-based predictor for 13C chemical shifts based on Grant-Paul additive parameters."""
    mol_with_h = Chem.AddHs(mol) if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() else mol
    shifts: Dict[int, float] = {}
    
    for atom in mol_with_h.GetAtoms():
        if atom.GetSymbol() == "C":
            idx = atom.GetIdx()
            if atom.GetIsAromatic():
                base = 128.5
                for n in atom.GetNeighbors():
                    if n.GetSymbol() == "O":
                        base += 25.0  # Phenol ipso ~153 ppm
                    elif n.GetSymbol() == "N":
                        base += 15.0
                    elif n.GetSymbol() == "C" and any(nn.GetSymbol() == "O" for nn in n.GetNeighbors()):
                        base -= 2.0
                shifts[idx] = round(base, 1)
                continue
                
            # Carbonyl
            if _has_double_bond_to(atom, "O"):
                if _has_single_bond_to(atom, "O") or _has_single_bond_to(atom, "N"):
                    shifts[idx] = 172.0  # Ester / Carboxylic acid / Amide
                else:
                    shifts[idx] = 205.0  # Ketone / Aldehyde
                continue

            # Nitrile (-C≡N)
            if _has_triple_bond_to(atom, "N"):
                shifts[idx] = 119.0
                continue
                
            if atom.GetHybridization() == Chem.HybridizationType.SP:
                shifts[idx] = 82.0  # Alkyne
            elif atom.GetHybridization() == Chem.HybridizationType.SP2:
                shifts[idx] = 125.0  # Alkene
            else:
                # sp3 aliphatic carbon
                h_cnt = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "H")
                base = 10.0 if h_cnt == 3 else (22.0 if h_cnt == 2 else (32.0 if h_cnt == 1 else 36.0))
                for n in atom.GetNeighbors():
                    nsym = n.GetSymbol()
                    if nsym == "O":
                        base += 45.0  # C-O ~55-70 ppm
                    elif nsym == "N":
                        base += 30.0
                    elif nsym == "Cl":
                        base += 32.0
                    elif nsym == "F":
                        base += 65.0
                    elif n.GetIsAromatic():
                        base += 18.0
                    elif nsym == "C" and _has_triple_bond_to(n, "N"):
                        base += 15.0  # alpha to nitrile (~30-37 ppm)
                shifts[idx] = round(base, 1)
                
    return shifts


def predict_conformer_nmr(
    mol: Chem.Mol,
    xyz_str: str,
    nucleus: str = "1H",
    spectrometer_mhz: float = 400.0,
    apply_3d_corrections: bool = True
) -> NMRSpectrum:
    """Predicts a 1D NMR spectrum for a single conformer incorporating 3D geometry effects.
    
    Combines:
    1. Online HOSE-code database predictions or robust offline rules.
    2. Topological symmetry and atom equivalence grouping.
    3. 3D Karplus dihedral J-coupling calculation.
    4. 3D McConnell ring current magnetic shielding and H-bond perturbations.
    """
    mol_with_h = Chem.AddHs(mol) if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() else mol
    symbols, coords = parse_xyz(xyz_str)
    can_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_with_h))
    
    eq_groups = get_atom_equivalence_groups(mol_with_h, nucleus=nucleus)
    labels = identify_proton_environments(mol_with_h) if nucleus.upper() in ["1H", "H"] else identify_carbon_environments(mol_with_h)
    
    # Step 1: Base shifts
    online_signals = predict_nmrdb_online(can_smiles, nucleus=nucleus)
    atom_base_shifts: Dict[int, float] = {}
    atom_multiplicities: Dict[int, str] = {}
    atom_couplings: Dict[int, List[float]] = {}
    
    if online_signals:
        for s in online_signals:
            delta = float(s.get("delta", 0.0))
            mult = str(s.get("multiplicity", "s"))
            js = [float(j_item.get("coupling", 7.0)) for j_item in s.get("js", [])]
            for atom_id in s.get("atoms", []):
                atom_base_shifts[atom_id] = delta
                atom_multiplicities[atom_id] = mult
                atom_couplings[atom_id] = js
                
    # Fallback for any missing atom
    if nucleus.upper() in ["1H", "H"]:
        empirical = predict_1h_empirical_offline(mol_with_h)
    else:
        empirical = predict_13c_empirical_offline(mol_with_h)
        
    for a_idx, val in empirical.items():
        if a_idx not in atom_base_shifts:
            atom_base_shifts[a_idx] = val
            atom_multiplicities[a_idx] = "s" if nucleus.upper() in ["13C", "C"] else "m"
            atom_couplings[a_idx] = [7.0] if nucleus.upper() in ["1H", "H"] else []
            
    # Step 2: 3D Geometrical Corrections (if 1H NMR and coordinates provided)
    if nucleus.upper() in ["1H", "H"] and apply_3d_corrections and len(coords) == mol_with_h.GetNumAtoms():
        all_h_indices = [a.GetIdx() for a in mol_with_h.GetAtoms() if a.GetSymbol() == "H"]
        # Ring current anisotropy shifts
        ring_shieldings = calculate_ring_current_shielding(mol_with_h, coords, all_h_indices)
        # Intramolecular H-bond downfield shifts
        h_bond_shifts = detect_intramolecular_h_bonds(mol_with_h, coords)
        # 3D Karplus couplings
        karplus_couplings = calculate_3d_karplus_couplings(mol_with_h, coords)
        
        for h_idx in all_h_indices:
            aniso = ring_shieldings.get(h_idx, 0.0)
            hbond = h_bond_shifts.get(h_idx, 0.0)
            atom_base_shifts[h_idx] = atom_base_shifts.get(h_idx, 1.0) + aniso + hbond
            
            # Update couplings from Karplus
            vic_js = [j for (a1, a2), j in karplus_couplings.items() if a1 == h_idx or a2 == h_idx]
            if vic_js:
                atom_couplings[h_idx] = [round(j, 1) for j in vic_js]
                
    # Step 3: Aggregate into discrete NMRSignals across equivalent groups
    final_signals: List[NMRSignal] = []
    for rep_idx, group in eq_groups.items():
        shifts = [atom_base_shifts.get(i, 0.0) for i in group]
        mean_shift = round(float(np.mean(shifts)), 3)
        integral = len(group)
        mult = atom_multiplicities.get(rep_idx, "s")
        js = atom_couplings.get(rep_idx, [])
        label = labels.get(rep_idx, "")
        
        sig = NMRSignal(
            nucleus=nucleus,
            delta=mean_shift,
            multiplicity=mult,
            coupling_j=js,
            integral=integral,
            atom_indices=group,
            label=label
        )
        final_signals.append(sig)
        
    # Sort signals descending by chemical shift (standard downfield -> upfield order)
    final_signals.sort(key=lambda s: s.delta, reverse=True)
    
    return NMRSpectrum(
        nucleus=nucleus,
        spectrometer_mhz=spectrometer_mhz,
        signals=final_signals
    )
