"""Topological symmetry detection and chemical equivalence grouping using RDKit."""

from typing import Dict, List, Tuple
from rdkit import Chem


def get_atom_equivalence_groups(mol: Chem.Mol, nucleus: str = "1H") -> Dict[int, List[int]]:
    """Groups atoms into chemical equivalence classes based on topological symmetry.
    
    Returns a dictionary mapping a group representative index to the list of equivalent atom indices.
    E.g. For ethanol (CCO), the 3 methyl protons are grouped together: {3: [3, 4, 5]},
    and the 2 methylene protons: {6: [6, 7]}.
    """
    mol_with_h = Chem.AddHs(mol) if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() else mol
    ranks = list(Chem.CanonicalRankAtoms(mol_with_h, breakTies=False))
    
    target_element = "H" if nucleus.upper() in ["1H", "H"] else "C"
    target_indices = [atom.GetIdx() for atom in mol_with_h.GetAtoms() if atom.GetSymbol() == target_element]
    
    if not target_indices:
        return {}

    # Disjoint Set (Union-Find) on target atom indices
    parent = {i: i for i in target_indices}
    
    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
        
    def union(i: int, j: int):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if root_i < root_j:
                parent[root_j] = root_i
            else:
                parent[root_i] = root_j

    # 1. Group atoms with identical canonical symmetry ranks
    rank_map: Dict[int, List[int]] = {}
    for idx in target_indices:
        r = ranks[idx]
        rank_map.setdefault(r, []).append(idx)
        
    for r, idx_list in rank_map.items():
        for k in range(1, len(idx_list)):
            union(idx_list[0], idx_list[k])
            
    # 2. Guarantee that equivalent hydrogens on rapidly rotating methyl groups are merged
    if target_element == "H":
        for atom in mol_with_h.GetAtoms():
            if atom.GetSymbol() == "C":
                h_neighbors = [
                    nbr.GetIdx() for nbr in atom.GetNeighbors()
                    if nbr.GetSymbol() == "H"
                ]
                if len(h_neighbors) == 3:
                    union(h_neighbors[0], h_neighbors[1])
                    union(h_neighbors[0], h_neighbors[2])

    # 3. Build final equivalence groups: representative_idx -> sorted list of atom indices
    group_dict: Dict[int, List[int]] = {}
    for idx in target_indices:
        root = find(idx)
        group_dict.setdefault(root, []).append(idx)
        
    result: Dict[int, List[int]] = {}
    for root, members in group_dict.items():
        members.sort()
        result[members[0]] = members
        
    return result


def identify_proton_environments(mol: Chem.Mol) -> Dict[int, str]:
    """Labels each proton with its chemical environment label (e.g. 'Methyl -CH3', 'Aromatic C-H', etc.)."""
    mol_with_h = Chem.AddHs(mol) if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() else mol
    labels: Dict[int, str] = {}
    
    for atom in mol_with_h.GetAtoms():
        if atom.GetSymbol() == "H":
            nbrs = atom.GetNeighbors()
            if not nbrs:
                labels[atom.GetIdx()] = "H"
                continue
            parent = nbrs[0]
            p_sym = parent.GetSymbol()
            
            if p_sym == "O":
                labels[atom.GetIdx()] = "Hydroxyl -OH"
            elif p_sym == "N":
                labels[atom.GetIdx()] = "Amine/Amide -NH"
            elif p_sym == "S":
                labels[atom.GetIdx()] = "Thiol -SH"
            elif p_sym == "C":
                if parent.GetIsAromatic():
                    labels[atom.GetIdx()] = "Aromatic C-H"
                elif parent.GetHybridization() == Chem.HybridizationType.SP:
                    labels[atom.GetIdx()] = "Alkyne ≡C-H"
                elif parent.GetHybridization() == Chem.HybridizationType.SP2:
                    # Check if aldehyde
                    has_db_o = any(
                        b.GetOtherAtom(parent).GetSymbol() == "O" and b.GetBondType() == Chem.BondType.DOUBLE
                        for b in parent.GetBonds()
                    )
                    if has_db_o:
                        labels[atom.GetIdx()] = "Aldehyde -CHO"
                    else:
                        labels[atom.GetIdx()] = "Alkene =C-H"
                else:
                    # sp3
                    h_cnt = sum(1 for n in parent.GetNeighbors() if n.GetSymbol() == "H")
                    if h_cnt == 3:
                        labels[atom.GetIdx()] = "Methyl -CH3"
                    elif h_cnt == 2:
                        labels[atom.GetIdx()] = "Methylene -CH2-"
                    else:
                        labels[atom.GetIdx()] = "Methine >CH-"
            else:
                labels[atom.GetIdx()] = f"{p_sym}-H"
                
    return labels


def identify_carbon_environments(mol: Chem.Mol) -> Dict[int, str]:
    """Labels each carbon with its functional group label (e.g. 'Carbonyl C=O', 'Aromatic C', 'Aliphatic C')."""
    mol_with_h = Chem.AddHs(mol) if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() else mol
    labels: Dict[int, str] = {}
    
    for atom in mol_with_h.GetAtoms():
        if atom.GetSymbol() == "C":
            idx = atom.GetIdx()
            if atom.GetIsAromatic():
                labels[idx] = "Aromatic Carbon"
                continue
                
            # Check for double bond to O (Carbonyl)
            has_carbonyl = False
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if other.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                    has_carbonyl = True
                    break
            if has_carbonyl:
                # Differentiate carboxylic/ester/amide vs ketone/aldehyde
                het_single = any(
                    b.GetOtherAtom(atom).GetSymbol() in ["O", "N", "S", "Cl"]
                    and b.GetBondType() == Chem.BondType.SINGLE
                    for b in atom.GetBonds()
                )
                if het_single:
                    labels[idx] = "Carbonyl (Ester/Acid/Amide)"
                else:
                    labels[idx] = "Carbonyl (Ketone/Aldehyde)"
                continue

            # Nitrile (-C#N)
            has_nitrile = any(
                b.GetOtherAtom(atom).GetSymbol() == "N" and b.GetBondType() == Chem.BondType.TRIPLE
                for b in atom.GetBonds()
            )
            if has_nitrile:
                labels[idx] = "Nitrile Carbon -C#N"
                continue
                
            if atom.GetHybridization() == Chem.HybridizationType.SP:
                labels[idx] = "Alkyne Carbon (sp)"
            elif atom.GetHybridization() == Chem.HybridizationType.SP2:
                labels[idx] = "Alkene Carbon (sp2)"
            else:
                h_cnt = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "H")
                if h_cnt == 3:
                    labels[idx] = "Aliphatic -CH3"
                elif h_cnt == 2:
                    labels[idx] = "Aliphatic -CH2-"
                elif h_cnt == 1:
                    labels[idx] = "Aliphatic >CH-"
                else:
                    labels[idx] = "Quaternary Carbon >C<"
                    
    return labels
