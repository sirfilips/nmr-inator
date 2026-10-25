"""Tests for topological symmetry and chemical equivalence detection."""

import pytest
from rdkit import Chem
from nmrinator.symmetry import (
    get_atom_equivalence_groups,
    identify_proton_environments,
    identify_carbon_environments
)


def test_ethanol_symmetry():
    # Ethanol: CH3-CH2-OH -> 3 methyl H, 2 methylene H, 1 OH
    mol = Chem.MolFromSmiles("CCO")
    h_groups = get_atom_equivalence_groups(mol, nucleus="1H")
    # There should be 3 proton groups: methyl (3H), methylene (2H), OH (1H)
    sizes = sorted(len(grp) for grp in h_groups.values())
    assert sizes == [1, 2, 3]
    
    c_groups = get_atom_equivalence_groups(mol, nucleus="13C")
    # Two distinct carbons
    assert len(c_groups) == 2


def test_acetone_symmetry():
    # Acetone: CH3-C(=O)-CH3 -> 6 equivalent methyl H, 2 equivalent methyl C, 1 carbonyl C
    mol = Chem.MolFromSmiles("CC(=O)C")
    h_groups = get_atom_equivalence_groups(mol, nucleus="1H")
    # Acetone methyl protons should be equivalent (6H) or two groups of 3H
    total_h = sum(len(g) for g in h_groups.values())
    assert total_h == 6
    
    c_groups = get_atom_equivalence_groups(mol, nucleus="13C")
    # Two carbon signals (carbonyl C and 2 methyl C)
    assert len(c_groups) == 2


def test_benzene_symmetry():
    # Benzene: all 6 H and all 6 C equivalent
    mol = Chem.MolFromSmiles("c1ccccc1")
    h_groups = get_atom_equivalence_groups(mol, nucleus="1H")
    assert len(h_groups) == 1
    assert len(list(h_groups.values())[0]) == 6
    
    c_groups = get_atom_equivalence_groups(mol, nucleus="13C")
    assert len(c_groups) == 1
    assert len(list(c_groups.values())[0]) == 6


def test_butyronitrile_and_dinitrile_symmetry():
    # Butyronitrile: CH3-CH2-CH2-C#N -> 3 distinct 1H signals (3H, 2H, 2H) = 7H total
    mol = Chem.MolFromSmiles("CCCC#N")
    h_groups = get_atom_equivalence_groups(mol, nucleus="1H")
    sizes = sorted(len(grp) for grp in h_groups.values())
    assert sizes == [2, 2, 3]
    assert sum(sizes) == 7
    
    # 4 distinct carbons
    c_groups = get_atom_equivalence_groups(mol, nucleus="13C")
    assert len(c_groups) == 4
    
    # Succinonitrile: N#C-CH2-CH2-C#N -> symmetric 4H for the two central CH2
    mol_sym = Chem.MolFromSmiles("N#CCCC#N")
    h_groups_sym = get_atom_equivalence_groups(mol_sym, nucleus="1H")
    assert len(h_groups_sym) == 1
    assert len(list(h_groups_sym.values())[0]) == 4

