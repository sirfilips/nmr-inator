"""Characteristic infrared vibrational functional group assignment rules."""

from typing import Optional, List, Tuple
from nmrinator.models import IRMode


# Standard diagnostic IR regions (wavenumber in cm^-1)
IR_REGIONS: List[Tuple[float, float, str]] = [
    (3550, 3700, "O-H stretch (Free hydroxyl)"),
    (3200, 3550, "O-H stretch (H-bonded) / N-H stretch (Amine/Amide)"),
    (3250, 3350, "≡C-H stretch (Terminal alkyne)"),
    (3000, 3150, "=C-H stretch (Alkene / Aromatic)"),
    (2840, 3000, "C-H stretch (sp3 Aliphatic alkane)"),
    (2690, 2830, "C-H stretch (Aldehyde Fermi doublet)"),
    (2200, 2270, "C≡N stretch (Nitrile) / C≡C stretch (Alkyne)"),
    (2100, 2200, "C≡C stretch (Internal alkyne)"),
    (1780, 1820, "C=O stretch (Acid chloride / Anhydride)"),
    (1735, 1750, "C=O stretch (Ester)"),
    (1705, 1725, "C=O stretch (Ketone / Aldehyde)"),
    (1680, 1715, "C=O stretch (Carboxylic acid)"),
    (1630, 1690, "C=O stretch (Amide) / C=C stretch (Alkene conjugated)"),
    (1600, 1680, "C=C stretch (Alkene)"),
    (1450, 1600, "Aromatic ring C=C stretch"),
    (1350, 1480, "C-H bending (Aliphatic CH3/CH2)"),
    (1000, 1300, "C-O stretch (Alcohol / Ether / Ester)")
]


def assign_ir_band(frequency_cm: float, intensity: float) -> str:
    """Assigns a characteristic vibrational band description based on scaled frequency and intensity."""
    # Only label modes with significant IR activity (intensity > 5 km/mol)
    if intensity < 5.0:
        return "Weak vibrational mode"
        
    for low, high, desc in IR_REGIONS:
        if low <= frequency_cm <= high:
            return desc
            
    if frequency_cm < 1000:
        return "Fingerprint / Skeletal deformation"
    elif frequency_cm > 3700:
        return "High-frequency overtone"
        
    return "Skeletal / Mixed vibration"


def annotate_modes(modes: List[IRMode]) -> List[IRMode]:
    """Annotates all IR modes with characteristic functional group vibration names."""
    for m in modes:
        if not m.assignment or m.assignment == "":
            m.assignment = assign_ir_band(m.frequency_scaled, m.intensity)
    return modes

