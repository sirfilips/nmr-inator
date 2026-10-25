"""Boltzmann ensemble averaging for NMR chemical shifts and scalar couplings."""

from typing import List, Optional, Dict
import numpy as np

from nmrinator.models import ConformerSpectralData, NMRSpectrum, NMRSignal


def compute_ensemble_nmr(
    conformers: List[ConformerSpectralData],
    nucleus: str = "1H",
    spectrometer_mhz: float = 400.0
) -> Optional[NMRSpectrum]:
    """Computes the Boltzmann-weighted ensemble NMR spectrum from individual conformer NMR spectra.
    
    In solution at room temperature, conformational exchange is fast on the NMR timescale,
    so observed chemical shifts <delta> and scalar couplings <J> are populations-weighted averages:
        <delta> = sum_k (P_k * delta_k)
        <J> = sum_k (P_k * J_k)
    """
    valid_confs = []
    for c in conformers:
        spec = c.nmr_1h if nucleus.upper() in ["1H", "H"] else c.nmr_13c
        if spec and spec.signals:
            valid_confs.append((c, spec))
            
    if not valid_confs:
        return None
        
    total_pop = sum(c.boltzmann_pop for c, _ in valid_confs)
    if total_pop <= 1e-6:
        total_pop = 100.0
        for c, _ in valid_confs:
            c.boltzmann_pop = 100.0 / len(valid_confs)
            
    # Group signals by their atom indices (which identify the specific functional nucleus)
    signal_groups: Dict[tuple, List[dict]] = {}
    
    for c, spec in valid_confs:
        weight = c.boltzmann_pop / total_pop
        for s in spec.signals:
            key = tuple(sorted(s.atom_indices))
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append({
                "weight": weight,
                "delta": s.delta,
                "multiplicity": s.multiplicity,
                "coupling_j": s.coupling_j,
                "integral": s.integral,
                "label": s.label
            })
            
    averaged_signals: List[NMRSignal] = []
    for atom_key, obs_list in signal_groups.items():
        avg_delta = sum(obs["weight"] * obs["delta"] for obs in obs_list)
        integral = obs_list[0]["integral"]
        label = obs_list[0]["label"]
        mult = obs_list[0]["multiplicity"]
        
        # Average J couplings if present
        avg_js = []
        max_j_len = max((len(obs["coupling_j"]) for obs in obs_list), default=0)
        for j_idx in range(max_j_len):
            j_vals = [obs["coupling_j"][j_idx] for obs in obs_list if len(obs["coupling_j"]) > j_idx]
            weights = [obs["weight"] for obs in obs_list if len(obs["coupling_j"]) > j_idx]
            if sum(weights) > 0:
                avg_j = sum(w * j for w, j in zip(weights, j_vals)) / sum(weights)
                avg_js.append(round(float(avg_j), 1))
                
        sig = NMRSignal(
            nucleus=nucleus,
            delta=round(float(avg_delta), 3),
            multiplicity=mult,
            coupling_j=avg_js,
            integral=integral,
            atom_indices=list(atom_key),
            label=label
        )
        averaged_signals.append(sig)
        
    averaged_signals.sort(key=lambda s: s.delta, reverse=True)
    
    return NMRSpectrum(
        nucleus=nucleus,
        spectrometer_mhz=spectrometer_mhz,
        signals=averaged_signals
    )

