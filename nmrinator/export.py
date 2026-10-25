"""Export utilities for CSV tables, JSON serialization, and JCAMP-DX standard files."""

import os
import csv
import json
from typing import Dict, Any, Optional

from nmrinator.models import SpectraResult
from nmrinator.io import export_to_jcamp_dx


def export_all_results(result: SpectraResult, output_dir: str, base_name: str = "nmrinator_result") -> Dict[str, str]:
    """Exports all spectral data to CSV, JSON, and standard JCAMP-DX files.
    
    Returns a dictionary mapping export formats to generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    exported_files = {}
    
    # 1. Full JSON serialization
    json_path = os.path.join(output_dir, f"{base_name}.json")
    result.save_json(json_path)
    exported_files["json"] = json_path
    
    # 2. 1H NMR CSV Table
    spec_1h = result.ensemble_nmr_1h or (result.conformers[0].nmr_1h if result.conformers else None)
    if spec_1h and spec_1h.signals:
        csv_1h_path = os.path.join(output_dir, f"{base_name}_1H_NMR.csv")
        with open(csv_1h_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Delta_ppm", "Multiplicity", "J_couplings_Hz", "Integral", "Atom_Indices", "Assignment"])
            for s in spec_1h.signals:
                j_str = ";".join(str(j) for j in s.coupling_j) if s.coupling_j else ""
                idx_str = ";".join(str(i) for i in s.atom_indices)
                writer.writerow([s.delta, s.multiplicity, j_str, s.integral, idx_str, s.label])
        exported_files["1h_csv"] = csv_1h_path
        
        # 1H JCAMP-DX
        if spec_1h.ppm_axis and spec_1h.intensity_axis:
            jdx_1h = export_to_jcamp_dx(
                spectrum_type="NMR",
                title=f"1H NMR Spectrum - {result.smiles or 'Molecule'}",
                x_axis=spec_1h.ppm_axis,
                y_axis=spec_1h.intensity_axis,
                nucleus="1H",
                frequency_mhz=result.spectrometer_mhz,
                solv_name=result.solvent
            )
            jdx_1h_path = os.path.join(output_dir, f"{base_name}_1H_NMR.jdx")
            with open(jdx_1h_path, "w", encoding="utf-8") as f:
                f.write(jdx_1h)
            exported_files["1h_jcamp"] = jdx_1h_path
            
    # 3. 13C NMR CSV Table
    spec_13c = result.ensemble_nmr_13c or (result.conformers[0].nmr_13c if result.conformers else None)
    if spec_13c and spec_13c.signals:
        csv_13c_path = os.path.join(output_dir, f"{base_name}_13C_NMR.csv")
        with open(csv_13c_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Delta_ppm", "Multiplicity", "Integral", "Atom_Indices", "Assignment"])
            for s in spec_13c.signals:
                idx_str = ";".join(str(i) for i in s.atom_indices)
                writer.writerow([s.delta, s.multiplicity, s.integral, idx_str, s.label])
        exported_files["13c_csv"] = csv_13c_path
        
        # 13C JCAMP-DX
        if spec_13c.ppm_axis and spec_13c.intensity_axis:
            jdx_13c = export_to_jcamp_dx(
                spectrum_type="NMR",
                title=f"13C NMR Spectrum - {result.smiles or 'Molecule'}",
                x_axis=spec_13c.ppm_axis,
                y_axis=spec_13c.intensity_axis,
                nucleus="13C",
                frequency_mhz=result.spectrometer_mhz / 4.0,  # 13C frequency is ~ 1/4 of 1H
                solv_name=result.solvent
            )
            jdx_13c_path = os.path.join(output_dir, f"{base_name}_13C_NMR.jdx")
            with open(jdx_13c_path, "w", encoding="utf-8") as f:
                f.write(jdx_13c)
            exported_files["13c_jcamp"] = jdx_13c_path
            
    # 4. IR Modes CSV Table & JCAMP-DX
    spec_ir = result.ensemble_ir or (result.conformers[0].ir if result.conformers else None)
    if spec_ir and spec_ir.modes:
        csv_ir_path = os.path.join(output_dir, f"{base_name}_IR_Modes.csv")
        with open(csv_ir_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Mode_Index", "Freq_Calc_cm-1", "Freq_Scaled_cm-1", "Intensity_km_mol", "Is_Imaginary", "Assignment"])
            for m in spec_ir.modes:
                writer.writerow([m.mode_index, m.frequency_calc, m.frequency_scaled, m.intensity, m.is_imaginary, m.assignment])
        exported_files["ir_csv"] = csv_ir_path
        
        # IR JCAMP-DX
        if spec_ir.wavenumber_axis and spec_ir.transmittance_axis:
            jdx_ir = export_to_jcamp_dx(
                spectrum_type="IR",
                title=f"IR Spectrum - {result.smiles or 'Molecule'}",
                x_axis=spec_ir.wavenumber_axis,
                y_axis=spec_ir.transmittance_axis,
                solv_name=result.solvent
            )
            jdx_ir_path = os.path.join(output_dir, f"{base_name}_IR.jdx")
            with open(jdx_ir_path, "w", encoding="utf-8") as f:
                f.write(jdx_ir)
            exported_files["ir_jcamp"] = jdx_ir_path

    # 5. Conformers summary CSV
    if len(result.conformers) > 1:
        csv_confs_path = os.path.join(output_dir, f"{base_name}_Conformers.csv")
        with open(csv_confs_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Conf_ID", "Energy_kcal_mol", "Boltzmann_Pop_pct"])
            for c in result.conformers:
                writer.writerow([c.conf_id, c.energy_kcalmol, c.boltzmann_pop])
        exported_files["conformers_csv"] = csv_confs_path

    return exported_files

