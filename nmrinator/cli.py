"""Command-line interface (CLI) for nmr-inator."""

import argparse
import sys
import os
import json
import re
from datetime import datetime

from nmrinator.config import SpectraConfig
from nmrinator.pipeline import run_nmrinator


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="nmr-inator: Quantum and Ensemble-Averaged NMR (1H, 13C) and IR Spectroscopy Prediction Suite."
    )
    
    # Input group
    input_group = parser.add_argument_group("Input Molecolare")
    input_group.add_argument("-s", "--smiles", type=str, help="Stringa SMILES della molecola.")
    input_group.add_argument("-x", "--xyz", type=str, help="Percorso del file .xyz contenente coordinate 3D.")
    input_group.add_argument("-m", "--mol", type=str, help="Percorso del file .mol o .sdf.")
    
    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "single"],
        default="full",
        help="Modalità di calcolo: 'full' (ensemble conformazionale Boltzmann) o 'single' (conformero unico)."
    )
    
    # Spectroscopy switches
    parser.add_argument("--no-1h", action="store_true", help="Disattiva la predizione NMR 1H.")
    parser.add_argument("--no-13c", action="store_true", help="Disattiva la predizione NMR 13C.")
    parser.add_argument("--no-ir", action="store_true", help="Disattiva il calcolo frequenze quantistiche IR.")
    
    # Parameters
    parser.add_argument("--mhz", type=float, default=400.0, help="Frequenza operativa dello spettrometro NMR in MHz (default: 400).")
    parser.add_argument("--method", type=str, default="b3lyp", help="Metodo DFT/HF per Psi4 (default: b3lyp).")
    parser.add_argument("--basis", type=str, default="6-31G(d)", help="Basis set per Psi4 (default: 6-31G(d)).")
    parser.add_argument("--solvent", type=str, default=None, help="Solvente implicito PCM (es. chloroform, water, dmso).")
    parser.add_argument("--confs", type=int, default=30, help="Numero di conformeri da generare in modalità full (default: 30).")
    parser.add_argument("--top-confs", type=int, default=5, help="Numero max di conformeri su cui calcolare l'IR quantistico (default: 5).")
    parser.add_argument("--threads", type=int, default=4, help="Numero di thread CPU per Psi4 (default: 4).")
    parser.add_argument("--temp", type=float, default=298.15, help="Temperatura in Kelvin per la distribuzione di Boltzmann (default: 298.15).")
    
    # Output
    parser.add_argument("-o", "--output", type=str, default="output_nmrinator", help="Directory in cui esportare file CSV, JSON e JCAMP-DX.")
    
    args = parser.parse_args()
    
    if not (args.smiles or args.xyz or args.mol):
        parser.error("È necessario fornire almeno uno tra --smiles, --xyz o --mol.")
        
    if not args.output or args.output == "output_nmrinator":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        smi_part = re.sub(r'[^a-zA-Z0-9]', '', args.smiles)[:10] if args.smiles else "mol"
        target_out_dir = os.path.join("output", f"session_{ts}_{smi_part}")
    else:
        target_out_dir = args.output

    os.makedirs(target_out_dir, exist_ok=True)

    config = SpectraConfig(
        mode=args.mode,
        smiles=args.smiles,
        xyz_path=args.xyz,
        mol_path=args.mol,
        calculate_1h_nmr=not args.no_1h,
        calculate_13c_nmr=not args.no_13c,
        calculate_ir=not args.no_ir,
        spectrometer_mhz=args.mhz,
        psi4_method=args.method,
        psi4_basis=args.basis,
        solvent=args.solvent,
        psi4_threads=args.threads,
        temperature_k=args.temp,
        n_conformers=args.confs,
        top_n_confs=args.top_confs,
        output_dir=target_out_dir,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("           NMR-INATOR: QUANTUM SPECTROSCOPY SUITE")
    print("=" * 60)
    print(f"Modalità:           {config.mode.upper()}")
    print(f"Input:              {args.smiles or args.xyz or args.mol}")
    print(f"Calcoli:            1H NMR: {config.calculate_1h_nmr} | 13C NMR: {config.calculate_13c_nmr} | IR: {config.calculate_ir}")
    print(f"Spettrometro:       {config.spectrometer_mhz:.0f} MHz")
    print(f"Metodo/Basis:       {config.psi4_method}/{config.psi4_basis}")
    if config.solvent:
        print(f"Solvente:           {config.solvent}")
    print(f"Cartella Output:    {config.output_dir}")
    print("=" * 60 + "\n")
    
    try:
        result = run_nmrinator(config)
    except Exception as e:
        print(f"\n[ERRORE] Esecuzione fallita: {e}", file=sys.stderr)
        sys.exit(1)

    # Save session metadata for GUI discovery and session history
    session_info = {
        "session_name": os.path.basename(config.output_dir),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "smiles": result.smiles,
        "mode": result.mode.upper(),
        "spectrometer_mhz": result.spectrometer_mhz,
        "psi4_method": result.psi4_method,
        "psi4_basis": result.psi4_basis,
        "solvent": result.solvent or "gas",
        "elapsed_time": result.elapsed_time,
        "n_conformers": len(result.conformers),
        "has_1h": result.ensemble_nmr_1h is not None,
        "has_13c": result.ensemble_nmr_13c is not None,
        "has_ir": result.ensemble_ir is not None
    }
    try:
        with open(os.path.join(config.output_dir, "session_info.json"), "w", encoding="utf-8") as f:
            json.dump(session_info, f, indent=2)
    except Exception:
        pass
        
    print("\n" + "=" * 60)
    print(f"  CALCOLO COMPLETATO CON SUCCESSO IN {result.elapsed_time}s")
    print("=" * 60)
    
    if result.ensemble_nmr_1h and result.ensemble_nmr_1h.signals:
        print(f"\n[1H NMR Peaks] ({len(result.ensemble_nmr_1h.signals)} segnali mediati):")
        for s in result.ensemble_nmr_1h.signals:
            j_txt = f"J={s.coupling_j} Hz" if s.coupling_j else ""
            print(f"  * δ {s.delta:6.2f} ppm ({s.integral}H, {s.multiplicity}) {j_txt:<15} - {s.label}")
            
    if result.ensemble_nmr_13c and result.ensemble_nmr_13c.signals:
        print(f"\n[13C NMR Peaks] ({len(result.ensemble_nmr_13c.signals)} segnali mediati):")
        for s in result.ensemble_nmr_13c.signals:
            print(f"  * δ {s.delta:6.1f} ppm ({s.integral}C) - {s.label}")
            
    if result.ensemble_ir and result.ensemble_ir.modes:
        strong_modes = [m for m in result.ensemble_ir.modes if m.intensity > 15.0 and not m.is_imaginary]
        print(f"\n[Bande Principali IR] ({len(strong_modes)} bande significative):")
        for m in strong_modes[:6]:
            print(f"  * ν {m.frequency_scaled:6.0f} cm⁻¹ (I={m.intensity:5.1f} km/mol) - {m.assignment}")
            
    if "exported_files" in result.metadata:
        print(f"\n[Esportazioni JCAMP-DX & CSV salvate in '{config.output_dir}':]")
        for k, path in result.metadata["exported_files"].items():
            print(f"  * [{k}] {os.path.basename(path)}")
            
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
