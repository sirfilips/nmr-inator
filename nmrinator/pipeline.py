"""Master orchestration pipeline for NMR and IR spectroscopic simulations."""

import os
import sys
import time
import logging
from typing import Optional, List
from rdkit import Chem
from tqdm import tqdm

from nmrinator.config import SpectraConfig
from nmrinator.models import (
    SpectraResult,
    ConformerSpectralData,
    NMRSpectrum,
    IRSpectrum
)
from nmrinator.io import load_molecule
from nmrinator.quantum.psi4_ir import calculate_psi4_ir
from nmrinator.quantum.psi4_runner import optimize_geometry_psi4
from nmrinator.nmr.predictor import predict_conformer_nmr
from nmrinator.nmr.simulation import simulate_nmr_spectrum
from nmrinator.nmr.averaging import compute_ensemble_nmr
from nmrinator.ir.scaling import get_scaling_factor
from nmrinator.ir.convolution import simulate_ir_spectrum
from nmrinator.ir.averaging import compute_ensemble_ir
from nmrinator.export import export_all_results

logger = logging.getLogger("nmrinator")


def run_nmrinator(config: SpectraConfig) -> SpectraResult:
    """Executes the complete spectroscopy simulation workflow according to the provided configuration."""
    start_time = time.time()
    
    # Configure logging
    log_level = logging.INFO if config.verbose else logging.WARNING
    logger.setLevel(log_level)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(log_level)
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    else:
        for h in logger.handlers:
            h.setLevel(log_level)
    
    logger.info("Avvio della pipeline nmr-inator...")
    logger.info(f"Modalità: {config.mode.upper()} | 1H NMR: {config.calculate_1h_nmr} | 13C NMR: {config.calculate_13c_nmr} | IR: {config.calculate_ir}")
    
    # Step 1: Load molecule
    mol, init_xyz, canonical_smiles = load_molecule(
        smiles=config.smiles,
        xyz_path=config.xyz_path,
        mol_path=config.mol_path,
        mol_block=config.mol_block
    )
    
    s_factor = config.ir_scaling_factor or get_scaling_factor(config.psi4_method, config.psi4_basis)
    conformers_data: List[ConformerSpectralData] = []
    
    # =========================================================================
    # MODALITÀ SINGOLA (Single Conformer Mode)
    # =========================================================================
    if config.mode.lower() == "single":
        logger.info("Esecuzione in Modalità Singola (Conformero Unico)...")
        opt_xyz = init_xyz
        opt_energy_kcal = 0.0
        
        # Optionally optimize geometry with Psi4
        if config.optimize_single_conformer:
            logger.info("Ottimizzazione quantistica del conformero con Psi4...")
            try:
                opt_xyz, opt_energy_hartree = optimize_geometry_psi4(
                    xyz_str=init_xyz,
                    method=config.psi4_method,
                    basis=config.psi4_basis,
                    solvent=config.solvent,
                    memory=config.psi4_memory,
                    threads=config.psi4_threads
                )
                opt_energy_kcal = opt_energy_hartree * 627.509474
            except Exception as e:
                logger.warning(f"Ottimizzazione Psi4 non riuscita ({e}), procedo con geometria iniziale.")
                
        # 1. 1H NMR
        spec_1h = None
        if config.calculate_1h_nmr:
            logger.info("Calcolo spettro 1H NMR...")
            raw_1h = predict_conformer_nmr(
                mol=mol,
                xyz_str=opt_xyz,
                nucleus="1H",
                spectrometer_mhz=config.spectrometer_mhz,
                apply_3d_corrections=config.apply_3d_corrections
            )
            spec_1h = simulate_nmr_spectrum(raw_1h, linewidth_hz=config.nmr_linewidth_hz)
            
        # 2. 13C NMR
        spec_13c = None
        if config.calculate_13c_nmr:
            logger.info("Calcolo spettro 13C NMR...")
            raw_13c = predict_conformer_nmr(
                mol=mol,
                xyz_str=opt_xyz,
                nucleus="13C",
                spectrometer_mhz=config.spectrometer_mhz,
                apply_3d_corrections=config.apply_3d_corrections
            )
            spec_13c = simulate_nmr_spectrum(raw_13c, linewidth_hz=config.nmr_linewidth_hz)
            
        # 3. IR (Psi4)
        spec_ir = None
        if config.calculate_ir:
            logger.info(f"Calcolo frequenze quantistiche IR con Psi4 ({config.psi4_method}/{config.psi4_basis})...")
            try:
                modes, _ = calculate_psi4_ir(
                    xyz_str=opt_xyz,
                    method=config.psi4_method,
                    basis=config.psi4_basis,
                    solvent=config.solvent,
                    memory=config.psi4_memory,
                    threads=config.psi4_threads,
                    scaling_factor=s_factor
                )
                spec_ir = simulate_ir_spectrum(modes=modes, scaling_factor=s_factor, fwhm=config.ir_fwhm)
            except Exception as e:
                logger.error(f"Errore durante il calcolo IR Psi4: {e}")
                
        conf_data = ConformerSpectralData(
            conf_id=1,
            energy_kcalmol=opt_energy_kcal,
            boltzmann_pop=100.0,
            xyz=opt_xyz,
            nmr_1h=spec_1h,
            nmr_13c=spec_13c,
            ir=spec_ir
        )
        conformers_data.append(conf_data)
        
        ensemble_1h = spec_1h
        ensemble_13c = spec_13c
        ensemble_ir = spec_ir

    # =========================================================================
    # MODALITÀ COMPLETA (Ensemble Conformer Boltzmann Mode via energy-inator)
    # =========================================================================
    else:
        logger.info("Esecuzione in Modalità Completa (Ensemble Boltzmann)...")
        try:
            from energyinator.config import CalculationConfig as EnergyConfig
            from energyinator.pipeline import run_pipeline as run_energyinator
        except ImportError:
            raise ImportError(
                "Il pacchetto 'energyinator' non è installato nell'ambiente Python. "
                "Installa la cartella 'energy-inator' con: pip install -e ../energy-inator"
            )

        logger.info(f"Generazione ed esplorazione conformazionale con energy-inator (N={config.n_conformers})...")
        energy_cfg = EnergyConfig(
            smiles=canonical_smiles,
            n_conformers=config.n_conformers,
            rmsd_threshold=config.rmsd_threshold,
            force_field=config.force_field,
            use_xtb=config.use_xtb,
            psi4_method=config.psi4_method,
            psi4_basis=config.psi4_basis,
            top_n=config.top_n_confs,
            sp_only=False,
            solvent=config.solvent,
            temperature_k=config.temperature_k,
            verbose=False
        )
        
        energy_result = run_energyinator(energy_cfg)
        raw_confs = energy_result.conformers
        if not raw_confs:
            raise RuntimeError("energy-inator non ha restituito conformeri validi.")
            
        logger.info(f"Ottenuti {len(raw_confs)} conformeri ottimizzati da energy-inator.")
        
        # Iterate over conformers to calculate NMR and IR
        for i, c_res in enumerate(raw_confs):
            c_id = c_res.conf_id
            c_energy = c_res.dft_energy_kcalmol or c_res.ff_energy
            c_pop = c_res.boltzmann_pop if c_res.boltzmann_pop is not None else (100.0 / len(raw_confs))
            c_xyz = c_res.xyz
            
            logger.info(f"Analisi Spettroscopica Conformero #{c_id} (Popolazione: {c_pop:.2f}%)...")
            
            # 1. 1H NMR for this conformer
            c_spec_1h = None
            if config.calculate_1h_nmr:
                raw_1h = predict_conformer_nmr(
                    mol=mol,
                    xyz_str=c_xyz,
                    nucleus="1H",
                    spectrometer_mhz=config.spectrometer_mhz,
                    apply_3d_corrections=config.apply_3d_corrections
                )
                c_spec_1h = simulate_nmr_spectrum(raw_1h, linewidth_hz=config.nmr_linewidth_hz)
                
            # 2. 13C NMR for this conformer
            c_spec_13c = None
            if config.calculate_13c_nmr:
                raw_13c = predict_conformer_nmr(
                    mol=mol,
                    xyz_str=c_xyz,
                    nucleus="13C",
                    spectrometer_mhz=config.spectrometer_mhz,
                    apply_3d_corrections=config.apply_3d_corrections
                )
                c_spec_13c = simulate_nmr_spectrum(raw_13c, linewidth_hz=config.nmr_linewidth_hz)
                
            # 3. IR (Psi4) - only compute if population is above threshold (avoids wasting CPU on <1% conformers)
            c_spec_ir = None
            if config.calculate_ir and c_pop >= config.min_boltzmann_pop and i < config.top_n_confs:
                logger.info(f"Calcolo frequenze IR Psi4 per conformero #{c_id} ({c_pop:.1f}%)...")
                try:
                    modes, _ = calculate_psi4_ir(
                        xyz_str=c_xyz,
                        method=config.psi4_method,
                        basis=config.psi4_basis,
                        solvent=config.solvent,
                        memory=config.psi4_memory,
                        threads=config.psi4_threads,
                        scaling_factor=s_factor
                    )
                    c_spec_ir = simulate_ir_spectrum(modes=modes, scaling_factor=s_factor, fwhm=config.ir_fwhm)
                except Exception as e:
                    logger.warning(f"Calcolo IR fallito per conformero #{c_id}: {e}")
                    
            conf_data = ConformerSpectralData(
                conf_id=c_id,
                energy_kcalmol=c_energy,
                boltzmann_pop=c_pop,
                xyz=c_xyz,
                nmr_1h=c_spec_1h,
                nmr_13c=c_spec_13c,
                ir=c_spec_ir
            )
            conformers_data.append(conf_data)
            
        # Compute Boltzmann Ensemble-Averaged spectra
        logger.info("Calcolo delle medie di Boltzmann spettrali...")
        ensemble_1h = None
        if config.calculate_1h_nmr:
            raw_ens_1h = compute_ensemble_nmr(conformers_data, nucleus="1H", spectrometer_mhz=config.spectrometer_mhz)
            if raw_ens_1h:
                ensemble_1h = simulate_nmr_spectrum(raw_ens_1h, linewidth_hz=config.nmr_linewidth_hz)
                
        ensemble_13c = None
        if config.calculate_13c_nmr:
            raw_ens_13c = compute_ensemble_nmr(conformers_data, nucleus="13C", spectrometer_mhz=config.spectrometer_mhz)
            if raw_ens_13c:
                ensemble_13c = simulate_nmr_spectrum(raw_ens_13c, linewidth_hz=config.nmr_linewidth_hz)
                
        ensemble_ir = None
        if config.calculate_ir:
            ensemble_ir = compute_ensemble_ir(conformers_data, scaling_factor=s_factor, fwhm=config.ir_fwhm)

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Pipeline completata con successo in {elapsed}s.")
    
    result = SpectraResult(
        smiles=canonical_smiles,
        mode=config.mode,
        temperature_k=config.temperature_k,
        spectrometer_mhz=config.spectrometer_mhz,
        psi4_method=config.psi4_method,
        psi4_basis=config.psi4_basis,
        solvent=config.solvent,
        ir_scaling_factor=s_factor,
        conformers=conformers_data,
        ensemble_nmr_1h=ensemble_1h,
        ensemble_nmr_13c=ensemble_13c,
        ensemble_ir=ensemble_ir,
        elapsed_time=elapsed
    )
    
    # Export results if output_dir specified
    if config.output_dir:
        export_files = export_all_results(result, config.output_dir)
        result.metadata["exported_files"] = export_files
        logger.info(f"File esportati in: {config.output_dir}")
        
    return result

