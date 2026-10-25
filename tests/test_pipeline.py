"""End-to-end pipeline integration tests."""

import pytest
import os
from nmrinator.config import SpectraConfig
from nmrinator.pipeline import run_nmrinator


def test_pipeline_single_mode_quick(tmp_path):
    out_dir = str(tmp_path / "out_quick")
    config = SpectraConfig(
        mode="single",
        smiles="C",  # Methane
        calculate_1h_nmr=True,
        calculate_13c_nmr=True,
        calculate_ir=False,  # Skip quantum IR for instantaneous unit test
        optimize_single_conformer=False,
        output_dir=out_dir,
        verbose=False
    )
    result = run_nmrinator(config)
    assert result.smiles == "C"
    assert result.mode == "single"
    assert result.ensemble_nmr_1h is not None
    assert len(result.ensemble_nmr_1h.signals) == 1
    assert result.ensemble_nmr_1h.signals[0].integral == 4  # 4 equivalent protons in methane!
    
    assert result.ensemble_nmr_13c is not None
    assert len(result.ensemble_nmr_13c.signals) == 1
    
    assert os.path.isfile(os.path.join(out_dir, "nmrinator_result.json"))
    assert os.path.isfile(os.path.join(out_dir, "nmrinator_result_1H_NMR.csv"))
    assert os.path.isfile(os.path.join(out_dir, "nmrinator_result_1H_NMR.jdx"))


def test_cli_subprocess_execution(tmp_path):
    import subprocess
    import sys
    out_dir = str(tmp_path / "cli_out")
    cmd = [
        sys.executable, "-m", "nmrinator.cli",
        "-s", "C",
        "--mode", "single",
        "--no-ir",
        "--threads", "2",
        "--temp", "298.15",
        "-o", out_dir
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0
    assert "CALCOLO COMPLETATO CON SUCCESSO" in proc.stdout
    assert os.path.isfile(os.path.join(out_dir, "nmrinator_result.json"))

