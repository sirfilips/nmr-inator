"""NMR-INATOR Studio: Professional Scientific Streamlit GUI for Quantum NMR and IR Spectroscopy."""

import os
import sys
import tempfile
import json
import re
import shutil
import subprocess
from datetime import datetime
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# Configure page with Google Material Symbols icon and scientific title
st.set_page_config(
    page_title="NMR-INATOR - Quantum Spectroscopy Studio",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="expanded"
)

from nmrinator.config import SpectraConfig
from nmrinator.models import SpectraResult
from nmrinator.pipeline import run_nmrinator
from nmrinator.visualization.plot_nmr import create_nmr_plot
from nmrinator.visualization.plot_ir import create_ir_plot
from nmrinator.io import export_to_jcamp_dx

# Ketcher 2D Editor
try:
    from streamlit_ketcher import st_ketcher
    HAS_KETCHER = True
except ImportError:
    HAS_KETCHER = False


def apply_scientific_theme():
    """Applies modern scientific design styling matching Energyinator Studio."""
    st.markdown("""
    <style>
        .stApp {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        .main-header-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 10px;
            padding: 18px 24px;
            color: #f8fafc;
            margin-bottom: 20px;
        }
        .main-title {
            font-size: 2.0rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #60a5fa 0%, #818cf8 50%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }
        .sub-title {
            font-size: 0.96rem;
            color: #94a3b8;
            font-weight: 400;
            margin: 0;
        }
        .metric-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            background: rgba(59, 130, 246, 0.12);
            color: #3b82f6;
            border: 1px solid rgba(59, 130, 246, 0.25);
            margin-right: 6px;
        }
        .structure-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 10px;
            margin-bottom: 12px;
        }
    </style>
    """, unsafe_allow_html=True)


def smiles_to_2d_molblock(smiles: str) -> str:
    """Generates clean 2D coordinates and converts a SMILES string to MolBlock V2000 for Ketcher."""
    if not smiles or not smiles.strip():
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return ""
        AllChem.Compute2DCoords(mol)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return ""


def mol_output_to_smiles(raw_output: str) -> str:
    """Converts Ketcher output (MolBlock or SMILES) to canonical SMILES string without stripping header lines."""
    if not raw_output:
        return ""
    clean = raw_output
    if "\n" not in clean:
        try:
            m = Chem.MolFromSmiles(clean.strip())
            if m:
                return Chem.MolToSmiles(m)
        except Exception:
            pass
        return clean.strip()
    
    # Try parsing molblock directly without strip()
    mol = Chem.MolFromMolBlock(clean, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMolBlock(clean.strip(), removeHs=False)
    if mol is None and not clean.startswith("\n"):
        mol = Chem.MolFromMolBlock("\n" + clean, removeHs=False)
        
    if mol is not None:
        try:
            return Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception:
            return Chem.MolToSmiles(mol)
    return ""


def render_plot_download_buttons(fig, base_name: str):
    """Renders high-resolution PNG and vector SVG download buttons for a Plotly chart."""
    col1, col2, _ = st.columns([1.5, 1.5, 4])
    try:
        png_data = fig.to_image(format="png", width=1200, height=650, scale=2)
        with col1:
            st.download_button(
                "Scarica Grafico (PNG)",
                data=png_data,
                file_name=f"{base_name}.png",
                mime="image/png",
                icon=":material/download:",
                use_container_width=True
            )
    except Exception as e:
        st.caption(f"Export PNG non disponibile: {e}")
        
    try:
        svg_data = fig.to_image(format="svg", width=1200, height=650)
        with col2:
            st.download_button(
                "Scarica Grafico (SVG)",
                data=svg_data,
                file_name=f"{base_name}.svg",
                mime="image/svg+xml",
                icon=":material/download:",
                use_container_width=True
            )
    except Exception as e:
        st.caption(f"Export SVG non disponibile: {e}")


def stop_active_calculation():
    """Terminates any active calculation process and sub-processes immediately."""
    proc = st.session_state.get("active_proc")
    if proc is not None:
        try:
            pid = proc.pid
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
            else:
                proc.terminate()
                proc.kill()
            st.warning("Calcolo quantistico interrotto dall'utente.")
        except Exception as e:
            st.error(f"Errore durante l'interruzione: {e}")
        st.session_state["active_proc"] = None
        st.session_state["is_running"] = False


def get_project_output_dir() -> str:
    """Returns absolute path to project's output/ directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(script_dir)
    out_dir = os.path.join(proj_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_available_sessions(output_base: str):
    """Scans output directory and returns list of valid session dictionaries sorted chronologically."""
    if not os.path.isdir(output_base):
        return []
    sessions = []
    for entry in sorted(os.listdir(output_base), reverse=True):
        full_path = os.path.join(output_base, entry)
        if not os.path.isdir(full_path):
            continue
        json_file = os.path.join(full_path, "nmrinator_result.json")
        if not os.path.exists(json_file):
            continue
        
        info_file = os.path.join(full_path, "session_info.json")
        info = {}
        if os.path.exists(info_file):
            try:
                with open(info_file, "r", encoding="utf-8", errors="replace") as f:
                    info = json.load(f)
            except Exception:
                pass
        if not info:
            try:
                with open(json_file, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                    info = {
                        "session_name": entry,
                        "timestamp": "-",
                        "smiles": data.get("smiles", "Molecola"),
                        "mode": data.get("mode", "FULL").upper(),
                        "spectrometer_mhz": data.get("spectrometer_mhz", 400.0),
                        "psi4_method": data.get("psi4_method", "b3lyp"),
                        "psi4_basis": data.get("psi4_basis", "6-31G(d)"),
                        "solvent": data.get("solvent") or "Gas",
                        "elapsed_time": data.get("elapsed_time", 0.0),
                        "has_1h": data.get("ensemble_nmr_1h") is not None,
                        "has_13c": data.get("ensemble_nmr_13c") is not None,
                        "has_ir": data.get("ensemble_ir") is not None
                    }
            except Exception:
                continue

        smi = info.get("smiles") or "Molecola"
        mode = info.get("mode", "FULL").upper()
        mhz = info.get("spectrometer_mhz", 400.0)
        label = f"{entry} | {smi} ({mode}, {mhz:.0f}MHz)"
        sessions.append({
            "dir_name": entry,
            "full_path": full_path,
            "label": label,
            "info": info
        })
    return sessions


def render_session_manager(output_base: str):
    """Renders the session archive explorer in the sidebar."""
    sessions = get_available_sessions(output_base)
    if not sessions:
        with st.sidebar.expander("Sessioni Archiviate (0)", expanded=False):
            st.caption("Nessuna sessione precedente trovata nella cartella output.")
        return

    with st.sidebar.expander(f"Sessioni Archiviate ({len(sessions)})", expanded=False):
        session_labels = [s["label"] for s in sessions]
        sel_label = st.selectbox(
            "Seleziona sessione salvata:",
            session_labels,
            key="sel_archived_session_box"
        )
        selected_session = next(s for s in sessions if s["label"] == sel_label)
        s_info = selected_session["info"]
        s_path = selected_session["full_path"]
        s_name = selected_session["dir_name"]

        # Mini card with session details
        st.markdown(
            f"<div style='font-size: 0.85rem; background: rgba(30, 41, 59, 0.7); padding: 8px 10px; border-radius: 6px; margin: 6px 0; border: 1px solid rgba(59, 130, 246, 0.2);'>"
            f"<b>SMILES:</b> <code>{s_info.get('smiles', '-')}</code><br>"
            f"<b>Modalità:</b> {s_info.get('mode', '-')}<br>"
            f"<b>Metodo:</b> {s_info.get('psi4_method', '-')}/{s_info.get('psi4_basis', '-')}<br>"
            f"<b>Spettri:</b> "
            f"{'¹H ' if s_info.get('has_1h') else ''}"
            f"{'¹³C ' if s_info.get('has_13c') else ''}"
            f"{'IR' if s_info.get('has_ir') else ''}"
            f"</div>",
            unsafe_allow_html=True
        )

        col_l1, col_l2 = st.columns([2.3, 1])
        with col_l1:
            if st.button("Carica Sessione", icon=":material/history:", type="primary", use_container_width=True, key=f"btn_load_{s_name}"):
                jpath = os.path.join(s_path, "nmrinator_result.json")
                try:
                    loaded_res = SpectraResult.from_json(jpath)
                    st.session_state["nmrinator_result"] = loaded_res
                    st.session_state["nmrinator_output_dir"] = s_path
                    st.session_state["active_session_name"] = s_name
                    if loaded_res.smiles:
                        st.session_state["active_smiles"] = loaded_res.smiles
                        st.session_state["ketcher_version"] = st.session_state.get("ketcher_version", 0) + 1
                    
                    # Load run.log if present
                    log_file = os.path.join(s_path, "run.log")
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                                st.session_state["calc_log_text"] = f.read()
                        except Exception:
                            st.session_state["calc_log_text"] = ""
                    else:
                        st.session_state["calc_log_text"] = ""

                    st.session_state["session_status_msg"] = f"Sessione '{s_name}' caricata con successo!"
                    st.rerun()
                except Exception as e:
                    st.error(f"Errore durante il caricamento: {e}")

        with col_l2:
            if st.button("Elimina", icon=":material/delete:", type="secondary", use_container_width=True, key=f"btn_del_{s_name}"):
                try:
                    shutil.rmtree(s_path, ignore_errors=True)
                    if st.session_state.get("active_session_name") == s_name:
                        st.session_state.pop("nmrinator_result", None)
                        st.session_state.pop("nmrinator_output_dir", None)
                        st.session_state.pop("active_session_name", None)
                    st.session_state["session_status_msg"] = f"Sessione '{s_name}' eliminata."
                    st.rerun()
                except Exception as e:
                    st.error(f"Errore durante eliminazione: {e}")


def main():
    apply_scientific_theme()

    # Initialize persistent session state for active molecule
    if "active_smiles" not in st.session_state:
        st.session_state["active_smiles"] = "CCO"
    if "last_synced_smiles" not in st.session_state:
        st.session_state["last_synced_smiles"] = "CCO"
    if "smiles_text_field" not in st.session_state:
        st.session_state["smiles_text_field"] = "CCO"
    if "ketcher_version" not in st.session_state:
        st.session_state["ketcher_version"] = 0
    if "active_proc" not in st.session_state:
        st.session_state["active_proc"] = None
    if "is_running" not in st.session_state:
        st.session_state["is_running"] = False
    if "calc_log_text" not in st.session_state:
        st.session_state["calc_log_text"] = ""

    st.markdown("""
    <div class="main-header-card">
        <h1 class="main-title">NMR-INATOR Studio</h1>
        <p class="sub-title">
            Simulazione e Predizione di Spettri <b>NMR (¹H, ¹³C)</b> e <b>IR</b> con Meccanica Quantistica (Psi4) e Medie di Popolazione di Boltzmann (Energyinator).
        </p>
    </div>
    """, unsafe_allow_html=True)

    output_base = get_project_output_dir()

    if "session_status_msg" in st.session_state:
        st.sidebar.success(st.session_state.pop("session_status_msg"))

    # Sidebar: Setup & Parameters
    st.sidebar.header("Configurazione Calcolo")

    # Archived Sessions Loader
    render_session_manager(output_base)

    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "Modalità di Simulazione",
        ["Modalità Completa (Ensemble Boltzmann)", "Modalità Singola (Conformero Unico)"],
        index=0,
        help="Modalità Completa: esplora i conformeri con energy-inator, calcola energie e pesa lo spettro finale su Boltzmann.\nModalità Singola: predice gli spettri direttamente sul singolo conformero inserito."
    )
    is_full_mode = "Completa" in mode

    st.sidebar.markdown("---")
    st.sidebar.subheader("Moduli Spettroscopici")
    calc_1h = st.sidebar.checkbox("Spettro ¹H NMR (Protonico)", value=True)
    calc_13c = st.sidebar.checkbox("Spettro ¹³C NMR (Carbonio-13)", value=True)
    calc_ir = st.sidebar.checkbox("Spettro Infrarosso (IR Vibrazionale - Psi4)", value=True)

    nmr_mhz = st.sidebar.selectbox("Frequenza Spettrometro (MHz)", [300.0, 400.0, 500.0, 600.0, 800.0], index=1)
    
    with st.sidebar.expander("Parametri Quantistici (Psi4 / Metodo)", expanded=False):
        psi4_method = st.selectbox("Metodo Quantistico", ["b3lyp", "hf", "pbe0", "wb97x-d"], index=0)
        psi4_basis = st.selectbox("Basis Set", ["6-31G(d)", "def2-SVP", "sto-3g", "3-21g", "6-311G(d,p)"], index=0)
        solvent = st.selectbox(
            "Solvente Implicito (PCM)",
            ["Nessuno (Fase Gassosa)", "Chloroform", "Water", "DMSO", "Methanol", "Toluene", "Dichloromethane"],
            index=0
        )
        solv_param = None if "Nessuno" in solvent else solvent.lower()
        threads = st.slider("Thread CPU Psi4", min_value=1, max_value=16, value=8)

    if is_full_mode:
        with st.sidebar.expander("Parametri Ensemble (Energyinator)", expanded=False):
            n_confs = st.slider("Numero Conformeri da Generare", 10, 100, 30, step=5)
            top_ir = st.slider("Max Conformeri per Calcolo IR", 1, 5, 2)
            temp_k = st.number_input("Temperatura (K)", value=298.15, step=1.0)
    else:
        n_confs = 1
        top_ir = 1
        temp_k = 298.15

    # Main Area: Molecule Input - Ketcher is primary option
    st.subheader("Input Struttura Molecolare")
    input_method = st.radio(
        "Metodo di inserimento:",
        ["Disegna Struttura 2D (Ketcher)", "Stringa SMILES", "Carica File (.xyz, .mol, .sdf)"],
        index=0,
        horizontal=True
    )

    smiles_input = None
    mol_block_input = None
    xyz_input = None

    if input_method == "Disegna Struttura 2D (Ketcher)":
        col_t1, col_t2 = st.columns([3, 1.2])
        with col_t2:
            st.caption("Template rapidi:")
            tmpl = st.selectbox(
                "Carica template:",
                ["Etanolo", "Acetone", "Benzene", "Toluene", "Acido Acetico", "2-Butanolo"],
                label_visibility="collapsed",
                key="tmpl_preset_select"
            )
            tmpl_smi = {
                "Etanolo": "CCO",
                "Acetone": "CC(=O)C",
                "Benzene": "c1ccccc1",
                "Toluene": "Cc1ccccc1",
                "Acido Acetico": "CC(=O)O",
                "2-Butanolo": "CCC(C)O"
            }.get(tmpl, "CCO")
            
            if st.button("Carica nel Canvas", icon=":material/refresh:", use_container_width=True):
                st.session_state["active_smiles"] = tmpl_smi
                st.session_state["ketcher_version"] = st.session_state.get("ketcher_version", 0) + 1
                st.rerun()

        with col_t1:
            st.caption("Disegna o modifica la molecola. Poi premi **'Applica Struttura nel Calcolo'** o il pulsante **'Apply'** di Ketcher:")

        curr_smi = st.session_state.get("active_smiles", "CCO")
        molblock_2d = smiles_to_2d_molblock(curr_smi)
        k_key = f"ketcher_editor_{st.session_state.get('ketcher_version', 0)}"
        
        if HAS_KETCHER:
            ketcher_output = st_ketcher(
                value=molblock_2d,
                height=430,
                molecule_format="MOLFILE",
                key=k_key
            )
            
            # Extract drawn molecule from Ketcher output
            drawn_smi = mol_output_to_smiles(ketcher_output) if ketcher_output else ""
            if drawn_smi and drawn_smi != curr_smi:
                # Synchronize internal session state immediately upon receiving updated Ketcher drawing
                st.session_state["active_smiles"] = drawn_smi
                curr_smi = drawn_smi
                
            col_apply1, col_apply2 = st.columns([1.8, 3.2])
            with col_apply1:
                btn_apply_k = st.button(
                    "Applica Struttura nel Calcolo",
                    icon=":material/check:",
                    type="secondary",
                    use_container_width=True,
                    key="btn_apply_ketcher"
                )
                if btn_apply_k and drawn_smi:
                    st.session_state["active_smiles"] = drawn_smi
                    st.session_state["ketcher_version"] = st.session_state.get("ketcher_version", 0) + 1
                    st.rerun()

            target_active = st.session_state["active_smiles"]
            active_mol = Chem.MolFromSmiles(target_active)
            formula = rdMolDescriptors.CalcMolFormula(active_mol) if active_mol else ""
            st.markdown(
                f"<div class='structure-card'>"
                f"<b>Struttura attiva nel calcolo:</b> <code>{target_active}</code> &nbsp;|&nbsp; "
                f"<b>Formula:</b> {formula}"
                f"</div>",
                unsafe_allow_html=True
            )
            smiles_input = target_active
        else:
            st.error("Modulo streamlit-ketcher non disponibile. Utilizzare la modalità SMILES.")
            smiles_input = "CCO"

    elif input_method == "Stringa SMILES":
        col1, col2 = st.columns([2.6, 1.4])
        
        current_active = st.session_state.get("active_smiles", "CCO")
        if "smiles_text_field" not in st.session_state or st.session_state.get("last_synced_smiles") != current_active:
            st.session_state["smiles_text_field"] = current_active
            st.session_state["last_synced_smiles"] = current_active

        with col1:
            smiles_val = st.text_input(
                "Inserisci SMILES (premi Invio per applicare):",
                key="smiles_text_field",
                placeholder="Es. CC(=O)C per acetone, CCCC#N per butirronitrile"
            )
            clean_smi = smiles_val.strip() if smiles_val else ""
            if clean_smi and clean_smi != current_active:
                test_mol = Chem.MolFromSmiles(clean_smi)
                if test_mol:
                    st.session_state["active_smiles"] = clean_smi
                    st.session_state["last_synced_smiles"] = clean_smi
                    st.session_state["ketcher_version"] = st.session_state.get("ketcher_version", 0) + 1
                    current_active = clean_smi
                else:
                    st.error("Stringa SMILES non valida. Verifica la formula inserita.")
            
            smiles_input = st.session_state.get("active_smiles", current_active)

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            quick_mol = st.selectbox(
                "Esempi rapidi:",
                [
                    "Etanolo (CCO)",
                    "Acetone (CC(=O)C)",
                    "Benzene (c1ccccc1)",
                    "Toluene (Cc1ccccc1)",
                    "Acido Acetico (CC(=O)O)",
                    "Butirronitrile (CCCC#N)"
                ],
                key="quick_mol_select"
            )
            if st.button("Carica Esempio", icon=":material/input:", use_container_width=True, key="btn_load_smiles_example"):
                sample_smi = quick_mol.split("(")[-1].replace(")", "").strip()
                st.session_state["active_smiles"] = sample_smi
                st.session_state["smiles_text_field"] = sample_smi
                st.session_state["last_synced_smiles"] = sample_smi
                st.session_state["ketcher_version"] = st.session_state.get("ketcher_version", 0) + 1
                st.rerun()
                    
        active_mol = Chem.MolFromSmiles(smiles_input) if smiles_input else None
        if active_mol:
            formula = rdMolDescriptors.CalcMolFormula(active_mol)
            st.markdown(
                f"<div class='structure-card'>"
                f"<b>Struttura attiva nel calcolo:</b> <code>{smiles_input}</code> &nbsp;|&nbsp; "
                f"<b>Formula:</b> {formula} &nbsp;|&nbsp; <b>SMILES Canonico:</b> <code>{Chem.MolToSmiles(active_mol)}</code>"
                f"</div>",
                unsafe_allow_html=True
            )
        elif smiles_input:
            st.warning(f"SMILES inserito non valido: {smiles_input}")

    else:
        uploaded_file = st.file_uploader("Carica file molecolare (.xyz, .mol, .sdf)", type=["xyz", "mol", "sdf"])
        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8")
            fname = uploaded_file.name.lower()
            if fname.endswith(".xyz"):
                xyz_input = content
            else:
                mol_block_input = content
            st.success(f"File caricato con successo: {uploaded_file.name}")

    st.markdown("---")
    
    # Launch and Stop control buttons with Google Material Symbols
    col_btn1, col_btn2, _ = st.columns([2.5, 1.2, 3.3])
    with col_btn1:
        start_btn = st.button(
            "Avvia Calcolo Spettroscopico",
            type="primary",
            icon=":material/play_arrow:",
            use_container_width=True,
            disabled=st.session_state.get("is_running", False)
        )
    with col_btn2:
        stop_btn = st.button(
            "STOP",
            type="secondary",
            icon=":material/stop:",
            use_container_width=True,
            help="Interrompe immediatamente il calcolo in corso"
        )

    if stop_btn:
        stop_active_calculation()
        st.rerun()

    if start_btn:
        if not (calc_1h or calc_13c or calc_ir):
            st.error("Selezionare almeno un modulo spettroscopico (¹H NMR, ¹³C NMR o IR) nella barra laterale.")
            return

        # Create dedicated session folder inside project's output/ directory
        output_base = get_project_output_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Resolve target structure based on current input method
        if input_method == "Disegna Struttura 2D (Ketcher)":
            # If Ketcher output contains a newly drawn structure, ensure it is synchronized
            latest_smi = mol_output_to_smiles(ketcher_output) if (HAS_KETCHER and ketcher_output) else ""
            final_target_smi = latest_smi if latest_smi else st.session_state.get("active_smiles", "CCO")
            st.session_state["active_smiles"] = final_target_smi
            calc_smiles = final_target_smi
            calc_mol_path = None
            calc_xyz_path = None
        elif input_method == "Stringa SMILES":
            calc_smiles = smiles_input or st.session_state.get("active_smiles", "CCO")
            calc_mol_path = None
            calc_xyz_path = None
        else:
            calc_smiles = None
            calc_mol_path = None
            calc_xyz_path = None

        if not (calc_smiles or xyz_input or mol_block_input):
            st.error("Inserire una molecola valida prima di avviare il calcolo.")
            return

        # Generate clean formula or slug for session folder name
        clean_slug = "mol"
        if calc_smiles:
            try:
                m = Chem.MolFromSmiles(calc_smiles)
                if m:
                    clean_slug = rdMolDescriptors.CalcMolFormula(m)
                else:
                    clean_slug = re.sub(r'[^a-zA-Z0-9]', '', calc_smiles)[:12]
            except Exception:
                clean_slug = re.sub(r'[^a-zA-Z0-9]', '', calc_smiles)[:12]

        session_name = f"session_{ts}_{clean_slug}"
        session_dir = os.path.join(output_base, session_name)
        os.makedirs(session_dir, exist_ok=True)

        if input_method == "Carica File (.xyz, .mol, .sdf)":
            if xyz_input:
                temp_xyz_path = os.path.join(session_dir, "input.xyz")
                with open(temp_xyz_path, "w", encoding="utf-8") as f:
                    f.write(xyz_input)
                calc_xyz_path = temp_xyz_path
            elif mol_block_input:
                temp_mol_path = os.path.join(session_dir, "input.mol")
                with open(temp_mol_path, "w", encoding="utf-8") as f:
                    f.write(mol_block_input)
                calc_mol_path = temp_mol_path

        # Prepare isolated subprocess execution
        cmd = [
            sys.executable, "-m", "nmrinator.cli",
            "--mode", "full" if is_full_mode else "single",
            "--mhz", str(nmr_mhz),
            "--method", str(psi4_method),
            "--basis", str(psi4_basis),
            "--threads", str(int(threads)),
            "--temp", str(float(temp_k)),
            "-o", session_dir
        ]
        if calc_smiles:
            cmd.extend(["-s", calc_smiles])
        elif calc_xyz_path:
            cmd.extend(["-x", calc_xyz_path])
        elif calc_mol_path:
            cmd.extend(["-m", calc_mol_path])

        if not calc_1h:
            cmd.append("--no-1h")
        if not calc_13c:
            cmd.append("--no-13c")
        if not calc_ir:
            cmd.append("--no-ir")
        if is_full_mode:
            cmd.extend(["--confs", str(int(n_confs)), "--top-confs", str(int(top_ir))])
        if solv_param:
            cmd.extend(["--solvent", str(solv_param)])

        prog_bar = st.progress(5, text="Inizializzazione del calcolo spettroscopico...")
        log_expander = st.expander("Console di Esecuzione (Avanzamento in tempo reale)", expanded=True)
        with log_expander:
            log_box = st.empty()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            st.session_state["active_proc"] = proc
            st.session_state["is_running"] = True

            all_lines = []
            curr_pct = 5

            for line in iter(proc.stdout.readline, ''):
                all_lines.append(line)
                lower_line = line.lower()

                if "avvio della pipeline" in lower_line:
                    curr_pct = 10
                    prog_bar.progress(curr_pct, text="Caricamento struttura molecolare e normalizzazione...")
                elif "modalità singola" in lower_line:
                    curr_pct = 25
                    prog_bar.progress(curr_pct, text="Modalità Singola: ottimizzazione geometrica conformero...")
                elif "generazione ed esplorazione conformazionale" in lower_line:
                    curr_pct = 20
                    prog_bar.progress(curr_pct, text="Esplorazione conformazionale con energy-inator (RDKit / MMFF94)...")
                elif "ottenuti" in lower_line and "conformeri" in lower_line:
                    curr_pct = 40
                    prog_bar.progress(curr_pct, text="Conformeri unici selezionati. Avvio predizioni spettroscopiche...")
                elif "analisi spettroscopica conformero" in lower_line:
                    curr_pct = min(80, max(curr_pct, curr_pct + 4))
                    prog_bar.progress(curr_pct, text="Analisi NMR conformero (correzioni 3D Karplus e anisotropia magnetica)...")
                elif "calcolo frequenze ir" in lower_line or "calcolo frequenze quantistiche" in lower_line:
                    curr_pct = 82
                    prog_bar.progress(curr_pct, text=f"Calcolo quantistico frequenze IR Psi4 ({psi4_method}/{psi4_basis})...")
                elif "medie di boltzmann" in lower_line:
                    curr_pct = 92
                    prog_bar.progress(curr_pct, text="Ponderazione statistica di Boltzmann e simulazione spettri...")
                elif "file esportati in" in lower_line or "completata con successo" in lower_line:
                    curr_pct = 98
                    prog_bar.progress(curr_pct, text="Esportazione file JCAMP-DX, CSV e JSON...")

                log_box.code("".join(all_lines[-22:]), language="log")

            proc.stdout.close()
            ret_code = proc.wait()

            full_log_str = "".join(all_lines)
            st.session_state["calc_log_text"] = full_log_str

            if ret_code == 0:
                prog_bar.progress(100, text="Calcolo completato con successo!")
                result_json = os.path.join(session_dir, "nmrinator_result.json")
                if os.path.exists(result_json):
                    res = SpectraResult.from_json(result_json)

                    # Save run.log permanently in session folder
                    try:
                        with open(os.path.join(session_dir, "run.log"), "w", encoding="utf-8") as f:
                            f.write(full_log_str)
                    except Exception:
                        pass

                    # Save session_info.json metadata
                    session_info = {
                        "session_name": session_name,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "smiles": res.smiles,
                        "formula": clean_slug,
                        "mode": res.mode.upper(),
                        "spectrometer_mhz": res.spectrometer_mhz,
                        "psi4_method": res.psi4_method,
                        "psi4_basis": res.psi4_basis,
                        "solvent": res.solvent or "gas",
                        "elapsed_time": res.elapsed_time,
                        "n_conformers": len(res.conformers),
                        "has_1h": res.ensemble_nmr_1h is not None,
                        "has_13c": res.ensemble_nmr_13c is not None,
                        "has_ir": res.ensemble_ir is not None
                    }
                    try:
                        with open(os.path.join(session_dir, "session_info.json"), "w", encoding="utf-8") as f:
                            json.dump(session_info, f, indent=2)
                    except Exception:
                        pass

                    st.session_state["nmrinator_result"] = res
                    st.session_state["nmrinator_output_dir"] = session_dir
                    st.session_state["active_session_name"] = session_name
                    st.session_state["session_status_msg"] = f"Sessione '{session_name}' archiviata con successo in output/!"
                    st.rerun()
                else:
                    st.error("Il processo è terminato ma 'nmrinator_result.json' non è stato trovato.")
            else:
                prog_bar.empty()
                st.error(f"Errore durante l'esecuzione del calcolo (Codice di uscita: {ret_code}).")
                st.code(full_log_str, language="log")

        except Exception as e:
            st.error(f"Errore durante l'esecuzione: {e}")
        finally:
            st.session_state["active_proc"] = None
            st.session_state["is_running"] = False

    # Render Results if available
    if "nmrinator_result" in st.session_state:
        result = st.session_state["nmrinator_result"]
        out_dir = st.session_state.get("nmrinator_output_dir", "")
        mol_label = result.smiles or "Molecola"
        active_sess = st.session_state.get("active_session_name", "")

        sess_pill = f"<span class='metric-pill'>Sessione: {active_sess}</span>" if active_sess else ""
        st.markdown(f"### Risultati Spettroscopici: `{mol_label}`")
        st.markdown(
            f"{sess_pill}"
            f"<span class='metric-pill'>Modalità: {result.mode.upper()}</span>"
            f"<span class='metric-pill'>Metodo: {result.psi4_method}/{result.psi4_basis}</span>"
            f"<span class='metric-pill'>Solvente: {result.solvent or 'Fase gassosa'}</span>"
            f"<span class='metric-pill'>Tempo: {result.elapsed_time}s</span>",
            unsafe_allow_html=True
        )
        st.write("")

        plotly_config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"spettro_{mol_label}",
                "height": 650,
                "width": 1200,
                "scale": 2
            },
            "displaylogo": False
        }

        tab_names = []
        if result.ensemble_nmr_1h:
            tab_names.append("¹H NMR")
        if result.ensemble_nmr_13c:
            tab_names.append("¹³C NMR")
        if result.ensemble_ir:
            tab_names.append("Spettro IR")
        if len(result.conformers) > 1:
            tab_names.append("Conformeri & Boltzmann")
        tab_names.append("Esportazione & Download")
        tab_names.append("Console e Log")

        tab_objects = st.tabs(tab_names)
        tab_idx = 0

        # Tab 1: 1H NMR
        if result.ensemble_nmr_1h:
            with tab_objects[tab_idx]:
                st.markdown(f"#### Spettro ¹H NMR ({result.spectrometer_mhz:.0f} MHz)")
                st.caption("Scala ppm invertita (convenzione sperimentale da 12 a 0 ppm). Passa il mouse sui picchi per visualizzare accoppiamenti J e assegnazioni.")
                fig_1h = create_nmr_plot(result.ensemble_nmr_1h, title=f"¹H NMR ({mol_label}) @ {result.spectrometer_mhz:.0f} MHz")
                st.plotly_chart(fig_1h, use_container_width=True, config=plotly_config)

                # Download buttons for PNG and SVG
                render_plot_download_buttons(fig_1h, f"{mol_label}_1H_NMR")

                st.markdown("##### Tabella dei Segnali ¹H")
                sig_data = []
                for s in result.ensemble_nmr_1h.signals:
                    j_str = ", ".join(f"{j:.1f}" for j in s.coupling_j) if s.coupling_j else "-"
                    sig_data.append({
                        "Chemical Shift δ (ppm)": f"{s.delta:.2f}",
                        "Integrazione": f"{s.integral}H",
                        "Molteplicità": s.multiplicity,
                        "Accoppiamento J (Hz)": j_str,
                        "Assegnazione Strutturale": s.label,
                        "Indici Atomi": str(s.atom_indices)
                    })
                df_1h = pd.DataFrame(sig_data)
                st.dataframe(df_1h, use_container_width=True)
            tab_idx += 1

        # Tab 2: 13C NMR
        if result.ensemble_nmr_13c:
            with tab_objects[tab_idx]:
                st.markdown("#### Spettro ¹³C NMR Disaccoppiato")
                st.caption("Scala ppm invertita (da 220 a 0 ppm).")
                fig_13c = create_nmr_plot(result.ensemble_nmr_13c, title=f"¹³C NMR Disaccoppiato ({mol_label})", show_integral=False)
                st.plotly_chart(fig_13c, use_container_width=True, config=plotly_config)

                # Download buttons for PNG and SVG
                render_plot_download_buttons(fig_13c, f"{mol_label}_13C_NMR")

                st.markdown("##### Tabella dei Carboni ¹³C")
                c_data = []
                for s in result.ensemble_nmr_13c.signals:
                    c_data.append({
                        "Chemical Shift δ (ppm)": f"{s.delta:.1f}",
                        "Integrazione": f"{s.integral}C",
                        "Assegnazione": s.label,
                        "Indici Atomi": str(s.atom_indices)
                    })
                df_13c = pd.DataFrame(c_data)
                st.dataframe(df_13c, use_container_width=True)
            tab_idx += 1

        # Tab 3: IR Spectrum
        if result.ensemble_ir:
            with tab_objects[tab_idx]:
                st.markdown("#### Spettro Infrarosso (IR) Vibrazionale")
                col_ir_opt, _ = st.columns([2, 4])
                with col_ir_opt:
                    ir_view = st.radio("Visualizzazione Spettro:", ["% Trasmittanza (%T)", "Assorbanza"], horizontal=True)
                
                plot_mode = "transmittance" if "% Trasmittanza" in ir_view else "absorbance"
                fig_ir = create_ir_plot(result.ensemble_ir, title=f"Spettro IR ({mol_label})", plot_type=plot_mode, show_band_labels=True)
                st.plotly_chart(fig_ir, use_container_width=True, config=plotly_config)

                # Download buttons for PNG and SVG
                render_plot_download_buttons(fig_ir, f"{mol_label}_IR_{plot_mode}")

                st.markdown("##### Modi Normali di Vibrazione (Frequenze e Intensità)")
                modes_data = []
                for m in result.ensemble_ir.modes:
                    modes_data.append({
                        "Modo #": m.mode_index,
                        "Freq Scalata (cm⁻¹)": f"{m.frequency_scaled:.1f}",
                        "Freq Calcolata (cm⁻¹)": f"{m.frequency_calc:.1f}",
                        "Intensità IR (km/mol)": f"{m.intensity:.2f}",
                        "Transizione Immaginaria": "Sì" if m.is_imaginary else "No",
                        "Assegnazione": m.assignment or "-"
                    })
                df_ir = pd.DataFrame(modes_data)
                st.dataframe(df_ir, use_container_width=True)
            tab_idx += 1

        # Tab 4: Conformers
        if len(result.conformers) > 1:
            with tab_objects[tab_idx]:
                st.markdown("#### Ensemble di Conformeri Ottimizzati (Energyinator)")
                conf_data = []
                min_e = min(c.energy_kcalmol for c in result.conformers)
                for c in result.conformers:
                    delta_e = c.energy_kcalmol - min_e
                    conf_data.append({
                        "Conformero #": c.conf_id,
                        "Energia (kcal/mol)": f"{c.energy_kcalmol:.2f}",
                        "ΔE (kcal/mol)": f"{delta_e:.2f}",
                        "Popolazione Boltzmann (%)": f"{c.boltzmann_pop:.2f}%",
                        "Spettro 1H": "Disponibile" if c.nmr_1h else "-",
                        "Spettro IR": "Disponibile" if c.ir else "-"
                    })
                st.dataframe(pd.DataFrame(conf_data), use_container_width=True)
            tab_idx += 1

        # Tab 5: Export & Download
        with tab_objects[tab_idx]:
            st.markdown("#### Esportazione e Download Spettri")
            st.info(
                f"Tutti i dati, file JCAMP-DX, tabelle e log sono archiviati in modo permanente in:\n`{out_dir}`\n\n"
                f"Formati supportati: JCAMP-DX (.jdx) per MestReNova/TopSpin, tabelle CSV, JSON completo e formati grafici PNG/SVG."
            )

            st.markdown("##### File Spettrali Interoperabili (JCAMP-DX & CSV)")
            col_d1, col_d2, col_d3 = st.columns(3)
            
            # 1H JCAMP-DX download
            if result.ensemble_nmr_1h and result.ensemble_nmr_1h.ppm_axis:
                jdx_1h = export_to_jcamp_dx(
                    spectrum_type="NMR",
                    title=f"1H NMR Spectrum - {mol_label}",
                    x_axis=result.ensemble_nmr_1h.ppm_axis,
                    y_axis=result.ensemble_nmr_1h.intensity_axis,
                    nucleus="1H",
                    frequency_mhz=result.spectrometer_mhz
                )
                with col_d1:
                    st.download_button(
                        "Scarica ¹H NMR (JCAMP-DX)",
                        data=jdx_1h,
                        file_name=f"{mol_label}_1H.jdx",
                        mime="chemical/x-jcamp-dx",
                        icon=":material/download:",
                        use_container_width=True
                    )

            # 13C JCAMP-DX download
            if result.ensemble_nmr_13c and result.ensemble_nmr_13c.ppm_axis:
                jdx_13c = export_to_jcamp_dx(
                    spectrum_type="NMR",
                    title=f"13C NMR Spectrum - {mol_label}",
                    x_axis=result.ensemble_nmr_13c.ppm_axis,
                    y_axis=result.ensemble_nmr_13c.intensity_axis,
                    nucleus="13C",
                    frequency_mhz=result.spectrometer_mhz / 4.0
                )
                with col_d2:
                    st.download_button(
                        "Scarica ¹³C NMR (JCAMP-DX)",
                        data=jdx_13c,
                        file_name=f"{mol_label}_13C.jdx",
                        mime="chemical/x-jcamp-dx",
                        icon=":material/download:",
                        use_container_width=True
                    )

            # IR JCAMP-DX download
            if result.ensemble_ir and result.ensemble_ir.wavenumber_axis:
                jdx_ir = export_to_jcamp_dx(
                    spectrum_type="IR",
                    title=f"IR Spectrum - {mol_label}",
                    x_axis=result.ensemble_ir.wavenumber_axis,
                    y_axis=result.ensemble_ir.transmittance_axis
                )
                with col_d3:
                    st.download_button(
                        "Scarica IR (JCAMP-DX)",
                        data=jdx_ir,
                        file_name=f"{mol_label}_IR.jdx",
                        mime="chemical/x-jcamp-dx",
                        icon=":material/download:",
                        use_container_width=True
                    )

            st.markdown("---")
            st.markdown("##### Grafici Pubblicabili (Vettoriali SVG e Raster PNG)")
            col_g1, col_g2, col_g3 = st.columns(3)
            
            # 1H Plot exports
            if result.ensemble_nmr_1h:
                fig_1h_exp = create_nmr_plot(result.ensemble_nmr_1h, title=f"¹H NMR ({mol_label})")
                with col_g1:
                    st.caption("Grafico ¹H NMR:")
                    png_1h = fig_1h_exp.to_image(format="png", width=1200, height=650, scale=2)
                    svg_1h = fig_1h_exp.to_image(format="svg", width=1200, height=650)
                    st.download_button("Scarica ¹H (PNG)", data=png_1h, file_name=f"{mol_label}_1H.png", mime="image/png", icon=":material/download:", use_container_width=True)
                    st.download_button("Scarica ¹H (SVG)", data=svg_1h, file_name=f"{mol_label}_1H.svg", mime="image/svg+xml", icon=":material/download:", use_container_width=True)

            # 13C Plot exports
            if result.ensemble_nmr_13c:
                fig_13c_exp = create_nmr_plot(result.ensemble_nmr_13c, title=f"¹³C NMR ({mol_label})", show_integral=False)
                with col_g2:
                    st.caption("Grafico ¹³C NMR:")
                    png_13c = fig_13c_exp.to_image(format="png", width=1200, height=650, scale=2)
                    svg_13c = fig_13c_exp.to_image(format="svg", width=1200, height=650)
                    st.download_button("Scarica ¹³C (PNG)", data=png_13c, file_name=f"{mol_label}_13C.png", mime="image/png", icon=":material/download:", use_container_width=True)
                    st.download_button("Scarica ¹³C (SVG)", data=svg_13c, file_name=f"{mol_label}_13C.svg", mime="image/svg+xml", icon=":material/download:", use_container_width=True)

            # IR Plot exports
            if result.ensemble_ir:
                fig_ir_exp = create_ir_plot(result.ensemble_ir, title=f"Spettro IR ({mol_label})", show_band_labels=True)
                with col_g3:
                    st.caption("Grafico IR:")
                    png_ir = fig_ir_exp.to_image(format="png", width=1200, height=650, scale=2)
                    svg_ir = fig_ir_exp.to_image(format="svg", width=1200, height=650)
                    st.download_button("Scarica IR (PNG)", data=png_ir, file_name=f"{mol_label}_IR.png", mime="image/png", icon=":material/download:", use_container_width=True)
                    st.download_button("Scarica IR (SVG)", data=svg_ir, file_name=f"{mol_label}_IR.svg", mime="image/svg+xml", icon=":material/download:", use_container_width=True)

            st.markdown("---")
            # JSON summary download
            json_str = json.dumps(result.to_dict(), indent=2)
            st.download_button(
                "Scarica Risultati Completi (JSON)",
                data=json_str,
                file_name=f"{mol_label}_results.json",
                mime="application/json",
                icon=":material/download:"
            )
            tab_idx += 1

        # Tab: Console e Log
        with tab_objects[tab_idx]:
            st.markdown("#### Console e Log di Calcolo")
            st.caption("Registro dettagliato di tutti i passaggi eseguiti durante l'analisi spettroscopica e quantistica.")
            log_content = st.session_state.get("calc_log_text", "")
            if not log_content and out_dir:
                log_file_cand = os.path.join(out_dir, "run.log")
                if os.path.exists(log_file_cand):
                    try:
                        with open(log_file_cand, "r", encoding="utf-8", errors="replace") as f:
                            log_content = f.read()
                    except Exception:
                        pass
            if log_content:
                st.code(log_content, language="log")
                st.download_button(
                    "Scarica Log (.log)",
                    data=log_content,
                    file_name=f"log_{mol_label}.log",
                    mime="text/plain",
                    icon=":material/download:"
                )
            else:
                st.info("Nessun log disponibile per la sessione corrente.")


if __name__ == "__main__":
    main()
