@echo off
title NMR-INATOR Studio
echo ===================================================
echo     Avvio di NMR-INATOR Studio (Streamlit GUI)...
echo ===================================================
echo.
cd /d "%~dp0"

:: 1. Rileva virtual environment locale (.venv)
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -m streamlit run gui\app_gui.py
    goto end
)
if exist ".venv\python.exe" (
    .venv\python.exe -m streamlit run gui\app_gui.py
    goto end
)

:: 2. Rileva ambiente condiviso con energy-inator (cartella adiacente)
if exist "..\energy-inator\.venv\Scripts\python.exe" (
    ..\energy-inator\.venv\Scripts\python.exe -m streamlit run gui\app_gui.py
    goto end
)
if exist "..\energy-inator\.venv\python.exe" (
    ..\energy-inator\.venv\python.exe -m streamlit run gui\app_gui.py
    goto end
)

:: 3. Fallback: usa l'ambiente Python/Conda attivo nel terminale
python -m streamlit run gui\app_gui.py

:end
pause

