@echo off
title NMR-INATOR Studio
echo ===================================================
echo     Avvio di NMR-INATOR Studio (Streamlit GUI)...
echo ===================================================
echo.
cd /d "%~dp0"
..\energy-inator\.venv\python.exe -m streamlit run gui\app_gui.py
pause

