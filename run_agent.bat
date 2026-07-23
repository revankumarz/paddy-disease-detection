@echo off
REM One-click launcher for the Paddy Disease Agent (Windows)
cd /d "%~dp0"
call venv\Scripts\activate
echo Starting PaddyNet - Paddy Disease Agent...
streamlit run agent\app.py
pause
