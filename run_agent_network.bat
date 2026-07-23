@echo off
REM Launch PaddyNet so OTHER devices on the same WiFi can open it.
REM On this PC:            http://localhost:8501
REM On phone/other PC:     http://192.168.1.8:8501   (same WiFi network)
cd /d "%~dp0"
call venv\Scripts\activate
echo ============================================================
echo   PaddyNet is starting...
echo   This computer : http://localhost:8501
echo   Other devices : http://192.168.1.8:8501   (same WiFi)
echo ============================================================
streamlit run agent\app.py --server.address=0.0.0.0 --server.port=8501
pause
