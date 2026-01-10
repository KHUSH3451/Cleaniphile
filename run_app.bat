@echo off
echo ==========================================
echo       Cleaniphile - Setup & Run
echo ==========================================

echo [1/2] Checking Dependencies...
python -m pip install -r requirements.txt

echo.
echo [2/2] Launching Application...
echo If the browser does not open, wait a few seconds...
python start_app.py

pause
