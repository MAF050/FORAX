@echo off
echo.
echo  FORAX - Forensic Analysis and eXtraction Platform
echo  ===================================================
echo  Starting Desktop Application...
echo.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
	echo [FORAX] ERROR: venv\Scripts\python.exe not found.
	echo Create the virtual environment and install requirements first.
	pause
	exit /b 1
)

"venv\Scripts\python.exe" desktop.py
pause


