@echo off
REM One-time setup: makes a Python 3.11 environment in .venv and installs the packages.
cd /d "%~dp0..\.."
py -3.11 -m venv .venv 2>nul || python -m venv .venv
if not exist .venv\Scripts\python.exe (
  echo Could not make a Python environment. Install Python 3.11 from python.org or use Anaconda, then run this again.
  pause & exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || .venv\Scripts\python.exe -m pip install numpy scipy numba pyamg pythermalcomfort pandas matplotlib
.venv\Scripts\python.exe -c "import sys,numpy,scipy,numba,pyamg,pythermalcomfort; print('SETUP OK, Python', sys.version.split()[0])"
pause
