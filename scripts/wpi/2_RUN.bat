@echo off
REM Runs round 8 (Delhi and Dubai). Keep this window open; minimizing it or disconnecting Remote Desktop is fine, signing out is not.
cd /d "%~dp0..\..\models\model2_advanced"
..\..\.venv\Scripts\python.exe launch_wpi.py --last-start 30 --hours 40 %*
pause
