@echo off
REM Stops every run started by 2_RUN.bat (finished runs stay saved; 2_RUN.bat resumes later).
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'launch_wpi|run_m2|run_venue|run_sens|run_model1' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }"
echo Stopped.
pause
