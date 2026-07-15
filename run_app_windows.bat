\
@echo off
setlocal
cd /d "%~dp0"

echo Sleep Stage QC - Windows launcher
echo Repository: %CD%
echo.

where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: conda was not found. Open Anaconda Prompt and run this file again.
  pause
  exit /b 1
)

call conda activate sleep_stage_qc_v2
if errorlevel 1 (
  echo ERROR: could not activate the sleep_stage_qc_v2 environment.
  echo Create/update it with: conda env update -f environment.yml --prune
  pause
  exit /b 1
)

echo Python:
where python
python -c "import sys; print(sys.executable)"
echo.

echo Starting app at http://127.0.0.1:8050
python -u dash_app\app.py

echo.
echo The app stopped. Review the message above and the logs folder if this was unexpected.
pause
