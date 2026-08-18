@echo off
setlocal
cd /d "%~dp0"

title Sleep Stage QC
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_windows.ps1"
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo The app stopped with an error. Review the message above.
    pause
)
exit /b %EXITCODE%
