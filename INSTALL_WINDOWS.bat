@echo off
setlocal
cd /d "%~dp0"

title Sleep Stage QC - Windows installer
echo.
echo ============================================================
echo  Sleep Stage QC - Windows installer
echo ============================================================
echo.
echo This will install/update:
echo   - the Sleep Stage QC app environment
echo   - the supported Somnotate environment
echo   - the supported Somnotate source from the official GitHub repo
echo.
echo You can safely run this installer again later to repair/update the setup.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install_windows.ps1"
set "EXITCODE=%ERRORLEVEL%"

echo.
if "%EXITCODE%"=="0" (
    echo ============================================================
    echo  Installation completed successfully.
    echo ============================================================
    echo.
    echo Next time, just double-click RUN_APP.bat
) else (
    echo ============================================================
    echo  Installation failed.
    echo ============================================================
    echo.
    echo Read the error above. Nothing in your recording data was modified.
)

echo.
pause
exit /b %EXITCODE%
