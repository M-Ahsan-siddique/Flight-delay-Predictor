@echo off
title Flight Delay Intelligence Dashboard
cd /d "%~dp0"

echo =========================================================
echo   LAUNCHING FLIGHT DELAY PREDICTION INTELLIGENCE SYSTEM
echo =========================================================
echo.

set "PY_EXE=C:\Users\Mahsa\AppData\Local\Programs\Python\Python313\python.exe"

if exist "%PY_EXE%" (
    echo [OK] Found Python at: "%PY_EXE%"
    goto :RUN
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_EXE=python"
    echo [OK] Using system python
    goto :RUN
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_EXE=py -3.13"
    echo [OK] Using Python Launcher
    goto :RUN
)

echo [ERROR] Could not find Python 3.13 executable.
echo Please check your Python installation.
pause
exit /b 1

:RUN
echo [OK] Starting Streamlit server...
echo [INFO] Opening dashboard at http://localhost:8501
echo.
"%PY_EXE%" -m streamlit run app.py --server.port 8501

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application encountered an error or was stopped.
    pause
)
