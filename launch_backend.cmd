@echo off
setlocal

cd /d "%~dp0"

if not defined APP_URL set "APP_URL=http://127.0.0.1:8000"
if not defined PYTHON_EXE set "PYTHON_EXE=.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Virtual environment Python was not found at "%PYTHON_EXE%".
    echo Restore the .venv folder, then run this file again.
    pause
    exit /b 1
)

echo Starting the cancer classification backend...
echo Backend URL: %APP_URL%
echo.
echo Keep this window open while using the interface.
echo Run open_frontend.cmd after this window says the frontend is available.
echo Press Ctrl+C in this window to stop the backend later.
echo.

"%PYTHON_EXE%" -m backend.web_app
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo Backend stopped with exit code %EXIT_CODE%.
) else (
    echo Backend stopped.
)
pause

endlocal & exit /b %EXIT_CODE%
