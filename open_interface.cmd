@echo off
setlocal

cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:8000"
set "PYTHON_EXE=.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Virtual environment Python was not found at "%PYTHON_EXE%".
    echo Restore the .venv folder, then run this file again.
    pause
    exit /b 1
)

echo Starting the cancer classification interface...
echo Backend URL: %APP_URL%
echo.
echo A new backend window will open.
echo Keep that backend window open while using the interface.
echo Press Ctrl+C in that backend window to stop the app later.
echo.

start "Cancer Classifier Backend" cmd /k ""%PYTHON_EXE%" -m backend.web_app"

timeout /t 4 /nobreak >nul
start "" "%APP_URL%"

echo The browser has been opened.
echo If the page is still loading, wait a few seconds and refresh once.
timeout /t 2 /nobreak >nul

endlocal
exit /b 0
