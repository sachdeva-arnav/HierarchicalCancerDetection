@echo off
setlocal

cd /d "%~dp0"

if not defined APP_URL set "APP_URL=http://127.0.0.1:8000"
if not defined HEALTH_URL set "HEALTH_URL=%APP_URL%/api/health"
if not defined MAX_WAIT_SECONDS set "MAX_WAIT_SECONDS=60"

echo Opening the cancer classification frontend...
echo Frontend URL: %APP_URL%
echo.
echo Waiting for the backend to become ready.
echo If it is not running yet, start launch_backend.cmd in another window.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline = (Get-Date).AddSeconds([int]$env:MAX_WAIT_SECONDS); do { try { $response = Invoke-WebRequest -Uri $env:HEALTH_URL -UseBasicParsing -TimeoutSec 2; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) { exit 0 } } catch { Start-Sleep -Seconds 1 } } while ((Get-Date) -lt $deadline); exit 1"

if errorlevel 1 (
    echo Backend did not respond at %HEALTH_URL% within %MAX_WAIT_SECONDS% seconds.
    echo Start launch_backend.cmd first, wait for the backend window to finish loading, then run this file again.
    if not defined NO_PAUSE pause
    exit /b 1
)

start "" "%APP_URL%"

echo Browser opened after the backend became ready.
timeout /t 2 /nobreak >nul

endlocal
exit /b 0
