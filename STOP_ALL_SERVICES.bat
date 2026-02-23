@echo off
echo ========================================
echo RAF ICD Extraction System - Shutdown
echo ========================================
echo.

echo Stopping all services...
echo.

REM Kill processes on port 4000 (React Frontend)
echo [1/3] Stopping React Frontend (Port 4000)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :4000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a 2>nul
    if %errorlevel% equ 0 (
        echo    ✓ Stopped process on port 4000
    )
)

REM Kill processes on port 9090 (Spring Boot Backend)
echo [2/3] Stopping Spring Boot Backend (Port 9090)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :9090 ^| findstr LISTENING') do (
    taskkill /F /PID %%a 2>nul
    if %errorlevel% equ 0 (
        echo    ✓ Stopped process on port 9090
    )
)

REM Kill processes on port 8500 (FastAPI Python Service)
echo [3/3] Stopping FastAPI Service (Port 8500)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8500 ^| findstr LISTENING') do (
    taskkill /F /PID %%a 2>nul
    if %errorlevel% equ 0 (
        echo    ✓ Stopped process on port 8500
    )
)

echo.
echo ========================================
echo All services have been stopped!
echo ========================================
echo.
pause
