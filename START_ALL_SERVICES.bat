@echo off
echo ========================================
echo RAF ICD Extraction System - Startup
echo ========================================
echo.

REM Check if services are running
echo Checking for existing services...
netstat -an | findstr ":4000" > nul
if %errorlevel% equ 0 (
    echo [WARNING] Port 4000 already in use
)

netstat -an | findstr ":9090" > nul
if %errorlevel% equ 0 (
    echo [WARNING] Port 9090 already in use
)

netstat -an | findstr ":8500" > nul
if %errorlevel% equ 0 (
    echo [WARNING] Port 8500 already in use
)

echo.
echo Starting all services...
echo.

REM Start Python FastAPI service
echo [1/3] Starting Python FastAPI service (Port 8500)...
start "FastAPI Service" cmd /k "cd ai_icd_extraction && ..\rafenv\Scripts\activate && python fastapi_service.py"
timeout /t 10 /nobreak > nul

REM Start Spring Boot backend
echo [2/3] Starting Spring Boot backend (Port 9090)...
start "Spring Boot Backend" cmd /k "cd backend && mvn spring-boot:run"
timeout /t 20 /nobreak > nul

REM Start React frontend
echo [3/3] Starting React frontend (Port 4000)...
start "React Frontend" cmd /k "cd frontend && npm start"

echo.
echo ========================================
echo All services are starting!
echo ========================================
echo.
echo Services:
echo - Frontend:   http://localhost:4000
echo - Backend:    http://localhost:9090
echo - Python API: http://localhost:8500
echo - API Docs:   http://localhost:8500/docs
echo.
echo Three terminal windows have opened.
echo Do NOT close them while using the system.
echo.
echo Press any key to exit this launcher...
pause > nul
