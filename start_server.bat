@echo off
title Industrial Damage Detection AOI Microservice
cd /d "%~dp0"
echo =====================================================================
echo Starting Industrial Optical Inspection (AOI) Server & Dashboard...
echo =====================================================================
echo.
echo Access URL: http://localhost:8000/dashboard/
echo API Docs:   http://localhost:8000/docs
echo.
.\.venv\Scripts\uvicorn.exe backend.app.main:app --host 127.0.0.1 --port 8000 --reload
pause
