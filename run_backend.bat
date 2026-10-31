@echo off
echo ========================================================
echo Starting FastAPI Backend on http://localhost:8000 ...
echo ========================================================
cd "%~dp0app\app\backend"
".\venv\Scripts\uvicorn.exe" server:app --reload --port 8000
pause
