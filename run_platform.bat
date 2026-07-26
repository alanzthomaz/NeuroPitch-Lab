@echo off
echo ==========================================================
echo ⚽ Starting FIFA World Cup 2026 Prediction Platform...
echo ==========================================================

:: Start backend FastAPI app in a new command window
echo 🚀 Starting FastAPI Backend...
start "FastAPI Backend" cmd /k "set PYTHONPATH=.&& uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

:: Start frontend Vite app in a new command window
echo 🎨 Starting Vite React Frontend...
cd frontend
start "React Frontend" cmd /k "npm run dev"
cd ..

echo.
echo 🔥 API Backend is running at: http://localhost:8000
echo 🔥 React Frontend is running at: http://localhost:5173
echo ==========================================================
echo Close the opened terminal windows to terminate the servers.
echo ==========================================================
