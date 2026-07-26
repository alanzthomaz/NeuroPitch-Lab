#!/bin/bash
# run_platform.sh
echo "=========================================================="
echo "⚽ Starting FIFA World Cup 2026 Prediction Platform..."
echo "=========================================================="

# Start backend FastAPI app
echo "🚀 Starting FastAPI Backend..."
PYTHONPATH=. uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# Navigate to frontend and start Vite server
echo "🎨 Starting Vite React Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "🔥 API Backend is running at: http://localhost:8000"
echo "🔥 React Frontend is running at: http://localhost:5173"
echo "=========================================================="
echo "Press Ctrl+C to terminate both servers."
echo "=========================================================="

# Graceful shutdown on Ctrl+C
trap "echo -e '\nStopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM

wait
