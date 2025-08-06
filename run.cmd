@echo off
echo 🚀 Starting Seismic Network Analyzer...

REM Start Flask backend in new terminal
start "Flask Backend" cmd /k "cd backend && python app.py"

REM Start React frontend in new terminal
start "React Frontend" cmd /k "cd frontend && npm start"
