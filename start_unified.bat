@echo off
echo Starting Unified Development Environment...
echo [Logs will appear below with prefixes]
echo.

:: Install dependencies first to avoid concurrent installation race conditions or missing modules
echo Installing backend dependencies...
pip install -r requirements.txt

echo Installing frontend dependencies...
cd frontend
call npm install
cd ..

:: Use npx concurrently to run all 3 services in one terminal with colors and prefixes
call npx -y concurrently ^
  --names "DJANGO,FASTAPI,REACT" ^
  --prefix-colors "green,blue,magenta" ^
  --kill-others ^
  "cd backend && python manage.py runserver 8000" ^
  "cd backend && uvicorn chat_api.main:app --reload --port 8001" ^
  "cd frontend && npm run dev"

pause
