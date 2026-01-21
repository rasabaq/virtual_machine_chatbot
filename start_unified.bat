@echo off
echo Starting Unified Development Environment...
echo [Logs will appear below with prefixes]
echo.

:: Use npx concurrently to run all 3 services in one terminal with colors and prefixes
call npx -y concurrently ^
  --names "DJANGO,FASTAPI,REACT" ^
  --prefix-colors "green,blue,magenta" ^
  --kill-others ^
  "cd backend && python manage.py runserver 8000" ^
  "cd backend && uvicorn chat_api.main:app --reload --port 8001" ^
  "cd frontend && npm run dev"

pause
