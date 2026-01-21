@echo off
echo Starting Services...

:: Start Django (Admin/ORM)
start "Django Server" cmd /k "cd backend && python manage.py runserver 8000"

:: Start FastAPI (Agent)
start "FastAPI Server" cmd /k "cd backend && uvicorn chat_api.main:app --reload --port 8001"

:: Start React Frontend
start "React Frontend" cmd /k "cd frontend && npm run dev"

echo All services started!
pause
