Write-Host "Starting Services..." -ForegroundColor Green

# Start Django (Admin/ORM)
Start-Process -FilePath "cmd" -ArgumentList "/k cd backend && python manage.py runserver 8000" -PassThru

# Start FastAPI (Agent)
Start-Process -FilePath "cmd" -ArgumentList "/k cd backend && uvicorn chat_api.main:app --reload --port 8001" -PassThru

# Start React Frontend
Start-Process -FilePath "cmd" -ArgumentList "/k cd frontend && npm run dev" -PassThru

Write-Host "All services started! Check the new windows." -ForegroundColor Green
