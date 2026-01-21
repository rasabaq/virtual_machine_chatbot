# Project Setup Guide

## Prerequisites
- **Python 3.10+** (Ensure pip is in PATH)
- **Node.js 18+** (Ensure npm is in PATH)
- **Git**

## Installation

### 1. Clone & Dependencies

**Backend**
```bash
pip install -r requirements.txt
cd backend
python manage.py migrate
```

**Frontend**
```bash
cd frontend
npm install
```

### 2. Environment Variables
Ensure `.env` exists in the root (or backend) with:
```
GOOGLE_API_KEY=your_key_here
DISCORD_TOKEN=your_token (optional if only using web)
```

## Running the Application

### Automatic (Windows)
Double-click `start_services.bat` in the project root.

### Manual
1. **Django** (Port 8000):
   ```bash
   cd backend
   python manage.py runserver 8000
   ```
2. **FastAPI** (Port 8001):
   ```bash
   cd backend
   uvicorn chat_api.main:app --reload --port 8001
   ```
3. **React** (Port 5173):
   ```bash
   cd frontend
   npm run dev
   ```

Access the app at: [http://localhost:5173](http://localhost:5173)
