@echo off
if not exist venv (
    echo Error: Virtual environment 'venv' not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo Activating environment...
call venv\Scripts\activate

echo Starting FastAPI Semantic Search Service...
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause
