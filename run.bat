@echo off
if not exist venv (
    echo Error: Virtual environment 'venv' not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo Activating environment...
call venv\Scripts\activate

echo Starting Semantic Search Service...
python backend/app.py

if %errorlevel% neq 0 (
    echo.
    echo Service stopped with an error.
    pause
)
