@echo off
echo Initializing Windows Virtual Environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment.
    exit /b %errorlevel%
)

echo Activating environment and installing dependencies...
call venv\Scripts\activate
pip install -r requirements.txt

echo.
echo Setup complete. You can now run process_data.bat or run.bat.
pause
