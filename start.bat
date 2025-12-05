@echo off
REM Sustainable AI - Energy Efficient Prompt Engineering
REM Windows Startup Script

echo ========================================
echo  Sustainable AI Energy Monitor
echo  Starting application...
echo ========================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
)

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

echo.
echo Starting Streamlit application...
echo.
echo Access the application at: http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.

REM Run the Streamlit application
streamlit run src\gui\app.py --server.port 8501 --server.address localhost

pause
