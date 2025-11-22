@echo off
echo ----------------------------------------------------------------
echo   OPAM (Online Purchasing-behavior Analysis And Management)
echo ----------------------------------------------------------------

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b
)

REM Setup Virtual Environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment found.
)

REM Activate Virtual Environment
call venv\Scripts\activate

REM Install Dependencies
echo Installing/Updating dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing dependencies.
    pause
    exit /b
)
echo Dependencies ready.

REM Check Data
if not exist data\transactions.csv (
    echo No data found in data\transactions.csv
    set /p choice="Generate sample data? (y/n): "
    if /i "%choice%"=="y" (
        echo Generating sample data...
        python scripts\generate_sample_data.py
    ) else (
        echo Please place 'transactions.csv' in the 'data' folder.
        pause
        exit /b
    )
)

REM Run Main System
echo Starting OPAM...
cd src
python run_all_systems.py
pause
