@echo off
title NutriCoach AI — IBM Granite Nutrition Advisor
color 0A

echo.
echo  ============================================================
echo.
echo     N   N  U   U  TTTTT  RRRR  IIIII  CCC  OOO   AA   CCC  HH  HH
echo     NN  N  U   U    T    R   R   III  C    O   O A  A C    HH  HH
echo     N N N  U   U    T    RRRR    III  C    O   O AAAA C    HHHHHH
echo     N  NN  U   U    T    R  R    III  C    O   O A  A C    HH  HH
echo     N   N   UUU     T    R   R  IIIII  CCC  OOO  A  A  CCC HH  HH
echo.
echo     NutriCoach AI  ^|  Powered by IBM Granite LLM
echo     Personalised Nutrition Planning with Agentic AI
echo     Following: HAM Principles + Full AI Ethics Guardrails
echo.
echo  ============================================================
echo.

:: ── Navigate to the directory where this .bat file lives ────────────────
cd /d "%~dp0"

:: ── STEP 1: Detect Python ───────────────────────────────────────────────
echo [1/6] Detecting Python installation...
set PYTHON_CMD=

:: Try 'py' launcher first (Windows Python Launcher — preferred)
py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    goto :python_found
)

:: Try 'python' directly
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :python_found
)

:: Try 'python3' (some setups)
python3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3
    goto :python_found
)

echo.
echo  [ERROR] Python is not installed or not in PATH.
echo  Please download Python 3.9+ from:  https://www.python.org/downloads/
echo  During installation, check "Add Python to PATH".
echo.
pause
exit /b 1

:python_found
for /f "tokens=*" %%i in ('%PYTHON_CMD% --version 2^>^&1') do echo  Found: %%i
echo.

:: ── STEP 2: Virtual Environment Setup ──────────────────────────────────
echo [2/6] Setting up virtual environment...

if not exist "venv\Scripts\activate.bat" (
    echo  Creating new virtual environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        echo  Try: %PYTHON_CMD% -m pip install --user virtualenv
        pause
        exit /b 1
    )
    echo  Virtual environment created.
) else (
    echo  Existing virtual environment found. Reusing it.
)
echo.

:: ── STEP 3: Activate Virtual Environment ───────────────────────────────
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo  [ERROR] Could not activate virtual environment.
    pause
    exit /b 1
)
echo  Virtual environment activated.
echo.

:: ── STEP 4: Upgrade pip silently ────────────────────────────────────────
echo [4/6] Upgrading pip (silent)...
python -m pip install --upgrade pip --quiet 2>nul
echo  pip is up to date.
echo.

:: ── STEP 5: Install Dependencies ────────────────────────────────────────
echo [5/6] Installing/verifying required packages...
echo  (This may take a minute on first run)
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [ERROR] Package installation failed.
    echo  Trying minimal install without IBM SDK...
    pip install flask flask-cors requests python-dotenv markdown
    if errorlevel 1 (
        echo  [ERROR] Even minimal install failed. Check your internet connection.
        pause
        exit /b 1
    )
)
echo.
echo  All packages installed successfully.
echo.

:: ── STEP 6: Environment File Setup ──────────────────────────────────────
echo [6/6] Checking configuration...

if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo  Created .env from template.
    ) else (
        echo WATSONX_API_KEY=your_ibm_cloud_api_key_here > .env
        echo WATSONX_PROJECT_ID=your_watsonx_project_id_here >> .env
        echo WATSONX_URL=https://us-south.ml.cloud.ibm.com >> .env
        echo FLASK_SECRET_KEY=nutrition-ai-secret-key-2024 >> .env
        echo FLASK_PORT=5000 >> .env
        echo  Created default .env file.
    )
    echo.
    echo  ┌─────────────────────────────────────────────────────┐
    echo  │  OPTIONAL: Add IBM watsonx.ai credentials to .env   │
    echo  │  for live IBM Granite LLM responses.                 │
    echo  │  Without credentials: app runs in OFFLINE MODE       │
    echo  │  (fully functional rule-based plan generator).       │
    echo  └─────────────────────────────────────────────────────┘
    echo.
) else (
    echo  .env file found.
)

:: ── Create saved_plans directory ────────────────────────────────────────
if not exist "saved_plans" (
    mkdir saved_plans
    echo  Created saved_plans\ directory.
)
echo.

:: ── Launch the Application ───────────────────────────────────────────────
echo  ============================================================
echo   Starting NutriCoach AI Server...
echo.
echo   Web Interface:    http://localhost:5000
echo   Health Check:     http://localhost:5000/api/health
echo   Saved Plans:      saved_plans\
echo   IBM Granite Model: ibm/granite-13b-instruct-v2
echo.
echo   Press Ctrl+C to stop the server.
echo  ============================================================
echo.

:: ── Auto-open browser after 3 seconds ───────────────────────────────────
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

:: ── Start Flask App ──────────────────────────────────────────────────────
python app.py

:: ── On Server Exit ───────────────────────────────────────────────────────
echo.
echo  NutriCoach AI server has stopped.
echo  Your saved plans are in:  saved_plans\
echo.
pause
