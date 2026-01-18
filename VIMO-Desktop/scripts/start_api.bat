@echo off
setlocal ENABLEDELAYEDEXPANSION

echo ==========================================
echo   VideoRAG Backend Launcher (Conda)
echo ==========================================

REM -------- CONFIG --------
set ENV_NAME=videorag
set PROJECT_DIR=%~dp0
set SRC_DIR=%PROJECT_DIR%
set REQUIREMENTS_FILE=%PROJECT_DIR%requirements.txt
REM ------------------------

REM 1) Check conda
where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Conda not found in PATH.
    echo Please install Anaconda or Miniconda first.
    exit /b 1
)

echo [OK] Conda found

REM 2) Check if env exists
conda env list | findstr /R /C:"^%ENV_NAME% " >nul
if errorlevel 1 (
    echo [INFO] Conda env "%ENV_NAME%" not found. Creating it...
    conda create -y -n %ENV_NAME% python=3.10
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment
        exit /b 1
    )
) else (
    echo [OK] Conda env "%ENV_NAME%" exists
)

REM 3) Activate env
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment
    exit /b 1
)

echo [OK] Conda env activated

REM 4) Install requirements (excluding torch)
if exist "%REQUIREMENTS_FILE%" (
    echo [INFO] Installing Python dependencies (excluding torch)...
    pip install --upgrade pip
    pip install -r "%REQUIREMENTS_FILE%"
) else (
    echo [ERROR] requirements.txt not found
    exit /b 1
)

REM 5) Optional: install torch CUDA 12.1
echo [INFO] Installing PyTorch (CUDA 12.1)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM 6) Run backend
echo ==========================================
echo [START] VideoRAG API
echo ==========================================
cd /d "%PROJECT_DIR%"
python -u -m src.api.videorag_api

REM If Python exits
echo ==========================================
echo [STOP] VideoRAG API stopped
echo ==========================================

endlocal
