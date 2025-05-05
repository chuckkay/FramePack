@echo off
setlocal

REM ================================================
REM install.bat — idempotent setup for QueueItUp_FramePack
REM ================================================

REM 1) Check for Conda
echo.
echo === Checking for Conda installation ===
where conda >nul 2>&1
if errorlevel 1 (
  echo ERROR: Conda not found. Install Miniconda:
  echo   https://docs.conda.io/en/latest/miniconda.html
  pause
  exit /b 1
)

REM 2) Verify repo root
echo.
echo === Verifying repository root ===
if not exist ".git" (
  echo ERROR: .git folder not found. Run from repo root.
  pause
  exit /b 1
)

REM 3) Pull latest
echo.
echo === Pulling latest changes ===
git pull || echo WARNING: git pull failed, continuing

REM 4) (Re)create the Conda env unconditionally, then pause
echo.
set "ENV_NAME=QueueItUp_FramePack"
echo === Creating or updating Conda env %ENV_NAME% ===
REM <-- “call” is required so control returns to this script
call conda create -n "%ENV_NAME%" python=3.10.6 pip=25.0 -y

echo.
echo --- Conda env creation finished. Continue Installation press any key---
pause

REM 5) Activate the environment
echo.
echo === Activating env %ENV_NAME% ===
call conda activate "%ENV_NAME%"
if errorlevel 1 (
  echo ERROR: could not activate env %ENV_NAME%
  pause
  exit /b 1
)

REM 6) Install CUDA + cuDNN (log & pause)
echo === Installing CUDA + cuDNN ===
call conda install -n "%ENV_NAME%" conda-forge::cuda-runtime=12.6 conda-forge::cudnn=9.8.0.87 -y > install_cuda.log 2>&1
echo log written to install_cuda.log
echo Continue Installation---
pause

REM 7) Install PyTorch & friends
echo.
echo === Installing PyTorch, Torchvision, Torchaudio ===
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
  echo ERROR: PyTorch install failed.
  pause
  exit /b 1
)

REM 8) Install other requirements
echo.
if exist requirements.txt (
  echo === Installing other Python dependencies ===
  pip install -r requirements.txt
) else (
  echo WARNING: requirements.txt not found, skipping
)

REM 9) Finish
echo.
echo ===== Setup Complete! =====
echo To in the future use: python Framepack_QueueItUp.py --server 127.0.0.1 --inbrowser
echo or use Framepack_QueueItUp.bat
echo ...
echo press any key to start Framepack_QueueItUp
echo ...
pause
call Framepack_QueueItUp.bat
pause
cmd /k
