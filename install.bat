@echo off
setlocal
setlocal EnableDelayedExpansion

REM ================================================
REM install.bat — idempotent setup for FramePack - QueueItUp Version
REM ================================================

REM 1) Check for Conda and install if not found
echo.
echo === Checking for Conda installation ===
where conda >nul 2>&1
if errorlevel 1 (
  echo Conda not found. Downloading and installing Miniconda...
  
  REM Download Miniconda installer
  curl -o miniconda.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
  if errorlevel 1 (
    echo ERROR: Failed to download Miniconda installer
    pause
    exit /b 1
  )
  
  REM Install Miniconda silently
  echo Installing Miniconda...
  start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniconda3
  if errorlevel 1 (
    echo ERROR: Miniconda installation failed
    pause
    exit /b 1
  )
  
  REM Delete the installer
  del miniconda.exe
  
  REM Add Conda to the current session's PATH
  set "PATH=%UserProfile%\Miniconda3;%UserProfile%\Miniconda3\Scripts;%UserProfile%\Miniconda3\Library\bin;%PATH%"
  
  echo Miniconda installed successfully!
  echo.
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

REM 4) (Re)create the Conda env unconditionally, then prompt
echo.
set "ENV_NAME=Framepack_QueueItUp"
echo === Creating or updating Conda env ===

REM Deactivate any active environment first
call conda deactivate

REM Remove existing environment if it exists
call conda env remove -n %ENV_NAME% -y

REM Create new environment
call conda create --name %ENV_NAME% python=3.10.6 pip=25.0 -y
if errorlevel 1 (
    echo ERROR: Failed to create Conda environment
    pause
    exit /b 1
)

echo.
echo --- Conda env creation finished ---
pause

REM 5) Activate the environment
echo.
echo === Activating environment ===
call conda activate %ENV_NAME%
if errorlevel 1 (
  echo ERROR: could not activate environment
  pause
  exit /b 1
)

REM 6) Install CUDA + cuDNN
echo === Installing CUDA + cuDNN ===
call conda install -n %ENV_NAME% conda-forge::cuda-runtime=12.6 conda-forge::cudnn=9.8.0.87 -y > install_cuda.log 2>&1
echo log written to install_cuda.log
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
echo Ready to start Framepack_QueueItUp?
pause

call Framepack_QueueItUp.bat
pause
cmd /k
