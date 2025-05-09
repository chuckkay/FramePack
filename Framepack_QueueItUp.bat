@echo off
setlocal

REM Activate the conda environment
echo Activating conda environment...
call conda activate FramePack_QueueItUp
if errorlevel 1 (
    echo Error: Failed to activate conda environment
    pause
    exit /b 1
)


REM Run the script
python Framepack_QueueItUp.py --server 127.0.0.1 --inbrowser

:done
pause
