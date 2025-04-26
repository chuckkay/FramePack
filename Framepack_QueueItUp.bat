@echo off

call C:\AI\framepack\environment.bat

cd C:\AI\framepack\webui

"%DIR%\python\python.exe" Framepack_QueueItUp.py --server 127.0.0.1 --inbrowser

:done
pause