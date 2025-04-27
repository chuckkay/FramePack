@echo off
call ..\environment.bat

"%DIR%\python\python.exe" Framepack_QueueItUp.py --server 127.0.0.1 --inbrowser

:done
pause
