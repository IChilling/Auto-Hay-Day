@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" goto setup
".venv\Scripts\python.exe" -I -c "import flet, cv2" >nul 2>&1
if errorlevel 1 goto setup
goto launch
:setup
call "Setup Hay Day.cmd" -NoLaunch
if errorlevel 1 exit /b 1
:launch
start "" ".venv\Scripts\pythonw.exe" -I "app.py"
