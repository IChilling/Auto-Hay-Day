@echo off
setlocal
set "REPORT_DIR=%LOCALAPPDATA%\HayDayAutomation\package-check"
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"
echo Checking bundled files, recognition, OCR, and all application pages...
echo This check does not connect to or control BlueStacks.
start "" /wait "%~dp0Hay Day Automation.exe" --self-test "%REPORT_DIR%\result.json"
if errorlevel 1 (
  echo The check reported a problem. Please share the report below.
) else (
  echo Installation check passed.
)
echo Report: %REPORT_DIR%\result.json
if exist "%REPORT_DIR%\result.json" type "%REPORT_DIR%\result.json"
pause
