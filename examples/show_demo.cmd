@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment was not found. See docs/DEMO_GUIDE_RU.md.
    pause
    exit /b 2
)
".venv\Scripts\python.exe" "examples\multi_agent_demo.py" --open
if errorlevel 1 (
    echo Demonstration failed. See the error above.
    pause
    exit /b 1
)
endlocal
