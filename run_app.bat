@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project virtual environment not found: .venv
    echo Create it with: python -m venv .venv
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app.py %*
exit /b %errorlevel%
