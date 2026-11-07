@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not defined MGD_PYTHON set "MGD_PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%MGD_PYTHON%" (
  echo [ERROR] Python environment not found. Run install_dependencies.bat first.
  if not defined MGD_NO_PAUSE pause
  exit /b 1
)

set "PYTHONPATH=%~dp0"
if not defined MGD_DATA_DIR set "MGD_DATA_DIR=%~dp0data"
if not defined MGD_MODEL_DIR set "MGD_MODEL_DIR=%~dp0models_weights"
if not defined MGD_OUTPUT_DIR (
  if exist "%~dp0outputs_mgd" (
    set "MGD_OUTPUT_DIR=%~dp0outputs_mgd"
  ) else (
    set "MGD_OUTPUT_DIR=%~dp0outputs"
  )
)

"%MGD_PYTHON%" -m mgd.verify_pipeline --epochs 1 --samples 6
set "MGD_EXIT=%ERRORLEVEL%"
if not "%MGD_EXIT%"=="0" echo [ERROR] Self-check failed with code %MGD_EXIT%.
if not defined MGD_NO_PAUSE pause
exit /b %MGD_EXIT%