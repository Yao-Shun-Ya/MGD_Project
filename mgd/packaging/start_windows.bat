@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0\..\.."
if not exist "venv\Scripts\python.exe" (
  echo [错误] 未找到 venv，请先运行 mgd\packaging\install_windows.bat
  exit /b 1
)

rem 数据与权重默认位于仓库根的 data / models_weights；可用环境变量覆盖。
if not defined MGD_DATA_DIR set "MGD_DATA_DIR=%CD%\data"
if not defined MGD_MODEL_DIR set "MGD_MODEL_DIR=%CD%\models_weights"
if not defined MGD_OUTPUT_DIR set "MGD_OUTPUT_DIR=%CD%\outputs"
set "PYTHONPATH=%CD%"

call "venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1
python -m mgd.adapters.gradio_app
