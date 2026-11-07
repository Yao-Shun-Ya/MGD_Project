@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not defined MGD_PYTHON set "MGD_PYTHON=%~dp0venv\Scripts\python.exe"
if exist "%MGD_PYTHON%" (
  "%MGD_PYTHON%" -c "import torch,monai,gradio,cv2" >nul 2>&1
  if not errorlevel 1 (
    echo [OK] Dependencies already installed.
    if not defined MGD_NO_PAUSE pause
    exit /b 0
  )
)

echo [INFO] Installing dependencies. This may take several minutes.
call "%~dp0mgd\packaging\install_windows.bat" %*
if errorlevel 1 (
  echo [ERROR] Installation failed. Check the messages above.
  if not defined MGD_NO_PAUSE pause
  exit /b 1
)
echo [OK] Installation complete. Run start_webui.bat or start_training_cli.bat.
if not defined MGD_NO_PAUSE pause
exit /b 0