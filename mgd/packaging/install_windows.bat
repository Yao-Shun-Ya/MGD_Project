@echo off
chcp 65001 >nul
setlocal
rem 在仓库根目录创建统一 venv；需求文件位于 mgd\requirements。
cd /d "%~dp0\..\.."
if not exist "mgd\requirements\base.txt" (
  echo [错误] 未找到 mgd\requirements\base.txt，请确认解压后目录结构完整。
  exit /b 1
)

set "TORCH_REQ=mgd\requirements\cuda.txt"
if /i "%~1"=="cpu" set "TORCH_REQ=mgd\requirements\cpu.txt"

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 python，请先安装 Python 3.10+ 并加入 PATH。
  exit /b 1
)

if not exist "venv\Scripts\python.exe" (
  echo [1/4] 创建虚拟环境 venv ...
  python -m venv venv
  if errorlevel 1 exit /b 1
)

call "venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1

echo [2/4] 升级 pip ...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

echo [3/4] 安装 PyTorch（%TORCH_REQ%）...
python -m pip install -r "%TORCH_REQ%"
if errorlevel 1 (
  echo [错误] PyTorch 安装失败。可改用 CPU 轮子：install_windows.bat cpu
  exit /b 1
)

echo [4/4] 安装公共依赖与 WebUI 依赖 ...
python -m pip install -r mgd\requirements\base.txt -r mgd\requirements\ui.txt
if errorlevel 1 exit /b 1

python -c "import cv2, torch, monai, gradio; print('cv2', cv2.__version__, '| torch', torch.__version__, '| monai', monai.__version__, '| gradio', gradio.__version__)"
if errorlevel 1 (
  echo [错误] 依赖自检失败。
  exit /b 1
)

echo.
echo [完成] 安装成功。启动 WebUI：mgd\packaging\start_windows.bat
exit /b 0
