@echo off
chcp 65001 >nul
color 0A
title 腺而易见 - Windows环境初始化
cls

echo.
echo    __  __  ______  _____  ____   _    _ 
echo   ^|  \/  ^|^|  ____^|^|_   _^|^|  _ \ ^| ^|  ^| ^|
echo   ^| \  / ^|^| ^|__     ^| ^|  ^| ^|__^| ^| ^|  ^| ^|
echo   ^| ^|\/^| ^|^|  __^|    ^| ^|  ^|  __  ^| ^|  ^| ^|
echo   ^| ^|  ^| ^|^| ^|____  _^| ^|_ ^| ^|__^| ^| ^|__^| ^|
echo   ^|_^|  ^|_^|^|______^|^|_____^|^|_____/   \____/ 
echo.
echo ============================================================
echo [INSTALLER] Meibomian Gland Analysis Research Platform
echo [VERSION] v1.0.0 (Windows Environment Setup)
echo ============================================================
echo.

REM 删除其他平台的代码文件夹
if exist "data\mac" (
    rmdir /s /q "data\mac"
    echo [%time%] [INFO] 已删除macOS代码
)
if exist "data\linux" (
    rmdir /s /q "data\linux"
    echo [%time%] [INFO] 已删除Linux代码
)

REM 将Windows代码平铺到根目录
echo [%time%] [INFO] 正在释放Windows代码...
xcopy /e /i /y "data\win\*" .

REM 1. 基础环境自检
echo [%time%] [INFO] 正在检测系统 Python 环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未检测到 Python，请先安装 Python 3.10 或以上版本！
    echo [ERROR] 请确保安装时勾选了 "Add Python to PATH"。
    pause
    exit
)

REM 2. 创建隔离运行环境
echo [%time%] [INFO] 正在创建虚拟环境 (venv)...
if not exist venv (
    python -m venv venv
    echo [%time%] [SUCCESS] 虚拟环境创建成功
) else (
    echo [%time%] [INFO] 检测到已存在 venv 文件夹，跳过创建步骤
)

REM 3. 激活并升级核心引擎
echo [%time%] [INFO] 正在激活环境并准备安装核心组件...
call .\venv\Scripts\activate
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

REM 4. 专项安装 PyTorch (针对显卡加速优化)
echo [%time%] [INFO] 正在从 PyTorch 官网下载计算核心 (CUDA 12.1)...
echo [%time%] [INFO] 这可能需要几分钟，请保持网络连接...
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

REM 5. 安装剩余依赖包
echo [%time%] [INFO] 正在根据要求安装 WebUI 组件与医疗影像库...
if exist requirements.txt (
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
) else (
    echo [WARNING] 未发现 requirements.txt，将安装基础组件...
    pip install gradio monai scikit-image Pillow numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo.
echo ============================================================
echo ✅ Windows环境初始化完成！
echo.
echo 提示：现在您可以直接关闭此窗口，
echo 然后双击 start_win.bat 来开启您的研发之旅
echo ============================================================
pause
