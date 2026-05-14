@echo off
:: 强制开启 UTF-8 模式，防止中文乱码
chcp 65001 >nul
color 0B
title “腺”而易见 - 研发平台启动器
cls

echo.
echo     __  __  ______  _____  ____   _    _ 
echo    ^|  \/  ^|^|  ____^|^|_   _^|^|  _ \ ^| ^|  ^| ^|
echo    ^| \  / ^|^| ^|__     ^| ^|  ^| ^|__^| ^| ^|  ^| ^|
echo    ^| ^|\/^| ^|^|  __^|    ^| ^|  ^|  __  ^| ^|  ^| ^|
echo    ^| ^|  ^| ^|^| ^|____  _^| ^|_ ^| ^|__^| ^| ^|__^| ^|
echo    ^|_^|  ^|_^|^|______^|^|_____^|^|_____/   \____/ 
echo.
echo ============================================================
echo [SYSTEM] Meibomian Gland Analysis Research Platform
echo [VERSION] v1.0.0 (Professional Edition)
echo [STATUS] Initializing Virtual Environment...
echo ============================================================
echo.

:: 激活虚拟环境
call .\venv\Scripts\activate

echo [%time%] [INFO] 系统依赖库自检完成.
echo [%time%] [INFO] 正在初始化计算引擎 (CUDA / CPU)
echo [%time%] [WARNING] 请保持此窗口开启，以维持后台服务运行.
echo.
echo ------------------------------------------------------------
echo 🔍 正在映射网页端口并载入 style.css，请稍候...
echo ------------------------------------------------------------

:: 自动探测可用端口（默认从 7860 开始，向上查找）
set "GRADIO_SERVER_PORT=7860"
for /L %%p in (7860,1,7999) do (
    netstat -ano | findstr /R /C:":%%p .*LISTENING" >nul
    if errorlevel 1 (
        set "GRADIO_SERVER_PORT=%%p"
        goto :port_found
    )
)

:port_found
echo [%time%] [INFO] 已分配服务端口: %GRADIO_SERVER_PORT%

:: 运行主程序
python app.py

pause