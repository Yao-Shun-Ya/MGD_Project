#!/bin/bash

# 强制开启 UTF-8 编码
export LANG="zh_CN.UTF-8"

# 颜色定义
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo ""
echo "    __  __  ______  _____  ____     _    _ "
echo "   |  \/  ||  ____||_   _||  _  \  | |  | |"
echo "   | \  / || |__     | |  | |__| | | |  | |"
echo "   | |\/| ||  __|    | |  |  __  | | |  | |"
echo "   | |  | || |____  _| |_ | |__| | | |__| |"
echo "   |_|  |_||______||_____||_____/   \____/ "
echo ""
echo "============================================================"
echo "[LAUNCHER] Meibomian Gland Analysis Research Platform"
echo "[VERSION] v1.0.0 (Launcher for macOS)"
echo "============================================================"
echo ""

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "${RED}[ERROR] 未检测到虚拟环境，请先运行 './setup_macos.sh' 配置环境！${NC}"
    read -p "按 Enter 键退出..."
    exit 1
fi

# 激活虚拟环境
echo "[INFO] 正在激活虚拟环境..."
source ./venv/bin/activate

# 启动应用
echo "[INFO] 正在启动 '腺'而易见 WebUI..."
echo "[INFO] 请稍候，正在加载模型和初始化界面..."
echo ""
echo "${GREEN}🚀 启动成功！Web 界面将在浏览器中打开${NC}"
echo "${GREEN}📝 访问地址: http://127.0.0.1:7860${NC}"
echo ""
echo "============================================================"
echo "提示：按 Ctrl+C 可停止服务"
echo "============================================================"
echo ""

# 运行应用
python3 app.py
