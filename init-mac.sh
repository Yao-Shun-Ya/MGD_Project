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
echo "[INSTALLER] Meibomian Gland Analysis Research Platform"
echo "[VERSION] v1.0.0 (macOS Environment Setup)"
echo "============================================================"
echo ""

# 删除其他平台的代码文件夹
if [ -d "data/win" ]; then
    rm -rf "data/win"
    echo "[INFO] 已删除Windows代码"
fi
if [ -d "data/linux" ]; then
    rm -rf "data/linux"
    echo "[INFO] 已删除Linux代码"
fi

# 将macOS代码平铺到根目录
echo "[INFO] 正在释放macOS代码..."
cp -r "data/mac"/* .

# 1. 基础环境自检
echo "[INFO] 正在检测系统 Python 环境..."
python3 --version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "${RED}[ERROR] 未检测到 Python，请先安装 Python 3.10 或以上版本！${NC}"
    echo "${RED}[ERROR] 请确保安装时勾选了 \"Add Python to PATH\"${NC}"
    read -p "按 Enter 键退出..."
    exit 1
fi

# 2. 创建隔离运行环境
echo "[INFO] 正在创建虚拟环境 (venv)..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "${GREEN}[SUCCESS] 虚拟环境创建成功${NC}"
else
    echo "[INFO] 检测到已存在 venv 文件夹，跳过创建步骤"
fi

# 3. 激活并升级核心引擎
echo "[INFO] 正在激活环境并准备安装核心组件..."
source ./venv/bin/activate
python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 专项安装 PyTorch (针对 macOS 优化)
echo "[INFO] 正在安装 PyTorch (支持 MPS 加速)..."
echo "[INFO] 这可能需要几分钟，请保持网络连接..."
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 安装剩余依赖包
echo "[INFO] 正在根据要求安装 WebUI 组件与医疗影像库..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "${YELLOW}[WARNING] 未发现 requirements.txt，将安装基础组件...${NC}"
    pip install gradio monai scikit-image Pillow numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

echo ""
echo "============================================================"
echo "${GREEN}✅ macOS环境初始化完成！${NC}"
echo ""
echo "提示：现在您可以直接关闭此窗口，"
echo "然后运行 ./start_macos.sh 开启您的研发之旅了"
echo "============================================================"
read -p "按 Enter 键退出..."
