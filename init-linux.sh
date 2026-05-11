#!/bin/bash

# 强制开启 UTF-8 编码
export LANG="zh_CN.UTF-8"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# 辅助函数
print_banner() {
    echo -e "${BLUE}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    🚀  腺而易见 - 初始化引擎                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_divider() {
    echo -e "${BLUE}────────────────────────────────────────────────────────────────${NC}"
}

# 主程序
main() {
    clear
    print_banner
    
    # 删除其他平台的代码文件夹
    echo -e "${YELLOW}[1/6] 清理其他平台代码...${NC}"
    if [ -d "data/win" ]; then
        rm -rf "data/win"
        echo -e "${GREEN}  ✓ 已删除Windows代码${NC}"
    fi
    if [ -d "data/mac" ]; then
        rm -rf "data/mac"
        echo -e "${GREEN}  ✓ 已删除macOS代码${NC}"
    fi
    
    # 将Linux代码平铺到根目录
    echo -e "${YELLOW}[2/6] 释放Linux代码...${NC}"
    cp -r "data/linux"/* .
    echo -e "${GREEN}  ✓ 代码释放完成${NC}"
    
    # 关闭 conda 环境
    echo -e "${YELLOW}[3/6] 清理 conda 环境...${NC}"
    if command -v conda &> /dev/null; then
        conda deactivate 2>/dev/null || true
        conda deactivate 2>/dev/null || true
        conda deactivate 2>/dev/null || true
        echo -e "${GREEN}  ✓ conda 已完全关闭${NC}"
    else
        echo -e "${YELLOW}  - 未检测到 conda，跳过${NC}"
    fi
    
    # 检查 Python
    echo -e "${YELLOW}[4/6] 检查 Python 环境...${NC}"
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: 未找到 python3，请先安装 Python 3.8+${NC}"
        exit 1
    fi
    PYTHON_VER=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    echo -e "${GREEN}  ✓ Python $PYTHON_VER 已就绪${NC}"
    
    # 管理虚拟环境
    echo -e "${YELLOW}[5/6] 配置虚拟环境...${NC}"
    VENV_DIR="./venv"
    
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}  → 创建虚拟环境...${NC}"
        python3 -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo -e "${RED}  ✗ 创建失败${NC}"
            exit 1
        fi
        
        echo -e "${YELLOW}  → 升级 pip...${NC}"
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip -q -i https://pypi.tuna.tsinghua.edu.cn/simple
        
        echo -e "${YELLOW}  → 安装 PyTorch...${NC}"
        pip install torch torchvision -q -i https://pypi.tuna.tsinghua.edu.cn/simple
        
        echo -e "${YELLOW}  → 安装依赖...${NC}"
        if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple
        else
            pip install gradio monai scikit-image Pillow numpy -q -i https://pypi.tuna.tsinghua.edu.cn/simple
        fi
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}  ✗ 依赖安装失败${NC}"
            exit 1
        fi
        echo -e "${GREEN}  ✓ 虚拟环境配置完成${NC}"
    else
        echo -e "${GREEN}  ✓ 虚拟环境已就绪${NC}"
        source "$VENV_DIR/bin/activate"
    fi
    
    echo ""
    print_divider
    echo -e "${GREEN}${BOLD}"
    echo "  ✨ Linux环境初始化完成！"
    echo ""
    echo "  现在您可以运行 ./run.sh 来启动程序"
    echo -e "${NC}"
}

# 启动
main "$@"
