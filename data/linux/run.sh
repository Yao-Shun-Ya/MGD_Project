#!/bin/bash

# ============================================================
# "腺"而易见 - 统一启动脚本
# 功能：先关闭 conda，自动管理 venv，支持交互/命令行双模式
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -------------------------- 颜色定义 --------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# -------------------------- 辅助函数 --------------------------
print_banner() {
    echo -e "${BLUE}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    🚀  \"腺\"而易见 - 训练引擎                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_divider() {
    echo -e "${BLUE}────────────────────────────────────────────────────────────────${NC}"
}

# -------------------------- 核心步骤 --------------------------

# 1. 关闭 conda 环境
clear_conda() {
    echo -e "${YELLOW}[1/4] 正在清理 conda 环境...${NC}"
    if command -v conda &> /dev/null; then
        conda deactivate 2>/dev/null || true
        conda deactivate 2>/dev/null || true
        conda deactivate 2>/dev/null || true
        echo -e "${GREEN}  ✓ conda 已完全关闭${NC}"
    else
        echo -e "${YELLOW}  - 未检测到 conda，跳过${NC}"
    fi
}

# 2. 检查 Python
check_python() {
    echo -e "${YELLOW}[2/4] 检查 Python 环境...${NC}"
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: 未找到 python3，请先安装 Python 3.8+${NC}"
        exit 1
    fi
    PYTHON_VER=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    echo -e "${GREEN}  ✓ Python $PYTHON_VER 已就绪${NC}"
}

# 3. 管理虚拟环境
setup_venv() {
    echo -e "${YELLOW}[3/4] 检查虚拟环境...${NC}"
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
        pip install --upgrade pip -q
        
        echo -e "${YELLOW}  → 安装依赖...${NC}"
        pip install -r requirements.txt -q
        if [ $? -ne 0 ]; then
            echo -e "${RED}  ✗ 依赖安装失败${NC}"
            exit 1
        fi
        echo -e "${GREEN}  ✓ 虚拟环境配置完成${NC}"
    else
        echo -e "${GREEN}  ✓ 虚拟环境已就绪${NC}"
        source "$VENV_DIR/bin/activate"
    fi
}

# 4. 选择模式并启动
run_program() {
    echo -e "${YELLOW}[4/4] 启动程序...${NC}"
    print_divider
    echo ""
    
    # 如果有参数，直接用命令行模式
    if [ $# -gt 0 ]; then
        echo -e "${CYAN}📌 命令行模式${NC}"
        echo -e "  执行: python3 train_cli.py $*"
        echo ""
        python3 train_cli.py "$@"
        return
    fi
    
    # 否则进入交互模式选择
    echo -e "${CYAN}请选择启动模式:${NC}"
    echo ""
    echo "  1) 🎮  交互式模式 (推荐)"
    echo "     - 预设管理、参数配置、批量训练"
    echo ""
    echo "  2) ⌨️  命令行模式"
    echo "     - 直接使用原 train_cli.py"
    echo ""
    echo "  3) ❌  退出"
    echo ""
    read -p "请输入选项 (1/2/3，默认 1): " choice
    choice=${choice:-1}
    
    case $choice in
        1)
            echo ""
            echo -e "${GREEN}🚀 启动交互式模式...${NC}"
            echo ""
            python3 train_cli_interactive.py
            ;;
        2)
            echo ""
            echo -e "${GREEN}🚀 启动命令行模式...${NC}"
            echo ""
            echo -e "${YELLOW}提示: 可在脚本后加参数可直接运行，如: ./run.sh --epochs 50 --batch-size 16${NC}"
            echo ""
            python3 train_cli.py
            ;;
        3)
            echo ""
            echo -e "${YELLOW}已退出${NC}"
            exit 0
            ;;
        *)
            echo ""
            echo -e "${RED}无效选项，启动交互式模式${NC}"
            echo ""
            python3 train_cli_interactive.py
            ;;
    esac
}

# -------------------------- 主程序 --------------------------
main() {
    clear
    print_banner
    
    clear_conda
    check_python
    setup_venv
    
    echo ""
    print_divider
    echo -e "${GREEN}${BOLD}"
    echo "  ✨ 环境准备完成！"
    echo -e "${NC}"
    
    run_program "$@"
}

# 启动
main "$@"
