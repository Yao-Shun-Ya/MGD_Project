#!/usr/bin/env python3
import sys
import os
import time
from typing import Dict, Any, List, Optional
from preset_manager import PresetManager, init_default_presets


# ANSI 颜色代码
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def clear_screen():
    """清除屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """打印程序横幅"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    \"腺\"而易见 - 训练引擎                      ║")
    print("║              交互式训练预设管理系统 v1.0                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")


def print_divider(char='─', length=70):
    """打印分隔线"""
    print(char * length)


def input_int(prompt: str, min_val: Optional[int] = None, max_val: Optional[int] = None, 
              default: Optional[int] = None) -> int:
    """安全输入整数"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default
            value = int(user_input)
            if min_val is not None and value < min_val:
                print(f"{Colors.RED}错误: 不能小于 {min_val}{Colors.RESET}")
                continue
            if max_val is not None and value > max_val:
                print(f"{Colors.RED}错误: 不能大于 {max_val}{Colors.RESET}")
                continue
            return value
        except ValueError:
            print(f"{Colors.RED}错误: 请输入有效的整数{Colors.RESET}")


def input_float(prompt: str, min_val: Optional[float] = None, max_val: Optional[float] = None,
                default: Optional[float] = None) -> float:
    """安全输入浮点数"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default
            value = float(user_input)
            if min_val is not None and value < min_val:
                print(f"{Colors.RED}错误: 不能小于 {min_val}{Colors.RESET}")
                continue
            if max_val is not None and value > max_val:
                print(f"{Colors.RED}错误: 不能大于 {max_val}{Colors.RESET}")
                continue
            return value
        except ValueError:
            print(f"{Colors.RED}错误: 请输入有效的数字{Colors.RESET}")


def input_choice(prompt: str, choices: List[str], default: Optional[int] = None) -> int:
    """安全输入选择"""
    while True:
        for i, choice in enumerate(choices, 1):
            print(f"  {i}. {choice}")
        user_input = input(prompt).strip()
        if not user_input and default is not None:
            return default
        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(choices):
                return idx
            print(f"{Colors.RED}错误: 请输入 1-{len(choices)} 之间的数字{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}错误: 请输入有效的数字{Colors.RESET}")


def input_bool(prompt: str, default: bool = True) -> bool:
    """安全输入布尔值"""
    yes_str = "是" if default else "否"
    no_str = "否" if default else "是"
    while True:
        user_input = input(f"{prompt} (1={yes_str}, 2={no_str}) [默认: {1 if default else 2}]: ").strip()
        if not user_input:
            return default
        if user_input == "1":
            return True
        elif user_input == "2":
            return False
        print(f"{Colors.RED}错误: 请输入 1 或 2{Colors.RESET}")


# 参数配置定义
PARAM_CONFIG = [
    {
        "key": "epochs",
        "name": "训练轮数",
        "type": "int",
        "min": 1,
        "max": 1000,
        "default": 100
    },
    {
        "key": "batch_size",
        "name": "批次大小",
        "type": "int",
        "min": 1,
        "max": 32,
        "default": 8
    },
    {
        "key": "lr",
        "name": "学习率",
        "type": "float",
        "min": 1e-6,
        "max": 0.1,
        "default": 0.001
    },
    {
        "key": "optimizer",
        "name": "优化器",
        "type": "choice",
        "choices": ["Adam", "AdamW", "SGD"],
        "default": 1
    },
    {
        "key": "scheduler",
        "name": "学习率调度器",
        "type": "choice",
        "choices": ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "CosineAnnealing", "StepDecay", "Fixed"],
        "default": 2
    },
    {
        "key": "loss",
        "name": "损失函数",
        "type": "choice",
        "choices": ["Dice", "BCE", "Focal", "Dice+BCE"],
        "default": 3
    },
    {
        "key": "precision",
        "name": "训练精度",
        "type": "choice",
        "choices": ["fp16", "fp32"],
        "default": 0
    },
    {
        "key": "augmentation",
        "name": "数据增强",
        "type": "bool",
        "default": True
    },
    {
        "key": "num_workers",
        "name": "数据加载线程数",
        "type": "int",
        "min": 0,
        "max": 32,
        "default": 8
    },
    {
        "key": "save_freq",
        "name": "保存频率 (每N轮)",
        "type": "int",
        "min": 1,
        "max": 100,
        "default": 10
    },
    {
        "key": "max_keep",
        "name": "保留Checkpoint数量",
        "type": "int",
        "min": 1,
        "max": 20,
        "default": 5
    },
    {
        "key": "weight_decay",
        "name": "权重衰减",
        "type": "float",
        "min": 0.0,
        "max": 0.1,
        "default": 1e-4
    },
    {
        "key": "pos_weight",
        "name": "正负样本权重",
        "type": "float",
        "min": 0.1,
        "max": 10.0,
        "default": 2.0
    }
]


def configure_parameters(initial_params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """逐项配置参数，支持返回上一级"""
    if initial_params is None:
        initial_params = {}
    
    params = initial_params.copy()
    param_idx = 0
    
    while param_idx < len(PARAM_CONFIG):
        clear_screen()
        print_banner()
        print(f"{Colors.CYAN}📝 参数配置 - 第 {param_idx + 1}/{len(PARAM_CONFIG)} 项{Colors.RESET}")
        print_divider()
        
        config = PARAM_CONFIG[param_idx]
        key = config["key"]
        name = config["name"]
        param_type = config["type"]
        default_val = initial_params.get(key, config.get("default"))
        
        print(f"\n{Colors.BOLD}{name}{Colors.RESET}")
        
        if param_idx > 0:
            print(f"{Colors.YELLOW}[提示] 输入 0 返回上一项修改{Colors.RESET}")
        
        value = None
        
        if param_type == "int":
            prompt = f"请输入 {name} [{config['min']}-{config['max']}, 默认: {default_val}]: "
            while True:
                user_input = input(prompt).strip()
                if user_input == "0" and param_idx > 0:
                    param_idx -= 2
                    break
                if not user_input:
                    params[key] = default_val
                    break
                try:
                    val = int(user_input)
                    if config["min"] <= val <= config["max"]:
                        params[key] = val
                        break
                    print(f"{Colors.RED}错误: 请输入 {config['min']}-{config['max']} 之间的整数{Colors.RESET}")
                except ValueError:
                    print(f"{Colors.RED}错误: 请输入有效的整数{Colors.RESET}")
        
        elif param_type == "float":
            prompt = f"请输入 {name} [{config['min']}-{config['max']}, 默认: {default_val}]: "
            while True:
                user_input = input(prompt).strip()
                if user_input == "0" and param_idx > 0:
                    param_idx -= 2
                    break
                if not user_input:
                    params[key] = default_val
                    break
                try:
                    val = float(user_input)
                    if config["min"] <= val <= config["max"]:
                        params[key] = val
                        break
                    print(f"{Colors.RED}错误: 请输入 {config['min']}-{config['max']} 之间的数字{Colors.RESET}")
                except ValueError:
                    print(f"{Colors.RED}错误: 请输入有效的数字{Colors.RESET}")
        
        elif param_type == "choice":
            choices = config["choices"]
            default_idx = config["default"]
            if key in initial_params:
                try:
                    default_idx = choices.index(initial_params[key])
                except ValueError:
                    pass
            
            print(f"\n请选择 {name}:")
            for i, choice in enumerate(choices, 1):
                marker = " [默认]" if i - 1 == default_idx else ""
                print(f"  {i}. {choice}{marker}")
            
            while True:
                user_input = input(f"请输入选项序号 (1-{len(choices)}): ").strip()
                if user_input == "0" and param_idx > 0:
                    param_idx -= 2
                    value = None
                    break
                if not user_input:
                    params[key] = choices[default_idx]
                    break
                try:
                    idx = int(user_input) - 1
                    if 0 <= idx < len(choices):
                        params[key] = choices[idx]
                        break
                    print(f"{Colors.RED}错误: 请输入 1-{len(choices)} 之间的数字{Colors.RESET}")
                except ValueError:
                    print(f"{Colors.RED}错误: 请输入有效的数字{Colors.RESET}")
        
        elif param_type == "bool":
            yes_val = default_val
            yes_str = "是" if yes_val else "否"
            no_str = "否" if yes_val else "是"
            default_choice = 1 if yes_val else 2
            
            while True:
                user_input = input(f"请选择 {name} (1={yes_str}, 2={no_str}) [默认: {default_choice}]: ").strip()
                if user_input == "0" and param_idx > 0:
                    param_idx -= 2
                    value = None
                    break
                if not user_input:
                    params[key] = yes_val
                    break
                if user_input == "1":
                    params[key] = True
                    break
                elif user_input == "2":
                    params[key] = False
                    break
                print(f"{Colors.RED}错误: 请输入 1 或 2{Colors.RESET}")
        
        param_idx += 1
    
    # 确认参数
    clear_screen()
    print_banner()
    print(f"{Colors.CYAN}📋 配置确认{Colors.RESET}")
    print_divider()
    print()
    
    for config in PARAM_CONFIG:
        key = config["key"]
        name = config["name"]
        value = params.get(key)
        print(f"  {name}: {Colors.GREEN}{value}{Colors.RESET}")
    
    print()
    print_divider()
    print("\n请选择:")
    print("  1. 确认并继续")
    print("  2. 重新配置")
    print("  3. 返回主菜单")
    
    choice = input_int("请输入选项: ", 1, 3)
    
    if choice == 1:
        return params
    elif choice == 2:
        return configure_parameters(initial_params)
    else:
        return None


def menu_select_preset(manager: PresetManager) -> Optional[Dict[str, Any]]:
    """选择已有预设菜单"""
    while True:
        clear_screen()
        print_banner()
        print(f"{Colors.CYAN}📁 选用已有预设{Colors.RESET}")
        print_divider()
        
        presets = manager.list_presets()
        
        if not presets:
            print(f"{Colors.YELLOW}当前没有预设，请先创建自定义预设{Colors.RESET}")
            print()
            input("按 Enter 返回主菜单...")
            return None
        
        print(f"\n共有 {len(presets)} 个预设:\n")
        for i, name in enumerate(presets, 1):
            print(f"  {i}. {name}")
        
        print()
        print_divider()
        print("\n请选择:")
        print("  输入预设序号: 选择该预设")
        print("  输入 0: 返回主菜单")
        
        choice = input_int("请输入: ", 0, len(presets))
        
        if choice == 0:
            return None
        
        selected_name = presets[choice - 1]
        preset = manager.get_preset(selected_name)
        
        # 自定义训练轮数
        clear_screen()
        print_banner()
        print(f"{Colors.CYAN}⚙️ 预设: {selected_name}{Colors.RESET}")
        print_divider()
        print("\n预设参数:")
        for config in PARAM_CONFIG:
            key = config["key"]
            name = config["name"]
            value = preset.get(key, "-")
            print(f"  {name}: {value}")
        
        print()
        print_divider()
        
        default_epochs = preset.get("epochs", 100)
        epochs = input_int(f"\n请输入本次训练的轮数 [默认: {default_epochs}]: ", 1, 1000, default_epochs)
        
        preset["epochs"] = epochs
        
        # 确认
        print()
        print_divider()
        print("\n请选择:")
        print("  1. 确认并开始训练")
        print("  2. 修改其他参数")
        print("  3. 返回")
        
        confirm = input_int("请输入选项: ", 1, 3)
        
        if confirm == 1:
            return preset
        elif confirm == 2:
            modified = configure_parameters(preset)
            if modified:
                return modified
        # else: continue


def menu_create_preset(manager: PresetManager) -> Optional[Dict[str, Any]]:
    """创建自定义预设菜单"""
    clear_screen()
    print_banner()
    print(f"{Colors.CYAN}✨ 新建自定义预设{Colors.RESET}")
    print_divider()
    
    params = configure_parameters()
    
    if params is None:
        return None
    
    # 命名预设
    while True:
        clear_screen()
        print_banner()
        print(f"{Colors.CYAN}🏷️  命名预设{Colors.RESET}")
        print_divider()
        print()
        
        name = input("请输入预设名称: ").strip()
        
        if not name:
            print(f"{Colors.RED}错误: 名称不能为空{Colors.RESET}")
            input("按 Enter 继续...")
            continue
        
        if name in manager.list_presets():
            print(f"{Colors.YELLOW}警告: 预设 '{name}' 已存在{Colors.RESET}")
            overwrite = input_bool("是否覆盖?", False)
            if not overwrite:
                continue
        
        manager.add_preset(name, params)
        print(f"\n{Colors.GREEN}✅ 预设 '{name}' 已保存!{Colors.RESET}")
        print()
        input("按 Enter 继续...")
        
        return params


def menu_batch_train(manager: PresetManager) -> Optional[List[Dict[str, Any]]]:
    """批量训练菜单"""
    while True:
        clear_screen()
        print_banner()
        print(f"{Colors.CYAN}🔄 批量训练{Colors.RESET}")
        print_divider()
        
        presets = manager.list_presets()
        
        if not presets:
            print(f"{Colors.YELLOW}当前没有预设，请先创建预设{Colors.RESET}")
            print()
            input("按 Enter 返回主菜单...")
            return None
        
        print(f"\n共有 {len(presets)} 个预设:\n")
        for i, name in enumerate(presets, 1):
            print(f"  {i}. {name}")
        
        print()
        print_divider()
        print("\n使用说明: 输入预设序号，多个序号用逗号分隔 (如: 1,3,5)")
        print("输入 0 返回主菜单")
        
        user_input = input("请选择: ").strip()
        
        if user_input == "0":
            return None
        
        try:
            indices = [int(x.strip()) - 1 for x in user_input.split(",")]
            selected_presets = []
            for idx in indices:
                if 0 <= idx < len(presets):
                    name = presets[idx]
                    preset = manager.get_preset(name)
                    preset["_name"] = name
                    selected_presets.append(preset)
            
            if not selected_presets:
                print(f"{Colors.RED}错误: 没有选择有效的预设{Colors.RESET}")
                input("按 Enter 继续...")
                continue
            
            # 确认
            clear_screen()
            print_banner()
            print(f"{Colors.CYAN}📋 批量训练确认{Colors.RESET}")
            print_divider()
            print(f"\n已选择 {len(selected_presets)} 个预设，将按顺序执行:\n")
            for i, preset in enumerate(selected_presets, 1):
                print(f"  {i}. {preset.get('_name', '未命名')}")
            
            print()
            print_divider()
            print("\n请选择:")
            print("  1. 确认并开始批量训练")
            print("  2. 重新选择")
            print("  3. 返回主菜单")
            
            confirm = input_int("请输入选项: ", 1, 3)
            
            if confirm == 1:
                return selected_presets
            elif confirm == 3:
                return None
            
        except ValueError:
            print(f"{Colors.RED}错误: 输入格式不正确{Colors.RESET}")
            input("按 Enter 继续...")


def run_training(params: Dict[str, Any], preset_name: str = "自定义"):
    """运行训练（模拟，实际调用 train_cli）"""
    clear_screen()
    print_banner()
    print(f"{Colors.CYAN}🚀 开始训练: {preset_name}{Colors.RESET}")
    print_divider()
    print()
    
    # 构建命令行参数
    cmd = ["python3", "train_cli.py"]
    
    param_map = {
        "epochs": "--epochs",
        "batch_size": "--batch-size",
        "lr": "--lr",
        "optimizer": "--optimizer",
        "scheduler": "--scheduler",
        "loss": "--loss",
        "precision": "--precision",
        "num_workers": "--num-workers",
        "save_freq": "--save-freq",
        "max_keep": "--max-keep",
        "weight_decay": "--weight-decay",
        "pos_weight": "--pos-weight"
    }
    
    for key, arg in param_map.items():
        if key in params:
            cmd.extend([arg, str(params[key])])
    
    if not params.get("augmentation", True):
        cmd.append("--no-augmentation")
    
    print(f"{Colors.YELLOW}执行命令:{' '.join(cmd)}{Colors.RESET}")
    print()
    print_divider()
    print()
    
    # 实际执行训练
    import subprocess
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode == 0
    except Exception as e:
        print(f"{Colors.RED}训练失败: {e}{Colors.RESET}")
        return False


def menu_main():
    """主菜单"""
    manager = PresetManager()
    init_default_presets()
    
    while True:
        clear_screen()
        print_banner()
        print(f"{Colors.CYAN}🏠 主菜单{Colors.RESET}")
        print_divider()
        
        presets = manager.list_presets()
        print(f"\n当前预设数量: {Colors.GREEN}{len(presets)}{Colors.RESET}\n")
        
        print("请选择操作:")
        print("  1. 选用已有预设")
        print("  2. 新建自定义预设")
        print("  3. 批量训练")
        print("  4. 退出程序")
        
        choice = input_int("\n请输入选项: ", 1, 4)
        
        if choice == 1:
            preset = menu_select_preset(manager)
            if preset:
                run_training(preset, "已选预设")
                print()
                input("按 Enter 返回主菜单...")
        
        elif choice == 2:
            preset = menu_create_preset(manager)
            if preset:
                # 询问是否立即训练
                print()
                if input_bool("是否立即使用此预设开始训练?", True):
                    run_training(preset, "新建预设")
                    print()
                    input("按 Enter 返回主菜单...")
        
        elif choice == 3:
            preset_list = menu_batch_train(manager)
            if preset_list:
                # 批量执行
                results = []
                for i, preset in enumerate(preset_list, 1):
                    name = preset.get("_name", f"预设{i}")
                    print(f"\n{Colors.MAGENTA}════════════════════════════════════════{Colors.RESET}")
                    print(f"{Colors.MAGENTA}批量训练 {i}/{len(preset_list)}: {name}{Colors.RESET}")
                    print(f"{Colors.MAGENTA}════════════════════════════════════════{Colors.RESET}")
                    success = run_training(preset, name)
                    results.append((name, success))
                    if i < len(preset_list):
                        time.sleep(2)
                
                # 汇总结果
                clear_screen()
                print_banner()
                print(f"{Colors.CYAN}📊 批量训练结果汇总{Colors.RESET}")
                print_divider()
                print()
                for name, success in results:
                    status = f"{Colors.GREEN}✅ 成功{Colors.RESET}" if success else f"{Colors.RED}❌ 失败{Colors.RESET}"
                    print(f"  {name}: {status}")
                print()
                print_divider()
                input("按 Enter 返回主菜单...")
        
        elif choice == 4:
            clear_screen()
            print(f"{Colors.GREEN}感谢使用 \"腺\"而易见 训练引擎!{Colors.RESET}")
            print()
            sys.exit(0)


if __name__ == "__main__":
    try:
        menu_main()
    except KeyboardInterrupt:
        clear_screen()
        print(f"\n{Colors.YELLOW}程序已中断{Colors.RESET}")
        sys.exit(0)
