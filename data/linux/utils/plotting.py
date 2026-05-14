"""
绘图工具 - 用于生成学术图表
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import List, Dict, Optional, Tuple

# 设置学术风格配置
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 11
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['figure.figsize'] = (8, 6)


def create_training_plot(
    loss_history: List[float],
    dice_history: Optional[List[float]] = None,
    lr_history: Optional[List[float]] = None,
    save_path: str = "training_plot.png",
    title: str = "训练过程",
    show: bool = False
) -> str:
    """
    创建训练过程的学术图表
    
    Args:
        loss_history: 损失历史
        dice_history: Dice系数历史
        lr_history: 学习率历史
        save_path: 保存路径
        title: 图表标题
        show: 是否显示图表
    
    Returns:
        保存的文件路径
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制损失曲线
    epochs = list(range(1, len(loss_history) + 1))
    line1, = ax1.plot(epochs, loss_history, 'b-', linewidth=2, label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    # 如果有Dice系数，在右侧Y轴绘制
    if dice_history:
        ax2 = ax1.twinx()
        line2, = ax2.plot(epochs, dice_history, 'r-', linewidth=2, label='Dice')
        ax2.set_ylabel('Dice Coefficient', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')
    else:
        ax1.legend(loc='best')
    
    # 如果有学习率，在第三个Y轴绘制（使用不同的比例
    if lr_history:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 80))
        line3, = ax3.plot(epochs, lr_history, 'g--', linewidth=1.5, label='Learning Rate')
        ax3.set_ylabel('Learning Rate', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        ax3.set_yscale('log')
        lines = [line1]
        labels = [l.get_label() for l in lines]
        if dice_history:
            lines.append(line2)
        lines.append(line3)
        ax1.legend(lines, labels, loc='best')
    
    plt.title(title, pad=15)
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()
    
    return save_path


def create_metric_comparison_plot(
    metrics_dict: Dict[str, List[float]],
    metric_name: str = "Metric",
    save_path: str = "comparison_plot.png",
    title: str = "模型对比",
    show: bool = False
) -> str:
    """
    创建多个模型/配置的对比图表
    
    Args:
        metrics_dict: 指标字典，键为配置名称，值为指标值列表
        metric_name: 指标名称
        save_path: 保存路径
        title: 图表标题
        show: 是否显示图表
    
    Returns:
        保存的文件路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(metrics_dict.keys())
    x_pos = np.arange(len(labels))
    values = [np.mean(v) for v in metrics_dict.values()]
    stds = [np.std(v) for v in metrics_dict.values()]
    
    # 绘制带误差棒的柱状图
    bars = ax.bar(x_pos, values, yerr=stds, align='center', alpha=0.7, capsize=5)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.4f}',
                 ha='center', va='bottom')
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel(metric_name)
    ax.set_title(title, pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()
    
    return save_path
