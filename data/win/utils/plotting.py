"""
论文绘图模块
用于生成医学顶刊标准的矢量图
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置全局样式
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8
matplotlib.rcParams['lines.linewidth'] = 0.8


def plot_metrics_bar(
    model_names,
    dice_scores,
    iou_scores=None,
    hd95_scores=None,
    save_path="metrics_comparison.pdf",
    dpi=300,
    bbox_inches="tight",
):
    """
    绘制不同模型的Dice/IoU/HD95对比柱状图
    
    Args:
        model_names: 模型名称列表
        dice_scores: Dice分数列表
        iou_scores: IoU分数列表（可选）
        hd95_scores: HD95分数列表（可选）
        save_path: 保存路径
        dpi: 分辨率
        bbox_inches: 边界框设置
    """
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制Dice
    rects1 = ax.bar(x - width, dice_scores, width, label='Dice', color='#1f77b4')
    
    # 绘制IoU
    if iou_scores is not None:
        rects2 = ax.bar(x, iou_scores, width, label='IoU', color='#ff7f0e')
    
    # 绘制HD95
    if hd95_scores is not None:
        rects3 = ax.bar(x + width, hd95_scores, width, label='HD95', color='#2ca02c')
    
    # 设置标签和标题
    ax.set_ylabel('Score')
    ax.set_xlabel('Model')
    ax.set_title('Segmentation Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    autolabel(rects1)
    if iou_scores is not None:
        autolabel(rects2)
    if hd95_scores is not None:
        autolabel(rects3)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, format='pdf')
    plt.close()
    return save_path


def plot_segmentation_comparison(
    original_image,
    gt_mask,
    pred_mask,
    save_path="segmentation_comparison.pdf",
    dpi=300,
    bbox_inches="tight",
):
    """
    绘制原图+GT+AI预测三栏对比图
    
    Args:
        original_image: 原始图像
        gt_mask: 真实标注
        pred_mask: 预测结果
        save_path: 保存路径
        dpi: 分辨率
        bbox_inches: 边界框设置
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # GT overlay - 半透明绿色
    img_gt = original_image.copy()
    if len(img_gt.shape) == 2:
        img_gt = np.dstack([img_gt, img_gt, img_gt])
    gt_mask_3ch = np.dstack([np.zeros_like(gt_mask), gt_mask, np.zeros_like(gt_mask)])
    axes[1].imshow(img_gt)
    axes[1].imshow(gt_mask_3ch, alpha=0.5)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction overlay - 半透明红色
    img_pred = original_image.copy()
    if len(img_pred.shape) == 2:
        img_pred = np.dstack([img_pred, img_pred, img_pred])
    pred_mask_3ch = np.dstack([pred_mask, np.zeros_like(pred_mask), np.zeros_like(pred_mask)])
    axes[2].imshow(img_pred)
    axes[2].imshow(pred_mask_3ch, alpha=0.5)
    axes[2].set_title('AI Prediction')
    axes[2].axis('off')
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, format='pdf')
    plt.close()
    return save_path


def plot_roc_curve(
    fpr,
    tpr,
    auc_value,
    save_path="roc_curve.pdf",
    dpi=300,
    bbox_inches="tight",
):
    """
    绘制ROC曲线带AUC值
    
    Args:
        fpr: 假阳性率
        tpr: 真阳性率
        auc_value: AUC值
        save_path: 保存路径
        dpi: 分辨率
        bbox_inches: 边界框设置
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'AUC = {auc_value:.3f}')
    ax.plot([0, 1], [0, 1], color='#7f7f7f', lw=1, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, format='pdf')
    plt.close()
    return save_path
