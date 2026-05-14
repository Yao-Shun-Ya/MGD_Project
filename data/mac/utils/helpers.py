"""
辅助工具模块
包含检查点管理、预设管理等功能
"""
import os
import glob
import json
import torch
from config import PRESET_FILE, BUILTIN_PRESETS


def cleanup_checkpoints(max_keep=3, model_name="UNet"):
    """
    按轮次保留最近的 N 个 checkpoint
    
    Args:
        max_keep: 保留的最大数量
        model_name: 模型名称
    """
    from config import MODEL_DIR
    
    model_dir = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_dir):
        return
    
    pattern = os.path.join(model_dir, f"{model_name.lower()}_epoch_*.pth")
    ckpt_files = glob.glob(pattern)
    if len(ckpt_files) <= max_keep:
        return
    
    sorted_ckpts = sorted(
        ckpt_files,
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )
    
    for stale_ckpt in sorted_ckpts[:-max_keep]:
        try:
            os.remove(stale_ckpt)
        except OSError:
            pass


def load_all_presets():
    """
    加载所有预设（内置 + 自定义）
    
    Returns:
        预设字典
    """
    presets = dict(BUILTIN_PRESETS)
    if os.path.exists(PRESET_FILE):
        try:
            with open(PRESET_FILE, "r", encoding="utf-8") as f:
                custom = json.load(f)
            if isinstance(custom, dict):
                for k, v in custom.items():
                    if isinstance(v, dict):
                        presets[k] = v
        except Exception:
            pass
    return presets


def save_custom_preset_file(all_presets):
    """
    保存自定义预设到文件
    
    Args:
        all_presets: 所有预设字典
    """
    custom = {k: v for k, v in all_presets.items() if k not in BUILTIN_PRESETS}
    with open(PRESET_FILE, "w", encoding="utf-8") as f:
        json.dump(custom, f, ensure_ascii=False, indent=2)


def sync_infer_controls(tile_mode, tile_size, tile_overlap, threshold, min_area, smooth_on, smooth_sigma, roi_only, tta_mode="关闭"):
    """
    同步推理参数控件
    
    Args:
        各种推理参数
    
    Returns:
        同步后的参数元组
    """
    tile_mode = bool(tile_mode)
    tile_size = int(max(128, min(int(tile_size), 512)))
    tile_overlap = float(max(0.1, min(float(tile_overlap), 0.7)))
    threshold = float(max(0.4, min(float(threshold), 0.95)))
    min_area = int(max(20, min(int(min_area), 500)))
    smooth_on = bool(smooth_on)
    smooth_sigma = float(max(0.0, min(float(smooth_sigma), 3.0)))
    roi_only = bool(roi_only)
    
    return (tile_mode, tile_size, tile_overlap, threshold, min_area, smooth_on, smooth_sigma, roi_only, tta_mode,
            tile_mode, tile_size, tile_overlap, threshold, min_area, smooth_on, smooth_sigma, roi_only, tta_mode)
