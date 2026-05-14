"""
通用帮助函数
"""
import os
import glob


def cleanup_checkpoints(max_keep=5):
    """
    清理旧的检查点，只保留最新的
    
    Args:
        max_keep: 保留的最大数量
    """
    ckpt_files = glob.glob("mgd_epoch_*.pth")
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
