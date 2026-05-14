#!/usr/bin/env python3
import json
import os
from typing import List, Dict, Any


PRESET_FILE = "training_presets.json"


class PresetManager:
    """预设管理类，负责预设的持久化存储和读取"""

    def __init__(self):
        self.presets: Dict[str, Dict[str, Any]] = {}
        self.load_presets()

    def load_presets(self) -> None:
        """从JSON文件加载预设"""
        if os.path.exists(PRESET_FILE):
            try:
                with open(PRESET_FILE, 'r', encoding='utf-8') as f:
                    self.presets = json.load(f)
            except Exception:
                self.presets = {}

    def save_presets(self) -> None:
        """保存预设到JSON文件"""
        with open(PRESET_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.presets, f, ensure_ascii=False, indent=2)

    def add_preset(self, name: str, params: Dict[str, Any]) -> bool:
        """添加新预设"""
        if name in self.presets:
            return False
        self.presets[name] = params
        self.save_presets()
        return True

    def update_preset(self, name: str, params: Dict[str, Any]) -> None:
        """更新预设"""
        self.presets[name] = params
        self.save_presets()

    def delete_preset(self, name: str) -> bool:
        """删除预设"""
        if name in self.presets:
            del self.presets[name]
            self.save_presets()
            return True
        return False

    def get_preset(self, name: str) -> Dict[str, Any]:
        """获取指定预设"""
        return self.presets.get(name, {})

    def list_presets(self) -> List[str]:
        """列出所有预设名称"""
        return sorted(self.presets.keys())

    def get_preset_count(self) -> int:
        """获取预设数量"""
        return len(self.presets)


def get_default_presets() -> Dict[str, Dict[str, Any]]:
    """获取默认预设"""
    return {
        "V100 32GB 优化配置": {
            "model": "UNet",
            "epochs": 100,
            "batch_size": 8,
            "lr": 0.001,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealing",
            "loss": "Dice+BCE",
            "precision": "fp16",
            "augmentation": True,
            "plot": True,
            "num_workers": 8,
            "save_freq": 10,
            "max_keep": 5,
            "weight_decay": 1e-4,
            "pos_weight": 2.0
        },
        "快速测试配置": {
            "model": "UNet",
            "epochs": 10,
            "batch_size": 2,
            "lr": 0.001,
            "optimizer": "Adam",
            "scheduler": "Fixed",
            "loss": "Dice",
            "precision": "fp32",
            "augmentation": False,
            "plot": True,
            "num_workers": 2,
            "save_freq": 5,
            "max_keep": 3,
            "weight_decay": 1e-4,
            "pos_weight": 2.0
        },
        "高精度训练配置": {
            "model": "AttentionUNet",
            "epochs": 200,
            "batch_size": 4,
            "lr": 0.0005,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "loss": "Dice+BCE",
            "precision": "fp32",
            "augmentation": True,
            "plot": True,
            "num_workers": 8,
            "save_freq": 20,
            "max_keep": 10,
            "weight_decay": 1e-4,
            "pos_weight": 2.0
        }
    }


def init_default_presets():
    """初始化默认预设（如果不存在）"""
    manager = PresetManager()
    if manager.get_preset_count() == 0:
        default_presets = get_default_presets()
        for name, params in default_presets.items():
            manager.add_preset(name, params)
        print("✅ 默认预设已初始化")


if __name__ == "__main__":
    init_default_presets()
    manager = PresetManager()
    print("当前预设列表:")
    for i, name in enumerate(manager.list_presets(), 1):
        print(f"  {i}. {name}")
