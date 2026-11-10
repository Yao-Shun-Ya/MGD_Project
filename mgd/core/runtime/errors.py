"""统一异常类型。外壳只需捕获 MGDError 即可给出面向用户的提示。"""


class MGDError(Exception):
    """所有业务异常的基类。"""

    def __init__(self, message: str, hint: str | None = None):
        self.hint = hint
        super().__init__(message)

    def user_message(self) -> str:
        if self.hint:
            return f"{self}\n提示：{self.hint}"
        return str(self)


class ConfigError(MGDError):
    """配置非法或缺失。"""


class DataIntegrityError(MGDError):
    """数据集结构不完整 / 配对不一致。必须中止，不允许静默降级。"""


class DeviceError(MGDError):
    """设备不可用或显式指定的设备不存在。"""


class ModelNotFoundError(MGDError):
    """找不到可用权重。"""


class ManifestError(MGDError):
    """权重 manifest 缺失、校验失败或状态为 pending。"""


class WeightLoadError(MGDError):
    """权重与架构不匹配 / 文件损坏。"""


class InferenceError(MGDError):
    """推理过程失败（显存、形状、数值异常）。"""


class TrainingError(MGDError):
    """训练过程失败。"""


class InteropDependencyError(MGDError):
    """互操作可选依赖（pydicom / highdicom 等）未安装。

    消息必须包含可执行的安装命令；禁止用 `except: pass` 静默吞掉导入失败。
    """
