
"""
Bundle ML 配置加载模块
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class MLConfig:
    """Bundle ML 配置类"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yml"

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    @property
    def dimensions(self) -> Dict[str, int]:
        """获取维度配置"""
        return self._config["dimensions"]

    @property
    def network(self) -> Dict[str, Any]:
        """获取网络配置"""
        return self._config["network"]

    @property
    def data(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self._config["data"]

    @property
    def training(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self._config["training"]

    @property
    def integration(self) -> Dict[str, Any]:
        """获取集成配置"""
        return self._config["integration"]

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


def get_default_config() -> MLConfig:
    """获取默认配置"""
    return MLConfig()
