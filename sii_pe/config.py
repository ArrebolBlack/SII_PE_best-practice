"""
配置系统：支持 YAML 文件 + 环境变量的层级化配置加载。

加载优先级（后者覆盖前者）：
  config/default.yaml < config.yaml（用户配置）< task.yaml < 环境变量 < CLI 参数

推荐方式：编辑 config.yaml 管理所有配置（包括 API Key）。
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """全局配置"""

    # LLM 配置
    api_keys: list[str] = field(default_factory=list)
    api_base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 1.0
    max_tokens: int = 8192

    # 评测配置
    max_concurrency: int = 10
    num_trials: int = 5
    val_data_path: str = ""

    # 优化器配置 — 独立的 LLM provider，用于生成/改进 prompt
    # 优化器应使用最强模型（如 Claude、GPT-4o、deepseek-reasoner）
    optimizer_api_keys: list[str] = field(default_factory=list)  # 为空时 fallback 到 api_keys
    optimizer_api_base_url: str = ""  # 为空时 fallback 到 api_base_url
    optimizer_model: str = "deepseek-chat"
    max_iterations: int = 20
    population_size: int = 50
    survivors_per_gen: int = 10

    # 输出配置
    result_dir: str = "results"
    log_dir: str = "logs"

    @classmethod
    def load(
        cls,
        yaml_path: Optional[str] = None,
        task_yaml_path: Optional[str] = None,
        cli_overrides: Optional[dict] = None,
    ) -> "Config":
        """
        层级化加载配置。

        加载顺序：default.yaml → config.yaml（用户配置）→ task.yaml → 环境变量 → CLI 参数

        用户配置文件查找顺序：
        1. yaml_path 参数指定的路径
        2. 当前工作目录的 config.yaml
        3. 项目根目录的 config.yaml
        """
        # 1. 从 default.yaml 加载基础配置
        default_yaml = os.path.join(
            os.path.dirname(__file__), "..", "config", "default.yaml"
        )
        merged = {}
        if os.path.exists(default_yaml):
            merged = cls._load_yaml(default_yaml)

        # 2. 自动查找用户 config.yaml
        user_yaml = None
        if yaml_path:
            # 显式指定路径
            user_yaml = yaml_path
        elif os.path.exists("config.yaml"):
            # 当前工作目录
            user_yaml = "config.yaml"
        elif os.path.exists(os.path.join(os.path.dirname(__file__), "..", "config.yaml")):
            # 项目根目录
            user_yaml = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

        if user_yaml and os.path.exists(user_yaml):
            user_cfg = cls._load_yaml(user_yaml)
            merged = cls._deep_merge(merged, user_cfg)
            logger.info(f"已加载用户配置: {os.path.abspath(user_yaml)}")

        # 3. task yaml 覆盖
        if task_yaml_path and os.path.exists(task_yaml_path):
            task_cfg = cls._load_yaml(task_yaml_path)
            merged = cls._deep_merge(merged, task_cfg)

        # 4. 环境变量覆盖
        env_overrides = cls._load_env()
        merged = cls._deep_merge(merged, env_overrides)

        # 5. CLI 参数覆盖
        if cli_overrides:
            merged = cls._deep_merge(merged, cli_overrides)

        # 展平嵌套结构
        flat = cls._flatten(merged)
        dataclass_fields = cls.__dataclass_fields__
        config = cls(**{k: v for k, v in flat.items() if k in dataclass_fields and v is not None})

        # 校验
        config._validate()
        return config

    def _validate(self):
        """校验配置，必要时输出警告。"""
        if not self.api_keys:
            logger.warning(
                "未配置 API Key。请在 config.yaml 中设置 llm.api_keys，或通过环境变量 SII_PE_API_KEYS 设置。\n"
                "示例: cp config.example.yaml config.yaml，然后编辑 config.yaml 填入你的 key。"
            )

        if self.api_keys and self.max_concurrency > 10 * len(self.api_keys):
            logger.warning(
                f"当前并发数 {self.max_concurrency} 超过 {len(self.api_keys)} 个 API Key "
                f"的建议上限（每 key 约 10 并发）。建议在 config.yaml 中添加更多 key 以保证负载均衡。"
            )

        # 优化器配置 fallback
        if not self.optimizer_api_keys:
            self.optimizer_api_keys = self.api_keys
            logger.info("优化器 API Key 未单独配置，复用评测 API Key")
        if not self.optimizer_api_base_url:
            self.optimizer_api_base_url = self.api_base_url

    def get_optimizer_api_keys(self) -> list[str]:
        """获取优化器的 API Keys（已处理 fallback）。"""
        return self.optimizer_api_keys or self.api_keys

    def get_optimizer_api_base_url(self) -> str:
        """获取优化器的 API Base URL（已处理 fallback）。"""
        return self.optimizer_api_base_url or self.api_base_url

    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _load_env() -> dict:
        """从 SII_PE_ 前缀的环境变量中加载配置。"""
        result = {}
        env_map = {
            "SII_PE_API_KEYS": "api_keys",
            "SII_PE_API_BASE_URL": "api_base_url",
            "SII_PE_MODEL": "model",
            "SII_PE_DATA_PATH": "val_data_path",
            "SII_PE_MAX_CONCURRENCY": "max_concurrency",
            "SII_PE_NUM_TRIALS": "num_trials",
            "SII_PE_MAX_ITERATIONS": "max_iterations",
            # 优化器独立配置
            "SII_PE_OPTIMIZER_API_KEYS": "optimizer_api_keys",
            "SII_PE_OPTIMIZER_API_BASE_URL": "optimizer_api_base_url",
            "SII_PE_OPTIMIZER_MODEL": "optimizer_model",
        }
        for env_key, config_key in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                if config_key in ("api_keys", "optimizer_api_keys"):
                    result[config_key] = [k.strip() for k in val.split(",") if k.strip()]
                elif config_key in ("max_concurrency", "num_trials", "max_iterations"):
                    result[config_key] = int(val)
                else:
                    result[config_key] = val
        return result

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """递归合并字典，override 中的值覆盖 base。"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _flatten(d: dict) -> dict:
        """将嵌套的 YAML 配置展平为 Config 字段。"""
        flat = {}
        section_map = {
            "llm": ["api_keys", "api_base_url", "model", "temperature", "max_tokens"],
            "evaluation": ["max_concurrency", "num_trials", "val_data_path"],
            "optimizer": ["optimizer_api_keys", "optimizer_api_base_url", "optimizer_model", "max_iterations", "population_size", "survivors_per_gen"],
            "output": ["result_dir", "log_dir"],
        }
        # 直接在顶层的字段
        for key, value in d.items():
            if not isinstance(value, dict):
                flat[key] = value

        # 从嵌套 section 中提取
        for section, fields in section_map.items():
            if section in d and isinstance(d[section], dict):
                for field_name in fields:
                    if field_name in d[section]:
                        flat[field_name] = d[section][field_name]

        return flat
