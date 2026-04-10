"""
任务抽象基类。

每个 PE 任务需实现此接口，提供 construct_prompt / parse_output / compute_metric。
评测器 (Evaluator) 通过此接口与任务逻辑解耦。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTask(ABC):
    """
    PE 任务的抽象基类。

    继承此类并实现所有抽象方法即可接入评测框架。
    对应考试中 Answer.py 的 construct_prompt + parse_output，
    额外添加了 metric 和数据处理方法。
    """

    @abstractmethod
    def construct_prompt(self, sample: dict, candidate) -> list[dict]:
        """
        构造 LLM 请求消息。

        参数:
            sample: 数据样本（已经 mask 过，不含答案）
            candidate: PromptCandidate 实例，包含 system_prompt 和 user_prompt_template

        返回:
            OpenAI Chat API messages 列表
        """

    @abstractmethod
    def parse_output(self, text: str) -> Any:
        """
        解析 LLM 原始输出为结构化结果。

        返回:
            解析后的预测结果；解析失败返回 None
        """

    @abstractmethod
    def compute_metric(self, prediction: Any, ground_truth: Any) -> float:
        """
        计算单样本得分。

        返回:
            得分，范围 [0, 1]
        """

    @abstractmethod
    def extract_ground_truth(self, sample: dict) -> Any:
        """从原始样本中提取真实答案。"""

    @abstractmethod
    def mask_sample(self, sample: dict) -> dict:
        """
        生成对模型可见的样本副本（隐藏答案）。

        例如 ARC 任务中去掉 test 的 output。
        默认返回原样本（无需 mask 的任务可不重写）。
        """

    def get_template_variables(self, sample: dict) -> dict:
        """
        从样本中提取 Jinja2 模板变量。

        子类可重写此方法以提供自定义变量，供 PromptCandidate.render() 使用。
        默认返回样本本身。
        """
        return sample
