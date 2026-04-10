"""
阶段 1：任务解析。

读取考试说明（文本），用 LLM 提取结构化任务规格。
"""

import json
import logging
from dataclasses import dataclass, field

from sii_pe.infra.client_pool import ClientPool
from sii_pe.infra.llm_caller import call_llm

logger = logging.getLogger(__name__)


@dataclass
class TaskSpec:
    """结构化任务规格。"""
    description: str = ""
    data_format: dict = field(default_factory=dict)
    metric_name: str = ""
    metric_direction: str = "higher_is_better"
    output_format: str = ""
    constraints: list[str] = field(default_factory=list)
    task_type: str = ""  # 如 "reranking", "grid_puzzle", "classification" 等


class TaskParser:
    """
    解析考试说明，提取任务描述、数据格式、评测指标、约束条件。
    """

    async def parse(self, instruction_text: str, pool: ClientPool, config) -> TaskSpec:
        """
        用 LLM 从自然语言说明中提取结构化任务规格。

        参数:
            instruction_text: 考试说明文本
            pool: 客户端池
            config: 配置

        返回:
            TaskSpec
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名任务分析专家。请从给定的考试说明中提取结构化的任务规格。"
                    "输出必须是 JSON 格式。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"请分析以下考试说明，提取关键信息：\n\n{instruction_text}\n\n"
                    f"输出 JSON 格式：\n"
                    f'{{\n'
                    f'  "description": "任务描述",\n'
                    f'  "data_format": {{"字段名": "字段说明"}},\n'
                    f'  "metric_name": "评测指标名称",\n'
                    f'  "metric_direction": "higher_is_better 或 lower_is_better",\n'
                    f'  "output_format": "期望的输出格式描述",\n'
                    f'  "constraints": ["约束条件1", "约束条件2"],\n'
                    f'  "task_type": "任务类型（如 reranking, grid_puzzle, classification）"\n'
                    f'}}'
                ),
            },
        ]

        try:
            response = await call_llm(
                pool, messages, model=config.optimizer_model, temperature=0.3, max_tokens=2048
            )
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end])

            spec = TaskSpec(
                description=data.get("description", ""),
                data_format=data.get("data_format", {}),
                metric_name=data.get("metric_name", ""),
                metric_direction=data.get("metric_direction", "higher_is_better"),
                output_format=data.get("output_format", ""),
                constraints=data.get("constraints", []),
                task_type=data.get("task_type", ""),
            )
            logger.info(f"任务解析完成: {spec.task_type} ({spec.metric_name})")
            return spec
        except Exception as e:
            logger.error(f"任务解析失败: {e}")
            return TaskSpec(description=instruction_text)
