"""
阶段 2：相关调研。

针对任务类型搜索相关方法和论文，生成调研报告。
"""

import json
import logging
from dataclasses import dataclass, field

from sii_pe.infra.client_pool import ClientPool
from sii_pe.infra.llm_caller import call_llm
from sii_pe.workflow.task_parser import TaskSpec

logger = logging.getLogger(__name__)


@dataclass
class ResearchReport:
    """调研报告。"""
    related_methods: list[dict] = field(default_factory=list)  # [{"title", "summary", "relevance"}]
    recommended_approaches: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    raw_summary: str = ""


class Researcher:
    """
    针对任务类型进行相关调研。

    使用 LLM 的内在知识生成相关方法和推荐方案。
    如果集成了 web search 工具，可以进一步获取最新文献。
    """

    async def research(self, task_spec: TaskSpec, pool: ClientPool, config) -> ResearchReport:
        """
        根据 TaskSpec 生成调研报告。

        参数:
            task_spec: 结构化任务规格
            pool: 客户端池
            config: 配置

        返回:
            ResearchReport
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名 AI 研究专家，擅长 Prompt Engineering 和 LLM 应用。"
                    "请针对给定任务提供相关方法的调研报告。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"任务描述：{task_spec.description}\n"
                    f"任务类型：{task_spec.task_type}\n"
                    f"评测指标：{task_spec.metric_name}\n\n"
                    f"请提供：\n"
                    f"1. 该类任务的相关方法（包括 prompt engineering 策略）\n"
                    f"2. 推荐的优化方向\n"
                    f"3. 相关论文或参考文献\n\n"
                    f"输出 JSON 格式：\n"
                    f'{{\n'
                    f'  "related_methods": [\n'
                    f'    {{"title": "方法名", "summary": "简要描述", "relevance": "与本任务的关联"}}\n'
                    f'  ],\n'
                    f'  "recommended_approaches": ["推荐方案1", "推荐方案2"],\n'
                    f'  "citations": ["参考文献1", "参考文献2"]\n'
                    f'}}'
                ),
            },
        ]

        try:
            response = await call_llm(
                pool, messages, model=config.optimizer_model, temperature=0.5, max_tokens=4096
            )
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end])

            report = ResearchReport(
                related_methods=data.get("related_methods", []),
                recommended_approaches=data.get("recommended_approaches", []),
                citations=data.get("citations", []),
                raw_summary=response,
            )
            logger.info(
                f"调研完成: {len(report.related_methods)} 个相关方法, "
                f"{len(report.recommended_approaches)} 个推荐方案"
            )
            return report
        except Exception as e:
            logger.error(f"调研失败: {e}")
            return ResearchReport(raw_summary=str(e))
