"""
阶段 5：报告生成。

生成 Markdown 格式的探索报告，包含任务描述、相关工作、实验日志和分析。
"""

import logging
from datetime import datetime

from sii_pe.core.population import Population
from sii_pe.infra.client_pool import ClientPool
from sii_pe.infra.llm_caller import call_llm
from sii_pe.infra.persistence import ExperimentLog
from sii_pe.workflow.task_parser import TaskSpec
from sii_pe.workflow.researcher import ResearchReport

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    生成探索报告（Markdown 格式）。
    """

    async def generate(
        self,
        task_spec: TaskSpec,
        research: ResearchReport,
        population: Population,
        experiment_log: ExperimentLog,
        pool: ClientPool,
        config,
    ) -> str:
        """
        生成完整的探索报告。

        包含：
        1. 任务描述
        2. 相关工作（引用文献）
        3. 实验日志（所有策略 + 分数表格）
        4. 分析与总结
        5. 最终最佳 prompt

        返回:
            Markdown 格式的报告文本
        """
        sections = []

        # 标题
        sections.append(f"# Prompt Engineering 探索报告\n\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 1. 任务描述
        sections.append("## 1. 任务描述\n")
        sections.append(f"- **任务类型**: {task_spec.task_type}")
        sections.append(f"- **评测指标**: {task_spec.metric_name}")
        sections.append(f"- **描述**: {task_spec.description}\n")

        # 2. 相关工作
        sections.append("## 2. 相关工作\n")
        if research.related_methods:
            for method in research.related_methods:
                sections.append(
                    f"### {method.get('title', '未知方法')}\n"
                    f"{method.get('summary', '')}\n"
                    f"**关联**: {method.get('relevance', '')}\n"
                )
        if research.citations:
            sections.append("### 参考文献\n")
            for i, cite in enumerate(research.citations, 1):
                sections.append(f"{i}. {cite}")
            sections.append("")

        # 3. 实验日志
        sections.append("## 3. 实验日志\n")
        if experiment_log.entries:
            sections.append("| # | 策略名称 | 得分 | 时间 |")
            sections.append("|---|---------|------|------|")
            for i, entry in enumerate(experiment_log.entries, 1):
                name = entry["candidate"].get("name", "unknown")
                score = entry["score"]
                timestamp = entry.get("timestamp", "")[:19]
                sections.append(f"| {i} | {name} | {score:.4f} | {timestamp} |")
            sections.append("")

        # 4. 分析
        sections.append("## 4. 分析与总结\n")
        if population.best:
            best_candidate, best_score = population.best
            sections.append(f"**最佳得分**: {best_score:.4f} (策略: {best_candidate.name})\n")

            # 用 LLM 生成分析
            try:
                trajectory = population.get_trajectory()
                analysis = await self._generate_analysis(trajectory, task_spec, pool, config)
                sections.append(analysis)
            except Exception as e:
                logger.warning(f"分析生成失败: {e}")
                sections.append("（分析生成失败）\n")

        # 5. 最佳 prompt
        sections.append("## 5. 最佳 Prompt\n")
        if population.best:
            best_candidate, _ = population.best
            sections.append(f"**策略名称**: {best_candidate.name}\n")
            sections.append(f"### 系统提示词\n```\n{best_candidate.system_prompt}\n```\n")
            sections.append(f"### 用户提示词模板\n```\n{best_candidate.user_prompt_template}\n```\n")

        return "\n".join(sections)

    async def _generate_analysis(
        self, trajectory: list[dict], task_spec: TaskSpec, pool: ClientPool, config
    ) -> str:
        """用 LLM 分析优化过程。"""
        trajectory_text = "\n".join(
            f"- {t['name']}: {t['score']:.4f}" for t in trajectory
        )

        messages = [
            {
                "role": "system",
                "content": "你是一名 Prompt Engineering 分析专家。请分析优化过程中的关键发现。",
            },
            {
                "role": "user",
                "content": (
                    f"任务: {task_spec.description}\n"
                    f"指标: {task_spec.metric_name}\n\n"
                    f"优化轨迹:\n{trajectory_text}\n\n"
                    f"请简要分析：\n"
                    f"1. 哪些因素对得分影响最大？\n"
                    f"2. 优化过程中的关键转折点是什么？\n"
                    f"3. 还有哪些可能的改进方向？"
                ),
            },
        ]

        response = await call_llm(
            pool, messages, model=config.optimizer_model, temperature=0.3, max_tokens=2048
        )
        return response
