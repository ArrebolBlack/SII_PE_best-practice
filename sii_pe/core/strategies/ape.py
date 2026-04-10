"""
APE 轨迹优化策略。

将历史 prompt + score 作为"优化轨迹"输入给 LLM，
让 LLM 分析优缺点并生成改进版 prompt。

复用自 Summer Prompt_APE.py 的核心逻辑，泛化为任务无关版本。
"""

import json
import logging

from sii_pe.core.prompt_candidate import PromptCandidate
from sii_pe.core.population import Population
from sii_pe.core.strategies.base import BaseStrategy
from sii_pe.infra.client_pool import ClientPool
from sii_pe.infra.llm_caller import call_llm

logger = logging.getLogger(__name__)


class APETrajectoryStrategy(BaseStrategy):
    """
    APE 风格轨迹优化。

    核心思路：让 LLM 看到之前所有尝试过的 prompt 及其得分，
    分析趋势，生成更好的 prompt。
    """

    def __init__(self, task_description: str, metric_name: str):
        """
        参数:
            task_description: 任务描述（用于构建优化器 prompt）
            metric_name: 评测指标名称（如 "NDCG@10", "Exact Match Accuracy"）
        """
        self.task_description = task_description
        self.metric_name = metric_name

    @property
    def name(self) -> str:
        return "ape_trajectory"

    async def generate_candidates(
        self, population: Population, pool: ClientPool, config
    ) -> list[PromptCandidate]:
        """根据优化轨迹生成改进的 prompt。"""
        trajectory = population.get_trajectory()
        generation = max((e.get("candidate", {}).get("generation", 0)
                         for e in population._entries), default=0) + 1

        messages = self._build_optimizer_prompt(trajectory, generation)

        try:
            response = await call_llm(
                pool,
                messages,
                model=config.optimizer_model,
                temperature=0.7,
                max_tokens=2048,
            )
            candidate = self._parse_optimizer_output(response, generation)
            logger.info(f"APE 生成新候选: {candidate.name}")
            return [candidate]
        except Exception as e:
            logger.error(f"APE 优化失败: {e}，使用 fallback")
            return [self._fallback_candidate(generation)]

    def _build_optimizer_prompt(self, trajectory: list[dict], generation: int) -> list[dict]:
        """构建优化器 prompt。"""
        system_prompt = (
            "你是一名提示词工程专家。你的任务是分析历史 prompt 的表现，"
            "找出影响分数的关键因素，并设计一个改进版的 prompt。"
        )

        # 构建轨迹文本
        if trajectory:
            history_text = "\n".join(
                f"迭代 {i+1}: 策略名称: {t['name']}, {self.metric_name}: {t['score']:.4f}\n"
                f"系统提示词: {t['system_prompt']}\n"
                f"用户提示词模板: {t['user_prompt_template']}\n"
                for i, t in enumerate(trajectory)
            )
        else:
            history_text = "暂无历史优化记录。\n"

        user_prompt = (
            f"当前迭代轮次：{generation}\n\n"
            f"任务描述：\n{self.task_description}\n\n"
            f"评估指标：{self.metric_name}，分数范围 [0,1]，越高越好。\n\n"
            f"历史优化轨迹：\n{history_text}\n\n"
            f"请分析历史策略的优缺点，提出改进建议，并设计一个新的提示词模板。\n"
            f"用户提示词模板中可以使用 Jinja2 占位符（如 {{{{ history }}}}、{{{{ candidates }}}} 等）。\n\n"
            f"输出格式为 JSON：\n"
            f'{{\n'
            f'  "name": "策略名称",\n'
            f'  "system_prompt": "完整的系统提示词",\n'
            f'  "user_prompt_template": "用户提示词模板",\n'
            f'  "analysis": "历史策略分析",\n'
            f'  "improvement": "改进思路"\n'
            f'}}'
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_optimizer_output(self, text: str, generation: int) -> PromptCandidate:
        """解析 LLM 输出的 JSON。"""
        # 提取 JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("LLM 输出中未找到 JSON")

        data = json.loads(text[start:end])

        return PromptCandidate(
            name=data.get("name", f"ape_gen{generation}"),
            system_prompt=data["system_prompt"],
            user_prompt_template=data["user_prompt_template"],
            generation=generation,
            metadata={
                "strategy": "ape_trajectory",
                "analysis": data.get("analysis", ""),
                "improvement": data.get("improvement", ""),
            },
        )

    def _fallback_candidate(self, generation: int) -> PromptCandidate:
        """解析失败时的回退候选。"""
        return PromptCandidate(
            name=f"ape_fallback_gen{generation}",
            system_prompt="你是一名任务专家，请根据提供的信息完成任务。",
            user_prompt_template="{{ history }}\n\n{{ candidates }}\n\n请给出你的答案。",
            generation=generation,
            metadata={"strategy": "ape_trajectory", "fallback": True},
        )
