"""
进化优化策略（AlphaEvolve + EvoPrompt 融合）。

使用 LLM 作为 mutation/crossover 算子，对高分 prompt 进行变异和交叉，
生成新一代候选。
"""

import json
import logging

from sii_pe.core.prompt_candidate import PromptCandidate
from sii_pe.core.population import Population
from sii_pe.core.strategies.base import BaseStrategy
from sii_pe.infra.client_pool import ClientPool
from sii_pe.infra.llm_caller import call_llm

logger = logging.getLogger(__name__)

# 可用的变异类型
MUTATION_TYPES = {
    "rephrase": "保留核心思想，用完全不同的措辞和句式重写",
    "simplify": "精简内容，去掉冗余，保留最有效的指令",
    "elaborate": "扩展细节，添加更具体的指导和约束",
    "restructure": "重新组织信息的呈现顺序和结构",
    "combine": "融合两个高分 prompt 的优点，取长补短",
}


class EvolutionaryStrategy(BaseStrategy):
    """
    进化优化策略。

    核心流程：
    1. 从种群中选择亲本（锦标赛选择）
    2. 用 LLM 作为变异算子生成 K 个变体
    3. 变体被评估后加入种群，自然选择
    """

    def __init__(
        self,
        task_description: str,
        metric_name: str,
        num_variants: int = 5,
        mutation_types: list[str] | None = None,
    ):
        self.task_description = task_description
        self.metric_name = metric_name
        self.num_variants = num_variants
        self.mutation_types = mutation_types or list(MUTATION_TYPES.keys())

    @property
    def name(self) -> str:
        return "evolutionary"

    async def generate_candidates(
        self, population: Population, pool: ClientPool, config
    ) -> list[PromptCandidate]:
        """通过变异/交叉生成新一代候选。"""
        if population.size == 0:
            logger.warning("种群为空，无法进化，返回空列表")
            return []

        # 选择亲本
        parents = population.get_diverse_sample(k=min(3, population.size))
        generation = max(p.generation for p, _ in parents) + 1

        variants = []
        # 对每种变异类型生成一个变体
        for i, mutation_type in enumerate(self.mutation_types[: self.num_variants]):
            try:
                messages = self._build_mutation_prompt(parents, mutation_type)
                response = await call_llm(
                    pool,
                    messages,
                    model=config.optimizer_model,
                    temperature=0.8,
                    max_tokens=2048,
                )
                variant = self._parse_variant(response, mutation_type, generation, parents)
                variants.append(variant)
                logger.info(f"进化变异 [{mutation_type}] 成功: {variant.name}")
            except Exception as e:
                logger.warning(f"进化变异 [{mutation_type}] 失败: {e}")

        logger.info(f"进化策略生成 {len(variants)} 个变体 (目标 {self.num_variants})")
        return variants

    def _build_mutation_prompt(
        self,
        parents: list[tuple[PromptCandidate, float]],
        mutation_type: str,
    ) -> list[dict]:
        """构建变异 prompt。"""
        mutation_desc = MUTATION_TYPES.get(mutation_type, "改进")

        parents_text = "\n".join(
            f"--- Prompt #{i+1} (得分: {score:.4f}) ---\n"
            f"系统提示词: {p.system_prompt}\n"
            f"用户提示词模板: {p.user_prompt_template}\n"
            for i, (p, score) in enumerate(parents)
        )

        system_prompt = (
            "你是一名提示词进化专家。你的任务是基于现有的高分 prompt，"
            "通过特定的变异操作生成一个新的改进版本。"
        )

        user_prompt = (
            f"任务描述：\n{self.task_description}\n\n"
            f"评估指标：{self.metric_name}\n\n"
            f"以下是当前表现最好的几个 prompt：\n{parents_text}\n\n"
            f"变异类型：{mutation_type} — {mutation_desc}\n\n"
            f"请基于上述高分 prompt，执行 [{mutation_type}] 变异，生成一个新的 prompt。\n"
            f"用户提示词模板中可以使用 Jinja2 占位符。\n\n"
            f"输出格式为 JSON：\n"
            f'{{\n'
            f'  "name": "变异后的策略名称",\n'
            f'  "system_prompt": "新的系统提示词",\n'
            f'  "user_prompt_template": "新的用户提示词模板",\n'
            f'  "reasoning": "变异思路说明"\n'
            f'}}'
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_variant(
        self,
        text: str,
        mutation_type: str,
        generation: int,
        parents: list[tuple[PromptCandidate, float]],
    ) -> PromptCandidate:
        """解析变异输出。"""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("LLM 输出中未找到 JSON")

        data = json.loads(text[start:end])

        return PromptCandidate(
            name=data.get("name", f"evo_{mutation_type}_gen{generation}"),
            system_prompt=data["system_prompt"],
            user_prompt_template=data["user_prompt_template"],
            generation=generation,
            parent_names=[p.name for p, _ in parents],
            metadata={
                "strategy": "evolutionary",
                "mutation_type": mutation_type,
                "reasoning": data.get("reasoning", ""),
            },
        )
