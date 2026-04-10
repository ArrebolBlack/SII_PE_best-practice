"""
网格搜索策略：系统性地探索 prompt 维度的笛卡尔积。

用户定义 prompt 的各个维度（角色、风格、格式等）及其可选值，
策略生成所有组合作为候选 prompt。
"""

import itertools
import logging

from sii_pe.core.prompt_candidate import PromptCandidate
from sii_pe.core.population import Population
from sii_pe.core.strategies.base import BaseStrategy
from sii_pe.infra.client_pool import ClientPool

logger = logging.getLogger(__name__)


class GridSearchStrategy(BaseStrategy):
    """
    网格搜索策略。

    示例 dimensions:
        {
            "role": {
                "expert": "你是一名专业的推荐系统专家",
                "analyst": "你是一名数据驱动的分析师",
            },
            "style": {
                "instruction": "请按以下步骤完成任务：",
                "question": "你认为用户最可能观看哪些电影？",
            },
        }
    """

    def __init__(
        self,
        dimensions: dict[str, dict[str, str]],
        system_prompt_template: str,
        user_prompt_template: str,
    ):
        """
        参数:
            dimensions: {维度名: {选项名: 选项值}} 字典
            system_prompt_template: 系统提示词 Jinja2 模板，可引用维度名
            user_prompt_template: 用户提示词 Jinja2 模板
        """
        self.dimensions = dimensions
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template

    @property
    def name(self) -> str:
        return "grid_search"

    async def generate_candidates(
        self, population: Population, pool: ClientPool, config
    ) -> list[PromptCandidate]:
        """生成所有维度组合的 PromptCandidate。"""
        dim_names = list(self.dimensions.keys())
        dim_options = [list(self.dimensions[d].keys()) for d in dim_names]

        candidates = []
        for combo in itertools.product(*dim_options):
            # 构建当前组合的变量映射
            combo_dict = dict(zip(dim_names, combo))
            combo_values = {
                name: self.dimensions[name][option]
                for name, option in combo_dict.items()
            }

            # 组合名称
            combo_name = "_".join(combo)

            # 渲染系统提示词
            from jinja2 import Template
            system_prompt = Template(self.system_prompt_template).render(**combo_values)

            candidate = PromptCandidate(
                name=f"grid_{combo_name}",
                system_prompt=system_prompt,
                user_prompt_template=self.user_prompt_template,
                generation=0,
                metadata={"strategy": "grid_search", "dimensions": combo_dict},
            )
            candidates.append(candidate)

        logger.info(f"网格搜索生成 {len(candidates)} 个候选")
        return candidates
