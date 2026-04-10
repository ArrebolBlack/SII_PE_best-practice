"""优化策略抽象基类。"""

from abc import ABC, abstractmethod

from sii_pe.core.prompt_candidate import PromptCandidate
from sii_pe.core.population import Population
from sii_pe.infra.client_pool import ClientPool


class BaseStrategy(ABC):
    """
    优化策略的抽象基类。

    每种策略负责根据当前种群状态生成新的 prompt 候选。
    """

    @abstractmethod
    async def generate_candidates(
        self,
        population: Population,
        pool: ClientPool,
        config,
    ) -> list[PromptCandidate]:
        """
        生成一批新的候选 prompt。

        参数:
            population: 当前种群
            pool: 客户端池（用于调用 LLM 生成新 prompt）
            config: 配置

        返回:
            新的 PromptCandidate 列表
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称。"""
