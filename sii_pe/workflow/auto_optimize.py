"""
阶段 4：自主优化。

封装 Layer 2 的 PromptOptimizer，提供自治优化循环。
先 grid search 快速筛选，再切换为 APE/进化精炼。
"""

import logging

from sii_pe.core.optimizer import PromptOptimizer
from sii_pe.core.population import Population
from sii_pe.core.prompt_candidate import PromptCandidate
from sii_pe.core.strategies.ape import APETrajectoryStrategy
from sii_pe.core.strategies.evolutionary import EvolutionaryStrategy
from sii_pe.infra.evaluator import Evaluator
from sii_pe.infra.persistence import EvalResult
from sii_pe.workflow.task_parser import TaskSpec

logger = logging.getLogger(__name__)


class AutoOptimize:
    """
    自主优化：封装 PromptOptimizer，管理优化生命周期。
    """

    async def run(
        self,
        evaluator: Evaluator,
        val_data: list[dict],
        initial_candidate: PromptCandidate,
        task_spec: TaskSpec,
        config,
    ) -> tuple[PromptCandidate, Population]:
        """
        运行自主优化循环。

        步骤:
        1. 初始化种群（种子 = initial_candidate）
        2. 用 APE 轨迹优化进行主循环
        3. 切换到进化策略进一步精炼
        4. 返回最佳候选和完整种群

        参数:
            evaluator: 评测器
            val_data: 验证集数据
            initial_candidate: 初始 prompt 候选
            task_spec: 任务规格
            config: 配置

        返回:
            (最佳候选, 种群)
        """
        population = Population(max_size=config.population_size)

        # 评估初始候选并加入种群
        logger.info("评估初始候选...")
        initial_result, _ = await evaluator.evaluate_prompt(val_data, initial_candidate)
        population.add(initial_candidate, initial_result.overall_score)

        # 阶段 1: APE 轨迹优化（前 2/3 迭代）
        ape_iterations = max(1, config.max_iterations * 2 // 3)
        ape_strategy = APETrajectoryStrategy(
            task_description=task_spec.description,
            metric_name=task_spec.metric_name,
        )

        logger.info(f"阶段 1: APE 轨迹优化 ({ape_iterations} 轮)")
        optimizer = PromptOptimizer(evaluator, ape_strategy, population, config)
        await optimizer.optimize(
            val_data,
            max_iterations=ape_iterations,
            early_stop_patience=max(3, ape_iterations // 3),
        )

        # 阶段 2: 进化策略精炼（剩余迭代）
        evo_iterations = config.max_iterations - ape_iterations
        if evo_iterations > 0 and population.size >= 2:
            evo_strategy = EvolutionaryStrategy(
                task_description=task_spec.description,
                metric_name=task_spec.metric_name,
                num_variants=min(5, evo_iterations),
            )
            logger.info(f"阶段 2: 进化策略精炼 ({evo_iterations} 轮)")
            optimizer.switch_strategy(evo_strategy)
            await optimizer.optimize(
                val_data,
                max_iterations=evo_iterations,
                early_stop_patience=max(2, evo_iterations // 2),
            )

        best_candidate, best_score = population.best
        logger.info(f"自主优化完成! 最佳: {best_candidate.name} (score={best_score:.4f})")

        return best_candidate, population
