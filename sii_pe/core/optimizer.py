"""
顶层优化器：编排 strategy + evaluator + population 的完整优化循环。

支持混合策略、早停和检查点保存。
"""

import logging
import os

from sii_pe.core.prompt_candidate import PromptCandidate
from sii_pe.core.population import Population
from sii_pe.core.strategies.base import BaseStrategy
from sii_pe.infra.evaluator import Evaluator
from sii_pe.infra.persistence import EvalResult, ExperimentLog

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    顶层优化器。

    编排策略、评测器和种群，运行完整的 prompt 优化循环。
    支持：
    - 多策略切换（如先 grid search 再进化）
    - 早停（连续 N 轮无改进）
    - 检查点保存
    """

    def __init__(
        self,
        evaluator: Evaluator,
        strategy: BaseStrategy,
        population: Population,
        config,
    ):
        self.evaluator = evaluator
        self.strategy = strategy
        self.population = population
        self.config = config
        self.experiment_log = ExperimentLog()

    async def optimize(
        self,
        val_data: list[dict],
        max_iterations: int | None = None,
        early_stop_patience: int = 5,
        checkpoint_interval: int = 1,
        sample_limit: int | None = None,
    ) -> tuple[PromptCandidate, EvalResult]:
        """
        运行优化循环。

        参数:
            val_data: 验证集数据
            max_iterations: 最大迭代次数
            early_stop_patience: 连续无改进的容忍轮数
            checkpoint_interval: 每 N 轮保存检查点
            sample_limit: 评测时只用部分数据（加速粗筛）

        返回:
            (最佳候选, 最佳评测结果)
        """
        max_iter = max_iterations or self.config.max_iterations
        best_score = -1.0
        best_result = None
        patience_counter = 0

        logger.info(
            f"开始优化: 策略={self.strategy.name}, "
            f"最大迭代={max_iter}, 早停耐心={early_stop_patience}"
        )

        for iteration in range(1, max_iter + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"迭代 {iteration}/{max_iter}")
            logger.info(f"{'='*50}")

            # 1. 生成新候选
            new_candidates = await self.strategy.generate_candidates(
                self.population, self.evaluator.pool, self.config
            )

            if not new_candidates:
                logger.warning(f"迭代 {iteration}: 未生成任何候选，跳过")
                continue

            # 2. 评估所有新候选
            for candidate in new_candidates:
                result, _ = await self.evaluator.evaluate_prompt(
                    val_data, candidate, sample_limit=sample_limit
                )
                self.population.add(candidate, result.overall_score)
                self.experiment_log.add(candidate.to_dict(), result)

                logger.info(
                    f"候选 '{candidate.name}': {result.overall_score:.4f} "
                    f"(trial scores: {[f'{s:.4f}' for s in result.trial_scores]})"
                )

            # 3. 检查是否改进
            current_best = self.population.best
            if current_best and current_best[1] > best_score:
                best_score = current_best[1]
                best_result = result
                patience_counter = 0
                logger.info(f"新最佳分数: {best_score:.4f} ({current_best[0].name})")
            else:
                patience_counter += 1
                logger.info(
                    f"未改进 ({patience_counter}/{early_stop_patience}), "
                    f"当前最佳: {best_score:.4f}"
                )

            # 4. 早停
            if patience_counter >= early_stop_patience:
                logger.info(f"早停: 连续 {early_stop_patience} 轮无改进")
                break

            # 5. 检查点
            if iteration % checkpoint_interval == 0:
                ckpt_dir = os.path.join(self.config.result_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.population.save(os.path.join(ckpt_dir, f"population_iter{iteration}.json"))
                self.experiment_log.save(os.path.join(ckpt_dir, f"experiment_log_iter{iteration}.json"))

        # 最终保存
        self.population.save(os.path.join(self.config.result_dir, "population_final.json"))
        self.experiment_log.save(os.path.join(self.config.result_dir, "experiment_log.json"))

        best_candidate, best_score = self.population.best
        logger.info(f"\n优化完成! 最佳候选: {best_candidate.name} (score={best_score:.4f})")

        return best_candidate, best_result

    def switch_strategy(self, new_strategy: BaseStrategy) -> None:
        """切换优化策略（用于混合策略）。"""
        logger.info(f"策略切换: {self.strategy.name} → {new_strategy.name}")
        self.strategy = new_strategy
