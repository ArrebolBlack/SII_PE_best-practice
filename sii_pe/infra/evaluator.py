"""
通用异步多试次评测器。

对每个样本运行 construct_prompt → LLM call → parse_output → metric 管线，
重复 num_trials 次，计算 mean/std。支持 sample_limit 快速粗筛。
"""

import asyncio
import logging
import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sii_pe.infra.client_pool import ClientPool
from sii_pe.infra.llm_caller import call_llm
from sii_pe.infra.persistence import EvalResult

logger = logging.getLogger(__name__)


class Evaluator:
    """
    通用异步评测器。

    通过 BaseTask 接口解耦任务逻辑，支持任意 PE 任务的评测。
    """

    def __init__(self, pool: ClientPool, task, config):
        """
        参数:
            pool: 客户端池
            task: BaseTask 实例（提供 construct_prompt, parse_output, compute_metric）
            config: Config 实例
        """
        self.pool = pool
        self.task = task
        self.config = config

    async def evaluate_prompt(
        self,
        val_data: list[dict],
        prompt_candidate,
        num_trials: int | None = None,
        max_concurrency: int | None = None,
        sample_limit: int | None = None,
    ) -> EvalResult:
        """
        评测一个 prompt 候选。

        参数:
            val_data: 验证集数据
            prompt_candidate: PromptCandidate 实例
            num_trials: 评测次数（覆盖 config 默认值）
            max_concurrency: 最大并发数（覆盖 config 默认值）
            sample_limit: 只评测前 N 个样本（用于快速粗筛）

        返回:
            EvalResult
        """
        num_trials = num_trials or self.config.num_trials
        max_concurrency = max_concurrency or self.config.max_concurrency

        # 可选：只用部分数据做快速评估
        data = val_data[:sample_limit] if sample_limit else val_data
        num_samples = len(data)

        logger.info(
            f"开始评测 '{prompt_candidate.name}': "
            f"{num_samples} 样本 x {num_trials} 试次, 并发 {max_concurrency}"
        )

        semaphore = asyncio.Semaphore(max_concurrency)
        sample_scores: dict[int, list[float]] = {i: [] for i in range(num_samples)}

        for trial in range(num_trials):
            logger.info(f"Trial {trial + 1}/{num_trials}")

            async def worker(idx: int, sample: dict) -> tuple[int, float]:
                async with semaphore:
                    try:
                        # 隐藏答案
                        visible = self.task.mask_sample(sample)
                        # 构建 prompt
                        messages = self.task.construct_prompt(visible, prompt_candidate)
                        # 调用 LLM
                        output_text = await call_llm(
                            self.pool,
                            messages,
                            model=self.config.model,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                        )
                        # 解析输出
                        prediction = self.task.parse_output(output_text)
                        if prediction is None:
                            logger.warning(f"样本 {idx} 输出解析失败")
                            return idx, 0.0
                        # 计算指标
                        ground_truth = self.task.extract_ground_truth(sample)
                        score = self.task.compute_metric(prediction, ground_truth)
                        return idx, score
                    except Exception as e:
                        logger.error(f"样本 {idx} (Trial {trial + 1}) 出错: {e}")
                        return idx, 0.0

            tasks = [worker(idx, sample) for idx, sample in enumerate(data)]
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=num_samples,
                desc=f"Trial {trial + 1}",
            ):
                idx, score = await coro
                sample_scores[idx].append(score)

            # 本次 trial 平均分
            trial_scores_list = [sample_scores[i][-1] for i in range(num_samples)]
            trial_avg = sum(trial_scores_list) / len(trial_scores_list) if trial_scores_list else 0.0
            logger.info(f"Trial {trial + 1} 平均分: {trial_avg:.4f}")

        # 计算统计
        trial_scores = []
        for t in range(num_trials):
            scores = [sample_scores[i][t] for i in range(num_samples)]
            trial_scores.append(sum(scores) / len(scores) if scores else 0.0)

        sample_stats = {}
        for idx in range(num_samples):
            arr = np.array(sample_scores[idx], dtype=float)
            sample_stats[idx] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

        overall_score = sum(trial_scores) / len(trial_scores) if trial_scores else 0.0

        result = EvalResult(
            overall_score=overall_score,
            trial_scores=trial_scores,
            sample_stats=sample_stats,
            num_trials=num_trials,
            num_samples=num_samples,
            metadata={
                "prompt_name": prompt_candidate.name,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "timestamp": datetime.now().isoformat(),
            },
        )

        logger.info(f"评测完成 '{prompt_candidate.name}': 总分 {overall_score:.4f}")
        return result, sample_scores
