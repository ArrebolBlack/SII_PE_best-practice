"""
5 阶段总调度器。

串联任务解析 → 相关调研 → 管线搭建 → 自主优化 → 报告生成。
"""

import json
import logging
import os
from datetime import datetime

from sii_pe.config import Config
from sii_pe.infra.client_pool import ClientPool
from sii_pe.infra.evaluator import Evaluator
from sii_pe.workflow.task_parser import TaskParser
from sii_pe.workflow.researcher import Researcher
from sii_pe.workflow.pipeline_setup import PipelineSetup
from sii_pe.workflow.auto_optimize import AutoOptimize
from sii_pe.workflow.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    5 阶段管线的总调度器。

    可由 Claude Code / AI agent 驱动，也可通过 CLI 独立运行。
    """

    async def run(
        self,
        instruction_text: str,
        val_data: list[dict],
        config: Config,
    ) -> dict:
        """
        运行完整的 5 阶段管线。

        参数:
            instruction_text: 考试说明文本
            val_data: 验证集数据
            config: 配置

        返回:
            {"best_prompt": PromptCandidate, "report": str, "best_score": float}
        """
        # 评测 pool：用于运行 prompt 得到评测结果
        eval_pool = ClientPool(config.api_keys, config.api_base_url)
        # 优化器 pool：用于生成/改进 prompt（独立的 provider，如 Claude/GPT-4o）
        opt_pool = ClientPool(
            config.get_optimizer_api_keys(),
            config.get_optimizer_api_base_url(),
        )

        # 准备输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(config.result_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # 阶段 1: 任务解析
        logger.info("=" * 60)
        logger.info("阶段 1/5: 任务解析")
        logger.info("=" * 60)
        task_spec = await TaskParser().parse(instruction_text, opt_pool, config)

        # 阶段 2: 相关调研
        logger.info("=" * 60)
        logger.info("阶段 2/5: 相关调研")
        logger.info("=" * 60)
        research = await Researcher().research(task_spec, opt_pool, config)

        # 阶段 3: 管线搭建
        logger.info("=" * 60)
        logger.info("阶段 3/5: 管线搭建与验证")
        logger.info("=" * 60)
        task, initial_candidate, baseline_result = await PipelineSetup().setup(
            task_spec, val_data, opt_pool, config
        )

        # 阶段 4: 自主优化
        logger.info("=" * 60)
        logger.info("阶段 4/5: 自主优化")
        logger.info("=" * 60)
        evaluator = Evaluator(eval_pool, task, config)
        best_candidate, population = await AutoOptimize().run(
            evaluator, val_data, initial_candidate, task_spec, config
        )

        # 阶段 5: 报告生成
        logger.info("=" * 60)
        logger.info("阶段 5/5: 报告生成")
        logger.info("=" * 60)
        report = await ReportGenerator().generate(
            task_spec, research, population,
            evaluator.experiment_log if hasattr(evaluator, 'experiment_log') else None,
            opt_pool, config
        )

        # 保存报告
        report_path = os.path.join(run_dir, "exploration_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"探索报告已保存: {report_path}")

        # 保存最佳 prompt
        best_prompt_path = os.path.join(run_dir, "best_prompt.json")
        with open(best_prompt_path, "w", encoding="utf-8") as f:
            json.dump(best_candidate.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"最佳 prompt 已保存: {best_prompt_path}")

        _, best_score = population.best
        logger.info(f"\n管线完成! 最佳分数: {best_score:.4f}")
        logger.info(f"结果目录: {run_dir}")

        return {
            "best_prompt": best_candidate,
            "report": report,
            "best_score": best_score,
            "run_dir": run_dir,
        }
