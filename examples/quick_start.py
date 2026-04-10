"""
快速上手示例：使用 ARC 任务演示评估和优化流程。
"""

import asyncio
import json
import os

from sii_pe.config import Config
from sii_pe.infra.client_pool import ClientPool
from sii_pe.infra.evaluator import Evaluator
from sii_pe.core.prompt_candidate import PromptCandidate
from sii_pe.core.optimizer import PromptOptimizer
from sii_pe.core.population import Population
from sii_pe.core.strategies.ape import APETrajectoryStrategy
from sii_pe.tasks.arc_puzzle import ARCPuzzleTask


async def main():
    # 1. 加载配置
    config = Config.load()

    # 2. 加载数据
    with open(config.val_data_path, "r", encoding="utf-8") as f:
        val_data = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"加载了 {len(val_data)} 个样本")

    # 3. 初始化组件
    pool = ClientPool(config.api_keys, config.api_base_url)
    task = ARCPuzzleTask()
    evaluator = Evaluator(pool, task, config)

    # 4. 定义初始 prompt
    initial = PromptCandidate(
        name="initial",
        system_prompt="你是一个 ARC 任务专家。请根据训练样本推断变换规则并应用于测试输入。",
        user_prompt_template=(
            "训练样本：\n{{ train_examples }}\n\n"
            "测试输入：\n{{ test_input_rows }}\n\n"
            "请输出预测网格，格式为 <grid> [[...],[...]] </grid>"
        ),
    )

    # 5. 评估初始 prompt
    print("评估初始 prompt...")
    result, _ = await evaluator.evaluate_prompt(val_data, initial, num_trials=2, sample_limit=5)
    print(f"初始分数: {result.overall_score:.4f}")

    # 6. 运行优化
    print("\n开始优化...")
    population = Population(max_size=20)
    population.add(initial, result.overall_score)

    strategy = APETrajectoryStrategy(
        task_description="ARC 网格谜题：根据训练样本推断变换规则并应用于测试输入",
        metric_name="Exact Match Accuracy",
    )

    optimizer = PromptOptimizer(evaluator, strategy, population, config)
    best, best_result = await optimizer.optimize(
        val_data, max_iterations=3, early_stop_patience=2
    )

    print(f"\n优化完成! 最佳: {best.name} (score={population.best[1]:.4f})")
    print(f"系统提示词: {best.system_prompt[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
