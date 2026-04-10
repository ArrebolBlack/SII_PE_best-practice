"""
CLI 入口：sii-pe evaluate / optimize / pipeline
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime

from sii_pe.config import Config


def setup_logging(log_dir: str = "logs"):
    """初始化日志系统。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"sii_pe_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


async def cmd_evaluate(args):
    """评估一个 prompt。"""
    import jsonlines
    from sii_pe.infra.client_pool import ClientPool
    from sii_pe.infra.evaluator import Evaluator
    from sii_pe.core.prompt_candidate import PromptCandidate
    from sii_pe.tasks.movie_reranking import MovieRerankingTask
    from sii_pe.tasks.arc_puzzle import ARCPuzzleTask

    config = Config.load(yaml_path=args.config)

    # 加载数据
    data_path = args.data or config.val_data_path
    with open(data_path, "r", encoding="utf-8") as f:
        val_data = [json.loads(line.strip()) for line in f if line.strip()]

    # 选择任务
    task_map = {"movie": MovieRerankingTask, "arc": ARCPuzzleTask}
    task = task_map[args.task]()

    # 加载 prompt
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    candidate = PromptCandidate.from_dict(prompt_data)

    # 评估
    pool = ClientPool(config.api_keys, config.api_base_url)
    evaluator = Evaluator(pool, task, config)
    result, sample_scores = await evaluator.evaluate_prompt(val_data, candidate)

    # 保存结果
    os.makedirs(config.result_dir, exist_ok=True)
    result.save_json(os.path.join(config.result_dir, "eval_result.json"))
    result.save_csv(os.path.join(config.result_dir, "eval_samples.csv"), sample_scores)

    print(f"\n评估完成! 总分: {result.overall_score:.4f}")
    print(f"Trial scores: {[f'{s:.4f}' for s in result.trial_scores]}")


async def cmd_optimize(args):
    """运行优化循环。"""
    import jsonlines
    from sii_pe.infra.client_pool import ClientPool
    from sii_pe.infra.evaluator import Evaluator
    from sii_pe.core.optimizer import PromptOptimizer
    from sii_pe.core.population import Population
    from sii_pe.core.prompt_candidate import PromptCandidate
    from sii_pe.core.strategies.ape import APETrajectoryStrategy
    from sii_pe.core.strategies.evolutionary import EvolutionaryStrategy
    from sii_pe.tasks.movie_reranking import MovieRerankingTask
    from sii_pe.tasks.arc_puzzle import ARCPuzzleTask

    config = Config.load(yaml_path=args.config)

    # 加载数据
    data_path = args.data or config.val_data_path
    with open(data_path, "r", encoding="utf-8") as f:
        val_data = [json.loads(line.strip()) for line in f if line.strip()]

    # 选择任务和策略
    task_map = {"movie": MovieRerankingTask, "arc": ARCPuzzleTask}
    task = task_map[args.task]()

    strategy_map = {
        "ape": lambda: APETrajectoryStrategy(
            task_description=f"{args.task} task",
            metric_name="score",
        ),
        "evolutionary": lambda: EvolutionaryStrategy(
            task_description=f"{args.task} task",
            metric_name="score",
        ),
    }
    strategy = strategy_map[args.strategy]()

    # 加载初始 prompt
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    initial = PromptCandidate.from_dict(prompt_data)

    # 优化
    pool = ClientPool(config.api_keys, config.api_base_url)
    evaluator = Evaluator(pool, task, config)
    population = Population(max_size=config.population_size)

    # 评估初始候选
    initial_result, _ = await evaluator.evaluate_prompt(val_data, initial)
    population.add(initial, initial_result.overall_score)

    optimizer = PromptOptimizer(evaluator, strategy, population, config)
    best_candidate, best_result = await optimizer.optimize(val_data)

    print(f"\n优化完成! 最佳: {best_candidate.name} (score={population.best[1]:.4f})")


async def cmd_pipeline(args):
    """运行完整 5 阶段管线。"""
    from sii_pe.workflow.orchestrator import PipelineOrchestrator

    config = Config.load(yaml_path=args.config)

    # 读取考试说明
    with open(args.instruction, "r", encoding="utf-8") as f:
        instruction_text = f.read()

    # 加载验证数据
    data_path = args.data or config.val_data_path
    with open(data_path, "r", encoding="utf-8") as f:
        val_data = [json.loads(line.strip()) for line in f if line.strip()]

    # 运行管线
    result = await PipelineOrchestrator().run(instruction_text, val_data, config)

    print(f"\n管线完成! 最佳分数: {result['best_score']:.4f}")
    print(f"结果目录: {result['run_dir']}")


def main():
    parser = argparse.ArgumentParser(
        description="SII PE Best Practice - 自动化 Prompt Engineering 优化框架"
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="评估一个 prompt")
    eval_parser.add_argument("--task", choices=["movie", "arc"], required=True, help="任务类型")
    eval_parser.add_argument("--prompt", required=True, help="PromptCandidate JSON 文件路径")
    eval_parser.add_argument("--data", help="验证数据路径（覆盖配置）")
    eval_parser.add_argument("--config", help="配置文件路径")

    # optimize
    opt_parser = subparsers.add_parser("optimize", help="运行优化循环")
    opt_parser.add_argument("--task", choices=["movie", "arc"], required=True, help="任务类型")
    opt_parser.add_argument("--strategy", choices=["ape", "evolutionary"], default="ape", help="优化策略")
    opt_parser.add_argument("--prompt", required=True, help="初始 PromptCandidate JSON")
    opt_parser.add_argument("--data", help="验证数据路径")
    opt_parser.add_argument("--config", help="配置文件路径")

    # pipeline
    pipe_parser = subparsers.add_parser("pipeline", help="运行完整 5 阶段管线")
    pipe_parser.add_argument("--instruction", required=True, help="考试说明文件路径")
    pipe_parser.add_argument("--data", required=True, help="验证数据路径")
    pipe_parser.add_argument("--config", help="配置文件路径")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging()

    cmd_map = {
        "evaluate": cmd_evaluate,
        "optimize": cmd_optimize,
        "pipeline": cmd_pipeline,
    }
    asyncio.run(cmd_map[args.command](args))


if __name__ == "__main__":
    main()
