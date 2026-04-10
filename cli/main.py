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


# ======================== Agent 命令 ========================


async def cmd_agent_init(args):
    """初始化 Agent 优化会话。"""
    from sii_pe.agent.session import Session

    session = Session(work_dir=args.work_dir)
    session.init(
        task=args.task,
        data_path=args.data,
        config_path=args.config,
    )
    print(f"会话已初始化: task={args.task}, data={args.data}")
    print(f"会话文件: {session.path}")
    print(f"\n下一步: 修改 Answer.py，然后运行 sii-pe agent evaluate --note '描述你的改动'")


async def cmd_agent_evaluate(args):
    """评测当前 Answer 文件。"""
    from sii_pe.agent.session import Session
    from sii_pe.agent.evaluate import evaluate_answer

    session = Session(work_dir=args.work_dir)
    if not session.is_initialized:
        print("错误: 会话未初始化。请先运行 sii-pe agent init")
        sys.exit(1)

    answer_path = args.answer or "Answer.py"
    if not os.path.exists(answer_path):
        print(f"错误: Answer 文件不存在: {answer_path}")
        sys.exit(1)

    print(f"评测 {answer_path} ...")
    summary = await evaluate_answer(
        session=session,
        answer_path=answer_path,
        note=args.note or "",
        num_trials=int(args.trials) if args.trials else None,
        sample_limit=int(args.samples) if args.samples else None,
    )

    print(f"\nRound {summary['round']}: {summary['score']:.4f} "
          f"({summary['arrow']}{abs(summary['delta']):.4f})")
    if summary["is_new_best"]:
        print(f"★ NEW BEST: {summary['best_score']:.4f}")
    else:
        print(f"Best: {summary['best_score']:.4f} ({os.path.basename(summary['best_file'])})")


async def cmd_agent_status(args):
    """查看当前会话状态。"""
    from sii_pe.agent.session import Session

    session = Session(work_dir=args.work_dir)
    if not session.is_initialized:
        print("会话未初始化。")
        return

    status = session.get_status()
    print(f"任务: {status['task']}")
    print(f"总轮数: {status['total_rounds']}")
    print(f"最佳分数: {status['best_score']:.4f}")
    if status["best_file"]:
        print(f"最佳文件: {os.path.basename(status['best_file'])}")
    if status["last_round"]:
        r = status["last_round"]
        print(f"\n最近一轮: Round {r['round']}, score={r['score']:.4f}, note={r.get('note', '')}")


async def cmd_agent_history(args):
    """查看完整评测历史。"""
    from sii_pe.agent.session import Session

    session = Session(work_dir=args.work_dir)
    print(session.get_trajectory_text())


async def cmd_agent_report(args):
    """从当前会话生成探索报告。"""
    from sii_pe.agent.session import Session

    session = Session(work_dir=args.work_dir)
    if not session.data.get("rounds"):
        print("暂无评测记录，无法生成报告。")
        return

    history = session.get_history()
    status = session.get_status()

    report_lines = [
        "# Agent 优化探索报告\n",
        f"任务: {status['task']}\n",
        f"总轮数: {status['total_rounds']}\n",
        f"最佳分数: {status['best_score']:.4f}\n",
        f"最佳文件: {os.path.basename(status['best_file']) if status['best_file'] else 'N/A'}\n",
        "\n## 评测历史\n",
        "| Round | 分数 | 文件 | 备注 |",
        "|-------|------|------|------|",
    ]
    for r in history:
        is_best = " ★" if r["file"] == status["best_file"] else ""
        report_lines.append(
            f"| {r['round']} | {r['score']:.4f}{is_best} | "
            f"{os.path.basename(r['file'])} | {r.get('note', '')} |"
        )

    report = "\n".join(report_lines)

    report_path = os.path.join(args.work_dir, "agent_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存: {report_path}")
    print(report)


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

    # agent 子命令
    agent_parser = subparsers.add_parser("agent", help="Agent 接管模式")
    agent_sub = agent_parser.add_subparsers(dest="agent_command", help="Agent 子命令")
    agent_parser.add_argument("--work-dir", default=".", help="工作目录（默认当前目录）")

    # agent init
    agent_init = agent_sub.add_parser("init", help="初始化优化会话")
    agent_init.add_argument("--task", choices=["movie", "arc"], required=True, help="任务类型")
    agent_init.add_argument("--data", required=True, help="验证数据路径")
    agent_init.add_argument("--config", help="配置文件路径")

    # agent evaluate
    agent_eval = agent_sub.add_parser("evaluate", help="评测 Answer 文件")
    agent_eval.add_argument("--answer", default="Answer.py", help="Answer 文件路径（默认 Answer.py）")
    agent_eval.add_argument("--note", default="", help="本轮备注")
    agent_eval.add_argument("--trials", help="评测次数（覆盖配置）")
    agent_eval.add_argument("--samples", help="评测样本数限制")

    # agent status
    agent_sub.add_parser("status", help="查看当前会话状态")

    # agent history
    agent_sub.add_parser("history", help="查看完整评测历史")

    # agent report
    agent_sub.add_parser("report", help="生成探索报告")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging()

    if args.command == "agent":
        if not args.agent_command:
            agent_parser.print_help()
            sys.exit(1)
        agent_cmd_map = {
            "init": cmd_agent_init,
            "evaluate": cmd_agent_evaluate,
            "status": cmd_agent_status,
            "history": cmd_agent_history,
            "report": cmd_agent_report,
        }
        # 传递 work_dir 给 agent 子命令
        asyncio.run(agent_cmd_map[args.agent_command](args))
    else:
        cmd_map = {
            "evaluate": cmd_evaluate,
            "optimize": cmd_optimize,
            "pipeline": cmd_pipeline,
        }
        asyncio.run(cmd_map[args.command](args))


if __name__ == "__main__":
    main()
