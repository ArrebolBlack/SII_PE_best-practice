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


def _load_config(args):
    """加载配置，检查 --config 指定的文件是否存在。"""
    if args.config and not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    return Config.load(yaml_path=args.config)


def setup_logging(log_dir: str = "logs"):
    """初始化日志系统。详细日志写文件，终端只显示 WARNING 以上。"""
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
    # 终端只显示 WARNING 以上，避免和 tqdm 进度条混杂
    logging.getLogger().handlers[1].setLevel(logging.WARNING)


async def cmd_evaluate(args):
    """评估一个 prompt。"""
    from sii_pe.infra.client_pool import ClientPool
    from sii_pe.infra.evaluator import Evaluator
    from sii_pe.core.prompt_candidate import PromptCandidate
    from sii_pe.tasks.movie_reranking import MovieRerankingTask
    from sii_pe.tasks.arc_puzzle import ARCPuzzleTask

    config = _load_config(args)

    # 加载数据
    data_path = args.data or config.val_data_path
    if not data_path:
        print("错误: 未指定数据文件。请通过 --data 参数或 config.yaml 中 evaluation.val_data_path 指定。")
        sys.exit(1)
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        sys.exit(1)

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            val_data = [json.loads(line.strip()) for line in f if line.strip()]
    except json.JSONDecodeError as e:
        print(f"错误: 数据文件格式错误: {e}")
        sys.exit(1)

    # 选择任务
    task_map = {"movie": MovieRerankingTask, "arc": ARCPuzzleTask}
    task = task_map[args.task]()

    # 加载 prompt
    if not os.path.exists(args.prompt):
        print(f"错误: Prompt 文件不存在: {args.prompt}")
        sys.exit(1)
    try:
        with open(args.prompt, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: Prompt JSON 解析失败: {e}")
        sys.exit(1)
    try:
        candidate = PromptCandidate.from_dict(prompt_data)
    except KeyError as e:
        print(f"错误: Prompt 文件缺少必填字段: {e}")
        print("提示: JSON 需包含 name, system_prompt, user_prompt_template 字段")
        sys.exit(1)

    # 评估
    pool = ClientPool(config.api_keys, config.api_base_url)
    evaluator = Evaluator(pool, task, config)
    result, sample_scores = await evaluator.evaluate_prompt(val_data, candidate)

    # 保存结果（文件名含任务和时间戳，避免覆盖）
    os.makedirs(config.result_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_prefix = os.path.join(config.result_dir, f"eval_{args.task}_{ts}")
    result.save_json(f"{result_prefix}_result.json")
    result.save_csv(f"{result_prefix}_samples.csv", sample_scores)

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

    config = _load_config(args)

    # 加载数据
    data_path = args.data or config.val_data_path
    if not data_path:
        print("错误: 未指定数据文件。请通过 --data 参数或 config.yaml 中 evaluation.val_data_path 指定。")
        sys.exit(1)
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        sys.exit(1)
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
    if not os.path.exists(args.prompt):
        print(f"错误: Prompt 文件不存在: {args.prompt}")
        sys.exit(1)
    try:
        with open(args.prompt, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: Prompt JSON 解析失败: {e}")
        sys.exit(1)
    try:
        initial = PromptCandidate.from_dict(prompt_data)
    except KeyError as e:
        print(f"错误: Prompt 文件缺少必填字段: {e}")
        sys.exit(1)

    # 优化
    pool = ClientPool(config.api_keys, config.api_base_url)
    evaluator = Evaluator(pool, task, config)
    population = Population(max_size=config.population_size)

    start_time = datetime.now()

    # 评估初始候选
    initial_result, _ = await evaluator.evaluate_prompt(val_data, initial)
    population.add(initial, initial_result.overall_score)

    optimizer = PromptOptimizer(evaluator, strategy, population, config)
    best_candidate, best_result = await optimizer.optimize(val_data)

    elapsed = datetime.now() - start_time
    print(f"\n优化完成! 耗时 {elapsed}, 最佳: {best_candidate.name} (score={population.best[1]:.4f})")


async def cmd_pipeline(args):
    """运行完整 5 阶段管线。"""
    from sii_pe.workflow.orchestrator import PipelineOrchestrator

    config = _load_config(args)

    # 读取考试说明
    if not os.path.exists(args.instruction):
        print(f"错误: 考试说明文件不存在: {args.instruction}")
        sys.exit(1)
    with open(args.instruction, "r", encoding="utf-8") as f:
        instruction_text = f.read()

    # 加载验证数据
    data_path = args.data or config.val_data_path
    if not data_path:
        print("错误: 未指定数据文件。请通过 --data 参数或 config.yaml 中 evaluation.val_data_path 指定。")
        sys.exit(1)
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        sys.exit(1)
    with open(data_path, "r", encoding="utf-8") as f:
        val_data = [json.loads(line.strip()) for line in f if line.strip()]

    start_time = datetime.now()

    # 运行管线
    result = await PipelineOrchestrator().run(instruction_text, val_data, config)

    elapsed = datetime.now() - start_time
    print(f"\n管线完成! 耗时 {elapsed}, 最佳分数: {result['best_score']:.4f}")
    print(f"结果目录: {result['run_dir']}")


# ======================== Agent 命令 ========================

# Answer.py 模板
_ARC_TEMPLATE = '''"""
ARC 网格谜题任务 — Answer.py

需要定义两个函数:
  - construct_prompt(d: dict) -> list[dict]
      将数据样本转为 OpenAI Chat API messages 列表
      格式: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
  - parse_output(text: str) -> list[list[int]] | None
      解析 LLM 原始输出为二维网格
"""


def construct_prompt(d: dict) -> list[dict]:
    """根据 ARC 样本构造 prompt。"""
    # d 包含 "train"（训练样本列表）和 "test"（测试样本列表）
    # d["test"] 中没有 "output"（已隐藏），只有 "input"

    train_text = ""
    for i, ex in enumerate(d.get("train", [])):
        inp = ex["input"]
        out = ex["output"]
        train_text += f"训练样本 {i+1}:\\n输入: {inp}\\n输出: {out}\\n\\n"

    test_input = d["test"][0]["input"]

    return [
        {"role": "system", "content": "你是一个 ARC 任务专家。请仔细观察训练样本中的输入输出对，推断变换规则。"},
        {"role": "user", "content": f"{train_text}测试输入: {test_input}\\n\\n请输出预测网格，格式为 <grid> [[...],[...]] </grid>"},
    ]


def parse_output(text: str):
    """解析 LLM 输出为二维整数网格。"""
    import re
    import ast

    m = re.search(r"<\\s*grid\\s*>\\s*(.*?)\\s*<\\s*/\\s*grid\\s*>", text, re.IGNORECASE | re.DOTALL)
    if m:
        try:
            return ast.literal_eval(m.group(1))
        except Exception:
            pass
    return None
'''

_MOVIE_TEMPLATE = '''"""
电影推荐重排序任务 — Answer.py

需要定义两个函数:
  - construct_prompt(d: dict) -> list[dict]
      将数据样本转为 OpenAI Chat API messages 列表
      格式: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
  - parse_output(text: str) -> list[int] | None
      解析 LLM 原始输出为电影 ID 列表（按兴趣从高到低排序）
"""


def construct_prompt(d: dict) -> list[dict]:
    """根据电影样本构造 prompt。"""
    # d 包含 "item_list"（用户历史）、"candidates"（候选电影）
    history = "\\n".join(f"- {title} (ID: {mid})" for mid, title in d["item_list"][-10:])
    candidates = "\\n".join(f"{mid}: {title}" for mid, title in d["candidates"])

    return [
        {"role": "system", "content": "你是一名推荐系统专家。根据用户观影历史，从候选列表中选出最感兴趣的电影。"},
        {"role": "user", "content": f"用户观影历史:\\n{history}\\n\\n候选电影:\\n{candidates}\\n\\n请按兴趣从高到低输出电影 ID，用逗号分隔。"},
    ]


def parse_output(text: str):
    """解析 LLM 输出为电影 ID 列表。"""
    import re
    text = text.replace("\\n", ",").replace("，", ",").replace(";", ",")
    ids = re.findall(r"\\b\\d+\\b", text)
    seen = set()
    result = []
    for id_str in ids:
        if id_str not in seen:
            seen.add(id_str)
            result.append(int(id_str))
    return result if result else None
'''


async def cmd_agent_init(args):
    """初始化 Agent 优化会话。"""
    from sii_pe.agent.session import Session

    session = Session(work_dir=args.work_dir)
    session.init(
        task=args.task,
        data_path=args.data,
        config_path=args.config,
    )

    # 自动生成模板 Answer.py（如不存在）
    answer_path = os.path.join(args.work_dir, "Answer.py")
    if not os.path.exists(answer_path):
        if args.task == "arc":
            template = _ARC_TEMPLATE
        else:
            template = _MOVIE_TEMPLATE
        with open(answer_path, "w", encoding="utf-8") as f:
            f.write(template)
        print(f"已生成模板: {answer_path}")

    print(f"会话已初始化: task={args.task}, data={args.data}")
    print(f"会话文件: {session.path}")
    print("\n下一步: 编辑 Answer.py，然后运行: sii-pe agent evaluate --note '描述你的改动'")


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
    try:
        summary = await evaluate_answer(
            session=session,
            answer_path=answer_path,
            note=args.note or "",
            num_trials=int(args.trials) if args.trials else None,
            sample_limit=int(args.samples) if args.samples else None,
        )
    except (SyntaxError, ImportError) as e:
        print(f"错误: Answer.py 加载失败: {e}")
        print("请检查文件语法和函数定义（construct_prompt, parse_output）。")
        sys.exit(1)
    except AttributeError as e:
        print(f"错误: {e}")
        sys.exit(1)

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
    if not session.is_initialized:
        print("会话未初始化。请先运行 sii-pe agent init")
        return
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
