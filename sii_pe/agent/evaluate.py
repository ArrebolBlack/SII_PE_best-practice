"""
Agent 评测：动态导入 Answer 文件，运行评测，更新会话状态。
"""

import importlib.util
import json
import logging
import os

from sii_pe.config import Config
from sii_pe.infra.client_pool import ClientPool
from sii_pe.infra.evaluator import Evaluator
from sii_pe.agent.session import Session
from sii_pe.tasks.base_task import BaseTask
from sii_pe.tasks.movie_reranking import MovieRerankingTask
from sii_pe.tasks.arc_puzzle import ARCPuzzleTask

logger = logging.getLogger(__name__)

# 任务类型到 BaseTask 实例的映射
_TASK_INSTANCES = {
    "movie": MovieRerankingTask(),
    "arc": ARCPuzzleTask(),
}


def load_answer_module(answer_path: str):
    """
    动态导入 Answer.py，返回 (construct_prompt, parse_output)。

    兼容两种形式：
    1. 考试格式：直接定义 construct_prompt(d) 和 parse_output(text) 函数
    2. 框架格式：定义 BaseTask 子类
    """
    answer_path = os.path.abspath(answer_path)
    if not os.path.exists(answer_path):
        raise FileNotFoundError(f"Answer 文件不存在: {answer_path}")

    spec = importlib.util.spec_from_file_location("answer", answer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "construct_prompt") or not hasattr(module, "parse_output"):
        raise AttributeError(
            f"{answer_path} 必须定义 construct_prompt(d) 和 parse_output(text) 两个函数"
        )

    return module.construct_prompt, module.parse_output


class AnswerWrapper(BaseTask):
    """
    将考试格式的 construct_prompt(d) / parse_output(text) 包装为 BaseTask 接口。

    复用已有 Task 实现的 compute_metric / extract_ground_truth / mask_sample。
    """

    def __init__(self, construct_prompt_fn, parse_output_fn, base_task: BaseTask):
        self._construct_prompt = construct_prompt_fn
        self._parse_output = parse_output_fn
        self._base_task = base_task

    def construct_prompt(self, sample, candidate=None):
        # 考试格式：construct_prompt 只接受 d
        return self._construct_prompt(sample)

    def parse_output(self, text):
        result = self._parse_output(text)
        return result if result else None

    def compute_metric(self, prediction, ground_truth):
        return self._base_task.compute_metric(prediction, ground_truth)

    def extract_ground_truth(self, sample):
        return self._base_task.extract_ground_truth(sample)

    def mask_sample(self, sample):
        return self._base_task.mask_sample(sample)


async def evaluate_answer(
    session: Session,
    answer_path: str,
    note: str = "",
    num_trials: int | None = None,
    sample_limit: int | None = None,
) -> dict:
    """
    评测一个 Answer 文件并更新会话。

    参数:
        session: 会话对象
        answer_path: Answer.py 文件路径
        note: 本轮备注
        num_trials: 评测次数
        sample_limit: 评测样本数限制

    返回:
        评测摘要 dict
    """
    # 加载配置
    config = Config.load(yaml_path=session.data.get("config_path"))

    # 加载数据
    data_path = session.data.get("data_path")
    with open(data_path, "r", encoding="utf-8") as f:
        val_data = [json.loads(line.strip()) for line in f if line.strip()]

    # 动态导入 Answer
    construct_prompt_fn, parse_output_fn = load_answer_module(answer_path)

    # 获取对应的 Task 实例（复用 compute_metric / extract_ground_truth / mask_sample）
    task_name = session.data.get("task", "arc")
    base_task = _TASK_INSTANCES.get(task_name, ARCPuzzleTask())
    task = AnswerWrapper(construct_prompt_fn, parse_output_fn, base_task)

    # 创建 dummy PromptCandidate 满足 Evaluator 接口
    from sii_pe.core.prompt_candidate import PromptCandidate
    dummy = PromptCandidate(
        name=f"agent_round_{len(session.data.get('rounds', [])) + 1}",
        system_prompt="",
        user_prompt_template="",
    )

    # 评测
    pool = ClientPool(config.api_keys, config.api_base_url)
    evaluator = Evaluator(pool, task, config)

    result, _ = await evaluator.evaluate_prompt(
        val_data,
        dummy,
        num_trials=num_trials,
        sample_limit=sample_limit,
    )

    # 更新会话
    summary = session.add_result(
        answer_file=answer_path,
        score=result.overall_score,
        note=note,
        trial_scores=result.trial_scores,
    )

    return summary
