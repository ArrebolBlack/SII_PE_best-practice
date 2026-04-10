"""
阶段 3：管线搭建。

根据 TaskSpec 自动配置评估管线，生成初始 prompt，验证管线通畅。
"""

import logging

from sii_pe.core.prompt_candidate import PromptCandidate
from sii_pe.infra.client_pool import ClientPool
from sii_pe.infra.evaluator import Evaluator
from sii_pe.infra.llm_caller import call_llm
from sii_pe.infra.persistence import EvalResult
from sii_pe.tasks.base_task import BaseTask
from sii_pe.tasks.movie_reranking import MovieRerankingTask
from sii_pe.tasks.arc_puzzle import ARCPuzzleTask
from sii_pe.workflow.task_parser import TaskSpec

logger = logging.getLogger(__name__)

# 已注册的任务类型
TASK_REGISTRY = {
    "reranking": MovieRerankingTask,
    "movie_reranking": MovieRerankingTask,
    "grid_puzzle": ARCPuzzleTask,
    "arc": ARCPuzzleTask,
    "arc_puzzle": ARCPuzzleTask,
}


class PipelineSetup:
    """
    管线搭建：选择任务适配器，生成初始 prompt，验证管线。
    """

    async def setup(
        self,
        task_spec: TaskSpec,
        val_data: list[dict],
        pool: ClientPool,
        config,
    ) -> tuple[BaseTask, PromptCandidate, EvalResult]:
        """
        搭建并验证评估管线。

        步骤:
        1. 根据 task_type 选择或创建 BaseTask 实现
        2. 生成初始 PromptCandidate
        3. 运行小规模评估验证管线通畅

        返回:
            (task, initial_candidate, baseline_result)
        """
        # 1. 选择任务适配器
        task = self._get_task(task_spec)
        logger.info(f"任务适配器: {task.__class__.__name__}")

        # 2. 生成初始 prompt
        initial_candidate = await self._generate_initial_prompt(task_spec, pool, config)
        logger.info(f"初始 prompt: {initial_candidate.name}")

        # 3. 验证管线
        evaluator = Evaluator(pool, task, config)
        logger.info("验证管线: 运行小规模评估 (5 样本, 1 试次)...")
        baseline_result, _ = await evaluator.evaluate_prompt(
            val_data,
            initial_candidate,
            num_trials=1,
            sample_limit=min(5, len(val_data)),
        )
        logger.info(f"管线验证通过! 基线分数: {baseline_result.overall_score:.4f}")

        return task, initial_candidate, baseline_result

    def _get_task(self, task_spec: TaskSpec) -> BaseTask:
        """根据 task_type 获取任务实例。"""
        task_type = task_spec.task_type.lower().replace(" ", "_")
        if task_type in TASK_REGISTRY:
            return TASK_REGISTRY[task_type]()
        # 默认尝试 ARC
        logger.warning(f"未知任务类型 '{task_spec.task_type}'，默认使用 ARCPuzzleTask")
        return ARCPuzzleTask()

    async def _generate_initial_prompt(
        self, task_spec: TaskSpec, pool: ClientPool, config
    ) -> PromptCandidate:
        """用 LLM 根据任务描述生成初始 prompt。"""
        import json

        # 根据任务类型确定模板变量名，确保生成的 prompt 使用正确的占位符
        task_type = task_spec.task_type.lower().replace(" ", "_")
        if task_type in ("grid_puzzle", "arc", "arc_puzzle"):
            template_hint = (
                "用户提示词模板必须使用以下 Jinja2 占位符：\n"
                "- {{ train_examples }}: 训练样本文本\n"
                "- {{ test_input_rows }}: 测试输入的行格式\n"
                "- {{ test_input_list }}: 测试输入的 Python 列表格式\n\n"
                "输出要求：预测网格以 <grid> [[...],[...]] </grid> 格式输出。"
            )
        elif task_type in ("reranking", "movie_reranking"):
            template_hint = (
                "用户提示词模板必须使用以下 Jinja2 占位符：\n"
                "- {{ history }}: 用户历史观影记录\n"
                "- {{ candidates }}: 候选电影列表\n\n"
                "输出要求：按兴趣从高到低输出电影 ID，用逗号分隔。"
            )
        else:
            template_hint = "用户提示词模板中可使用 Jinja2 占位符。"

        messages = [
            {
                "role": "system",
                "content": "你是一名 Prompt Engineering 专家。请为给定任务设计一个初始提示词。",
            },
            {
                "role": "user",
                "content": (
                    f"任务：{task_spec.description}\n"
                    f"输出格式：{task_spec.output_format}\n"
                    f"约束：{', '.join(task_spec.constraints) if task_spec.constraints else '无'}\n\n"
                    f"{template_hint}\n\n"
                    f"请设计一个初始提示词，输出 JSON：\n"
                    f'{{\n'
                    f'  "system_prompt": "系统提示词",\n'
                    f'  "user_prompt_template": "用户提示词模板"\n'
                    f'}}'
                ),
            },
        ]

        try:
            response = await call_llm(
                pool, messages, model=config.optimizer_model, temperature=0.5, max_tokens=1024
            )
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end])
            return PromptCandidate(
                name="initial_auto",
                system_prompt=data["system_prompt"],
                user_prompt_template=data["user_prompt_template"],
                metadata={"strategy": "auto_generated"},
            )
        except Exception as e:
            logger.warning(f"自动生成初始 prompt 失败: {e}，使用默认模板")
            return self._get_fallback_prompt(task_type)

    def _get_fallback_prompt(self, task_type: str) -> PromptCandidate:
        """返回任务类型对应的默认 fallback prompt。"""
        if task_type in ("grid_puzzle", "arc", "arc_puzzle"):
            return PromptCandidate(
                name="initial_default",
                system_prompt="你是一个 ARC 任务专家。请仔细观察训练样本中的输入输出对，推断变换规则，并将该规则应用到测试输入上。",
                user_prompt_template=(
                    "训练样本：\n{{ train_examples }}\n\n"
                    "测试输入：\n{{ test_input_rows }}\n\n"
                    "请分析训练样本的变换规律，然后对测试输入应用相同的变换。\n"
                    "将预测结果以 <grid> [[...],[...]] </grid> 格式输出。"
                ),
                metadata={"strategy": "default_fallback"},
            )
        else:
            return PromptCandidate(
                name="initial_default",
                system_prompt="请根据提供的信息完成任务。",
                user_prompt_template="任务信息：\n{{ history }}\n\n候选：\n{{ candidates }}\n\n请给出你的答案。",
                metadata={"strategy": "default_fallback"},
            )
