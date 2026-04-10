"""
电影推荐重排序任务（Summer 2025 考试）。

指标：NDCG@10
数据格式：{"user_id", "item_list": [[id, title], ...], "target_item": [id, title], "candidates": [[id, title], ...]}
"""

import math
import re
from typing import Any

from sii_pe.tasks.base_task import BaseTask


class MovieRerankingTask(BaseTask):
    """电影推荐重排序任务：根据用户历史，对候选电影按兴趣排序。"""

    def __init__(self, k: int = 10):
        self.k = k

    def construct_prompt(self, sample: dict, candidate) -> list[dict]:
        """使用 candidate 的模板构造 prompt。"""
        variables = self.get_template_variables(sample)
        return candidate.render(variables)

    def parse_output(self, text: str) -> list[int] | None:
        """
        鲁棒解析：替换所有分隔符，提取数字 ID，去重保序。

        复用自 Summer Answer.py 的解析逻辑。
        """
        if not isinstance(text, str):
            return None
        text = text.replace("\n", ",").replace("，", ",").replace(";", ",").replace("、", ",")
        candidates = re.findall(r"\b\d+\b", text)
        seen = set()
        parsed = []
        for id_str in candidates:
            if id_str not in seen:
                seen.add(id_str)
                parsed.append(int(id_str))
        return parsed if parsed else None

    def compute_metric(self, prediction: list[int], ground_truth: int) -> float:
        """
        NDCG@K 计算。

        假设只有一个相关项（target_item），IDCG = 1.0。
        """
        relevance = [1 if item_id == ground_truth else 0 for item_id in prediction[: self.k]]
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
        idcg = 1.0  # 理想情况：唯一相关项在第一位，1/log2(2) = 1.0
        return dcg / idcg if idcg > 0 else 0.0

    def extract_ground_truth(self, sample: dict) -> int:
        return sample["target_item"][0]

    def mask_sample(self, sample: dict) -> dict:
        # 电影数据不需要 mask，答案不在样本结构中直接暴露给模型
        return sample

    def get_template_variables(self, sample: dict) -> dict:
        """提取模板变量：history 和 candidates。"""
        history = "\n".join(
            f"- {title} (ID: {mid})"
            for mid, title in reversed(sample["item_list"][-10:])
        )
        candidates = "\n".join(
            f"{mid}: {title}" for mid, title in sample["candidates"]
        )
        return {"history": history, "candidates": candidates}
