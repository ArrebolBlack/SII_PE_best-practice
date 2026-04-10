"""
结果持久化：EvalResult 和 ExperimentLog。

支持 JSON 和 CSV 格式的评测结果保存与加载。
"""

import csv
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """单次评测的完整结果。"""

    overall_score: float
    trial_scores: list[float]
    sample_stats: dict[int, dict]  # {sample_idx: {"mean": x, "std": y}}
    num_trials: int
    num_samples: int
    metadata: dict = field(default_factory=dict)  # model, temperature, prompt_name, timestamp

    def save_json(self, path: str) -> None:
        """保存详细结果为 JSON。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # 将 int keys 转为 str 以兼容 JSON
        data = {
            "overall_score": self.overall_score,
            "trial_scores": self.trial_scores,
            "sample_stats": {str(k): v for k, v in self.sample_stats.items()},
            "num_trials": self.num_trials,
            "num_samples": self.num_samples,
            "metadata": self.metadata,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"评测结果 JSON 已保存: {path}")

    def save_csv(self, path: str, sample_scores: dict[int, list[float]] = None) -> None:
        """保存逐样本统计为 CSV。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["Sample_ID"]
            if sample_scores:
                header += [f"Trial_{i+1}" for i in range(self.num_trials)]
            header += ["Mean", "Std"]
            writer.writerow(header)
            for idx in range(self.num_samples):
                row = [idx]
                if sample_scores and idx in sample_scores:
                    row += [f"{s:.4f}" for s in sample_scores[idx]]
                stats = self.sample_stats.get(idx, self.sample_stats.get(str(idx), {}))
                row += [f"{stats.get('mean', 0):.4f}", f"{stats.get('std', 0):.4f}"]
                writer.writerow(row)
        logger.info(f"样本统计 CSV 已保存: {path}")

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentLog:
    """
    追踪整个优化过程的所有评测结果。

    每次调用 add() 都会记录一个 (PromptCandidate, EvalResult) 对，
    用于生成优化轨迹和最终报告。
    """

    def __init__(self):
        self.entries: list[dict] = []

    def add(self, candidate_dict: dict, result: EvalResult) -> None:
        """记录一次评测。"""
        self.entries.append({
            "candidate": candidate_dict,
            "score": result.overall_score,
            "trial_scores": result.trial_scores,
            "metadata": result.metadata,
            "timestamp": datetime.now().isoformat(),
        })

    def get_best(self) -> dict | None:
        """返回得分最高的记录。"""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e["score"])

    def get_trajectory(self) -> list[dict]:
        """
        返回优化轨迹（按时间顺序），用于 APE 优化器。

        格式: [{"name": ..., "score": ..., "system_prompt": ..., "user_prompt_template": ...}, ...]
        """
        trajectory = []
        for entry in self.entries:
            c = entry["candidate"]
            trajectory.append({
                "name": c.get("name", "unknown"),
                "score": entry["score"],
                "system_prompt": c.get("system_prompt", ""),
                "user_prompt_template": c.get("user_prompt_template", ""),
            })
        return trajectory

    def save(self, path: str) -> None:
        """保存完整实验日志为 JSON。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)
        logger.info(f"实验日志已保存: {path}")

    @classmethod
    def load(cls, path: str) -> "ExperimentLog":
        """从 JSON 加载实验日志。"""
        log = cls()
        with open(path, "r", encoding="utf-8") as f:
            log.entries = json.load(f)
        return log
