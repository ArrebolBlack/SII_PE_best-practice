"""
Agent 会话状态管理。

管理 session.json，记录每轮评测的文件、分数、备注。
"""

import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

SESSION_FILE = "session.json"


class Session:
    """
    优化会话状态。

    通过 session.json 持久化，让 agent 不需要自己记住历史。
    """

    def __init__(self, work_dir: str = "."):
        self.work_dir = work_dir
        self.path = os.path.join(work_dir, SESSION_FILE)
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"rounds": [], "best_score": -1.0, "best_file": None}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def init(self, task: str, data_path: str, config_path: str | None = None) -> None:
        """初始化新会话。"""
        self.data = {
            "task": task,
            "data_path": os.path.abspath(data_path),
            "config_path": os.path.abspath(config_path) if config_path else None,
            "created_at": datetime.now().isoformat(),
            "rounds": [],
            "best_score": -1.0,
            "best_file": None,
        }
        self._save()
        logger.info(f"会话初始化: task={task}, data={data_path}")

    @property
    def is_initialized(self) -> bool:
        return "task" in self.data and len(self.data.get("rounds", [])) >= 0

    def add_result(self, answer_file: str, score: float, note: str = "",
                   trial_scores: list[float] | None = None) -> dict:
        """
        记录一轮评测结果。

        返回本轮摘要。
        """
        round_num = len(self.data["rounds"]) + 1
        prev_best = self.data["best_score"]

        # 保存 Answer.py 版本快照
        abs_answer = os.path.abspath(answer_file)
        snapshot_dir = os.path.join(self.work_dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_name = f"Answer_v{round_num}.py"
        snapshot_path = os.path.join(snapshot_dir, snapshot_name)
        if os.path.exists(abs_answer):
            import shutil
            shutil.copy2(abs_answer, snapshot_path)
            logger.info(f"Answer 快照已保存: {snapshot_path}")

        entry = {
            "round": round_num,
            "file": abs_answer,
            "snapshot": snapshot_name,
            "score": score,
            "note": note,
            "trial_scores": trial_scores or [],
            "timestamp": datetime.now().isoformat(),
        }
        self.data["rounds"].append(entry)

        # 更新最佳
        is_new_best = score > self.data["best_score"]
        if is_new_best:
            self.data["best_score"] = score
            self.data["best_file"] = os.path.abspath(answer_file)

        self._save()

        # 计算变化
        delta = score - prev_best if prev_best >= 0 else score
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"

        summary = {
            "round": round_num,
            "score": score,
            "delta": delta,
            "arrow": arrow,
            "is_new_best": is_new_best,
            "best_score": self.data["best_score"],
            "best_file": self.data["best_file"],
        }

        logger.info(
            f"Round {round_num}: {score:.4f} ({arrow}{abs(delta):.4f}), "
            f"Best: {self.data['best_score']:.4f}"
            + (" ★ NEW BEST" if is_new_best else "")
        )
        return summary

    def get_status(self) -> dict:
        """获取当前状态摘要。"""
        return {
            "task": self.data.get("task", "unknown"),
            "total_rounds": len(self.data.get("rounds", [])),
            "best_score": self.data["best_score"],
            "best_file": self.data["best_file"],
            "last_round": self.data["rounds"][-1] if self.data.get("rounds") else None,
        }

    def get_history(self) -> list[dict]:
        """返回完整历史。"""
        return self.data.get("rounds", [])

    def get_trajectory_text(self) -> str:
        """返回格式化的历史轨迹文本，供 agent 参考。"""
        lines = []
        for r in self.data.get("rounds", []):
            lines.append(
                f"Round {r['round']}: score={r['score']:.4f}, "
                f"file={os.path.basename(r['file'])}, "
                f"note={r.get('note', '')}"
            )
        return "\n".join(lines) if lines else "暂无评测记录"
