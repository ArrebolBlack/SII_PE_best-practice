"""
种群管理器：维护 prompt 候选数据库。

支持添加、选择、轨迹导出和持久化。
灵感来源：AlphaEvolve 的 candidate database。
"""

import json
import logging
import random
from typing import Optional

from sii_pe.core.prompt_candidate import PromptCandidate

logger = logging.getLogger(__name__)


class Population:
    """
    管理 prompt 候选种群。

    每个候选关联一个评测分数，支持多种选择策略。
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._entries: list[dict] = []  # [{"candidate": PromptCandidate, "score": float}, ...]

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def max_generation(self) -> int:
        """返回种群中最大的 generation 值。"""
        return max((e["candidate"].generation for e in self._entries), default=0)

    @property
    def best(self) -> tuple[PromptCandidate, float] | None:
        if not self._entries:
            return None
        entry = max(self._entries, key=lambda e: e["score"])
        return entry["candidate"], entry["score"]

    def add(self, candidate: PromptCandidate, score: float) -> None:
        """添加候选到种群。如果超过最大容量，移除最差的。"""
        self._entries.append({"candidate": candidate, "score": score})
        logger.info(f"种群添加: {candidate.name} (score={score:.4f}), 当前大小: {self.size}")

        # 超过容量时移除最差的
        if self.size > self.max_size:
            self._entries.sort(key=lambda e: e["score"], reverse=True)
            removed = self._entries.pop()
            logger.debug(f"种群已满，移除最差: {removed['candidate'].name} (score={removed['score']:.4f})")

    def get_top_k(self, k: int) -> list[tuple[PromptCandidate, float]]:
        """返回得分最高的 k 个候选。"""
        sorted_entries = sorted(self._entries, key=lambda e: e["score"], reverse=True)
        return [(e["candidate"], e["score"]) for e in sorted_entries[:k]]

    def get_diverse_sample(self, k: int) -> list[tuple[PromptCandidate, float]]:
        """
        选择得分高且互相不同的 k 个候选。

        策略：从 top 2k 中随机采样 k 个，平衡质量与多样性。
        """
        top = self.get_top_k(min(k * 2, self.size))
        if len(top) <= k:
            return top
        return random.sample(top, k)

    def tournament_select(self, tournament_size: int = 3) -> PromptCandidate:
        """锦标赛选择：随机选 tournament_size 个候选，返回最优者。"""
        if not self._entries:
            raise ValueError("种群为空，无法选择")
        participants = random.sample(self._entries, min(tournament_size, self.size))
        winner = max(participants, key=lambda e: e["score"])
        return winner["candidate"]

    def get_trajectory(self) -> list[dict]:
        """
        返回优化轨迹（按添加顺序），用于 APE 优化器。

        格式: [{"name": ..., "score": ..., "system_prompt": ..., "user_prompt_template": ...}, ...]
        """
        return [
            {
                "name": e["candidate"].name,
                "score": e["score"],
                "system_prompt": e["candidate"].system_prompt,
                "user_prompt_template": e["candidate"].user_prompt_template,
            }
            for e in self._entries
        ]

    def save(self, path: str) -> None:
        """保存种群为 JSON。"""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = [
            {"candidate": e["candidate"].to_dict(), "score": e["score"]}
            for e in self._entries
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"种群已保存: {path} ({self.size} 个候选)")

    def load(self, path: str) -> None:
        """从 JSON 加载种群。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._entries = [
            {"candidate": PromptCandidate.from_dict(e["candidate"]), "score": e["score"]}
            for e in data
        ]
        logger.info(f"种群已加载: {path} ({self.size} 个候选)")
