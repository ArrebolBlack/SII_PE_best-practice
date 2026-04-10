"""
PromptCandidate 数据模型。

表示一个可被优化的 prompt 候选，包含系统提示词和用户提示词模板。
支持 Jinja2 模板渲染。
"""

import json
from dataclasses import dataclass, field

from jinja2 import Template


@dataclass
class PromptCandidate:
    """
    一个可被优化的 prompt 候选。

    属性:
        name: 策略名称
        system_prompt: 系统提示词
        user_prompt_template: 用户提示词模板（含 Jinja2 占位符）
        generation: 进化代数
        parent_names: 父代名称列表（用于追溯进化路径）
        metadata: 额外信息（analysis, improvement 等）
    """

    name: str
    system_prompt: str
    user_prompt_template: str
    generation: int = 0
    parent_names: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def render(self, variables: dict) -> list[dict]:
        """
        用 Jinja2 渲染模板，返回 OpenAI Chat API messages 列表。

        参数:
            variables: 模板变量字典，如 {"history": "...", "candidates": "..."}

        返回:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        rendered = Template(self.user_prompt_template).render(**variables)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": rendered},
        ]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "generation": self.generation,
            "parent_names": self.parent_names,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptCandidate":
        return cls(
            name=d["name"],
            system_prompt=d["system_prompt"],
            user_prompt_template=d["user_prompt_template"],
            generation=d.get("generation", 0),
            parent_names=d.get("parent_names", []),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return f"PromptCandidate(name='{self.name}', gen={self.generation})"
