"""
ARC 网格谜题任务（Autumn 2025 考试）。

指标：Exact Match Accuracy
数据格式：{"train": [{"input": grid, "output": grid}, ...], "test": [{"input": grid, "output": grid}]}
"""

import ast
import re
from copy import deepcopy
from typing import Any

from sii_pe.tasks.base_task import BaseTask


class ARCPuzzleTask(BaseTask):
    """ARC 网格谜题任务：根据训练样本推断变换规则，应用于测试输入。"""

    def construct_prompt(self, sample: dict, candidate) -> list[dict]:
        """使用 candidate 的模板构造 prompt。"""
        variables = self.get_template_variables(sample)
        return candidate.render(variables)

    def parse_output(self, text: str) -> list[list[int]] | None:
        """
        解析 LLM 输出为二维整数网格。

        解析优先级：<grid>...</grid> > 文本中的首个嵌套列表
        复用自 Autumn Answer_test.py 的解析逻辑。
        """
        if not isinstance(text, str):
            return None

        # 1) 优先抓取 <grid> ... </grid>
        m = re.search(
            r"<\s*grid\s*>\s*(.*?)\s*<\s*/\s*grid\s*>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        candidate_text = m.group(1) if m else None

        # 2) 回退：搜索首个双层列表
        if candidate_text is None:
            m2 = re.search(
                r"\[\s*(\[\s*(?:-?\d+\s*,\s*)*-?\d+\s*\]\s*(?:,\s*\[\s*(?:-?\d+\s*,\s*)*-?\d+\s*\]\s*)+)\s*\]",
                text,
                flags=re.DOTALL,
            )
            if m2:
                candidate_text = "[" + m2.group(1) + "]"

        if candidate_text is None:
            return None

        # 3) 安全解析
        try:
            obj = ast.literal_eval(candidate_text)
        except Exception:
            return None

        # 4) 验证：非空二维整数列表，行长度一致
        if not (isinstance(obj, list) and all(isinstance(r, list) for r in obj) and len(obj) > 0):
            return None
        try:
            grid = [[int(x) for x in row] for row in obj]
        except Exception:
            return None

        w = len(grid[0])
        if any(len(row) != w for row in grid):
            return None

        return grid

    def compute_metric(self, prediction: list[list[int]], ground_truth: list[list[int]]) -> float:
        """Exact Match: 形状一致且每个元素都相等返回 1.0，否则 0.0。"""
        if not isinstance(prediction, list) or not isinstance(ground_truth, list):
            return 0.0
        if len(prediction) != len(ground_truth):
            return 0.0
        for r1, r2 in zip(prediction, ground_truth):
            if not isinstance(r1, list) or not isinstance(r2, list):
                return 0.0
            if len(r1) != len(r2):
                return 0.0
            if any(a != b for a, b in zip(r1, r2)):
                return 0.0
        return 1.0

    def extract_ground_truth(self, sample: dict) -> list[list[int]]:
        return sample["test"][0]["output"]

    def mask_sample(self, sample: dict) -> dict:
        """隐藏 test 的 output，只保留 input。"""
        x = deepcopy(sample)
        x["test"] = [{"input": sample["test"][0]["input"]}]
        return x

    def get_template_variables(self, sample: dict) -> dict:
        """提取模板变量：train_examples 和 test_input。"""

        def grid_to_str(g):
            rows_txt = "\n".join(" ".join(str(x) for x in row) for row in g)
            py_txt = repr([[int(x) for x in row] for row in g])
            return rows_txt, py_txt

        train_chunks = []
        for ex in sample.get("train", []):
            in_rows, in_list = grid_to_str(ex["input"])
            out_rows, out_list = grid_to_str(ex.get("output", []))
            chunk = (
                f"输入：\n{in_rows}\nPython列表：{in_list}\n"
                f"输出：\n{out_rows}\nPython列表：{out_list}"
            )
            train_chunks.append(chunk)

        test_input = sample["test"][0]["input"]
        t_rows, t_list = grid_to_str(test_input)

        return {
            "train_examples": "\n\n".join(train_chunks),
            "test_input_rows": t_rows,
            "test_input_list": t_list,
        }
