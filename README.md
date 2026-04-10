# SII PE Best Practice

自动化 Prompt Engineering 优化框架 —— 借鉴 AlphaEvolve、DSPy、EvoPrompt 等思想，实现 prompt 的自动迭代优化。

> 从 [SII_PE_2025_summer](https://github.com/ArrebolBlack/SII_PE_2025_summer) 和 [SII_PE_Exam_2025_Autumn](https://github.com/ArrebolBlack/SII_PE_Exam_2025_Autumn) 中提炼的最佳实践。

## 特性

- **通用异步评测框架**：支持高并发 prompt 评测，多 API Key 负载均衡
- **多种优化策略**：网格搜索、APE 轨迹优化、进化优化（AlphaEvolve/EvoPrompt）
- **5 阶段端到端工作流**：任务解析 → 相关调研 → 管线搭建 → 自主优化 → 报告生成
- **可扩展任务接口**：通过继承 `BaseTask` 快速适配新任务

## 架构

```
┌─────────────────────────────────────────────────┐
│              Layer 3: 工作流 (Workflow)            │
│  任务解析 → 调研 → 管线搭建 → 自主优化 → 报告生成   │
├─────────────────────────────────────────────────┤
│              Layer 2: 优化核心 (Core)              │
│  PromptCandidate · Population · Strategies       │
│  GridSearch │ APE Trajectory │ Evolutionary       │
├─────────────────────────────────────────────────┤
│              Layer 1: 基础设施 (Infra)             │
│  ClientPool (多Key负载均衡) · LLM Caller          │
│  Evaluator (异步多试次) · Persistence              │
├─────────────────────────────────────────────────┤
│              任务适配器 (Tasks)                     │
│  BaseTask → MovieReranking │ ARCPuzzle │ Custom   │
└─────────────────────────────────────────────────┘
```

## 快速开始

### 安装

```bash
git clone https://github.com/ArrebolBlack/SII_PE_best-practice.git
cd SII_PE_best-practice
pip install -r requirements.txt
```

### 配置

```bash
# 设置 API Key（支持多个 key 负载均衡）
export SII_PE_API_KEYS=sk-key1,sk-key2,sk-key3

# 可选：自定义 API 地址
export SII_PE_API_BASE_URL=https://api.deepseek.com
```

### 使用

#### 评估一个 prompt

```bash
sii-pe evaluate --task arc --prompt prompt.json --data val.jsonl
```

#### 运行优化

```bash
sii-pe optimize --task arc --strategy ape --prompt initial_prompt.json --data val.jsonl
```

#### 运行完整 5 阶段管线

```bash
sii-pe pipeline --instruction exam_instruction.txt --data val.jsonl
```

### Python API

```python
import asyncio
from sii_pe.config import Config
from sii_pe.infra.client_pool import ClientPool
from sii_pe.infra.evaluator import Evaluator
from sii_pe.core.prompt_candidate import PromptCandidate
from sii_pe.tasks.arc_puzzle import ARCPuzzleTask

async def main():
    config = Config.load()
    pool = ClientPool(config.api_keys, config.api_base_url)
    task = ARCPuzzleTask()
    evaluator = Evaluator(pool, task, config)

    candidate = PromptCandidate(
        name="my_prompt",
        system_prompt="你是 ARC 任务专家。",
        user_prompt_template="{{ train_examples }}\n\n{{ test_input_rows }}",
    )

    result, _ = await evaluator.evaluate_prompt(val_data, candidate)
    print(f"Score: {result.overall_score:.4f}")

asyncio.run(main())
```

## 自定义任务

继承 `BaseTask` 即可适配新任务：

```python
from sii_pe.tasks.base_task import BaseTask

class MyTask(BaseTask):
    def construct_prompt(self, sample, candidate):
        variables = self.get_template_variables(sample)
        return candidate.render(variables)

    def parse_output(self, text):
        # 解析 LLM 输出
        ...

    def compute_metric(self, prediction, ground_truth):
        # 计算得分 [0, 1]
        ...

    def extract_ground_truth(self, sample):
        return sample["answer"]

    def mask_sample(self, sample):
        # 隐藏答案
        masked = sample.copy()
        del masked["answer"]
        return masked
```

## 多 Key 负载均衡

当并发数超过单个 API Key 的限制时，框架自动将请求分配到不同 Key：

```bash
# 设置 3 个 key，总并发能力 = 3 × 10 = 30
export SII_PE_API_KEYS=sk-key1,sk-key2,sk-key3
```

如果并发数超过所有 Key 的总容量，框架会自动提示添加更多 Key。

## 相关项目

- [SII_PE_2025_summer](https://github.com/ArrebolBlack/SII_PE_2025_summer) - 电影推荐重排序 PE 考试（NDCG@10: 0.7306）
- [SII_PE_Exam_2025_Autumn](https://github.com/ArrebolBlack/SII_PE_Exam_2025_Autumn) - ARC 网格谜题 PE 考试框架

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ArrebolBlack/SII_PE_best-practice&type=Date)](https://star-history.com/#ArrebolBlack/SII_PE_best-practice&Date)

## License

MIT
