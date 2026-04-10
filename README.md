# SII PE Best Practice

自动化 Prompt Engineering 优化框架 —— 借鉴 AlphaEvolve、DSPy、EvoPrompt 等思想，实现 prompt 的自动迭代优化。

> 从 [SII_PE_2025_summer](https://github.com/ArrebolBlack/SII_PE_2025_summer) 和 [SII_PE_Exam_2025_Autumn](https://github.com/ArrebolBlack/SII_PE_Exam_2025_Autumn) 中提炼的最佳实践。

## 特性

- **通用异步评测框架**：支持高并发 prompt 评测，多 API Key 负载均衡
- **多种优化策略**：网格搜索、APE 轨迹优化、进化优化（AlphaEvolve/EvoPrompt）
- **5 阶段端到端工作流**：任务解析 → 相关调研 → 管线搭建 → 自主优化 → 报告生成
- **Agent 接管模式**：支持 Claude Code / Codex 等 coding agent 自主驱动优化，可修改任意代码
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
pip install -e .  # 安装 sii-pe CLI 命令
```

### 配置

```bash
# 复制配置模板，填入你的 API Key
cp config.example.yaml config.yaml
# 编辑 config.yaml，将 sk-your-key-here 替换为你的真实 key
```

配置文件支持：
- **多 Key 负载均衡**：在 `api_keys` 列表中添加多个 key，框架自动分配请求
- **独立优化器模型**：评测和优化器可使用不同的 API provider（如评测用 DeepSeek，优化器用 Claude Opus）
- **随时调整**：修改 `config.yaml` 后下次运行即生效，无需重启

也可以通过环境变量覆盖（适用于 CI/CD）：

```bash
export SII_PE_API_KEYS=sk-key1,sk-key2
```

### 使用

框架提供两种优化模式：

> 项目自带示例数据，位于 `examples/data/` 目录。以下命令可直接运行体验：
> - ARC 任务示例：`examples/data/arc_sample.jsonl` + `examples/data/arc_prompt.json`
> - Movie 任务示例：`examples/data/movie_sample.jsonl` + `examples/data/movie_prompt.json`

#### 模式 A：Python 自动优化

框架内置优化循环，自动生成和评测 prompt 变体：

```bash
# 评估一个 prompt（使用示例数据）
sii-pe evaluate --task arc --prompt examples/data/arc_prompt.json --data examples/data/arc_sample.jsonl

# 运行优化（支持 ape / evolutionary 策略）
sii-pe optimize --task arc --strategy ape --prompt examples/data/arc_prompt.json --data examples/data/arc_sample.jsonl

# 运行完整 5 阶段管线（任务解析 → 调研 → 搭建 → 优化 → 报告）
sii-pe pipeline --instruction examples/data/arc_instruction.txt --data examples/data/arc_sample.jsonl
```

#### 模式 B：Agent 接管优化（推荐）

让 Claude Code / Codex 等 coding agent 自主驱动优化，可修改任意代码（不仅是 prompt 文本）：

```bash
# 1. 初始化优化会话（自动生成模板 Answer.py）
sii-pe agent init --task arc --data examples/data/arc_sample.jsonl
# 如需指定配置: --config config.yaml

# 2. 编写 Answer.py（定义 construct_prompt 和 parse_output 两个函数）
#    - construct_prompt(d: dict) -> list[dict]: 返回 OpenAI Chat API messages 列表
#      格式: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
#    - parse_output(text: str) -> Any: 解析 LLM 原始输出为结构化结果

# 3. 评测当前版本
sii-pe agent evaluate --note "baseline: 基本 few-shot prompt"

# 4. 修改代码后再次评测
sii-pe agent evaluate --note "添加了 Chain-of-Thought 推理"

# 5. 查看状态和历史
sii-pe agent status
sii-pe agent history

# 6. 生成探索报告
sii-pe agent report
```

Agent 模式的优势：
- 可修改 `construct_prompt`、`parse_output`、数据处理逻辑等**任意代码**
- 框架自动追踪所有评测历史，无需手动记录
- 支持 `--samples 5` 快速验证，有提升再用全量数据确认
- 详见 [AGENT.md](AGENT.md) 获取完整工作流说明

### Python API

```python
import asyncio
import json
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

    # 加载验证数据（使用示例数据）
    with open("examples/data/arc_sample.jsonl", "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f if line.strip()]

    candidate = PromptCandidate(
        name="my_prompt",
        system_prompt="你是 ARC 任务专家。",
        user_prompt_template="{{ train_examples }}\n\n{{ test_input_rows }}",
    )

    result, _ = await evaluator.evaluate_prompt(val_data, candidate)
    print(f"Score: {result.overall_score:.4f}")

asyncio.run(main())
```

更多示例见 `examples/quick_start.py`。

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

在 `config.yaml` 中添加多个 key，框架自动将请求分配到不同 Key：

```yaml
llm:
  api_keys:
    - sk-key1
    - sk-key2
    - sk-key3
  # 总并发能力 = 3 × 10 = 30
```

新增 key 直接在列表中添加，删除则移除。如果并发数超过所有 Key 的总容量，框架会自动提示添加更多 Key。

评测和优化器支持使用不同的 API provider：

```yaml
llm:
  api_keys: [sk-deepseek-key]
  api_base_url: "https://api.deepseek.com"
  model: "deepseek-chat"

optimizer:
  optimizer_api_keys: [sk-anthropic-key]
  optimizer_api_base_url: "https://api.anthropic.com/v1"
  optimizer_model: "claude-opus-4-6"
```

## 相关项目

- [SII_PE_2025_summer](https://github.com/ArrebolBlack/SII_PE_2025_summer) - 电影推荐重排序 PE 考试（NDCG@10: 0.7306）
- [SII_PE_Exam_2025_Autumn](https://github.com/ArrebolBlack/SII_PE_Exam_2025_Autumn) - ARC 网格谜题 PE 考试框架

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ArrebolBlack/SII_PE_best-practice&type=Date)](https://star-history.com/#ArrebolBlack/SII_PE_best-practice&Date)

## License

MIT
