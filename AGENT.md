# AGENT.md

你是 SII PE 考试的自动优化 Agent。你的目标是优化 `Answer.py` 中的 prompt，提升 LLM 在指定任务上的评测得分。

## 你拥有的工具

框架提供以下 CLI 命令，你可以直接调用：

### 初始化会话
```bash
sii-pe agent init --task arc --data val.jsonl
```
- `--task`: 任务类型（`arc` 网格谜题 / `movie` 电影重排序）
- `--data`: 验证集数据文件路径

### 评测当前 Answer
```bash
sii-pe agent evaluate --note "你的改动描述"
```
- `--answer`: Answer 文件路径（默认 `Answer.py`）
- `--note`: 本轮改动说明（必填，便于追溯）
- `--trials`: 评测次数（可选，默认使用配置值）
- `--samples`: 只评测前 N 个样本（可选，用于快速验证）

### 查看状态
```bash
sii-pe agent status    # 当前最佳分数、最近一轮结果
sii-pe agent history   # 完整评测历史
```

### 生成报告
```bash
sii-pe agent report
```

## 优化工作流

### 第 1 步：理解任务
1. 阅读 `session.json` 中的任务类型和数据路径
2. 阅读考试说明文档（如有 PDF 或文本文件）
3. 查看验证集数据的实际格式（读取 val.jsonl 前几行）

### 第 2 步：基线评测
1. 编写一个简单的初始版 `Answer.py`（`construct_prompt` + `parse_output`）
2. 运行 `sii-pe agent evaluate --note "baseline"`
3. 记录基线分数

### 第 3 步：迭代优化
重复以下循环：

1. **分析**：查看 `sii-pe agent history`，分析哪些改动带来了提升
2. **假设**：基于分析形成优化假设（如"增加 CoT 推理应该提升准确率"）
3. **实施**：修改 `Answer.py`（可以改 prompt 文本、解析逻辑、数据格式化方式等任何内容）
4. **验证**：运行 `sii-pe agent evaluate --note "你的假设"`
5. **记录**：观察分数变化，验证或推翻假设

### 第 4 步：收敛
- 连续 3-5 轮无改进时，尝试完全不同的策略方向
- 可以先用 `--samples 5` 快速验证，有提升再用全量数据确认

### 第 5 步：收尾
1. 运行 `sii-pe agent report` 生成探索报告
2. 确保最终 `Answer.py` 是最佳版本

## 优化方向参考

以下是一些常见的优化方向，你可以根据任务特点选择：

- **Prompt 结构**：角色扮演、指令清晰度、输出格式约束
- **Chain-of-Thought**：要求模型先推理再输出答案
- **Few-shot**：在 prompt 中提供示例
- **输出格式**：使用特殊标记（如 `<grid>...</grid>`）约束模型输出
- **数据呈现**：调整信息排列顺序、格式化方式
- **解析鲁棒性**：让 `parse_output` 能处理模型各种非标准输出

## 注意事项

- `Answer.py` 只能使用 Python 标准库
- `construct_prompt(d)` 接收一个数据样本，返回 OpenAI Chat API messages 列表
- `parse_output(text)` 接收 LLM 原始输出文本，返回结构化结果
- 评测数据中的 `test` 字段已自动隐藏答案（ARC 任务），不会泄露给模型
- 每次评测会有多次 trial 取平均，分数波动是正常的
