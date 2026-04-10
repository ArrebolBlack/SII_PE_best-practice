"""
Mock 集成测试：用假 API 跑通 agent 完整流程。

测试覆盖：
1. agent init → 创建 session.json
2. agent evaluate → 评测 Answer.py，更新 session
3. agent status → 查看状态
4. agent history → 查看历史
5. agent report → 生成报告

以及底层组件：
- ClientPool 负载均衡
- PromptCandidate 渲染
- Population 管理
- AnswerWrapper 动态导入
"""

import asyncio
import json
import os
import pytest
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 测试 1: Session 状态管理
# ============================================================
def test_session():
    from sii_pe.agent.session import Session

    with tempfile.TemporaryDirectory() as tmpdir:
        session = Session(work_dir=tmpdir)

        # init
        session.init(task="arc", data_path="/fake/data.jsonl")
        assert session.data["task"] == "arc"
        assert os.path.exists(os.path.join(tmpdir, "session.json"))

        # add_result
        s = session.add_result("Answer_v1.py", 0.1, "baseline")
        assert s["round"] == 1
        assert s["score"] == 0.1
        assert s["is_new_best"]

        # add_result 2: 更好
        s = session.add_result("Answer_v2.py", 0.3, "added CoT")
        assert s["round"] == 2
        assert s["is_new_best"]
        assert session.data["best_score"] == 0.3

        # add_result 3: 更差
        s = session.add_result("Answer_v3.py", 0.2, "bad change")
        assert not s["is_new_best"]
        assert session.data["best_score"] == 0.3  # best 不变

        # status
        status = session.get_status()
        assert status["total_rounds"] == 3
        assert status["best_score"] == 0.3

        # history
        history = session.get_history()
        assert len(history) == 3

        # trajectory text
        text = session.get_trajectory_text()
        assert "Round 1" in text
        assert "0.1000" in text

        print("  PASS  test_session")


# ============================================================
# 测试 2: ClientPool 负载均衡
# ============================================================
@pytest.mark.asyncio
async def test_client_pool():
    from sii_pe.infra.client_pool import ClientPool

    pool = ClientPool(
        api_keys=["fake-key-1", "fake-key-2"],
        base_url="https://fake-api.example.com",
        max_per_key=2,
    )

    assert pool.total_capacity == 4

    # 正常 acquire/release
    client, idx = await pool.acquire()
    assert pool.active_requests == 1
    await pool.release(idx)
    assert pool.active_requests == 0

    # 上下文管理器
    async with pool.get_client() as c:
        assert pool.active_requests == 1
    assert pool.active_requests == 0

    print("  PASS  test_client_pool")


# ============================================================
# 测试 3: PromptCandidate 渲染
# ============================================================
def test_prompt_candidate():
    from sii_pe.core.prompt_candidate import PromptCandidate

    c = PromptCandidate(
        name="test",
        system_prompt="你是专家",
        user_prompt_template="历史：{{ history }}\n候选：{{ candidates }}",
    )

    messages = c.render({"history": "电影A", "candidates": "电影B, 电影C"})
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "电影A" in messages[1]["content"]
    assert "电影B" in messages[1]["content"]

    # to_dict / from_dict 往返
    d = c.to_dict()
    c2 = PromptCandidate.from_dict(d)
    assert c2.name == c.name
    assert c2.system_prompt == c.system_prompt

    print("  PASS  test_prompt_candidate")


# ============================================================
# 测试 4: Population 种群管理
# ============================================================
def test_population():
    from sii_pe.core.population import Population
    from sii_pe.core.prompt_candidate import PromptCandidate

    pop = Population(max_size=5)

    # 添加
    for i in range(6):
        c = PromptCandidate(name=f"p{i}", system_prompt=f"sp{i}", user_prompt_template=f"upt{i}")
        pop.add(c, score=i * 0.1)

    # 超过 max_size 应自动裁剪
    assert pop.size == 5

    # best
    best_c, best_s = pop.best
    assert best_s == 0.5  # 最高分

    # top_k
    top2 = pop.get_top_k(2)
    assert len(top2) == 2
    assert top2[0][1] >= top2[1][1]

    # trajectory
    traj = pop.get_trajectory()
    assert len(traj) == 5

    # save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "pop.json")
        pop.save(path)
        pop2 = Population(max_size=5)
        pop2.load(path)
        assert pop2.size == 5

    print("  PASS  test_population")


# ============================================================
# 测试 5: EvalResult 持久化
# ============================================================
def test_persistence():
    from sii_pe.infra.persistence import EvalResult, ExperimentLog
    from sii_pe.core.prompt_candidate import PromptCandidate

    result = EvalResult(
        overall_score=0.42,
        trial_scores=[0.4, 0.44],
        sample_stats={0: {"mean": 0.5, "std": 0.1}, 1: {"mean": 0.34, "std": 0.0}},
        num_trials=2,
        num_samples=2,
        metadata={"model": "test"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # JSON
        result.save_json(os.path.join(tmpdir, "result.json"))
        with open(os.path.join(tmpdir, "result.json")) as f:
            loaded = json.load(f)
        assert abs(loaded["overall_score"] - 0.42) < 1e-6

        # CSV
        result.save_csv(os.path.join(tmpdir, "result.csv"), sample_scores={0: [0.5, 0.5], 1: [0.0, 0.0]})

        # ExperimentLog
        log = ExperimentLog()
        c = PromptCandidate(name="test", system_prompt="s", user_prompt_template="u")
        log.add(c.to_dict(), result)
        assert log.get_best()["score"] == 0.42
        traj = log.get_trajectory()
        assert len(traj) == 1

        log.save(os.path.join(tmpdir, "log.json"))
        log2 = ExperimentLog.load(os.path.join(tmpdir, "log.json"))
        assert len(log2.entries) == 1

    print("  PASS  test_persistence")


# ============================================================
# 测试 6: Config 加载
# ============================================================
def test_config():
    from sii_pe.config import Config

    # 清理可能残留的 env
    _saved = {}
    for k in ["SII_PE_API_KEYS", "SII_PE_MODEL"]:
        if k in os.environ:
            _saved[k] = os.environ.pop(k)

    try:
        # 默认配置
        config = Config.load()
        assert config.api_base_url == "https://api.deepseek.com"
        assert config.model == "deepseek-chat"
        assert config.max_concurrency == 10

        # optimizer fallback
        assert config.get_optimizer_api_keys() == []  # 没设 key
        assert config.get_optimizer_api_base_url() == "https://api.deepseek.com"

        # YAML 加载
        with tempfile.TemporaryDirectory() as tmpdir2:
            yaml_path = os.path.join(tmpdir2, "test.yaml")
            with open(yaml_path, "w") as f:
                f.write("llm:\n  model: test-model\nevaluation:\n  num_trials: 3\n")
            config = Config.load(yaml_path=yaml_path)
            assert config.model == "test-model"
            assert config.num_trials == 3
    finally:
        # 恢复 env
        os.environ.update(_saved)

    print("  PASS  test_config")


# ============================================================
# 测试 7: ARC Task 逻辑
# ============================================================
def test_arc_task():
    from sii_pe.tasks.arc_puzzle import ARCPuzzleTask
    from sii_pe.core.prompt_candidate import PromptCandidate

    task = ARCPuzzleTask()

    # mask_sample
    sample = {
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3, 4]], "output": [[4, 3]]}],
    }
    masked = task.mask_sample(sample)
    assert "output" not in masked["test"][0]
    assert masked["test"][0]["input"] == [[3, 4]]

    # extract_ground_truth
    gt = task.extract_ground_truth(sample)
    assert gt == [[4, 3]]

    # compute_metric
    assert task.compute_metric([[4, 3]], [[4, 3]]) == 1.0
    assert task.compute_metric([[1, 2]], [[4, 3]]) == 0.0

    # parse_output: <grid> 标签
    text = "some text <grid> [[1, 2], [3, 4]] </grid> more text"
    result = task.parse_output(text)
    assert result == [[1, 2], [3, 4]]

    # parse_output: 纯列表
    text2 = "result is [[5, 6], [7, 8]]"
    result2 = task.parse_output(text2)
    assert result2 == [[5, 6], [7, 8]]

    # parse_output: 失败
    assert task.parse_output("no grid here") is None

    # get_template_variables
    candidate = PromptCandidate(
        name="test", system_prompt="sys", user_prompt_template="{{ train_examples }}\n{{ test_input_rows }}"
    )
    variables = task.get_template_variables(masked)
    assert "train_examples" in variables
    assert "test_input_rows" in variables

    print("  PASS  test_arc_task")


# ============================================================
# 测试 8: Movie Task 逻辑
# ============================================================
def test_movie_task():
    from sii_pe.tasks.movie_reranking import MovieRerankingTask

    task = MovieRerankingTask()

    sample = {
        "user_id": 1,
        "item_list": [[100, "Movie A"], [200, "Movie B"]],
        "target_item": [300, "Movie C"],
        "candidates": [[300, "Movie C"], [400, "Movie D"], [500, "Movie E"]],
    }

    # parse_output
    assert task.parse_output("[300, 400, 500]") == [300, 400, 500]
    assert task.parse_output("300，400、500") == [300, 400, 500]  # 中文标点
    assert task.parse_output(None) is None

    # compute_metric: NDCG@10
    score = task.compute_metric([300, 400, 500], 300)
    assert score == 1.0  # target 在第一位

    score2 = task.compute_metric([400, 300, 500], 300)
    assert 0 < score2 < 1.0  # target 在第二位

    score3 = task.compute_metric([400, 500], 300)
    assert score3 == 0.0  # target 不在列表中

    # extract_ground_truth
    assert task.extract_ground_truth(sample) == 300

    # get_template_variables
    variables = task.get_template_variables(sample)
    assert "history" in variables
    assert "candidates" in variables
    assert "Movie B" in variables["history"]

    print("  PASS  test_movie_task")


# ============================================================
# 测试 9: 动态导入 Answer.py
# ============================================================
def test_answer_import():
    from sii_pe.agent.evaluate import load_answer_module

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建一个测试用 Answer.py
        answer_path = os.path.join(tmpdir, "Answer.py")
        with open(answer_path, "w") as f:
            f.write("""
def construct_prompt(d):
    return [{"role": "user", "content": str(d)}]

def parse_output(text):
    return [[1, 2], [3, 4]]
""")
        cp, po = load_answer_module(answer_path)
        assert cp({"test": 1}) == [{"role": "user", "content": "{'test': 1}"}]
        assert po("whatever") == [[1, 2], [3, 4]]

        # 缺少函数的文件
        bad_path = os.path.join(tmpdir, "BadAnswer.py")
        with open(bad_path, "w") as f:
            f.write("x = 1\n")
        try:
            load_answer_module(bad_path)
            assert False, "应该抛异常"
        except AttributeError:
            pass

    print("  PASS  test_answer_import")


# ============================================================
# 测试 10: Agent 完整流程（Mock API）
# ============================================================
@pytest.mark.asyncio
async def test_agent_flow():
    """用 mock API 跑通 init → evaluate → status → history → report"""
    from unittest.mock import AsyncMock, patch
    from sii_pe.agent.session import Session
    from sii_pe.agent.evaluate import evaluate_answer

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建假数据
        data_path = os.path.join(tmpdir, "val.jsonl")
        with open(data_path, "w") as f:
            for _ in range(3):
                f.write(json.dumps({
                    "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
                    "test": [{"input": [[3, 4]], "output": [[4, 3]]}],
                }) + "\n")

        # 创建测试用 Answer.py
        answer_path = os.path.join(tmpdir, "Answer.py")
        with open(answer_path, "w") as f:
            f.write("""
def construct_prompt(d):
    return [{"role": "user", "content": "solve this"}]

def parse_output(text):
    return [[4, 3]]
""")

        # 创建 Answer_v2.py（输出错误答案）
        answer_v2_path = os.path.join(tmpdir, "Answer_v2.py")
        with open(answer_v2_path, "w") as f:
            f.write("""
def construct_prompt(d):
    return [{"role": "user", "content": "solve this"}]

def parse_output(text):
    return [[1, 1]]
""")

        # 提供假 API key，让 Config 和 ClientPool 能正常工作
        os.environ["SII_PE_API_KEYS"] = "fake-test-key"

        # 初始化 session（需要在 env 设好之后）
        session = Session(work_dir=tmpdir)
        session.init(task="arc", data_path=data_path)

        # Mock call_llm: 返回假 LLM 输出
        mock_response = "<grid> [[4, 3]] </grid>"

        with patch("sii_pe.infra.evaluator.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            # 第一轮：正确答案
            summary1 = await evaluate_answer(
                session, answer_path, note="baseline", num_trials=1
            )
            assert summary1["round"] == 1
            assert summary1["score"] > 0, f"Expected score > 0, got {summary1['score']}"
            assert summary1["is_new_best"]

        # 第二轮：错误答案
        with patch("sii_pe.infra.evaluator.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "some output"

            summary2 = await evaluate_answer(
                session, answer_v2_path, note="bad change", num_trials=1
            )
            assert summary2["round"] == 2
            assert summary2["score"] == 0.0
            assert not summary2["is_new_best"]

        # 清理环境变量
        del os.environ["SII_PE_API_KEYS"]

        # 验证 session 状态
        status = session.get_status()
        assert status["total_rounds"] == 2
        assert status["best_score"] > 0

        history = session.get_history()
        assert len(history) == 2

        # 生成报告
        report_text = session.get_trajectory_text()
        assert "Round 1" in report_text
        assert "Round 2" in report_text

    print("  PASS  test_agent_flow")


# ============================================================
# 运行所有测试
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("开始 Mock 集成测试")
    print("=" * 50)

    errors = []

    # 同步测试
    sync_tests = [
        test_session,
        test_prompt_candidate,
        test_population,
        test_persistence,
        test_config,
        test_arc_task,
        test_movie_task,
        test_answer_import,
    ]

    for test in sync_tests:
        try:
            test()
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            errors.append((test.__name__, e))

    # 异步测试
    async_tests = [
        test_client_pool,
        test_agent_flow,
    ]

    for test in async_tests:
        try:
            asyncio.run(test())
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAIL  {test.__name__}: {e}")
            errors.append((test.__name__, e))

    print("\n" + "=" * 50)
    if errors:
        print(f"失败: {len(errors)} 个测试")
        for name, e in errors:
            print(f"  {name}: {e}")
    else:
        print("全部 10 个测试通过!")
    print("=" * 50)
