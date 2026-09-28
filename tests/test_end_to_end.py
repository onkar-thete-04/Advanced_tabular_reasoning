"""End-to-end integration tests: agent loop, evaluation, multi-stage training."""

from config import Config, ExecConfig, TrainingConfig
from src.agent.loop import AgentLoop
from src.agent.parser import parse_steps, extract_answer, extract_tool_calls
from src.constants import STEP_ANSWER, STEP_TOOL_CALL, STEP_TOOL_RESPONSE, STEP_THINK
from src.execution.executor import PythonExecutor
from src.evaluation.benchmarks import evaluate_policy
from src.evaluation.metrics import accuracy, rouge_l, exact_match, compute_metrics
from src.data.schema import Sample
from src.rl.multistage import MultiStageTrainer, pass_at_k
from src.constants import TaskType


def test_parser_round_trip():
    text = (
        "<think>plan</think>"
        "<tool_call>print(1)</tool_call>"
        "<tool_response>1</tool_response>"
        "<answer>1</answer>"
    )
    steps = parse_steps(text)
    kinds = [s.kind for s in steps]
    assert kinds == [STEP_THINK, STEP_TOOL_CALL, STEP_TOOL_RESPONSE, STEP_ANSWER]
    assert extract_answer(text) == "1"
    assert extract_tool_calls(text) == ["print(1)"]


def test_parser_accepts_function_call_marker():
    text = "<function_call>print(2)</function_call><answer>2</answer>"
    assert extract_tool_calls(text) == ["print(2)"]


def test_agent_loop_closed_loop():
    def gen(prompt):
        if "tool_response" not in prompt:
            return "<tool_call>\nprint(6 * 7)\n</tool_call>"
        return "<answer>42</answer>"

    loop = AgentLoop(gen, PythonExecutor(ExecConfig(use_subprocess=False)), max_rounds=5)
    result = loop.run("compute", "t.csv")
    assert result.answer == "42" and result.rounds == 2


def test_agent_loop_stops_without_tool_call():
    loop = AgentLoop(lambda p: "no markers here", PythonExecutor(ExecConfig(use_subprocess=False)))
    result = loop.run("q", "t")
    assert result.error and not result.answer


def test_metrics_basic():
    assert accuracy(["60", "3"], ["60", "3"]) == 1.0
    assert exact_match(["a"], ["b"]) == 0.0
    assert 0.0 <= rouge_l(["the cat sat"], ["the cat sat"]) <= 1.0
    results = compute_metrics(["Acc", "EM"], ["1"], ["1"])
    assert {r.name for r in results} == {"Acc", "EM"}


def test_evaluate_policy_internal():
    samples = [
        {"question": "sum?", "table_ref": "t", "reference": "60", "table_path": "t.csv"},
    ]
    metrics = evaluate_policy(lambda q, t: "60", samples, "internal", input_mode="table_path")
    assert metrics["Acc"] == 1.0


def test_pass_at_k():
    assert pass_at_k(8, 8, 8) == 1.0
    assert 0.0 <= pass_at_k(2, 8, 8) <= 1.0


def test_multistage_runs_three_stages():
    samples = []
    for i in range(20):
        tt = TaskType.GENERAL.value if i % 3 == 0 else TaskType.TABLE_QA_PYTHON.value
        s = Sample(sample_id=f"s{i}", question="q", table_ref="t",
                   task_type=tt, difficulty_score=(i % 10) / 10.0)
        samples.append(s)

    calls = {"sft": 0, "rl": 0}
    trainer = MultiStageTrainer(
        TrainingConfig(),
        sft_fn=lambda data: calls.__setitem__("sft", calls["sft"] + 1),
        rl_stage_fn=lambda stage: calls.__setitem__("rl", calls["rl"] + 1) or {"ran": True},
        evaluate_fn=lambda data: {"n": len(data)},
    )
    history = trainer.run(samples)
    assert calls["sft"] == 1 and calls["rl"] == 3
    assert len(history) == 4


def test_config_round_trip():
    cfg = Config()
    restored = Config.from_dict(cfg.to_dict())
    assert restored.rl.entropy_bonus_C_H == cfg.rl.entropy_bonus_C_H
    assert restored.training.stage_termination_step == 200
