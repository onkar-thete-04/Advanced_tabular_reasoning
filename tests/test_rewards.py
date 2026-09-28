"""Tests for the task-adaptive reward system (Section 3.4)."""

from config import RewardConfig
from src.constants import (
    FeedbackMethod, TaskType,
    STEP_TOOL_CALL, STEP_TOOL_RESPONSE, STEP_ANSWER,
)
from src.data.schema import Trajectory, Step
from src.rewards.process_reward import ProcessStepReward
from src.rewards.regularization import Regularizer
from src.rewards.router import RewardRouter
from src.rewards.rule_based import RuleBasedReward, exact_match, numeric_match
from src.rewards.criteria_judge import (
    CriteriaInjectedRewardModel, build_judge_prompt, parse_judge_output,
)
from src.rewards.aggregator import RewardAggregator


def test_process_reward_values():
    pr = ProcessStepReward(RewardConfig())
    assert pr.reward(function_correct=False, exec_success=False).value == -0.2
    assert pr.reward(function_correct=True, exec_success=True).value == 0.1
    assert pr.reward(function_correct=True, exec_success=False).value == -0.1


def test_process_reward_terminal_is_zero():
    pr = ProcessStepReward(RewardConfig())
    assert pr.reward(True, True, is_terminal=True).value == 0.0


def test_router_maps_all_task_types():
    expected = {
        TaskType.GENERAL: FeedbackMethod.LLM_EVAL,
        TaskType.MATH_LOGIC: FeedbackMethod.RULE,
        TaskType.CODING: FeedbackMethod.RUN_AND_RULE,
        TaskType.TABLE_QA_PYTHON: FeedbackMethod.RUN_AND_LLM_EVAL,
        TaskType.SQL: FeedbackMethod.RUN_AND_RULE,
        TaskType.TABLE_QA_WITH_LABEL: FeedbackMethod.RULE,
        TaskType.QA_WITH_LABEL: FeedbackMethod.RULE,
    }
    for tt, method in expected.items():
        assert RewardRouter.route(tt.value) == method


def test_rule_exact_and_numeric_match():
    assert exact_match(" 60 ", "60")
    assert not exact_match("61", "60")
    assert numeric_match("the answer is 60.0", "60")
    assert not numeric_match("61", "60")


def test_rule_terminal_reward():
    router = RewardRouter()
    r = router.terminal_reward(TaskType.TABLE_QA_WITH_LABEL.value, "60", "60")
    assert r.value == 1.0
    r2 = router.terminal_reward(TaskType.TABLE_QA_WITH_LABEL.value, "61", "60")
    assert r2.value == 0.0


def test_judge_prompt_contains_criteria_and_response():
    prompt = build_judge_prompt("must be accurate", "the answer is 42")
    assert "must be accurate" in prompt and "the answer is 42" in prompt


def test_parse_judge_output_valid_and_invalid():
    score, expl, valid = parse_judge_output('{"score": 8, "explanation": "ok"}')
    assert valid and score == 8 and expl == "ok"
    _, _, valid2 = parse_judge_output("not json")
    assert not valid2


def test_criteria_judge_normalizes_score():
    judge = CriteriaInjectedRewardModel(
        llm_fn=lambda p: '{"score": 5, "explanation": "mid"}', config=RewardConfig()
    )
    result = judge.score("criteria", "response")
    assert abs(result.score - 0.5) < 1e-9 and result.valid_json


def test_regularization_plot_penalty():
    reg = Regularizer(RewardConfig())
    b = reg.compute("plt.plot([1,2]); plt.savefig('x.png')", is_plot=True, need_plot=False)
    assert b.plot == RewardConfig().reg_plot_weight
    assert reg.penalty("plt.plot([1,2])", is_plot=True, need_plot=False) < 0


def test_regularization_no_plot_penalty_when_needed():
    reg = Regularizer(RewardConfig())
    b = reg.compute("plt.plot([1,2])", is_plot=True, need_plot=True)
    assert b.plot == 0.0


def test_regularization_repetition_and_wait():
    reg = Regularizer(RewardConfig())
    b = reg.compute("wait wait wait wait wait")
    assert b.wait_token > 0


def test_aggregator_total_return():
    traj = Trajectory(sample_id="1", question="q", table_ref="t", need_plot=False)
    traj.add_step(Step(kind=STEP_TOOL_CALL, content="print(1)", exec_success=True, function_correct=True))
    traj.add_step(Step(kind=STEP_TOOL_RESPONSE, content="1"))
    traj.add_step(Step(kind=STEP_ANSWER, content="1"))
    agg = RewardAggregator()
    ret = agg.aggregate(traj, terminal_reward=1.0)
    # +0.1 process + 1.0 terminal = 1.1
    assert abs(ret.total - 1.1) < 1e-9
