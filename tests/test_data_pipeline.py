"""Tests for the data engineering pipeline (Section 2 / 3.2)."""

from config import DataConfig
from src.constants import (
    DIFFICULTY_HIGH, DIFFICULTY_LOW, DIFFICULTY_MEDIUM,
    STEP_THINK, STEP_TOOL_CALL, STEP_TOOL_RESPONSE, STEP_ANSWER,
)
from src.data.schema import Sample, Step, Trajectory
from src.data.filtering import (
    length_filter, dedup, rule_clean, language_align, output_quality_filter,
    estimate_difficulty, classify_difficulty, resample_difficulty, run_input_filtering,
)
from src.data.composition import compose
from src.data.synthesis import validate_trajectory, AgenticDataSynthesizer
from src.data.labeling import ConsensusLabeler, extract_final_answer
from src.data.augmentation import present_table, alternate_special_tokens


def _sample(i, question="q", lang="en", out_lang="en", **kw):
    return Sample(
        sample_id=f"s{i}", question=question, table_ref="t", reference_answer="a",
        language=lang, output_language=out_lang, **kw,
    )


def test_length_filter():
    short = _sample(0, "hello world")
    long = _sample(1, "word " * 20000)
    kept = length_filter([short, long], max_tokens=100)
    assert short in kept and long not in kept


def test_dedup_removes_near_duplicates():
    cfg = DataConfig()
    a = _sample(0, "What is the total revenue for 2023?")
    b = _sample(1, "what is the total revenue for 2023?")
    c = _sample(2, "List all employees hired after 2020.")
    out = dedup([a, b, c], cfg)
    assert len(out) == 2


def test_rule_clean_removes_harmful_and_identity():
    harmful = _sample(0, "How to make a bomb")
    identity = _sample(1, "I am an AI language model")
    good = _sample(2, "What is the average?")
    out = rule_clean([harmful, identity, good])
    assert out == [good]


def test_language_align():
    ok = _sample(0, "q", lang="en", out_lang="en")
    bad = _sample(1, "q", lang="zh", out_lang="en")
    assert language_align([ok, bad]) == [ok]


def test_output_quality_filter_threshold():
    samples = [_sample(0), _sample(1)]
    scores = {"s0": 0.9, "s1": 0.5}
    out = output_quality_filter(samples, lambda s: scores[s.sample_id], threshold=0.7)
    assert [s.sample_id for s in out] == ["s0"]


def test_difficulty_classification_boundaries():
    cfg = DataConfig()
    assert classify_difficulty(0.2, cfg) == DIFFICULTY_HIGH
    assert classify_difficulty(0.5, cfg) == DIFFICULTY_MEDIUM
    assert classify_difficulty(0.9, cfg) == DIFFICULTY_LOW


def test_estimate_difficulty_average():
    s = _sample(0)
    m = estimate_difficulty(s, lambda _: "r", lambda _s, _r: 0.4, num_rollouts=5)
    assert abs(m - 0.4) < 1e-9


def test_resample_distribution():
    cfg = DataConfig()
    samples = []
    for i in range(30):
        s = _sample(i)
        s.difficulty = DIFFICULTY_HIGH if i < 10 else (DIFFICULTY_MEDIUM if i < 20 else DIFFICULTY_LOW)
        samples.append(s)
    out = resample_difficulty(samples, cfg, seed=0)
    counts = {d: 0 for d in (DIFFICULTY_HIGH, DIFFICULTY_MEDIUM, DIFFICULTY_LOW)}
    for s in out:
        counts[s.difficulty] += 1
    total = len(out)
    assert abs(counts[DIFFICULTY_MEDIUM] / total - 0.6) < 0.1


def test_composition_ratios():
    pools = {
        "general": [_sample(i, "g") for i in range(100)],
        "math_logic": [_sample(i, "m") for i in range(100)],
        "code": [_sample(i, "c") for i in range(100)],
        "table_task": [_sample(i, "t") for i in range(100)],
        "sql": [_sample(i, "s") for i in range(100)],
        "qa_with_answer": [_sample(i, "a") for i in range(100)],
        "table_agent": [_sample(i, "ta") for i in range(100)],
    }
    out = compose(pools, total=100, seed=1)
    assert len(out) == 100


def test_trajectory_validation_ok():
    traj = Trajectory(sample_id="1", question="q", table_ref="t")
    traj.add_step(Step(kind=STEP_THINK, content="plan"))
    traj.add_step(Step(kind=STEP_TOOL_CALL, content="print(1)", exec_success=True, function_correct=True))
    traj.add_step(Step(kind=STEP_TOOL_RESPONSE, content="1"))
    traj.add_step(Step(kind=STEP_ANSWER, content="1"))
    result = validate_trajectory(traj)
    assert result.ok, result.errors


def test_trajectory_validation_rejects_missing_answer():
    traj = Trajectory(sample_id="1", question="q", table_ref="t")
    traj.add_step(Step(kind=STEP_TOOL_CALL, content="print(1)"))
    traj.add_step(Step(kind=STEP_TOOL_RESPONSE, content="1"))
    result = validate_trajectory(traj)
    assert not result.ok


def test_trajectory_validation_rejects_error_before_answer():
    traj = Trajectory(sample_id="1", question="q", table_ref="t")
    traj.add_step(Step(kind=STEP_TOOL_CALL, content="print(1)"))
    traj.add_step(Step(kind=STEP_TOOL_RESPONSE, content="Traceback: ValueError: bad"))
    traj.add_step(Step(kind=STEP_ANSWER, content="1"))
    result = validate_trajectory(traj)
    assert not result.ok


def test_trajectory_validation_rejects_read_csv_after_first_round():
    traj = Trajectory(sample_id="1", question="q", table_ref="t")
    traj.add_step(Step(kind=STEP_TOOL_CALL, content="print(1)"))
    traj.add_step(Step(kind=STEP_TOOL_RESPONSE, content="1"))
    traj.add_step(Step(kind=STEP_TOOL_CALL, content="pd.read_csv('x.csv')"))
    traj.add_step(Step(kind=STEP_ANSWER, content="1"))
    result = validate_trajectory(traj)
    assert not result.ok


def test_synthesizer_returns_none_on_invalid():
    synth = AgenticDataSynthesizer()
    s = _sample(0)
    steps = [Step(kind=STEP_TOOL_CALL, content="print(1)"), Step(kind=STEP_TOOL_RESPONSE, content="1")]
    assert synth.synthesize(s, steps) is None


def test_consensus_labeler():
    models = {
        "a": lambda q: "<answer>42</answer>",
        "b": lambda q: "<answer>42</answer>",
        "c": lambda q: "<answer>7</answer>",
    }
    labeler = ConsensusLabeler(models)
    result = labeler.label("q")
    assert result.kept and result.answer == "42" and result.num_agree == 2


def test_consensus_labeler_rejects_no_agreement():
    models = {
        "a": lambda q: "<answer>1</answer>",
        "b": lambda q: "<answer>2</answer>",
        "c": lambda q: "<answer>3</answer>",
    }
    assert not ConsensusLabeler(models).label("q").kept


def test_extract_final_answer():
    assert extract_final_answer("<answer> hi </answer>") == "hi"


def test_table_input_diversity_path():
    s = _sample(0)
    s.metadata["table_path"] = "/data/x.csv"
    present_table(s, "static", force_mode="path")
    assert s.table_ref == "/data/x.csv"


def test_special_token_alternation():
    text = "<tool_call>code</tool_call>"
    out = alternate_special_tokens(text)
    assert out == text or out == "<function_call>code</function_call>"
