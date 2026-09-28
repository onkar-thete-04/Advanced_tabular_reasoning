"""Tests for the before/after tabular evaluation."""

import json

import pytest

from src.data.synthetic_tabular import make_items
from src.evaluation.tabular_eval import (
    compare,
    evaluate_items,
    format_report,
    write_report,
)


def test_evaluate_items_all_correct():
    items = make_items(10, seed=2, id_prefix="eval")
    metrics = evaluate_items(items, lambda it: f"<answer>{it.reference}</answer>")
    assert metrics == {"n": 10, "accuracy": 1.0, "format_rate": 1.0}


def test_evaluate_items_unformatted():
    items = make_items(4, seed=2, id_prefix="eval")
    metrics = evaluate_items(items, lambda it: "no answer here")
    assert metrics["accuracy"] == 0.0
    assert metrics["format_rate"] == 0.0


def test_evaluate_items_wrong_but_formatted():
    items = make_items(4, seed=2, id_prefix="eval")
    metrics = evaluate_items(items, lambda it: "<answer>999999</answer>")
    assert metrics["accuracy"] == 0.0
    assert metrics["format_rate"] == 1.0


def test_compare_delta():
    base = {"n": 4, "accuracy": 0.0, "format_rate": 0.5}
    trained = {"n": 4, "accuracy": 0.5, "format_rate": 1.0}
    result = compare(base, trained)
    assert result["delta"]["accuracy"] == pytest.approx(0.5)
    assert result["delta"]["format_rate"] == pytest.approx(0.5)


def test_format_report_and_write(tmp_path):
    result = compare(
        {"n": 8, "accuracy": 0.1, "format_rate": 0.2},
        {"n": 8, "accuracy": 0.6, "format_rate": 0.9},
    )
    text = format_report(result)
    assert "accuracy" in text and "format_rate" in text and "base" in text

    path = tmp_path / "report.json"
    write_report(str(path), result)
    assert json.loads(path.read_text(encoding="utf-8"))["delta"]["accuracy"] == pytest.approx(0.5)


def test_main_eval_tabular_mock(capsys):
    from main import main

    rc = main(["eval-tabular", "--mock", "--n", "8"])
    assert rc == 0
    assert "accuracy" in capsys.readouterr().out


def test_main_eval_tabular_mock_hard_task(capsys):
    from main import main

    rc = main(["eval-tabular", "--mock", "--n", "8", "--task", "filter_aggregate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "accuracy" in out


def test_evaluate_arm_metrics():
    from src.evaluation.tabular_eval import evaluate_arm

    items = make_items(4, seed=2, id_prefix="eval")
    m = evaluate_arm(items, lambda it: f"<answer>{it.reference}</answer>")
    assert m["n"] == 4
    assert m["accuracy"] == 1.0
    assert m["format_rate"] == 1.0
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["numeric_parse_rate"] == 1.0
    assert set(m["per_op_accuracy"]) <= {"sum", "count", "max", "min"}
    lo, hi = m["accuracy_ci"]
    assert 0.0 <= lo <= hi <= 1.0


def test_evaluate_arm_mae_on_wrong_answers():
    from src.evaluation.tabular_eval import evaluate_arm

    items = make_items(4, seed=2, id_prefix="eval")
    m = evaluate_arm(items, lambda it: f"<answer>{int(it.reference) + 3}</answer>")
    assert m["accuracy"] == 0.0
    assert m["format_rate"] == 1.0
    assert m["mae"] == pytest.approx(3.0)
    assert m["rmse"] == pytest.approx(3.0)


def test_evaluate_arm_pass_at_k():
    from src.evaluation.tabular_eval import evaluate_arm

    items = make_items(4, seed=2, id_prefix="eval")
    m0 = evaluate_arm(
        items, lambda it: "<answer>999999</answer>", pass_k=8,
        sample_fn=lambda it, k: ["<answer>999999</answer>"] * k,
    )
    assert m0["pass_at_k"] == 0.0
    assert m0["pass_at_1"] == 0.0
    assert m0["pass_k"] == 8

    m1 = evaluate_arm(
        items, lambda it: "<answer>999999</answer>", pass_k=8,
        sample_fn=lambda it, k: [f"<answer>{it.reference}</answer>"] * k,
    )
    assert m1["pass_at_k"] == 1.0
    assert m1["pass_at_1"] == 1.0


def test_evaluate_arm_omits_pass_metrics_when_off():
    from src.evaluation.tabular_eval import evaluate_arm

    items = make_items(3, seed=2, id_prefix="eval")
    m = evaluate_arm(items, lambda it: f"<answer>{it.reference}</answer>")
    assert "pass_at_k" not in m


def test_format_benchmark():
    from src.evaluation.tabular_eval import format_benchmark

    result = {
        "task": "simple",
        "n": 4,
        "arms": {
            "base": {"accuracy": 0.0, "format_rate": 0.0, "mae": 0.0, "rmse": 0.0,
                     "numeric_parse_rate": 0.0, "per_op_accuracy": {},
                     "accuracy_ci": (0.0, 0.0), "format_ci": (0.0, 0.0)},
            "sft+rl": {"accuracy": 1.0, "format_rate": 1.0, "mae": 0.0, "rmse": 0.0,
                       "numeric_parse_rate": 1.0, "per_op_accuracy": {},
                       "accuracy_ci": (0.0, 1.0), "format_ci": (0.0, 1.0)},
        },
    }
    text = format_benchmark(result)
    assert "base" in text and "sft+rl" in text and "accuracy" in text


def test_main_eval_tabular_mock_three_arms(capsys):
    from main import main

    rc = main(["eval-tabular", "--mock", "--n", "8", "--task", "filter_aggregate",
               "--pass-k", "8"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "base" in out and "sft" in out and "sft+rl" in out
    assert "pass_at_k" in out


def test_main_eval_tabular_mock_writes_json(tmp_path, capsys):
    import json as _json
    from main import main

    out = tmp_path / "eval.json"
    rc = main(["eval-tabular", "--mock", "--n", "6", "--pass-k", "8",
               "--json", str(out)])
    assert rc == 0
    data = _json.loads(out.read_text(encoding="utf-8"))
    assert set(data["arms"]) == {"base", "sft", "sft+rl"}
    assert data["arms"]["sft+rl"]["accuracy"] == 1.0


def test_load_run_info_malformed_json_returns_none(tmp_path):
    from main import _load_run_info

    (tmp_path / "run_info.json").write_text("{not valid json", encoding="utf-8")
    assert _load_run_info(str(tmp_path / "rl_adapter")) is None
