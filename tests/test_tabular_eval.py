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
