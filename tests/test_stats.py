"""Tests for the pure evaluation statistics."""

import pytest

from src.evaluation import stats


def test_wilson_ci_known_value():
    lo, hi = stats.wilson_ci(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3)
    assert hi == pytest.approx(0.943, abs=1e-3)


def test_wilson_ci_bounds_and_empty():
    assert stats.wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = stats.wilson_ci(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= lo < 1.0


def test_pass_at_k():
    assert stats.pass_at_k([True, False], 8) == pytest.approx(0.5)
    assert stats.pass_at_k([False, False], 8) == 0.0
    assert stats.pass_at_k([True], 8) == 1.0
    assert stats.pass_at_k([], 8) == 0.0
    assert stats.pass_at_k([True], 0) == 0.0


def test_mae_rmse():
    mae, rmse = stats.mae_rmse([0, 2])
    assert mae == pytest.approx(1.0)
    assert rmse == pytest.approx(2 ** 0.5)
    assert stats.mae_rmse([]) == (0.0, 0.0)


def test_per_op_accuracy():
    out = stats.per_op_accuracy([("sum", True), ("sum", False), ("max", True)])
    assert out == {"sum": 0.5, "max": 1.0}


def test_mean():
    assert stats.mean([1, 2, 3]) == pytest.approx(2.0)
    assert stats.mean([]) == 0.0
