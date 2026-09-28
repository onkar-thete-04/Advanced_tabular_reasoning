"""Tests for the deterministic synthetic tabular task."""

import pytest
import torch

from src.data.synthetic_tabular import (
    _compute,
    extract_answer,
    item_meta,
    item_prompt,
    make_items,
    make_reward_fn,
    make_trajectories,
)
from src.training.sft import trajectory_to_sft_example


def test_make_items_deterministic():
    a = [(i.signature, i.reference) for i in make_items(20, seed=7)]
    b = [(i.signature, i.reference) for i in make_items(20, seed=7)]
    assert a == b


def test_references_are_correct():
    for it in make_items(50, seed=1):
        assert it.reference == str(_compute(it.op, it.values))
        assert it.op in {"sum", "count", "max", "min"}


def test_train_eval_disjoint():
    train = make_items(40, seed=1, id_prefix="train")
    held = make_items(
        40, seed=999, id_prefix="eval", exclude={i.signature for i in train}
    )
    assert len(held) == 40
    assert not ({i.signature for i in train} & {i.signature for i in held})


def test_prompt_and_trajectory_share_format():
    it = make_items(1, seed=3)[0]
    example = trajectory_to_sft_example(make_trajectories([it])[0])
    assert example.prompt == item_prompt(it)
    assert example.completion == f"<answer>{it.reference}</answer>"


def test_extract_answer():
    assert extract_answer("junk <answer> 60 </answer> tail") == "60"
    assert extract_answer("no answer here") is None
    assert extract_answer("") is None


def test_reward_fn_correct_wrong_and_missing():
    it = make_items(1, seed=5)[0]
    reward = make_reward_fn()
    meta = item_meta(it)
    assert reward(meta, f"<answer>{it.reference}</answer>") == pytest.approx(1.0)
    assert reward(meta, "<answer>999999</answer>") == pytest.approx(0.2)
    assert reward(meta, "no answer here") == pytest.approx(0.0)


def test_extract_number():
    from src.data.synthetic_tabular import extract_number, extract_numbers

    assert extract_number("sum is 42.") == "42"
    assert extract_number("no digits here") is None
    assert extract_numbers("1 and 2 and 3") == ["1", "2", "3"]


def test_reward_cold_start_credit_without_tag():
    it = make_items(1, seed=5)[0]
    reward = make_reward_fn()
    meta = item_meta(it)
    assert reward(meta, f"the answer is {it.reference}") == pytest.approx(0.5)
    assert reward(meta, "the answer is 999999") == pytest.approx(0.05)
    assert reward(meta, "i cannot answer") == pytest.approx(0.0)


def test_reward_creates_group_variance_for_formatless_policy():
    from src.rl.advantage import group_advantage

    it = make_items(1, seed=5)[0]
    meta = item_meta(it)
    reward = make_reward_fn()
    responses = [
        f"the answer is {it.reference}",
        "the answer is 999999",
        "no idea",
        f"<answer>{it.reference}</answer>",
        "<answer>999999</answer>",
    ]
    rewards = torch.tensor([reward(meta, r) for r in responses])
    assert len(set(rewards.tolist())) > 1
    adv = group_advantage(rewards)
    assert float(adv.abs().sum()) > 0.0


def test_filter_aggregate_selectivity():
    items = make_items(40, seed=12, task="filter_aggregate")
    assert len(items) == 40
    for it in items:
        n_rows = len(it.table_csv.splitlines()) - 1
        assert 1 <= len(it.passing_values) <= n_rows - 1


def test_filter_aggregate_reference_is_correct():
    from src.data.synthetic_tabular import _passes

    for it in make_items(40, seed=11, task="filter_aggregate"):
        lines = it.table_csv.splitlines()
        header = lines[0].split(",")
        rows = [list(map(int, line.split(","))) for line in lines[1:]]
        fi = header.index(it.filter_col)
        ai = header.index(it.column)
        passing = [
            r[ai] for r in rows if _passes(r[fi], it.filter_op, it.filter_threshold)
        ]
        assert passing == it.passing_values
        assert it.reference == str(_compute(it.op, passing))


def test_filter_aggregate_avg_is_integer():
    avgs = [
        it for it in make_items(200, seed=13, task="filter_aggregate")
        if it.op == "avg"
    ]
    assert avgs, "generator produced no avg items"
    for it in avgs:
        assert sum(it.passing_values) % len(it.passing_values) == 0
        assert it.reference == str(sum(it.passing_values) // len(it.passing_values))


def test_filter_aggregate_deterministic_and_disjoint():
    a = [(i.signature, i.reference) for i in make_items(30, seed=5, task="filter_aggregate")]
    b = [(i.signature, i.reference) for i in make_items(30, seed=5, task="filter_aggregate")]
    assert a == b

    train = make_items(40, seed=1, task="filter_aggregate", id_prefix="train")
    held = make_items(
        40, seed=9, task="filter_aggregate", id_prefix="eval",
        exclude={i.signature for i in train},
    )
    assert len(held) == 40
    assert not ({i.signature for i in train} & {i.signature for i in held})
