"""Tests for the shared SFT -> RL orchestration loop."""

from src.training.loop import run_sft_then_rl


class _FakeTrainer:
    def __init__(self):
        self.calls = []

    def train_step(self, prompts, metas):
        self.calls.append((list(prompts), list(metas)))
        return {"step": len(self.calls), "loss": 0.1, "mean_reward": 0.5}


def test_run_sft_then_rl_orders_and_counts():
    order = []
    trainer = _FakeTrainer()
    items = [{"q": f"q{i}", "ref": str(i)} for i in range(4)]

    seen = []
    history = run_sft_then_rl(
        trainer,
        items,
        lambda it: it["q"],
        lambda it: {"reference": it["ref"]},
        rl_steps=3,
        prompts_per_step=2,
        sft_fn=lambda: order.append("sft"),
        on_step=lambda m: seen.append(m["step"]),
    )

    assert order == ["sft"]
    assert len(history) == 3
    assert seen == [1, 2, 3]
    assert all(len(prompts) == 2 for prompts, _ in trainer.calls)
    assert all(len(metas) == 2 for _, metas in trainer.calls)


def test_run_sft_then_rl_empty_items_is_noop():
    trainer = _FakeTrainer()
    history = run_sft_then_rl(
        trainer, [], lambda x: "", lambda x: {}, rl_steps=5
    )
    assert history == []
    assert trainer.calls == []


def test_run_sft_then_rl_zero_rl_steps_runs_sft():
    order = []
    trainer = _FakeTrainer()
    history = run_sft_then_rl(
        trainer,
        [{"q": "q", "ref": "0"}],
        lambda it: it["q"],
        lambda it: {"reference": it["ref"]},
        rl_steps=0,
        sft_fn=lambda: order.append("sft"),
    )
    assert order == ["sft"]
    assert history == []
    assert trainer.calls == []
