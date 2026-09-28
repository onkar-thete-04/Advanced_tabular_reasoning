"""Shared SFT -> RL orchestration.

Kept free of any HuggingFace import so it can be unit-tested with fake
policies/trainers on CPU. ``kaggle/train_kaggle.py`` (and any future ``main.py``
command) call this so the run logic lives in exactly one place.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence


def run_sft_then_rl(
    trainer,
    items: Sequence,
    prompt_builder: Callable[[object], str],
    meta_builder: Callable[[object], dict],
    rl_steps: int = 20,
    prompts_per_step: int = 1,
    sft_fn: Optional[Callable[[], None]] = None,
    on_step: Optional[Callable[[Dict[str, float]], None]] = None,
) -> List[Dict[str, float]]:
    """Run an optional SFT warm-up, then ``rl_steps`` RL updates.

    Returns the list of per-step metric dicts produced by ``trainer.train_step``.
    ``sft_fn`` (if given) always runs first, giving the policy format alignment
    before RL — the paper's Section 3.5 ordering.
    """
    if sft_fn is not None:
        sft_fn()

    history: List[Dict[str, float]] = []
    n = len(items)
    if n == 0:
        return history
    prompts_per_step = max(1, int(prompts_per_step))

    for step in range(int(rl_steps)):
        batch = [
            items[(step * prompts_per_step + j) % n] for j in range(prompts_per_step)
        ]
        metrics = trainer.train_step(
            [prompt_builder(it) for it in batch],
            [meta_builder(it) for it in batch],
        )
        history.append(metrics)
        if on_step is not None:
            on_step(metrics)
    return history
