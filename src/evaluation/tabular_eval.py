"""Before/after evaluation on the synthetic tabular task.

Compares a base policy against a trained (LoRA-adapted) policy on the held-out
synthetic set, reporting exact-match accuracy and format-validity rate.

The scoring logic is HuggingFace-free and unit-testable; ``hf_policy_fn`` wraps
an ``HFPolicy`` into a ``policy_fn(item) -> str`` used by ``evaluate_items``.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, Sequence

from src.data.synthetic_tabular import TabularItem, extract_answer, item_prompt
from src.rewards.rule_based import numeric_match

PolicyFn = Callable[[TabularItem], str]


def hf_policy_fn(policy, max_new_tokens: int = 64) -> PolicyFn:
    """Wrap an ``HFPolicy`` into a greedy ``policy_fn(item) -> str``."""
    import torch

    tokenizer = policy.tokenizer
    model = policy.model
    if hasattr(model, "eval"):
        model.eval()

    def fn(item: TabularItem) -> str:
        prompt = item_prompt(item)
        enc = tokenizer(prompt, return_tensors="pt").to(policy.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            out[0, enc.input_ids.shape[1]:], skip_special_tokens=True
        )

    return fn


def evaluate_items(items: Sequence[TabularItem], policy_fn: PolicyFn) -> Dict[str, float]:
    """Return accuracy and format-validity rate over ``items``."""
    n = len(items)
    matches = 0
    formatted = 0
    for item in items:
        prediction = policy_fn(item) or ""
        answer = extract_answer(prediction)
        if answer is not None:
            formatted += 1
            if numeric_match(answer, item.reference):
                matches += 1
    return {
        "n": n,
        "accuracy": (matches / n) if n else 0.0,
        "format_rate": (formatted / n) if n else 0.0,
    }


def compare(base: Dict[str, float], trained: Dict[str, float]) -> Dict[str, dict]:
    deltas = {k: trained[k] - base[k] for k in ("accuracy", "format_rate")}
    return {"base": base, "trained": trained, "delta": deltas}


def format_report(result: Dict[str, dict]) -> str:
    base, trained, delta = result["base"], result["trained"], result["delta"]
    lines = [
        f"held-out items: {base['n']}",
        f"{'metric':<13}{'base':>8}{'trained':>10}{'delta':>9}",
        f"{'accuracy':<13}{base['accuracy']:>8.3f}{trained['accuracy']:>10.3f}{delta['accuracy']:>+9.3f}",
        f"{'format_rate':<13}{base['format_rate']:>8.3f}{trained['format_rate']:>10.3f}{delta['format_rate']:>+9.3f}",
    ]
    return "\n".join(lines)


def write_report(path: str, result: Dict[str, dict]) -> None:
    import os

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
