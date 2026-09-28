"""Before/after evaluation on the synthetic tabular task.

Compares a base policy against a trained (LoRA-adapted) policy on the held-out
synthetic set, reporting exact-match accuracy and format-validity rate.

The scoring logic is HuggingFace-free and unit-testable; ``hf_policy_fn`` wraps
an ``HFPolicy`` into a ``policy_fn(item) -> str`` used by ``evaluate_items``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

from src.data.synthetic_tabular import TabularItem, extract_answer, item_prompt
from src.rewards.rule_based import numeric_match

PolicyFn = Callable[[TabularItem], str]


@dataclass
class GenMeter:
    """Accumulates generated-token counts and wall time for one arm."""

    calls: int = 0
    tokens: int = 0
    seconds: float = 0.0

    def record(self, tokens: int, seconds: float) -> None:
        self.calls += 1
        self.tokens += int(tokens)
        self.seconds += float(seconds)

    def reset(self) -> None:
        self.calls = 0
        self.tokens = 0
        self.seconds = 0.0

    @property
    def tokens_per_sec(self) -> float:
        return (self.tokens / self.seconds) if self.seconds > 0 else 0.0


def hf_policy_fn(policy, max_new_tokens: int = 64, meter: Optional[GenMeter] = None) -> PolicyFn:
    """Wrap an ``HFPolicy`` into a greedy ``policy_fn(item) -> str``."""
    import torch

    tokenizer = policy.tokenizer
    model = policy.model
    if hasattr(model, "eval"):
        model.eval()

    def fn(item: TabularItem) -> str:
        prompt = item_prompt(item)
        enc = tokenizer(prompt, return_tensors="pt").to(policy.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[0, enc.input_ids.shape[1]:]
        if meter is not None:
            meter.record(int(gen.shape[-1]), time.perf_counter() - t0)
        return tokenizer.decode(gen, skip_special_tokens=True)

    return fn


def hf_sample_fn(
    policy,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    meter: Optional[GenMeter] = None,
):
    """Wrap an ``HFPolicy`` into ``sample_fn(item, k) -> List[str]``."""
    import torch

    tokenizer = policy.tokenizer
    model = policy.model
    if hasattr(model, "eval"):
        model.eval()

    def fn(item: TabularItem, k: int) -> list:
        prompt = item_prompt(item)
        enc = tokenizer(prompt, return_tensors="pt").to(policy.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=int(k),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, enc.input_ids.shape[1]:]
        if meter is not None:
            meter.record(int(gen.numel()), time.perf_counter() - t0)
        return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]

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


def _numeric_error(prediction: str, reference: str) -> Optional[float]:
    try:
        return abs(float(str(prediction).strip()) - float(str(reference).strip()))
    except (TypeError, ValueError):
        return None


def evaluate_arm(
    items: Sequence[TabularItem],
    policy_fn: PolicyFn,
    pass_k: int = 0,
    sample_fn: Optional[Callable[[TabularItem, int], Sequence[str]]] = None,
) -> Dict[str, object]:
    """Greedy metrics always; pass@k metrics when ``pass_k > 0`` and ``sample_fn`` given."""
    from src.evaluation.stats import mae_rmse, per_op_accuracy, wilson_ci

    n = len(items)
    matches = formatted = parsed = 0
    errors: list = []
    op_pairs: list = []
    first_ok: list = []
    any_ok: list = []

    for item in items:
        prediction = policy_fn(item) or ""
        answer = extract_answer(prediction)
        ok = False
        if answer is not None:
            formatted += 1
            err = _numeric_error(answer, item.reference)
            if err is not None:
                parsed += 1
                errors.append(err)
            if numeric_match(answer, item.reference):
                ok = True
                matches += 1
        op_pairs.append((item.op, ok))

        if pass_k > 0 and sample_fn is not None:
            samples = list(sample_fn(item, pass_k) or [])
            oks = []
            for s in samples:
                sa = extract_answer(s)
                oks.append(bool(sa is not None and numeric_match(sa, item.reference)))
            first_ok.append(oks[0] if oks else False)
            any_ok.append(any(oks))

    mae, rmse = mae_rmse(errors)
    metrics: Dict[str, object] = {
        "n": n,
        "accuracy": (matches / n) if n else 0.0,
        "accuracy_ci": list(wilson_ci(matches, n)),
        "format_rate": (formatted / n) if n else 0.0,
        "format_ci": list(wilson_ci(formatted, n)),
        "per_op_accuracy": per_op_accuracy(op_pairs),
        "numeric_parse_rate": (parsed / n) if n else 0.0,
        "mae": mae,
        "rmse": rmse,
    }
    if pass_k > 0 and sample_fn is not None:
        k = len(any_ok)
        metrics["pass_k"] = pass_k
        metrics["pass_at_k"] = (sum(any_ok) / k) if k else 0.0
        metrics["pass_at_1"] = (sum(first_ok) / k) if k else 0.0
        metrics["passk_ci"] = list(wilson_ci(sum(any_ok), k))
    return metrics


def format_benchmark(result: Dict[str, object]) -> str:
    """Render the multi-arm benchmark result as a text table."""
    arms: Dict[str, dict] = result["arms"]
    names = list(arms)
    base = names[0]
    rows = ["accuracy", "format_rate"]
    if "pass_at_k" in arms[base]:
        rows += ["pass_at_k", "pass_at_1"]
    rows += ["mae", "rmse", "numeric_parse_rate"]

    header = f"{'metric':<18}" + "".join(f"{nm:>12}" for nm in names) + f"{'delta':>12}"
    lines = [
        f"held-out items: {result.get('n', 0)}  task: {result.get('task', 'simple')}",
        header,
    ]
    for r in rows:
        cells = "".join(f"{float(arms[nm].get(r, 0.0)):>12.3f}" for nm in names)
        delta = float(arms[names[-1]].get(r, 0.0)) - float(arms[base].get(r, 0.0))
        lines.append(f"{r:<18}{cells}{delta:>+12.3f}")

    lines.append(
        "accuracy 95% CI: "
        + "; ".join(
            f"{nm}={tuple(round(float(x), 3) for x in arms[nm].get('accuracy_ci', (0.0, 0.0)))}"
            for nm in names
        )
    )

    ops = sorted({op for nm in names for op in arms[nm].get("per_op_accuracy", {})})
    for op in ops:
        cells = "".join(
            f"{float(arms[nm].get('per_op_accuracy', {}).get(op, 0.0)):>12.3f}" for nm in names
        )
        lines.append(f"{'op:' + op:<18}{cells}")

    if all("tokens_per_sec" in arms[nm] for nm in names):
        lines.append(
            "tokens/s: " + "; ".join(f"{nm}={float(arms[nm]['tokens_per_sec']):.1f}" for nm in names)
        )
    training = result.get("training")
    if training:
        lines.append("training: " + ", ".join(f"{k}={v}" for k, v in training.items()))
    return "\n".join(lines)


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
