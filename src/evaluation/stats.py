"""Dependency-free statistics for the tabular benchmark."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple


def mean(values: Sequence[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion, clamped to [0, 1]."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pass_at_k(correct: Sequence[bool], k: int) -> float:
    """Fraction of items with at least one correct sample among k."""
    flags = list(correct)
    if not flags or k <= 0:
        return 0.0
    return sum(1 for c in flags if c) / len(flags)


def mae_rmse(errors: Sequence[float]) -> Tuple[float, float]:
    vals = list(errors)
    if not vals:
        return (0.0, 0.0)
    mae = sum(abs(v) for v in vals) / len(vals)
    rmse = math.sqrt(sum(v * v for v in vals) / len(vals))
    return (mae, rmse)


def per_op_accuracy(pairs: Sequence[Tuple[str, bool]]) -> Dict[str, float]:
    grouped: Dict[str, list] = {}
    for op, ok in pairs:
        grouped.setdefault(op, []).append(1.0 if ok else 0.0)
    return {op: sum(vals) / len(vals) for op, vals in grouped.items()}
