"""Data composition and mixture management (Section 2.3, Figure 3).

Final composition ratios:
    General 5%, Math & Logic 20%, Code 5%, Table Task 30%,
    SQL 10%, QA with Answer 20%, Table Agent 10%
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

from src.constants import DATA_COMPOSITION
from src.data.schema import Sample


def compose(
    samples_by_category: Dict[str, Sequence[Sample]],
    ratios: Dict[str, float] = DATA_COMPOSITION,
    total: int | None = None,
    seed: int = 42,
) -> List[Sample]:
    """Mix category pools into the target ratio.

    Args:
        samples_by_category: category -> list of samples.
        ratios: category -> fraction (should sum to 1.0).
        total: target mixture size; defaults to the largest category size.
        seed: RNG seed.
    """
    rng = random.Random(seed)
    if total is None:
        total = max((len(v) for v in samples_by_category.values()), default=0)
    out: List[Sample] = []
    for category, ratio in ratios.items():
        pool = list(samples_by_category.get(category, []))
        if not pool:
            continue
        target = int(round(total * ratio))
        if target <= 0:
            continue
        if len(pool) >= target:
            out.extend(rng.sample(pool, target))
        else:
            out.extend(pool)
            out.extend(rng.choices(pool, k=target - len(pool)))
    rng.shuffle(out)
    return out


def actual_distribution(samples: Sequence[Sample], key: str = "task_type") -> Dict[str, float]:
    """Report the realized distribution of a categorical field."""
    counts: Dict[str, int] = {}
    for s in samples:
        value = str(getattr(s, key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}
