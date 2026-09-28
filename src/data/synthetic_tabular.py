"""Deterministic synthetic tabular task for demos and smoke training.

The paper trains on large, largely gated corpora (Spider, BIRD, TableBench ...).
For a self-contained, offline demonstration we generate a small table-arithmetic
task with exact integer answers: the reward signal is unambiguous, the data is
reproducible from a seed, and a training curve is meaningful.

Task: given a small integer CSV table and a question ("sum/count/max/min of
column X"), emit ``<answer>N</answer>``. The terminal reward is exact numeric
match via the existing ``RewardRouter`` (Table-QA-With-Label -> Rule), plus a
small format bonus so early RL has non-zero signal.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from config import RewardConfig
from src.constants import STEP_ANSWER, TaskType
from src.data.schema import Sample, Step, Trajectory
from src.rewards.router import RewardRouter

OPS = ("sum", "count", "max", "min")
COLUMNS = ("a", "b", "c")

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d+")


@dataclass
class TabularItem:
    sample_id: str
    question: str
    table_csv: str
    column: str
    op: str
    values: List[int]
    reference: str

    @property
    def signature(self) -> str:
        return f"{self.op}:{self.column}:{','.join(map(str, self.values))}"


def _compute(op: str, values: Sequence[int]) -> int:
    if op == "sum":
        return int(sum(values))
    if op == "count":
        return len(values)
    if op == "max":
        return int(max(values))
    if op == "min":
        return int(min(values))
    raise ValueError(f"unknown op: {op!r}")


def _make_table(rng: random.Random, n_rows: int, n_cols: int = 3):
    header = COLUMNS[:n_cols]
    rows = [[rng.randint(1, 20) for _ in range(n_cols)] for _ in range(n_rows)]
    lines = [",".join(header)] + [",".join(map(str, r)) for r in rows]
    return "\n".join(lines), header, rows


def make_items(
    n: int,
    seed: int,
    id_prefix: str = "train",
    exclude: Optional[Set[str]] = None,
) -> List[TabularItem]:
    """Generate ``n`` deterministic items, skipping excluded signatures."""
    rng = random.Random(seed)
    exclude = set(exclude or ())
    items: List[TabularItem] = []
    attempts = 0
    while len(items) < n and attempts < n * 50:
        attempts += 1
        n_rows = rng.randint(3, 6)
        table_csv, header, rows = _make_table(rng, n_rows)
        col_idx = rng.randrange(len(header))
        column = header[col_idx]
        values = [r[col_idx] for r in rows]
        op = rng.choice(OPS)
        sig = f"{op}:{column}:{','.join(map(str, values))}"
        if sig in exclude:
            continue
        reference = str(_compute(op, values))
        items.append(
            TabularItem(
                sample_id=f"{id_prefix}-{len(items)}",
                question=f"What is the {op} of column {column}?",
                table_csv=table_csv,
                column=column,
                op=op,
                values=values,
                reference=reference,
            )
        )
        exclude.add(sig)
    return items


def item_prompt(item: TabularItem) -> str:
    """Prompt shared by SFT and RL (must stay identical for both)."""
    return f"{item.question}\n{item.table_csv}"


def item_meta(item: TabularItem) -> dict:
    return {"sample_id": item.sample_id, "reference": item.reference}


def items_to_samples(items: Sequence[TabularItem]) -> List[Sample]:
    return [
        Sample(
            sample_id=it.sample_id,
            question=it.question,
            table_ref=it.table_csv,
            reference_answer=it.reference,
            task_type=TaskType.TABLE_QA_WITH_LABEL.value,
        )
        for it in items
    ]


def make_trajectory(item: TabularItem) -> Trajectory:
    """A minimal gold trajectory: just the formatted final answer."""
    return Trajectory(
        sample_id=item.sample_id,
        question=item.question,
        table_ref=item.table_csv,
        task_type=TaskType.TABLE_QA_WITH_LABEL.value,
        final_answer=item.reference,
        steps=[Step(kind=STEP_ANSWER, content=item.reference)],
    )


def make_trajectories(items: Sequence[TabularItem]) -> List[Trajectory]:
    return [make_trajectory(it) for it in items]


def extract_answer(text: str) -> Optional[str]:
    m = _ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else None


def extract_numbers(text: str) -> List[str]:
    """All signed integers appearing in ``text`` (in order)."""
    return _NUMBER_RE.findall(text or "")


def extract_number(text: str) -> Optional[str]:
    nums = extract_numbers(text)
    return nums[0] if nums else None


def _as_int(value) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _numbers_equal(numbers: Sequence[str], reference: str) -> bool:
    ref = _as_int(reference)
    if ref is None:
        return str(reference).strip() in {str(n).strip() for n in numbers}
    return any(_as_int(n) == ref for n in numbers)


def make_reward_fn(
    config: Optional[RewardConfig] = None,
    format_bonus: float = 0.2,
    partial_bonus: float = 0.05,
    unformatted_correct_reward: float = 0.5,
):
    """Return ``reward_fn(meta, response) -> float``.

    Reward tiers (Section 3.3-style shaping so RL escapes a zero-reward cold
    start):

    - ``<answer>`` tag + correct answer  -> 1.0
    - ``<answer>`` tag + wrong answer    -> ``format_bonus`` (0.2)
    - no tag, a number matches reference -> ``unformatted_correct_reward`` (0.5)
    - no tag, some other number          -> ``partial_bonus`` (0.05)
    - no number at all                   -> 0.0

    The untagged tiers matter: without them every no-tag rollout scored 0.0, so
    a whole group shared one reward, group-normalized advantages collapsed to
    zero, and the RL step produced no gradient at all (verified). Shaping keeps
    the group non-degenerate until the policy learns to emit the tag.
    """
    router = RewardRouter(config=config)

    def reward(meta: dict, response: str) -> float:
        reference = str(meta.get("reference", ""))
        answer = extract_answer(response)
        if answer is not None:
            result = router.terminal_reward(
                TaskType.TABLE_QA_WITH_LABEL.value,
                prediction=answer,
                reference=reference,
                numeric=True,
            )
            return float(result.value) if result.value > 0 else float(format_bonus)
        if _numbers_equal(extract_numbers(response), reference):
            return float(unformatted_correct_reward)
        if extract_number(response) is not None:
            return float(partial_bonus)
        return 0.0

    return reward
