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
AGG_OPS = OPS + ("avg",)
FILTER_OPS = (">", "<", ">=", "<=", "==", "!=")
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
    task: str = "simple"
    filter_col: Optional[str] = None
    filter_op: Optional[str] = None
    filter_threshold: Optional[int] = None
    passing_values: Optional[List[int]] = None

    @property
    def signature(self) -> str:
        base = f"{self.task}:{self.op}:{self.column}:{','.join(map(str, self.values))}"
        if self.filter_col is not None:
            base += f"|{self.filter_col}{self.filter_op}{self.filter_threshold}"
        return base


def _compute(op: str, values: Sequence[int]) -> int:
    if op == "sum":
        return int(sum(values))
    if op == "count":
        return len(values)
    if op == "max":
        return int(max(values))
    if op == "min":
        return int(min(values))
    if op == "avg":
        if not values:
            raise ValueError("avg of an empty sequence")
        total = int(sum(values))
        if total % len(values):
            raise ValueError("avg is not an integer")
        return total // len(values)
    raise ValueError(f"unknown op: {op!r}")


def _passes(value: int, op: str, threshold: int) -> bool:
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    raise ValueError(f"unknown filter op: {op!r}")


def _choose_threshold(rng: random.Random, values: Sequence[int], op: str) -> Optional[int]:
    """Pick a threshold with 1 <= passing <= len(values)-1, or None."""
    n = len(values)
    cands = []
    if op in (">", "<", ">=", "<="):
        for t in range(min(values) - 1, max(values) + 1):
            if 1 <= sum(1 for v in values if _passes(v, op, t)) <= n - 1:
                cands.append(t)
    else:  # "==" or "!=": threshold must be an existing value
        for t in sorted(set(values)):
            if 1 <= sum(1 for v in values if _passes(v, op, t)) <= n - 1:
                cands.append(t)
    return rng.choice(cands) if cands else None


def _make_table(rng: random.Random, n_rows: int, n_cols: int = 3):
    header = COLUMNS[:n_cols]
    rows = [[rng.randint(1, 20) for _ in range(n_cols)] for _ in range(n_rows)]
    lines = [",".join(header)] + [",".join(map(str, r)) for r in rows]
    return "\n".join(lines), header, rows


def _make_item(
    rng: random.Random, sample_id: str, exclude: Set[str], task: str
) -> Optional[TabularItem]:
    if task == "filter_aggregate":
        n_rows = rng.randint(5, 8)
        table_csv, header, rows = _make_table(rng, n_rows)
        fcol_idx = rng.randrange(len(header))
        fcol = header[fcol_idx]
        fcol_vals = [r[fcol_idx] for r in rows]
        fop = rng.choice(FILTER_OPS)
        thr = _choose_threshold(rng, fcol_vals, fop)
        if thr is None:
            return None
        agg = rng.choice(AGG_OPS)
        agg_idx = rng.randrange(len(header))
        passing = [
            r[agg_idx] for r in rows if _passes(r[fcol_idx], fop, thr)
        ]
        if not (1 <= len(passing) <= n_rows - 1):
            return None
        try:
            ref = _compute(agg, passing)
        except ValueError:
            return None
        agg_col = header[agg_idx]
        values = [r[agg_idx] for r in rows]
        sig = (
            f"filter_aggregate:{agg}:{agg_col}:{','.join(map(str, values))}"
            f"|{fcol}{fop}{thr}"
        )
        if sig in exclude:
            return None
        return TabularItem(
            sample_id=sample_id,
            question=(
                f"What is the {agg} of column {agg_col} for rows where "
                f"column {fcol} {fop} {thr}?"
            ),
            table_csv=table_csv,
            column=agg_col,
            op=agg,
            values=values,
            reference=str(ref),
            task="filter_aggregate",
            filter_col=fcol,
            filter_op=fop,
            filter_threshold=thr,
            passing_values=passing,
        )

    # simple (default): RNG call order preserved from the original generator
    n_rows = rng.randint(3, 6)
    table_csv, header, rows = _make_table(rng, n_rows)
    col_idx = rng.randrange(len(header))
    column = header[col_idx]
    values = [r[col_idx] for r in rows]
    op = rng.choice(OPS)
    sig = f"simple:{op}:{column}:{','.join(map(str, values))}"
    if sig in exclude:
        return None
    return TabularItem(
        sample_id=sample_id,
        question=f"What is the {op} of column {column}?",
        table_csv=table_csv,
        column=column,
        op=op,
        values=values,
        reference=str(_compute(op, values)),
        task="simple",
    )


def make_items(
    n: int,
    seed: int,
    id_prefix: str = "train",
    exclude: Optional[Set[str]] = None,
    task: str = "simple",
) -> List[TabularItem]:
    """Generate ``n`` deterministic items, skipping excluded signatures."""
    if task not in ("simple", "filter_aggregate"):
        raise ValueError(f"unknown task: {task!r}")
    rng = random.Random(seed)
    exclude = set(exclude or ())
    items: List[TabularItem] = []
    attempts = 0
    while len(items) < n and attempts < n * 200:
        attempts += 1
        it = _make_item(rng, f"{id_prefix}-{len(items)}", exclude, task)
        if it is None:
            continue
        items.append(it)
        exclude.add(it.signature)
    return items


def item_prompt(item: TabularItem) -> str:
    """Prompt shared by SFT and RL (must stay identical for both)."""
    return f"{item.question}\n{item.table_csv}"


def item_meta(item: TabularItem) -> dict:
    meta = {"sample_id": item.sample_id, "reference": item.reference}
    if item.passing_values is not None:
        meta["passing_values"] = list(item.passing_values)
    return meta


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
