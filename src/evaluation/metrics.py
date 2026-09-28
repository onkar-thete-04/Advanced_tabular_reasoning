"""Evaluation metrics (Section 4).

Implements: Acc, EX (execution accuracy), Rge (Rouge), EM (exact match),
GPT (LLM-judged), ECR (chart-generation metric).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from src.data.labeling import normalize_answer


def accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, r in zip(predictions, references) if normalize_answer(p) == normalize_answer(r)
    )
    return correct / len(predictions)


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """EM: identical to accuracy for normalized exact string match."""
    return accuracy(predictions, references)


def execution_accuracy(
    predictions: Sequence[str], references: Sequence[str], executor=None
) -> float:
    """EX: fraction of predictions whose execution output matches the reference."""
    from src.execution.executor import PythonExecutor

    executor = executor or PythonExecutor()
    if not predictions:
        return 0.0
    correct = 0
    for code, ref in zip(predictions, references):
        res = executor.execute(code)
        if res.success and normalize_answer(res.stdout) == normalize_answer(ref):
            correct += 1
    return correct / len(predictions)


def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Rge: mean Rouge-L F1 (uses rouge_score if available, else LCS fallback)."""
    if not predictions:
        return 0.0
    try:
        from rouge_score import rouge_scorer  # type: ignore

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)
        ]
        return sum(scores) / len(scores)
    except Exception:
        return sum(_lcs_f1(p, r) for p, r in zip(predictions, references)) / len(predictions)


def _lcs_f1(pred: str, ref: str) -> float:
    a, b = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def gpt_eval(predictions: Sequence[str], judge_fn: Callable[[str], float]) -> float:
    """GPT: mean judge score over predictions."""
    if not predictions:
        return 0.0
    return sum(float(judge_fn(p)) for p in predictions) / len(predictions)


def ecr(predictions: Sequence[str], references: Sequence[str]) -> float:
    """ECR (chart generation): normalized match of chart specs.

    Without an image renderer we compare the normalized textual chart
    specification (type + series values).
    """
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, r in zip(predictions, references) if normalize_answer(p) == normalize_answer(r)
    )
    return correct / len(predictions)


METRIC_REGISTRY = {
    "Acc": accuracy,
    "EX": execution_accuracy,
    "Rge": rouge_l,
    "EM": exact_match,
    "ECR": ecr,
}


@dataclass
class MetricResult:
    name: str
    value: float


def compute_metrics(
    metric_names: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str],
    judge_fn: Optional[Callable[[str], float]] = None,
) -> List[MetricResult]:
    results: List[MetricResult] = []
    for name in metric_names:
        if name == "GPT":
            if judge_fn is None:
                continue
            results.append(MetricResult(name, gpt_eval(predictions, judge_fn)))
        elif name in METRIC_REGISTRY:
            results.append(MetricResult(name, METRIC_REGISTRY[name](predictions, references)))
    return results
