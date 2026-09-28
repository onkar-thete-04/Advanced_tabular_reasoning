"""Rule-based reward function (Section 3.4.2).

For tasks with clear labels, deterministic terminal verification is used:
exact matching or execution-based verification (e.g., SQL execution followed by
rule extraction), producing high-precision supervision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from config import RewardConfig
from src.data.labeling import normalize_answer
from src.execution.executor import ExecutionResult, PythonExecutor


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def numeric_match(prediction: str, reference: str, tol: float = 1e-6) -> bool:
    """Compare the last number appearing in each string."""
    def _last_number(text: str) -> Optional[float]:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return float(nums[-1]) if nums else None

    p, r = _last_number(prediction), _last_number(reference)
    if p is None or r is None:
        return False
    return abs(p - r) <= tol


@dataclass
class RuleReward:
    correct: bool
    score: float
    detail: str = ""


class RuleBasedReward:
    """Deterministic reward for label-bearing tasks."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def _score(self, correct: bool, detail: str = "") -> RuleReward:
        value = (
            self.config.rule_correct_reward if correct else self.config.rule_incorrect_reward
        )
        return RuleReward(correct=correct, score=value, detail=detail)

    def score_answer(self, prediction: str, reference: str, numeric: bool = False) -> RuleReward:
        correct = numeric_match(prediction, reference) if numeric else exact_match(prediction, reference)
        return self._score(correct, "exact/numeric match")

    def score_execution(
        self,
        code: str,
        reference: str,
        executor: Optional[PythonExecutor] = None,
        numeric: bool = False,
    ) -> RuleReward:
        """Run & Rule: execute code, then rule-extract from stdout."""
        executor = executor or PythonExecutor()
        result: ExecutionResult = executor.execute(code)
        if not result.success:
            return self._score(False, f"execution failed: {result.error}")
        correct = numeric_match(result.stdout, reference) if numeric else exact_match(result.stdout, reference)
        return self._score(correct, "execution output match")

    def score_sql(self, prediction: str, reference: str, executor: Optional[PythonExecutor] = None) -> RuleReward:
        """Run & Rule for SQL. Falls back to normalized SQL string match.

        A real database executor can be plugged in; without one we compare the
        normalized query text.
        """
        if executor is None:
            return self.score_answer(prediction, reference)
        return self.score_execution(prediction, reference, executor)
