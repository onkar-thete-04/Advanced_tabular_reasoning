"""Policy regularization R_reg(t) (Section 3.4.3).

A composite penalty applied at every step to prevent degenerate behaviors in
code-execution-centric interactions:
    * length control            (overly verbose intermediate outputs)
    * repetition suppression    (repetitive action patterns)
    * invalid-reflection mitigation (excessive "wait"-like tokens)
    * opportunistic plotting    (need_plot supervision signal)

The paper specifies these qualitatively; weights are configurable defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from config import RewardConfig
from src.data.filtering import token_len

_WAIT_RE = re.compile(r"\bwait\b", re.IGNORECASE)


@dataclass
class RegBreakdown:
    length: float = 0.0
    repetition: float = 0.0
    wait_token: float = 0.0
    plot: float = 0.0

    @property
    def total(self) -> float:
        return self.length + self.repetition + self.wait_token + self.plot


class Regularizer:
    """Computes the unified per-step regularization penalty R_reg(t)."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def _length_penalty(self, text: str) -> float:
        tokens = token_len(text)
        budget = self.config.reg_length_budget_tokens
        if tokens <= budget:
            return 0.0
        return self.config.reg_length_weight * (tokens - budget) / budget

    @staticmethod
    def _repetition_ratio(text: str, n: int = 4) -> float:
        words = text.split()
        if len(words) < n * 2:
            return 0.0
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        if not grams:
            return 0.0
        repeats = len(grams) - len(set(grams))
        return repeats / len(grams)

    def _repetition_penalty(self, text: str) -> float:
        return self.config.reg_repetition_weight * self._repetition_ratio(text)

    def _wait_penalty(self, text: str) -> float:
        # Allow a couple of natural uses; penalize excessive reflection.
        count = len(_WAIT_RE.findall(text))
        excess = max(0, count - 2)
        return self.config.reg_wait_token_weight * excess

    def _plot_penalty(self, is_plot: bool, need_plot: bool) -> float:
        if is_plot and not need_plot:
            return self.config.reg_plot_weight
        return 0.0

    def compute(
        self, text: str, is_plot: bool = False, need_plot: bool = False
    ) -> RegBreakdown:
        return RegBreakdown(
            length=self._length_penalty(text),
            repetition=self._repetition_penalty(text),
            wait_token=self._wait_penalty(text),
            plot=self._plot_penalty(is_plot, need_plot),
        )

    def penalty(self, text: str, is_plot: bool = False, need_plot: bool = False) -> float:
        """Return R_reg(t) as a non-positive value (a penalty)."""
        return -self.compute(text, is_plot, need_plot).total
