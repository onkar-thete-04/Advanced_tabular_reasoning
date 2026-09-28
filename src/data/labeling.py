"""Reliable label generation via multi-model consensus (Section 3.2.3).

For each query, responses are generated using multiple strong models
(DeepSeek, GPT-4o, Qwen-Max) and their final answers extracted. Only samples
for which at least two models produce identical final answers are retained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from config import DataConfig

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_final_answer(response: str) -> str:
    """Extract the text inside <answer> ... </answer>, if present."""
    m = _ANSWER_RE.search(response)
    if m:
        return m.group(1).strip()
    return response.strip()


def normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.strip().lower())


@dataclass
class ConsensusResult:
    kept: bool
    answer: Optional[str]
    votes: Dict[str, str]
    num_agree: int


class ConsensusLabeler:
    """Generates labels via a multi-model consensus strategy."""

    def __init__(
        self,
        model_fns: Dict[str, Callable[[str], str]],
        config: Optional[DataConfig] = None,
    ):
        self.model_fns = model_fns
        self.config = config or DataConfig()

    def label(self, query: str) -> ConsensusResult:
        votes: Dict[str, str] = {}
        for name, fn in self.model_fns.items():
            try:
                response = fn(query)
                votes[name] = normalize_answer(extract_final_answer(response))
            except Exception:
                continue

        counts: Dict[str, int] = {}
        for ans in votes.values():
            counts[ans] = counts.get(ans, 0) + 1
        if not counts:
            return ConsensusResult(False, None, votes, 0)

        best_answer, best_count = max(counts.items(), key=lambda kv: kv[1])
        kept = best_count >= self.config.consensus_min_agree
        return ConsensusResult(kept, best_answer if kept else None, votes, best_count)

    def label_batch(self, queries: List[str]) -> List[ConsensusResult]:
        return [self.label(q) for q in queries]
