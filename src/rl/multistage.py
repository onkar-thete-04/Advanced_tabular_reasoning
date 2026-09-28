"""Multi-stage training framework (Section 3.5).

Stage 0 : SFT warm-up on ~3% of the full dataset (format alignment).
Stage 1 : RL dominated by general reasoning data.
Stage 2 : RL with significantly increased Table-agentic data.
Stage 3 : RL exclusively on difficult/borderline samples (Stage 2 score < 5).

Each stage terminates at step 200. Between stages, data is filtered using the
current policy: keep only samples with pass@k in [3, 7].
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from config import TrainingConfig
from src.data.schema import Sample


def pass_at_k(num_correct: int, num_samples: int, k: int) -> float:
    """Unbiased pass@k estimator (Codex-style)."""
    if num_samples - num_correct < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k)
    import math

    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def score_to_pass_count(score: float, num_samples: int) -> int:
    """Map a 0-1 sample score to a pseudo pass-count for filtering."""
    return int(round(max(0.0, min(1.0, score)) * num_samples))


@dataclass
class Stage:
    index: int
    name: str
    data: List[Sample] = field(default_factory=list)
    description: str = ""


class MultiStageTrainer:
    """Orchestrates the SFT warm-up and three RL stages."""

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        sft_fn: Optional[Callable[[List[Sample]], None]] = None,
        rl_stage_fn: Optional[Callable[[Stage], dict]] = None,
        evaluate_fn: Optional[Callable[[List[Sample]], dict]] = None,
    ):
        self.config = config or TrainingConfig()
        self.sft_fn = sft_fn
        self.rl_stage_fn = rl_stage_fn
        self.evaluate_fn = evaluate_fn
        self.history: List[dict] = []

    def sample_sft_subset(self, samples: Sequence[Sample]) -> List[Sample]:
        n = max(1, int(round(len(samples) * self.config.sft_warmup_fraction)))
        rng = random.Random(42)
        return rng.sample(list(samples), min(n, len(samples)))

    def build_stages(self, samples: Sequence[Sample]) -> List[Stage]:
        general = [s for s in samples if s.task_type == "General"]
        table_agentic = [s for s in samples if s.task_type == "Table-QA-Python"]
        difficult = [s for s in samples if (s.difficulty_score or 0.0) < self.config.stage3_threshold_score]

        return [
            Stage(0, "SFT Warm-up", self.sample_sft_subset(samples), "format alignment (~3%)"),
            Stage(1, "RL Stage 1", general or list(samples), "general reasoning dominant"),
            Stage(2, "RL Stage 2", table_agentic or list(samples), "Table-agentic dominant"),
            Stage(3, "RL Stage 3", difficult, "difficult/borderline (Stage 2 score < 5)"),
        ]

    def filter_by_passk(self, samples: Sequence[Sample]) -> List[Sample]:
        """Keep samples whose pass@k falls within [min, max] (Section 3.5)."""
        kept = []
        for s in samples:
            score = s.difficulty_score if s.difficulty_score is not None else 0.5
            num_correct = score_to_pass_count(score, self.config.passk_k)
            p = pass_at_k(num_correct, self.config.passk_k, self.config.passk_k)
            # interpret pass@k as a 0-1 rate; retain the informative middle band
            band = p * self.config.passk_k
            if self.config.passk_retain_min <= band <= self.config.passk_retain_max:
                kept.append(s)
        return kept

    def run(self, samples: Sequence[Sample]) -> List[dict]:
        stages = self.build_stages(samples)

        # Stage 0: SFT warm-up
        sft_stage = stages[0]
        if self.sft_fn is not None:
            self.sft_fn(sft_stage.data)
        self.history.append({"stage": 0, "name": sft_stage.name, "num_samples": len(sft_stage.data)})

        current_data = list(samples)
        for stage in stages[1:]:
            stage.data = [s for s in stage.data if s in current_data] or stage.data
            metrics = {"stage": stage.index, "name": stage.name, "num_samples": len(stage.data)}
            if self.rl_stage_fn is not None:
                metrics.update(self.rl_stage_fn(stage) or {})
            metrics["terminated_at_step"] = self.config.stage_termination_step
            self.history.append(metrics)

            if self.evaluate_fn is not None:
                eval_metrics = self.evaluate_fn(stage.data) or {}
                self.history[-1].update({f"eval_{k}": v for k, v in eval_metrics.items()})

            # Filter data for the next stage using the current policy.
            current_data = self.filter_by_passk(stage.data)

        return self.history
