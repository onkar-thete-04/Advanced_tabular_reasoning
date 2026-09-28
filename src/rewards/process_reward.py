"""Process step reward (Section 3.4.2).

A lightweight rule-based step reward scores intermediate actions according to
execution state and progress:
    * incorrect function selected                  -> -0.2
    * correct function + exec success + non-final  -> +0.1
    * correct function + exec failure              -> -0.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import RewardConfig


@dataclass
class StepReward:
    value: float
    reason: str


class ProcessStepReward:
    """Computes per-step shaping rewards for agentic rollouts."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def reward(
        self,
        function_correct: bool,
        exec_success: bool,
        is_terminal: bool = False,
    ) -> StepReward:
        # Terminal steps are scored by the task-adaptive terminal verifier
        # (Section 3.4.3), so no process reward is applied here.
        if is_terminal:
            return StepReward(0.0, "terminal step (use terminal reward)")

        if not function_correct:
            return StepReward(self.config.reward_wrong_function, "incorrect function")

        if exec_success:
            return StepReward(
                self.config.reward_correct_progress, "correct function, successful progress"
            )

        return StepReward(
            self.config.reward_correct_exec_fail, "correct function, execution failed"
        )
