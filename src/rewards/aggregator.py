"""Reward aggregation (Section 3.4.3).

For an agentic trajectory tau with T interaction steps, the total return is the
sum of per-step rewards, with policy regularization applied at every step:

    R(t) = R_base(t) + R_reg(t)
    Total = sum_{t=1}^{T} R(t)

For intermediate steps (t < T), R_base(t) is the process step reward. For the
final step (t = T), R_base(t) = R_terminal(tau), produced by the task-adaptive
verifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.constants import STEP_TOOL_CALL, STEP_ANSWER
from src.data.schema import Trajectory
from src.rewards.process_reward import ProcessStepReward
from src.rewards.regularization import Regularizer


@dataclass
class StepRewardRecord:
    index: int
    r_base: float
    r_reg: float

    @property
    def total(self) -> float:
        return self.r_base + self.r_reg


@dataclass
class TrajectoryReturn:
    total: float
    terminal_reward: float
    records: List[StepRewardRecord] = field(default_factory=list)


class RewardAggregator:
    """Aggregates process/terminal rewards and regularization over a trajectory."""

    def __init__(
        self,
        process_reward: Optional[ProcessStepReward] = None,
        regularizer: Optional[Regularizer] = None,
    ):
        self.process = process_reward or ProcessStepReward()
        self.regularizer = regularizer or Regularizer()

    def aggregate(
        self,
        trajectory: Trajectory,
        terminal_reward: float,
        need_plot: Optional[bool] = None,
    ) -> TrajectoryReturn:
        need_plot = trajectory.need_plot if need_plot is None else need_plot
        records: List[StepRewardRecord] = []
        total = 0.0
        n = len(trajectory.steps)

        for i, step in enumerate(trajectory.steps):
            is_terminal = (i == n - 1) and (step.kind == STEP_ANSWER)

            if is_terminal:
                r_base = terminal_reward
            elif step.kind == STEP_TOOL_CALL:
                r_base = self.process.reward(
                    function_correct=bool(step.function_correct),
                    exec_success=bool(step.exec_success),
                    is_terminal=False,
                ).value
            else:
                r_base = 0.0

            r_reg = self.regularizer.penalty(
                step.content, is_plot=step.is_plot, need_plot=need_plot
            )
            records.append(StepRewardRecord(i, r_base, r_reg))
            total += r_base + r_reg

        return TrajectoryReturn(total=total, terminal_reward=terminal_reward, records=records)
