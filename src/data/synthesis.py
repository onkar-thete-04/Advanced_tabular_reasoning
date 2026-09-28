"""Agentic data synthesis (Section 3.2.1 / Figure 4).

Builds the standardized interaction trajectory:
    question + table -> [<think> <tool_call> <tool_response>]* -> <answer>

Format constraints enforced (Figure 4 "Data Cleaning"):
    * Special tokens appear in pairs.
    * The last round must include <answer> and </answer>.
    * The table address in pd.read_csv() only appears in the first round.
    * The tool execution result before the final answer must not contain errors.
    * Token count and interaction count must not exceed thresholds.
    * Code passes the security check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from config import DataConfig
from src.constants import (
    STEP_THINK, STEP_TOOL_CALL, STEP_TOOL_RESPONSE, STEP_ANSWER,
    THINK_OPEN, THINK_CLOSE, TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
    TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE, ANSWER_OPEN, ANSWER_CLOSE,
)
from src.data.filtering import token_len
from src.data.schema import Step, Trajectory, Sample
from src.execution.executor import check_security, detect_plot


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]

    def __bool__(self) -> bool:  # allow `if validate(...)`
        return self.ok


_READ_CSV_RE = re.compile(r"pd\.read_csv\s*\(")
_ERROR_RE = re.compile(r"(traceback|\berror\b|exception)", re.IGNORECASE)


def build_trajectory(sample: Sample, steps: Optional[List[Step]] = None) -> Trajectory:
    """Construct a Trajectory from a Sample (steps optional)."""
    traj = Trajectory(
        sample_id=sample.sample_id,
        question=sample.question,
        table_ref=sample.table_ref,
        table_input_mode="info",
        task_type=sample.task_type,
        need_plot=sample.need_plot,
    )
    if steps:
        traj.steps = list(steps)
    return traj


def validate_trajectory(traj: Trajectory, config: Optional[DataConfig] = None) -> ValidationResult:
    """Validate all Figure 4 format constraints."""
    config = config or DataConfig()
    errors: List[str] = []

    # (a) special tokens paired + last round contains <answer>
    if not traj.is_terminal:
        errors.append("Last round must include <answer> and </answer>.")

    # (b) pd.read_csv table address only in the first round
    for idx, step in enumerate(traj.steps):
        if step.kind == STEP_TOOL_CALL and _READ_CSV_RE.search(step.content):
            if idx != 0:
                errors.append(
                    "The table address in pd.read_csv() may only appear in the first round."
                )

    # (c) no error messages in tool responses before the final answer
    for step in traj.steps:
        if step.kind == STEP_TOOL_RESPONSE and _ERROR_RE.search(step.content):
            errors.append(
                "Tool execution result before the final answer must not contain errors."
            )
            break

    # (d) token / interaction limits
    if traj.num_tool_calls > config.max_interaction_rounds:
        errors.append(
            f"Interaction count {traj.num_tool_calls} exceeds limit "
            f"{config.max_interaction_rounds}."
        )
    if token_len(traj.to_text()) > config.max_trajectory_tokens:
        errors.append("Trajectory exceeds the token threshold.")

    # (e) security check on every tool call
    for step in traj.steps:
        if step.kind == STEP_TOOL_CALL:
            safe, reason = check_security(step.content)
            if not safe:
                errors.append(f"Security check failed: {reason}")

    # (f) answer present
    if not traj.final_answer and traj.is_terminal:
        traj.final_answer = traj.steps[-1].content

    return ValidationResult(ok=not errors, errors=errors)


def attach_execution_observation(
    traj: Trajectory, code: str, observation: str, exec_success: bool, function_correct: bool = True
) -> None:
    """Append a tool_call + tool_response pair with execution metadata."""
    traj.add_step(
        Step(
            kind=STEP_TOOL_CALL,
            content=code,
            exec_success=exec_success,
            function_correct=function_correct,
            is_plot=detect_plot(code),
        )
    )
    traj.add_step(Step(kind=STEP_TOOL_RESPONSE, content=observation))


class AgenticDataSynthesizer:
    """Orchestrates trajectory construction + validation."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()

    def synthesize(self, sample: Sample, steps: List[Step]) -> Optional[Trajectory]:
        traj = build_trajectory(sample, steps)
        result = validate_trajectory(traj, self.config)
        if not result.ok:
            return None
        return traj
