"""Data structures for TableGPT-R1 trajectories and samples.

Implements the standardized agentic interaction trajectory of Section 3.2.1:
    question + table  ->  [<think>..</think> <tool_call>..</tool_call>
                          <tool_response>..</tool_response>] *  ->  <answer>..</answer>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.constants import (
    THINK_OPEN, THINK_CLOSE,
    TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
    TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE,
    ANSWER_OPEN, ANSWER_CLOSE,
    STEP_THINK, STEP_TOOL_CALL, STEP_TOOL_RESPONSE, STEP_ANSWER,
)


@dataclass
class Step:
    """A single step in an agentic trajectory."""

    kind: str                       # think | tool_call | tool_response | answer
    content: str
    exec_success: Optional[bool] = None       # for tool_call steps
    function_correct: Optional[bool] = None   # for tool_call steps (process reward)
    is_plot: bool = False                     # whether code emits visualization
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        if self.kind == STEP_THINK:
            return f"{THINK_OPEN}{self.content}{THINK_CLOSE}"
        if self.kind == STEP_TOOL_CALL:
            return f"{TOOL_CALL_OPEN}\n{self.content}\n{TOOL_CALL_CLOSE}"
        if self.kind == STEP_TOOL_RESPONSE:
            return f"{TOOL_RESPONSE_OPEN}\n{self.content}\n{TOOL_RESPONSE_CLOSE}"
        if self.kind == STEP_ANSWER:
            return f"{ANSWER_OPEN}{self.content}{ANSWER_CLOSE}"
        return self.content


@dataclass
class Trajectory:
    """A full agentic trajectory for one query (Section 3.2.1)."""

    sample_id: str
    question: str
    table_ref: str                       # table path or static info text
    table_input_mode: str = "info"       # "info" | "path"
    steps: List[Step] = field(default_factory=list)
    final_answer: str = ""
    task_type: str = "General"
    need_plot: bool = False
    difficulty: Optional[str] = None
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        parts = [self.question, self.table_ref]
        parts.extend(s.to_text() for s in self.steps)
        return "\n".join(parts)

    @property
    def num_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == STEP_TOOL_CALL)

    @property
    def is_terminal(self) -> bool:
        return bool(self.steps) and self.steps[-1].kind == STEP_ANSWER

    def add_step(self, step: Step) -> None:
        self.steps.append(step)
        if step.kind == STEP_ANSWER and not self.final_answer:
            self.final_answer = step.content


@dataclass
class Sample:
    """A raw/filtered training sample before trajectory construction."""

    sample_id: str
    question: str
    table_ref: str
    reference_answer: str = ""
    task_type: str = "General"
    need_plot: bool = False
    language: str = "en"
    output_language: str = "en"
    quality_score: Optional[float] = None
    difficulty_score: Optional[float] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
