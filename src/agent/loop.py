"""Agentic inference loop (Section 3.2.1).

Implements the closed-loop think-act-observe cycle. Given a model generation
function and a Python executor, the loop:
    1. builds the prompt (with the table as info or file path),
    2. generates a continuation,
    3. parses <tool_call> blocks, executes them, and feeds observations back
       inside <tool_response>,
    4. repeats until an <answer> is produced or max rounds is reached.

This enables truly agentic interaction with tabular data (Section 3.2.2): when
only a path is provided, the model must autonomously load the table, inspect its
structure, and determine what information to retrieve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from config import ExecConfig
from src.agent.parser import extract_answer, extract_tool_calls
from src.constants import (
    THINK_OPEN, TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
    TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE, ANSWER_OPEN, ANSWER_CLOSE,
)
from src.execution.executor import PythonExecutor
from src.execution.error_format import format_error


@dataclass
class AgentResult:
    answer: str
    rounds: int
    transcript: List[str] = field(default_factory=list)
    error: Optional[str] = None


class AgentLoop:
    """Runs the iterative think-act-observe loop."""

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        executor: Optional[PythonExecutor] = None,
        config: Optional[ExecConfig] = None,
        max_rounds: int = 10,
    ):
        self.generate_fn = generate_fn
        self.executor = executor or PythonExecutor(config)
        self.config = config or ExecConfig()
        self.max_rounds = max_rounds

    @staticmethod
    def build_initial_prompt(question: str, table_ref: str, system_prompt: str = "") -> str:
        parts = []
        if system_prompt:
            parts.append(system_prompt)
        parts.append(question)
        parts.append(table_ref)
        parts.append(THINK_OPEN)  # auto-included opening (HF model card)
        return "\n".join(parts)

    def run(self, question: str, table_ref: str, system_prompt: str = "") -> AgentResult:
        prompt = self.build_initial_prompt(question, table_ref, system_prompt)
        transcript: List[str] = []
        answer = ""

        for rnd in range(self.max_rounds):
            completion = self.generate_fn(prompt)
            transcript.append(completion)

            answer = extract_answer(completion)
            if answer:
                return AgentResult(answer=answer, rounds=rnd + 1, transcript=transcript)

            calls = extract_tool_calls(completion)
            if not calls:
                # No tool call and no answer: stop to avoid infinite loop.
                return AgentResult(answer="", rounds=rnd + 1, transcript=transcript,
                                   error="no tool_call or answer produced")

            observations = []
            for code in calls:
                result = self.executor.execute(code)
                obs = result.observation
                observations.append(
                    f"{TOOL_RESPONSE_OPEN}\n{obs}\n{TOOL_RESPONSE_CLOSE}"
                )
            prompt = prompt + "\n" + completion + "\n" + "\n".join(observations) + "\n" + THINK_OPEN

        return AgentResult(answer=answer, rounds=self.max_rounds, transcript=transcript,
                           error="max rounds reached")
