"""Special-token trajectory parser (Section 3.2.1).

Parses model output containing:
    <think>...</think> | <tool_call>...</tool_call> |
    <tool_response>...</tool_response> | <answer>...</answer>

Also accepts the alternating markers <function_call> and the Qwen3 thinking
close token (Section 3.2.2 / HF model card).
"""

from __future__ import annotations

import re
from typing import List

from src.constants import (
    THINK_OPEN, THINK_CLOSES, TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
    FUNCTION_CALL_OPEN, FUNCTION_CALL_CLOSE,
    TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE, ANSWER_OPEN, ANSWER_CLOSE,
    STEP_THINK, STEP_TOOL_CALL, STEP_TOOL_RESPONSE, STEP_ANSWER,
)
from src.data.schema import Step

_TOOL_CALL_RE = re.compile(
    rf"{re.escape(TOOL_CALL_OPEN)}(.*?){re.escape(TOOL_CALL_CLOSE)}"
    rf"|{re.escape(FUNCTION_CALL_OPEN)}(.*?){re.escape(FUNCTION_CALL_CLOSE)}",
    re.DOTALL,
)
_TOOL_RESPONSE_RE = re.compile(
    rf"{re.escape(TOOL_RESPONSE_OPEN)}(.*?){re.escape(TOOL_RESPONSE_CLOSE)}", re.DOTALL
)
_ANSWER_RE = re.compile(
    rf"{re.escape(ANSWER_OPEN)}(.*?){re.escape(ANSWER_CLOSE)}", re.DOTALL
)
_THINK_RE = re.compile(
    rf"{re.escape(THINK_OPEN)}(.*?)(?:{'|'.join(re.escape(c) for c in THINK_CLOSES)})",
    re.DOTALL,
)


def extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else ""


def extract_tool_calls(text: str) -> List[str]:
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        calls.append((m.group(1) or m.group(2) or "").strip())
    return calls


def parse_steps(text: str) -> List[Step]:
    """Parse text into an ordered list of Steps by token position."""
    spans = []
    for m in _THINK_RE.finditer(text):
        spans.append((m.start(), Step(kind=STEP_THINK, content=m.group(1).strip())))
    for m in _TOOL_CALL_RE.finditer(text):
        code = (m.group(1) or m.group(2) or "").strip()
        spans.append((m.start(), Step(kind=STEP_TOOL_CALL, content=code)))
    for m in _TOOL_RESPONSE_RE.finditer(text):
        spans.append((m.start(), Step(kind=STEP_TOOL_RESPONSE, content=m.group(1).strip())))
    for m in _ANSWER_RE.finditer(text):
        spans.append((m.start(), Step(kind=STEP_ANSWER, content=m.group(1).strip())))
    spans.sort(key=lambda kv: kv[0])
    return [step for _, step in spans]


def parse_trajectory_text(text: str) -> dict:
    """Convenience parse into steps + final answer."""
    steps = parse_steps(text)
    return {"steps": steps, "answer": extract_answer(text)}
