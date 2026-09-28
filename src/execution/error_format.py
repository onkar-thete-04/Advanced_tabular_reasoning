"""Error message formatting (Section 3.2.2: Error Message Diversity).

Injects heterogeneous Python error formats, including minimal exceptions and
full stack traces, to improve robustness in diagnosing and recovering from
execution failures.
"""

from __future__ import annotations

import traceback
from typing import Optional


def format_error(
    exc: Optional[BaseException] = None,
    style: str = "full",
    exc_type: Optional[str] = None,
    message: Optional[str] = None,
) -> str:
    """Format an error as either a minimal exception or a full traceback.

    Args:
        exc: the exception instance (preferred).
        style: "minimal" -> "TypeError: msg"; "full" -> full traceback.
        exc_type / message: used when no exception object is available.
    """
    if exc is not None:
        if style == "minimal":
            return f"{type(exc).__name__}: {exc}"
        return "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ).strip()

    etype = exc_type or "ExecutionError"
    msg = message or ""
    return f"{etype}: {msg}" if style == "minimal" else f"{etype}: {msg}"


def minimal_error(exc_type: str, message: str) -> str:
    return format_error(style="minimal", exc_type=exc_type, message=message)


def full_error(exc_type: str, message: str) -> str:
    return format_error(style="full", exc_type=exc_type, message=message)
