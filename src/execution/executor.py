"""Sandboxed Python code execution environment.

Section 3.2.1: executable Python code is wrapped in <tool_call> tags to invoke
the Python execution environment; the environment returns the outcome
(success or error) within <tool_response> tags.

Section 3.4.2: rule-based reward uses execution-based verification.
Figure 4: "Check the security of the code."
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

from config import ExecConfig


# Patterns that indicate visualization code (used for the need_plot penalty,
# Section 3.4.3 opportunistic-plotting suppression).
PLOT_PATTERNS = (
    "matplotlib", "seaborn", "plt.", "pyplot", ".plot(", "savefig",
    "figure(", "barplot", "histplot", "sns.",
)

# Security denylist (Figure 4: Security Check).
SECURITY_DENYLIST = (
    "os.system", "subprocess", "shutil.rmtree", "__import__",
    "socket.", "requests.", "urllib.request", "ctypes",
    "os.remove", "os.rmdir", "os.unlink",
)


@dataclass
class ExecutionResult:
    """Outcome of executing a code snippet."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    is_plot: bool = False
    timed_out: bool = False

    @property
    def observation(self) -> str:
        """Text placed inside <tool_response> (Section 3.2.1)."""
        if self.timed_out:
            return "ExecutionError: execution timed out."
        if self.success:
            return self.stdout if self.stdout else "(no output)"
        return f"{self.error}\n{self.stderr}".strip()


def detect_plot(code: str) -> bool:
    """Return True if code appears to generate a visualization."""
    lowered = code.lower()
    return any(p in lowered for p in PLOT_PATTERNS)


def check_security(code: str) -> tuple[bool, Optional[str]]:
    """Simple static security check (Figure 4).

    Returns (is_safe, reason_if_unsafe).
    """
    lowered = code.lower()
    for bad in SECURITY_DENYLIST:
        if bad.lower() in lowered:
            return False, f"Unsafe construct detected: '{bad}'"
    return True, None


class PythonExecutor:
    """Executes Python snippets in an isolated subprocess."""

    def __init__(self, config: Optional[ExecConfig] = None):
        self.config = config or ExecConfig()

    def execute(self, code: str, workdir: Optional[str] = None) -> ExecutionResult:
        """Execute `code` and capture the outcome.

        The snippet runs in a fresh subprocess with a timeout. When
        `use_subprocess` is False (tests / trusted code) it runs in-process.
        """
        is_plot = detect_plot(code)
        safe, reason = check_security(code)
        if not safe:
            return ExecutionResult(
                success=False, error=f"SecurityError: {reason}", is_plot=is_plot
            )

        if not self.config.use_subprocess:
            return self._execute_inprocess(code, is_plot)

        tmp_dir = tempfile.mkdtemp(prefix="tablegpt_exec_")
        script_path = os.path.join(tmp_dir, "_snippet.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if not self.config.allow_network:
            # Best-effort network isolation hint (not a hard sandbox).
            env["no_proxy"] = "*"

        try:
            proc = subprocess.run(
                [sys.executable, "-I", script_path],
                cwd=workdir or tmp_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error="ExecutionError: execution timed out.",
                is_plot=is_plot,
                timed_out=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ExecutionResult(
                success=False, error=f"ExecutionError: {exc}", is_plot=is_plot
            )

        stdout = (proc.stdout or "")[: self.config.max_output_chars]
        stderr = (proc.stderr or "")[: self.config.max_output_chars]
        success = proc.returncode == 0
        error = None if success else self._extract_error(stderr)
        return ExecutionResult(
            success=success, stdout=stdout, stderr=stderr, error=error, is_plot=is_plot
        )

    def _execute_inprocess(self, code: str, is_plot: bool) -> ExecutionResult:
        import io
        import contextlib

        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                exec(code, {"__name__": "__main__"})  # noqa: S102 - trusted test path
        except Exception as exc:
            return ExecutionResult(
                success=False,
                stdout=buf_out.getvalue()[: self.config.max_output_chars],
                stderr=buf_err.getvalue()[: self.config.max_output_chars],
                error=f"{type(exc).__name__}: {exc}",
                is_plot=is_plot,
            )
        return ExecutionResult(
            success=True,
            stdout=buf_out.getvalue()[: self.config.max_output_chars],
            stderr=buf_err.getvalue()[: self.config.max_output_chars],
            is_plot=is_plot,
        )

    @staticmethod
    def _extract_error(stderr: str) -> str:
        """Extract the final exception line from a traceback."""
        lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
        return lines[-1] if lines else "ExecutionError"
