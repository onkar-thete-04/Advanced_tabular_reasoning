"""Tests for the code execution environment (Section 3.2.1 / 3.4.2)."""

from config import ExecConfig
from src.execution.executor import PythonExecutor, check_security, detect_plot
from src.execution.error_format import format_error, minimal_error, full_error


def _exec(**kw):
    cfg = ExecConfig(**kw)
    return PythonExecutor(cfg)


def test_execute_success_inprocess():
    ex = _exec(use_subprocess=False)
    result = ex.execute("print(2 + 3)")
    assert result.success and result.stdout.strip() == "5"


def test_execute_error_inprocess():
    ex = _exec(use_subprocess=False)
    result = ex.execute("raise ValueError('boom')")
    assert not result.success
    assert "ValueError" in (result.error or "")


def test_security_blocks_dangerous_code():
    ex = _exec(use_subprocess=False)
    result = ex.execute("import os; os.system('echo hi')")
    assert not result.success and "SecurityError" in (result.error or "")


def test_detect_plot():
    assert detect_plot("import matplotlib.pyplot as plt; plt.plot([1,2])")
    assert not detect_plot("print(1)")


def test_check_security():
    safe, reason = check_security("print(1)")
    assert safe and reason is None
    unsafe, reason2 = check_security("import subprocess")
    assert not unsafe and reason2


def test_error_formatting_styles():
    assert minimal_error("ValueError", "x") == "ValueError: x"
    assert "ValueError: x" in full_error("ValueError", "x")


def test_execute_subprocess_success():
    ex = _exec(use_subprocess=True)
    result = ex.execute("print('hello from child process')")
    assert result.success and "hello from child process" in result.stdout
