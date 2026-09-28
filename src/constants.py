"""Global constants for TableGPT-R1.

Sources: Section 3.2.1 (data format), Section 3.4.1 (taxonomy / Table 1).
"""

from enum import Enum

# ---------------------------------------------------------------------------
# Special tokens (Section 3.2.1, Figure 4)
# ---------------------------------------------------------------------------
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# Qwen3 chat template auto-includes the opening <think> and closes with this
# token; the paper also writes </think>. Both are accepted by the parser.
QWEN_THINK_CLOSE = " response"
THINK_CLOSES = ("</think>", QWEN_THINK_CLOSE, "<｜end▁of▁thinking｜>")
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

# Rotating equivalent marker (Section 3.2.2: Special Token Alternation)
FUNCTION_CALL_OPEN = "<function_call>"
FUNCTION_CALL_CLOSE = "</function_call>"

# All accepted openings for a tool call, in alternation order.
TOOL_CALL_OPENINGS = (TOOL_CALL_OPEN, FUNCTION_CALL_OPEN)
TOOL_CALL_CLOSINGS = (TOOL_CALL_CLOSE, FUNCTION_CALL_CLOSE)

# Step kinds
STEP_THINK = "think"
STEP_TOOL_CALL = "tool_call"
STEP_TOOL_RESPONSE = "tool_response"
STEP_ANSWER = "answer"


# ---------------------------------------------------------------------------
# Task-adaptive reward taxonomy (Table 1, Section 3.4.1)
# ---------------------------------------------------------------------------
class TaskType(str, Enum):
    GENERAL = "General"
    MATH_LOGIC = "Math-Logic"
    CODING = "Coding"
    TABLE_QA_PYTHON = "Table-QA-Python"
    SQL = "SQL"
    TABLE_QA_WITH_LABEL = "Table-QA-With-Label"
    QA_WITH_LABEL = "QA-with-Label"


class FeedbackMethod(str, Enum):
    LLM_EVAL = "LLM-eval"
    RULE = "Rule"
    RUN_AND_RULE = "Run & Rule"
    RUN_AND_LLM_EVAL = "Run & LLM-eval"


# Table 1: task type -> feedback method (Section 3.4.2)
TASK_ROUTING = {
    TaskType.GENERAL: FeedbackMethod.LLM_EVAL,
    TaskType.MATH_LOGIC: FeedbackMethod.RULE,
    TaskType.CODING: FeedbackMethod.RUN_AND_RULE,
    TaskType.TABLE_QA_PYTHON: FeedbackMethod.RUN_AND_LLM_EVAL,
    TaskType.SQL: FeedbackMethod.RUN_AND_RULE,
    TaskType.TABLE_QA_WITH_LABEL: FeedbackMethod.RULE,
    TaskType.QA_WITH_LABEL: FeedbackMethod.RULE,
}

# Data composition ratio (Figure 3)
DATA_COMPOSITION = {
    "general": 0.05,
    "math_logic": 0.20,
    "code": 0.05,
    "table_task": 0.30,
    "sql": 0.10,
    "qa_with_answer": 0.20,
    "table_agent": 0.10,
}

# Difficulty buckets (Section 2.2)
DIFFICULTY_HIGH = "high"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_LOW = "low"
