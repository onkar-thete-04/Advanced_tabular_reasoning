"""Task-adaptive reward routing (Section 3.4.1, Table 1).

Routes each task to the most suitable terminal verifier:
    General              -> LLM-eval
    Math-Logic           -> Rule
    Coding               -> Run & Rule
    Table-QA-Python      -> Run & LLM-eval
    SQL                  -> Run & Rule
    Table-QA-With-Label  -> Rule
    QA-with-Label        -> Rule
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.constants import FeedbackMethod, TaskType, TASK_ROUTING
from src.rewards.criteria_judge import CriteriaInjectedRewardModel
from src.rewards.rule_based import RuleBasedReward
from src.execution.executor import PythonExecutor
from config import RewardConfig


@dataclass
class TerminalReward:
    value: float          # normalized [0, 1] for optimization
    method: FeedbackMethod
    detail: str = ""


class RewardRouter:
    """Selects and applies the terminal verifier for a task."""

    def __init__(
        self,
        rule_reward: Optional[RuleBasedReward] = None,
        judge: Optional[CriteriaInjectedRewardModel] = None,
        executor: Optional[PythonExecutor] = None,
        config: Optional[RewardConfig] = None,
    ):
        self.rule = rule_reward or RuleBasedReward(config)
        self.judge = judge
        self.executor = executor or PythonExecutor()
        self.config = config or RewardConfig()

    @staticmethod
    def route(task_type: str) -> FeedbackMethod:
        try:
            return TASK_ROUTING[TaskType(task_type)]
        except ValueError:
            return FeedbackMethod.LLM_EVAL

    def terminal_reward(
        self,
        task_type: str,
        prediction: str,
        reference: str = "",
        criteria: str = "",
        code: str = "",
        numeric: bool = False,
    ) -> TerminalReward:
        method = self.route(task_type)

        if method == FeedbackMethod.RULE:
            r = self.rule.score_answer(prediction, reference, numeric=numeric)
            return TerminalReward(r.score, method, r.detail)

        if method == FeedbackMethod.RUN_AND_RULE:
            r = self.rule.score_execution(code or prediction, reference, self.executor, numeric)
            return TerminalReward(r.score, method, r.detail)

        if method == FeedbackMethod.RUN_AND_LLM_EVAL:
            if code:
                res = self.executor.execute(code)
                if not res.success:
                    return TerminalReward(0.0, method, f"execution failed: {res.error}")
            if self.judge is None:
                raise RuntimeError("Table-QA-Python requires a criteria-injected judge.")
            j = self.judge.score(criteria, prediction)
            return TerminalReward(j.score, method, j.explanation)

        # FeedbackMethod.LLM_EVAL
        if self.judge is None:
            raise RuntimeError("General task requires a criteria-injected judge.")
        j = self.judge.score(criteria, prediction)
        return TerminalReward(j.score, method, j.explanation)
