"""Criteria-Injected Reward Model (Section 3.4.2, Figure 5).

For open-ended tabular questions lacking reliable labels, a reward model is
conditioned on explicit evaluation criteria generated in advance by a stronger
teacher model, and scores the response by checking criterion satisfaction.

Prompt template (Figure 5) and strict JSON output contract are reproduced
verbatim below.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from config import RewardConfig

# Figure 5 prompt template (verbatim structure).
JUDGE_PROMPT_TEMPLATE = """## You are a reward model. Your task is to evaluate the quality of the
assistant's response based on the following criteria:
{criteria}

The assistant's response is as follows: {response}

Assign a numeric score between 0 and 10, where 0 is the worst and 10 is the
best. Then provide a concise explanation for the score.
## Output strictly in JSON format with the following keys:
"score": numeric value (0-10)
"explanation": a brief text explaining the score
Do not include any other text outside the JSON. The explanation should be
short and focused on the criteria.
## Example of correct output:
{{ "score": 8, "explanation": "The response is mostly accurate but misses one key detail." }}"""


@dataclass
class JudgeResult:
    score: float          # normalized to [0, 1]
    raw_score: float      # original 0-10
    explanation: str
    valid_json: bool


def build_judge_prompt(criteria: str, response: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(criteria=criteria, response=response)


def parse_judge_output(text: str) -> tuple[Optional[float], str, bool]:
    """Parse the strict JSON judge output.

    Returns (raw_score_0_10, explanation, valid_json).
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, "", False
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "", False
    score = data.get("score")
    explanation = str(data.get("explanation", ""))
    if not isinstance(score, (int, float)):
        return None, explanation, False
    return float(score), explanation, True


class CriteriaInjectedRewardModel:
    """LLM judge conditioned on pre-generated criteria."""

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        config: Optional[RewardConfig] = None,
    ):
        self.llm_fn = llm_fn
        self.config = config or RewardConfig()

    def score(self, criteria: str, response: str) -> JudgeResult:
        if self.llm_fn is None:
            raise RuntimeError(
                "No judge LLM configured. Provide llm_fn or set up an API client."
            )
        prompt = build_judge_prompt(criteria, response)
        raw = self.llm_fn(prompt)
        raw_score, explanation, valid = parse_judge_output(raw)
        if raw_score is None:
            return JudgeResult(score=0.0, raw_score=0.0, explanation=explanation, valid_json=False)

        # Clamp to the configured [0, 10] range, then normalize to [0, 1].
        lo, hi = self.config.judge_score_min, self.config.judge_score_max
        raw_score = max(lo, min(hi, raw_score))
        norm = (raw_score - lo) / (hi - lo) if hi > lo else 0.0
        return JudgeResult(score=norm, raw_score=raw_score, explanation=explanation, valid_json=valid)
