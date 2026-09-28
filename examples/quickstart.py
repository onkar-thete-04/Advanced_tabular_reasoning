"""Quickstart: build an agentic trajectory, score it, and run the agent loop.

Run from the project root:
    python examples/quickstart.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.constants import (
    TaskType, STEP_THINK, STEP_TOOL_CALL, STEP_TOOL_RESPONSE, STEP_ANSWER,
)
from src.data.schema import Trajectory, Step
from src.execution.executor import PythonExecutor
from src.rewards.router import RewardRouter
from src.rewards.aggregator import RewardAggregator
from src.agent.loop import AgentLoop


def main() -> None:
    cfg = Config()

    # 1. Build a validated agentic trajectory (Section 3.2.1).
    traj = Trajectory(sample_id="demo", question="Sum of val?", table_ref="examples/sample_table.csv",
                      need_plot=False)
    traj.add_step(Step(kind=STEP_THINK, content="Load the table and sum the val column."))
    traj.add_step(Step(kind=STEP_TOOL_CALL,
                       content="import pandas as pd\ndf = pd.read_csv('examples/sample_table.csv')\nprint(df['val'].sum())",
                       exec_success=True, function_correct=True))
    traj.add_step(Step(kind=STEP_TOOL_RESPONSE, content="60"))
    traj.add_step(Step(kind=STEP_ANSWER, content="60"))
    traj.final_answer = "60"
    print("Trajectory text:\n", traj.to_text())

    # 2. Terminal reward via the task-adaptive router (Section 3.4).
    router = RewardRouter(config=cfg.reward)
    terminal = router.terminal_reward(
        TaskType.TABLE_QA_WITH_LABEL.value, prediction=traj.final_answer, reference="60"
    )
    print(f"\nTerminal reward: {terminal.value} via {terminal.method.value}")

    # 3. Aggregate process + terminal + regularization (Section 3.4.3).
    agg = RewardAggregator()
    ret = agg.aggregate(traj, terminal_reward=terminal.value)
    print(f"Total trajectory return: {ret.total:.3f}")

    # 4. Run the closed-loop agent on the same table (Section 3.2.1).
    def gen(prompt: str) -> str:
        if "tool_response" not in prompt:
            return ("<tool_call>\nimport pandas as pd\n"
                    "df = pd.read_csv('examples/sample_table.csv')\n"
                    "print(int(df['val'].sum()))\n</tool_call>")
        return "<answer>60</answer>"

    loop = AgentLoop(gen, PythonExecutor(cfg.exec), cfg.exec, max_rounds=4)
    result = loop.run("Sum of val?", "examples/sample_table.csv")
    print(f"\nAgent answer: {result.answer!r} in {result.rounds} rounds")


if __name__ == "__main__":
    main()
