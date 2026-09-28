"""TableGPT-R1 command-line entry point.

Subcommands:
    data       Run the data engineering pipeline on a synthetic fixture.
    rewards    Demonstrate the task-adaptive reward system.
    rl-smoke   Verify the hybrid RL objective on hand-computed tensors.
    agent      Run the agentic think-act-observe loop on a small table.
    eval       Evaluate a policy over a benchmark (mock or model policy).
    eval-tabular  Before/after eval on the synthetic tabular task.
    pipeline   End-to-end smoke test of all components.
    config     Print the resolved configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

# Ensure project root on sys.path (allows `import config`, `from src...`).
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_samples(n: int = 30):
    from src.data.schema import Sample
    from src.constants import TaskType

    samples = []
    for i in range(n):
        samples.append(
            Sample(
                sample_id=f"s{i}",
                question=f"What is the sum of column B for row group {i}?",
                table_ref="id,val\n1,10\n2,20\n3,30",
                reference_answer=str(60 + i),
                task_type=TaskType.TABLE_QA_WITH_LABEL.value,
                language="en",
                output_language="en",
            )
        )
    return samples


def _mock_scorer(sample, response=None):
    """Reference-quality scorer: references are assumed high quality."""
    import hashlib

    h = int(hashlib.md5(sample.sample_id.encode()).hexdigest(), 16)
    return 0.75 + (h % 25) / 100.0  # 0.75 - 1.00


def _mock_rollout_scorer(sample, response=None):
    """Difficulty scorer: varies across samples to span all difficulty buckets."""
    import hashlib

    h = int(hashlib.md5(sample.sample_id.encode()).hexdigest(), 16)
    return (h % 100) / 100.0  # 0.00 - 1.00


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_config(args, cfg: Config):
    print(json.dumps(cfg.to_dict(), indent=2, default=str))


def cmd_data(args, cfg: Config):
    from src.data.filtering import (
        run_input_filtering, output_quality_filter, assign_difficulty, resample_difficulty,
    )
    from src.data.composition import actual_distribution

    samples = _make_samples(50)

    def rollout_fn(s):
        return s.reference_answer

    samples = run_input_filtering(samples, cfg.data)
    samples = output_quality_filter(samples, _mock_scorer, cfg.data.quality_retention_threshold)
    samples = assign_difficulty(samples, rollout_fn, _mock_rollout_scorer, cfg.data)
    samples = resample_difficulty(samples, cfg.data)

    dist = {}
    for s in samples:
        dist[s.difficulty] = dist.get(s.difficulty, 0) + 1
    print(f"Input samples: 50 -> filtered: {len(samples)}")
    print("Difficulty distribution:", json.dumps(dist))


def cmd_rewards(args, cfg: Config):
    from src.rewards.process_reward import ProcessStepReward
    from src.rewards.regularization import Regularizer
    from src.rewards.router import RewardRouter
    from src.constants import TaskType

    pr = ProcessStepReward(cfg.reward)
    reg = Regularizer(cfg.reward)
    router = RewardRouter(config=cfg.reward)

    print("Process rewards:")
    for correct, success, label in [
        (False, False, "wrong function"),
        (True, True, "correct + success"),
        (True, False, "correct + exec fail"),
    ]:
        r = pr.reward(correct, success)
        print(f"  {label:22s} -> {r.value:+.2f}  ({r.reason})")

    print("\nRegularization (need_plot=False, plot code emitted):")
    b = reg.compute("import matplotlib.pyplot as plt; plt.plot([1,2]); plt.savefig('x.png')",
                    is_plot=True, need_plot=False)
    print(f"  total penalty = {-b.total:.3f}  breakdown={b}")

    print("\nReward routing (Table 1):")
    for tt in TaskType:
        print(f"  {tt.value:22s} -> {router.route(tt.value).value}")

    r = router.terminal_reward(
        TaskType.TABLE_QA_WITH_LABEL.value, prediction="60", reference="60"
    )
    print(f"\nRule terminal reward (correct answer): {r.value} via {r.method.value}")


def cmd_rl_smoke(args, cfg: Config):
    import torch
    from src.rl.objective import HybridRLObjective
    from src.rl.advantage import group_advantage

    torch.manual_seed(0)
    G, T = 4, 5
    logprobs = torch.zeros(G, T)
    old_logprobs = torch.zeros(G, T)
    mask = torch.ones(G, T)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv = group_advantage(rewards)

    obj = HybridRLObjective(cfg.rl)
    out = obj(logprobs, old_logprobs, adv, mask, step=0)
    print(f"step 0  loss={float(out.loss):.4f} policy={float(out.policy_loss):.4f} "
          f"entropy_active={out.entropy_active} mean_ratio={float(out.mean_ratio):.4f}")
    assert abs(float(out.mean_ratio) - 1.0) < 1e-6, "ratio must be 1 when policies match"

    out2 = obj(logprobs, old_logprobs, adv, mask, step=cfg.rl.entropy_terms_start_step)
    print(f"step {cfg.rl.entropy_terms_start_step}  loss={float(out2.loss):.4f} "
          f"entropy_bonus={float(out2.entropy_bonus):.6f} active={out2.entropy_active}")
    assert out2.entropy_active and not out.entropy_active
    print("RL objective smoke test passed.")


def cmd_agent(args, cfg: Config):
    from src.agent.loop import AgentLoop
    from src.execution.executor import PythonExecutor

    def mock_generate(prompt: str) -> str:
        # First turn: emit a tool call; second turn: emit the answer.
        if "tool_response" not in prompt:
            return (
                "I will load the table and compute the sum.\n"
                "<tool_call>\nimport pandas as pd\n"
                "from io import StringIO\n"
                "df = pd.read_csv(StringIO('id,val\\n1,10\\n2,20\\n3,30'))\n"
                "print(int(df['val'].sum()))\n</tool_call>"
            )
        return "<answer>60</answer>"

    loop = AgentLoop(mock_generate, PythonExecutor(cfg.exec), cfg.exec, max_rounds=5)
    result = loop.run("What is the sum of val?", "table.csv")
    print(f"Agent rounds: {result.rounds}  answer: {result.answer!r}  error: {result.error}")
    assert result.answer == "60", "agent should recover the answer"
    print("Agent loop smoke test passed.")


def cmd_eval(args, cfg: Config):
    from src.evaluation.benchmarks import evaluate_policy

    samples = [
        {"question": "sum?", "table_ref": "t.csv", "reference": "60", "table_path": "t.csv"},
        {"question": "count?", "table_ref": "t.csv", "reference": "3", "table_path": "t.csv"},
    ]

    def mock_policy(question, table_ref):
        return "60" if "sum" in question else "3"

    metrics = evaluate_policy(mock_policy, samples, "internal", input_mode="table_info")
    print("Internal benchmark (mock policy):", json.dumps(metrics))


def cmd_eval_tabular(args, cfg: Config):
    from src.data.synthetic_tabular import make_items
    from src.evaluation.tabular_eval import (
        compare, evaluate_items, format_report, hf_policy_fn, write_report,
    )

    items = make_items(args.n, seed=args.seed, id_prefix="eval")

    if args.mock:
        base = evaluate_items(items, lambda it: "<answer>-1</answer>")
        trained = evaluate_items(items, lambda it: f"<answer>{it.reference}</answer>")
        result = compare(base, trained)
        result["mock"] = True
    else:
        from config import QuantConfig
        from src.rl.trainer import HFPolicy
        from src.training.quant import load_adapter

        model_id = args.model or cfg.rl.smoke_test_model
        policy = HFPolicy(
            model_id, device=args.device, quant=QuantConfig(enabled=args.qlora)
        )
        policy_fn = hf_policy_fn(policy, args.max_new_tokens)
        base = evaluate_items(items, policy_fn)
        if args.adapter:
            load_adapter(policy.model, args.adapter)
        trained = evaluate_items(items, policy_fn)
        result = compare(base, trained)
        result["adapter"] = args.adapter

    print(format_report(result))
    if args.json:
        write_report(args.json, result)
        print(f"wrote {args.json}")


def cmd_pipeline(args, cfg: Config):
    print("== End-to-end smoke test ==")
    cmd_data(args, cfg)
    print()
    cmd_rewards(args, cfg)
    print()
    cmd_rl_smoke(args, cfg)
    print()
    cmd_agent(args, cfg)
    print()
    cmd_eval(args, cfg)
    print("\nPipeline smoke test passed.")


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TableGPT-R1 reproduction (arXiv:2512.20312)")
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("config", "Print resolved configuration"),
        ("data", "Run the data engineering pipeline on a synthetic fixture"),
        ("rewards", "Demonstrate the task-adaptive reward system"),
        ("rl-smoke", "Verify the hybrid RL objective"),
        ("agent", "Run the agentic think-act-observe loop"),
        ("eval", "Evaluate a policy over a benchmark"),
        ("eval-tabular", "Before/after eval on the synthetic tabular task"),
        ("pipeline", "End-to-end smoke test of all components"),
    ]:
        if name == "eval-tabular":
            et = sub.add_parser(name, help=help_text)
            et.add_argument("--model", default=None, help="HuggingFace model id")
            et.add_argument("--adapter", default=None, help="path to a LoRA adapter")
            et.add_argument("--qlora", action="store_true", default=False)
            et.add_argument("--n", type=int, default=32, help="held-out items")
            et.add_argument("--seed", type=int, default=123)
            et.add_argument("--max-new-tokens", type=int, default=64)
            et.add_argument("--device", default=None)
            et.add_argument("--json", default=None, help="write the report to this path")
            et.add_argument("--mock", action="store_true", default=False,
                            help="run without a model (plumbing smoke test)")
        else:
            sub.add_parser(name, help=help_text)
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config()
    dispatch = {
        "config": cmd_config,
        "data": cmd_data,
        "rewards": cmd_rewards,
        "rl-smoke": cmd_rl_smoke,
        "agent": cmd_agent,
        "eval": cmd_eval,
        "eval-tabular": cmd_eval_tabular,
        "pipeline": cmd_pipeline,
    }
    dispatch[args.command](args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
