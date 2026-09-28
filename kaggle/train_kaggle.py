"""Kaggle / GPU demo entry point for TableGPT-R1.

Runs the paper's Section 3.5 ordering — an SFT warm-up followed by one RL stage —
on a deterministic synthetic tabular task with an optional QLoRA policy. The SFT
warm-up trains the *same* policy instance in place, so RL continues from the
warmed-up weights without reloading the model.

Example (Kaggle, T4 x2):
    python kaggle/train_kaggle.py --policy-size 3b --qlora --rl-steps 20

Outputs (under ``--out``): metrics.json, metrics.csv, optional curves.png, and
the saved LoRA adapter(s).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config, QuantConfig  # noqa: E402

POLICY_SIZES = {
    "0.5b": "Qwen/Qwen2.5-0.5B",
    "1.5b": "Qwen/Qwen2.5-1.5B",
    "3b": "Qwen/Qwen2.5-3B",
    "7b": "Qwen/Qwen2.5-7B",
    "8b": "Qwen/Qwen3-8B",
}

METRIC_COLUMNS = [
    "step", "loss", "policy_loss", "mean_reward",
    "mean_ratio", "mean_entropy", "entropy_active",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TableGPT-R1 Kaggle demo trainer")
    p.add_argument("--policy-size", choices=sorted(POLICY_SIZES), default="3b")
    p.add_argument("--model", default=None, help="override the HuggingFace model id")
    p.add_argument("--qlora", dest="qlora", action="store_true", default=True)
    p.add_argument("--no-qlora", dest="qlora", action="store_false")
    p.add_argument("--no-sft", dest="sft", action="store_false", default=True)
    p.add_argument("--sft-samples", type=int, default=256)
    p.add_argument("--sft-epochs", type=int, default=3)
    p.add_argument("--rl-steps", type=int, default=40)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--task", choices=["simple", "filter_aggregate"], default="simple")
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-eval", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs/kaggle_demo")
    p.add_argument("--plot", dest="plot", action="store_true", default=True)
    p.add_argument("--no-plot", dest="plot", action="store_false")
    return p


def _write_metrics(out_dir: str, history: List[dict]) -> None:
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if not history:
        return
    columns = [c for c in METRIC_COLUMNS if any(c in row for row in history)]
    columns += sorted({k for row in history for k in row if k not in columns})
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _maybe_plot(out_dir: str, history: List[dict]) -> Optional[str]:
    if not history:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib unavailable; skipping curves.png")
        return None

    steps = [int(r.get("step", i + 1)) for i, r in enumerate(history)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, [r.get("mean_reward", 0.0) for r in history])
    axes[0].set_title("mean reward")
    axes[1].plot(steps, [r.get("loss", 0.0) for r in history])
    axes[1].set_title("loss")
    axes[2].plot(steps, [r.get("mean_entropy", 0.0) for r in history])
    axes[2].set_title("mean entropy")
    for ax in axes:
        ax.set_xlabel("RL step")
    fig.tight_layout()
    path = os.path.join(out_dir, "curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] curves -> {path}")
    return path


def _adapter_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return round(total / 1e6, 3)


def write_run_info(out_dir: str, info: dict) -> str:
    path = os.path.join(out_dir, "run_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    cfg = Config()
    cfg.rl.group_size = args.group_size
    cfg.rl.max_new_tokens = args.max_new_tokens
    cfg.training.sft_epochs = args.sft_epochs
    qcfg = QuantConfig(enabled=args.qlora)

    model_id = args.model or POLICY_SIZES[args.policy_size]
    print(f"[setup] model={model_id} qlora={args.qlora} sft={args.sft} "
          f"device={args.device or 'auto'} out={args.out}")

    from src.data.synthetic_tabular import (
        item_meta, item_prompt, make_items, make_reward_fn, make_trajectories,
    )
    from src.rl.trainer import HFPolicy, RLTrainer
    from src.training.loop import run_sft_then_rl
    from src.training.quant import is_peft_model, preferred_amp_dtype
    from src.training.sft import SFTWarmup

    if args.qlora:
        amp = preferred_amp_dtype()
        print(f"[precision] 4-bit compute + mixed precision = {amp} "
              f"(Turing/T4 -> fp16, Ampere+ -> bf16)")

    train_items = make_items(
        args.n_train, seed=args.seed, id_prefix="train", task=args.task
    )
    eval_items = make_items(
        args.n_eval, seed=args.seed + 10_000, id_prefix="eval",
        exclude={it.signature for it in train_items}, task=args.task,
    )
    print(f"[data] train={len(train_items)} eval={len(eval_items)} "
          f"(disjoint signatures)")

    policy = HFPolicy(
        model_id, device=args.device, quant=qcfg,
        gradient_checkpointing=qcfg.gradient_checkpointing,
    )

    def trainable_count() -> int:
        return sum(p.numel() for p in policy.model.parameters() if p.requires_grad)

    total = sum(p.numel() for p in policy.model.parameters())
    print(f"[model] trainable={trainable_count():,} / total={total:,} "
          f"({100.0 * trainable_count() / max(1, total):.3f}%)")

    trainer = RLTrainer(policy, reward_fn=make_reward_fn(cfg.reward), config=cfg.rl)
    sft = SFTWarmup(cfg.training)

    timing = {"sft_seconds": 0.0, "rl_seconds": 0.0}

    def sft_fn() -> None:
        started = time.perf_counter()
        print(f"[sft] warm-up on {min(args.sft_samples, len(train_items))} examples "
              f"x {args.sft_epochs} epoch(s)")
        sft.train(
            make_trajectories(train_items[: args.sft_samples]),
            model=policy.model,
            tokenizer=policy.tokenizer,
            quant=qcfg,
            output_dir=os.path.join(args.out, "sft_adapter"),
        )
        timing["sft_seconds"] = time.perf_counter() - started

    history: List[dict] = []

    def on_step(metrics: dict) -> None:
        history.append(dict(metrics))
        print(
            f"[rl] step {int(metrics.get('step', len(history))):>3}  "
            f"loss={metrics.get('loss', 0.0):+.4f}  "
            f"reward={metrics.get('mean_reward', 0.0):.3f}  "
            f"ratio={metrics.get('mean_ratio', 0.0):.3f}  "
            f"entropy={metrics.get('mean_entropy', 0.0):.3f}"
        )

    started_all = time.perf_counter()
    run_sft_then_rl(
        trainer, train_items, item_prompt, item_meta,
        rl_steps=args.rl_steps, prompts_per_step=args.prompts_per_step,
        sft_fn=sft_fn if args.sft else None, on_step=on_step,
    )
    timing["rl_seconds"] = max(0.0, time.perf_counter() - started_all - timing["sft_seconds"])

    _write_metrics(args.out, history)
    if args.plot:
        _maybe_plot(args.out, history)

    if is_peft_model(policy.model):
        adapter_dir = os.path.join(args.out, "rl_adapter")
        policy.model.save_pretrained(adapter_dir)
        print(f"[save] adapter -> {adapter_dir}")

    trainable = trainable_count()
    info = {
        "model": model_id,
        "task": args.task,
        "qlora": args.qlora,
        "sft_seconds": round(timing["sft_seconds"], 2),
        "rl_seconds": round(timing["rl_seconds"], 2),
        "trainable_params": trainable,
        "trainable_pct": round(100.0 * trainable / max(1, total), 4),
        "sft_adapter_mb": _adapter_mb(os.path.join(args.out, "sft_adapter")),
        "rl_adapter_mb": _adapter_mb(os.path.join(args.out, "rl_adapter")),
        "n_train": len(train_items),
        "n_eval": len(eval_items),
        "sft_samples": args.sft_samples,
        "sft_epochs": args.sft_epochs,
        "rl_steps": args.rl_steps,
        "group_size": args.group_size,
    }
    print(f"[save] run_info -> {write_run_info(args.out, info)}")

    print(f"[done] {len(history)} RL steps. Artifacts in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
