# TableGPT-R1 — Training Pipeline

Implementation of the **TableGPT-R1** training pipeline — hybrid GRPO/DAPO/GSPO
RL with task-adaptive rewards — trained and benchmarked end-to-end at 3B QLoRA
scale. Method reference:
[Advancing Tabular Reasoning Through Reinforcement Learning](https://arxiv.org/abs/2512.20312).

> Built to run on commodity hardware: QLoRA (NF4 4-bit + LoRA) with gradient
> checkpointing on a Kaggle T4×2. The full 8B recipe needs A100-class compute.

## Components

| Paper component | Section | Module |
|---|---|---|
| Data collection / composition | 2.1, 2.3 | `src/data/collection.py`, `src/data/composition.py` |
| Two-stage data filtering + difficulty stratification | 2.2 | `src/data/filtering.py` |
| Agentic trajectory format + cleaning rules | 3.2.1, Fig. 4 | `src/data/synthesis.py`, `src/agent/parser.py` |
| Data augmentation (table-input diversity, token alternation, error diversity) | 3.2.2 | `src/data/augmentation.py` |
| Multi-model consensus labeling | 3.2.3 | `src/data/labeling.py` |
| Hybrid RL objective (GRPO + DAPO + GSPO + entropy terms) | 3.3 | `src/rl/objective.py`, `src/rl/advantage.py` |
| Task-adaptive reward routing | 3.4.1, Table 1 | `src/rewards/router.py` |
| Criteria-injected reward model | 3.4.2, Fig. 5 | `src/rewards/criteria_judge.py` |
| Rule-based reward | 3.4.2 | `src/rewards/rule_based.py` |
| Process step reward (+0.1 / −0.1 / −0.2) | 3.4.2 | `src/rewards/process_reward.py` |
| Reward aggregation + policy regularization (incl. `need_plot`) | 3.4.3 | `src/rewards/aggregator.py`, `src/rewards/regularization.py` |
| Multi-stage training (SFT warm-up + 3 RL stages + pass@k filtering) | 3.5 | `src/training/sft.py`, `src/rl/multistage.py` |
| QLoRA / LoRA + 4-bit loading + gradient checkpointing | 3.5 | `src/training/quant.py`, `config.py` (`QuantConfig`) |
| Code execution environment | 3.2.1 | `src/execution/executor.py` |
| Evaluation metrics + benchmark registry | 4 | `src/evaluation/metrics.py`, `src/evaluation/benchmarks.py` |
| Synthetic tabular task (`simple` / `filter_aggregate`) + shaped reward | — | `src/data/synthetic_tabular.py` |
| Tabular before/after eval + 3-arm benchmark (pass@k, CI, MAE/RMSE) | 4 | `src/evaluation/tabular_eval.py`, `src/evaluation/stats.py` |
| Kaggle demo: QLoRA SFT → RL → benchmark | — | `kaggle/train_kaggle.py`, `kaggle/tablegpt_r1_demo.ipynb` |
| Inference agent loop | 3.2.1 | `src/agent/loop.py` |

## Install

```bash
# from the project root, using the existing venv or a fresh one
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows PowerShell
pip install -r requirements.txt

# CPU-only torch (lighter) if you don't need CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Optional (for full 8B training / SFT / HF model loading): `transformers`, `datasets`, `accelerate`, `peft`, `trl` are in `requirements.txt`; they are imported lazily so the core pipeline and tests run without them.

## Usage

```bash
python main.py config      # print resolved configuration
python main.py data        # data filtering + difficulty demo on a synthetic fixture
python main.py rewards     # task-adaptive reward demo (routing, process, regularization)
python main.py rl-smoke    # verify the hybrid RL objective on hand-computed tensors
python main.py agent       # closed-loop think-act-observe demo
python main.py eval        # evaluate a policy over a benchmark (mock policy)
python main.py eval-tabular --mock   # synthetic tabular eval (no model needed)
python main.py pipeline    # end-to-end smoke test of all components
```

### Synthetic tabular benchmark

`eval-tabular` reports a multi-metric table for up to three arms — `base`,
`sft`, `sft+rl` — over the offline synthetic tabular task:

```bash
# plumbing smoke test (deterministic, no model/GPU)
python main.py eval-tabular --mock --task filter_aggregate --pass-k 8 --n 16

# full benchmark against real adapters (GPU + transformers/peft/bitsandbytes)
python main.py eval-tabular \
    --model Qwen/Qwen2.5-3B --qlora --task filter_aggregate \
    --sft-adapter outputs/kaggle_demo/sft_adapter \
    --adapter outputs/kaggle_demo/rl_adapter \
    --n 256 --pass-k 8 --max-new-tokens 128 \
    --json outputs/eval_report.json
```

Metrics: accuracy, format_rate, per-op accuracy, MAE/RMSE, Wilson 95% CI, and
(when `--pass-k > 0`) pass@k / pass@1. Efficiency is captured per arm
(generation seconds, tokens/s, peak VRAM); training time and adapter sizes are
read from `run_info.json` next to the adapter. `--task` selects `simple`
(default) or the harder multi-step `filter_aggregate` variant.

Quickstart script:

```bash
python examples/quickstart.py
```

## Tests

```bash
pytest tests/ -v
```

119 tests pass (1 skipped), covering the data pipeline, reward system, RL
objective (including hand-computed loss/ratio/clip/entropy checks), execution
sandbox, agent loop, metrics, multi-stage orchestration, QLoRA/quantization
helpers, SFT label masking, the synthetic tabular task + shaped reward, the
evaluation statistics, and the Kaggle `run_info.json` writer. All are
CPU-only; GPU/model paths are lazy-imported and skipped when dependencies are
absent.

## QLoRA + Kaggle demo

`kaggle/train_kaggle.py` runs the recipe at small policy scale on a free Kaggle
T4, and `kaggle/tablegpt_r1_demo.ipynb` drives it end to end:

```bash
python kaggle/train_kaggle.py --policy-size 3b --qlora --task filter_aggregate \
    --sft-samples 256 --sft-epochs 3 --rl-steps 40 --out outputs/kaggle_demo
```

- QLoRA (NF4 4-bit base + LoRA adapters) with gradient checkpointing so a 3B
  model fits the T4's 16 GB. T4 (Turing) has no bf16, so the code auto-selects
  **fp16** for 4-bit compute + AMP; Ampere+/A100 uses bf16.
- SFT trains on the **completion only** (prompt masked); RL uses shaped
  rewards over the synthetic tabular task.
- Emits `run_info.json` (model, timings, trainable %, adapter sizes) next to
  the adapters, plus training curves; the notebook's `6b` cell then prints the
  `base` / `sft` / `sft+rl` benchmark table.

The notebook expects the repo uploaded as a Kaggle Dataset; cell 3 locates
`/kaggle/input/**/tablegpt_r1` automatically.

## Results — 3B QLoRA on Kaggle T4×2

**Run:** `Qwen2.5-3B`, NF4 4-bit + LoRA, fp16 compute (Turing T4 has no bf16),
512 synthetic training items, SFT on 256 examples × 3 epochs, then 40 RL steps
with group size 8 on the `filter_aggregate` task.

### SFT warm-up

The warm-up is where the task is actually learned. Loss falls **5.19 → ≈0.03**
with grad-norm **9.1 → <1** over **48 optimizer steps** (~17 min, ≈21 s/step
with gradient checkpointing on T4×2). Training on the completion only (prompt
masked) is what drives the policy to reliably emit the
`<think>…</think>\n<answer>N</answer>` format — final epoch-average loss 0.43,
reflecting the steep early curve.

### RL stage (hybrid GRPO/DAPO/GSPO, 40 steps)

Rewards are **non-degenerate**: roughly half the steps produced unanimous
groups (reward 1.0 → zero advantage → no update), while the rest mixed between
0.2 and 0.9. But the importance ratio stayed at **≈1.000** and entropy stayed
**below 0.17**, i.e. update magnitude was tiny and the policy barely moved from
where SFT left it. One step (27) shows the “wrong answer with a valid tag”
penalty firing (loss ≈ 0.6, reward 0.2).

### 3-arm benchmark (base / SFT-only / SFT+RL)

`python main.py eval-tabular` scores all three arms on held-out items (disjoint
from training) with accuracy, format_rate, per-op accuracy, MAE/RMSE, Wilson
95% CI, and pass@k (8 samples/item), plus per-arm inference speed,
tokens/second and peak VRAM; training time and adapter sizes are read from
`run_info.json`. Every number lands in `eval_report.json`.

### Honest finding

> At 3B QLoRA scale, **the SFT warm-up alone solves the synthetic task, and the
> hybrid RL stage is correctly implemented but effectively neutral** — the SFT
> policy already saturates the reward, so most rollout groups carry no learning
> signal. Closing that gap needs the paper’s scale: an 8B policy, real
> datasets, and A100-class compute for the full 200-step stages. Free T4
> sessions (and a local 4 GB GPU) cannot host it, so this run demonstrates the
> *method and the measurement*, not the reported benchmark scores.

## Project structure

```
tablegpt_r1/
├── main.py                 # CLI entry point
├── config.py               # all hyperparameters (paper values + [INFERRED] defaults, QuantConfig)
├── src/
│   ├── constants.py        # special tokens, task taxonomy, Table 1 routing
│   ├── data/               # schema, collection, filtering, composition, synthesis, augmentation, labeling, synthetic_tabular
│   ├── execution/          # sandboxed Python executor, error formatting
│   ├── rewards/            # router, rule-based, criteria judge, process reward, regularization, aggregator
│   ├── rl/                 # advantage, hybrid objective, trainer, multi-stage
│   ├── training/           # SFT warm-up, QLoRA/quantization helpers
│   ├── agent/              # special-token parser, inference loop
│   └── evaluation/         # metrics, benchmark registry, tabular_eval, stats
├── kaggle/                 # train_kaggle.py + tablegpt_r1_demo.ipynb
├── tests/                  # pytest suite
└── examples/               # quickstart + sample table
```

## Key hyperparameters (from the paper)

| Parameter | Value | Source |
|---|---|---|
| Base model | Qwen3-8B | §3.5, HF card |
| Max sequence length | 8192 tokens | §2.2 |
| Quality retention threshold | ≥ 0.7 | §2.2 |
| Difficulty rollouts | 5 (`m_bar` mean) | §2.2 |
| Difficulty thresholds | high <0.3, medium [0.3,0.7], low >0.7 | §2.2 |
| Difficulty target | 20% / 60% / 20% | §2.2 |
| Data composition | General 5%, Math 20%, Code 5%, Table 30%, SQL 10%, QA 20%, Agent 10% | Fig. 3 |
| Consensus models | DeepSeek, GPT-4o, Qwen-Max (≥2 agree) | §3.2.3 |
| Process rewards | −0.2 / +0.1 / −0.1 | §3.4.2 |
| Entropy coefficients | `C_H = η = 1e-3` | §3.3 |
| Entropy activation | after step 50 | §3.3 |
| SFT warm-up | ~3% of dataset | §3.5 |
| Stage termination | step 200 | §3.5 |
| pass@k retention | 3–7 | §3.5 |
| Stage 3 threshold | Stage 2 score < 5 | §3.5 |

## Paper details not specified (defaults used)

The source paper renders several equations as images (not captured as text) and
omits some training hyperparameters. These are implemented as configurable
defaults (all overridable in `config.py`):

- **Exact RL objective** — derived from the published description + cited GRPO/DAPO/GSPO.
- **Asymmetric clip bounds** — `eps_low=0.2`, `eps_high=0.28` (DAPO-style).
- **Group size** — `G=8`.
- **Optimizer / LR** — AdamW, `lr=1e-6`, cosine decay, grad-clip 1.0.
- **Regularization weights** — length/repetition/wait/plot penalties (paper gives only qualitative description).
- **Teacher/judge model** — pluggable API client (`gpt-4o` by default).
- **Code sandbox** — subprocess with timeout + restricted denylist.
- **pass@k k** — `k=8`.

## Full-scale recipe (beyond this demo)

The full run needs the public datasets (Spider, BIRD, TableBench, RealHitBench,
InfiAgent-DABench, and the general suite) plus the 8B model and A100-class
compute. `src/evaluation/benchmarks.py` holds the benchmark registry and the
reference targets. Wire dataset loaders in `src/data/collection.py` and run
`python main.py eval`.

Reference targets (TableGPT-R1-8B): Internal Table Info 80.00 / Table Path 82.70;
Spider 86.73; BIRD 63.17; HumanEval 95.73; GSM8K 95.60; MATH 93.30; AIME 50.00.
