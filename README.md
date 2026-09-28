# TableGPT-R1 — Reproduction

Paper-faithful implementation of **TableGPT-R1: Advancing Tabular Reasoning Through Reinforcement Learning** ([arXiv:2512.20312](https://arxiv.org/abs/2512.20312)).

> No official *training* code was released (only model weights at
> [`tablegpt/TableGPT-R1`](https://huggingface.co/tablegpt/TableGPT-R1)), so this
> repository reconstructs the training framework from the paper specification.

## What is reproduced

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
| Code execution environment | 3.2.1 | `src/execution/executor.py` |
| Evaluation metrics + benchmark registry | 4 | `src/evaluation/metrics.py`, `src/evaluation/benchmarks.py` |
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
python main.py pipeline    # end-to-end smoke test of all components
```

Quickstart script:

```bash
python examples/quickstart.py
```

## Tests

```bash
pytest tests/ -v
```

58 tests cover the data pipeline, reward system, RL objective (including
hand-computed loss/ratio/clip/entropy checks), execution sandbox, agent loop,
metrics, and multi-stage orchestration.

## Project structure

```
tablegpt_r1/
├── main.py                 # CLI entry point
├── config.py               # all hyperparameters (paper values + [INFERRED] defaults)
├── src/
│   ├── constants.py        # special tokens, task taxonomy, Table 1 routing
│   ├── data/               # schema, collection, filtering, composition, synthesis, augmentation, labeling
│   ├── execution/          # sandboxed Python executor, error formatting
│   ├── rewards/            # router, rule-based, criteria judge, process reward, regularization, aggregator
│   ├── rl/                 # advantage, hybrid objective, trainer, multi-stage
│   ├── training/           # SFT warm-up
│   ├── agent/              # special-token parser, inference loop
│   └── evaluation/         # metrics, benchmark registry
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
defaults and documented in `paper_workspace/01_algorithm_extraction.yaml`:

- **Exact RL objective** — reconstructed from prose + cited GRPO/DAPO/GSPO.
- **Asymmetric clip bounds** — `eps_low=0.2`, `eps_high=0.28` (DAPO-style).
- **Group size** — `G=8`.
- **Optimizer / LR** — AdamW, `lr=1e-6`, cosine decay, grad-clip 1.0.
- **Regularization weights** — length/repetition/wait/plot penalties (paper gives only qualitative description).
- **Teacher/judge model** — pluggable API client (`gpt-4o` by default).
- **Code sandbox** — subprocess with timeout + restricted denylist.
- **pass@k k** — `k=8`.

## Reproducing paper results

Full reproduction requires the public datasets (Spider, BIRD, TableBench,
RealHitBench, InfiAgent-DABench, and the general suite) plus the 8B model and a
large GPU. `src/evaluation/benchmarks.py` defines the benchmark registry and the
Table 2/3 reference targets. Wire dataset loaders in
`src/data/collection.py` and run `python main.py eval`.

Reference targets (TableGPT-R1-8B): Internal Table Info 80.00 / Table Path 82.70;
Spider 86.73; BIRD 63.17; HumanEval 95.73; GSM8K 95.60; MATH 93.30; AIME 50.00.

## Artifacts

Phase 0–3 analysis lives in `../paper_workspace/`:
`reference_search.yaml`, `01_algorithm_extraction.yaml`, `02_concept_analysis.yaml`,
`03_implementation_plan.yaml`.
