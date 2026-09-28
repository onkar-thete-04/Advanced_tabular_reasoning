"""Central configuration for TableGPT-R1.

All values are taken from the paper where specified; unspecified values are
marked with their [INFERRED] default and documented in
paper_workspace/01_algorithm_extraction.yaml -> missing_but_critical.

Usage:
    from config import Config
    cfg = Config()
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class DataConfig:
    """Section 2: Training Data Construction."""

    # Section 2.2 - Input-based filtering
    max_seq_length_tokens: int = 8192
    dedup_use_minhash: bool = True
    minhash_num_perm: int = 128          # [INFERRED] standard MinHash default
    minhash_threshold: float = 0.9       # [INFERRED] near-duplicate threshold

    # Section 2.2 - Output-based filtering
    quality_retention_threshold: float = 0.7
    difficulty_num_rollouts: int = 5
    difficulty_high_max: float = 0.3
    difficulty_medium_min: float = 0.3
    difficulty_medium_max: float = 0.7
    difficulty_low_min: float = 0.7

    # Section 2.2 - target distribution
    target_high: float = 0.20
    target_medium: float = 0.60
    target_low: float = 0.20

    # Section 3.2.3 - multi-model consensus
    consensus_models: List[str] = field(
        default_factory=lambda: ["deepseek", "gpt-4o", "qwen-max"]
    )
    consensus_min_agree: int = 2

    # Section 3.2.2 - augmentation toggles
    augment_table_input_diversity: bool = True
    augment_system_prompt_variation: bool = True
    augment_special_token_alternation: bool = True
    augment_error_message_diversity: bool = True

    # Agentic cleaning limits (Figure 4)
    max_interaction_rounds: int = 10      # [INFERRED] token/interaction limit
    max_trajectory_tokens: int = 8192


@dataclass
class RewardConfig:
    """Section 3.4: Task-Adaptive Reward System."""

    # Process step reward (Section 3.4.2)
    reward_wrong_function: float = -0.2
    reward_correct_progress: float = 0.1
    reward_correct_exec_fail: float = -0.1

    # Criteria-injected reward model (Figure 5)
    judge_score_min: int = 0
    judge_score_max: int = 10
    judge_use_api: bool = True
    judge_model: str = "gpt-4o"           # [INFERRED] teacher/judge endpoint
    judge_temperature: float = 0.0

    # Regularization R_reg(t) (Section 3.4.3) - weights qualitative in paper
    reg_length_weight: float = 0.01       # [INFERRED]
    reg_repetition_weight: float = 0.1    # [INFERRED]
    reg_wait_token_weight: float = 0.05   # [INFERRED]
    reg_plot_weight: float = 1.0          # [INFERRED] strong negative
    reg_length_budget_tokens: int = 2048  # [INFERRED] per-step budget

    # Terminal reward scaling for rule-based 0/1 -> numeric
    rule_correct_reward: float = 1.0
    rule_incorrect_reward: float = 0.0

    # Data-quality scoring scale (Section 2.2 uses 0-1)
    quality_score_max: float = 1.0


@dataclass
class RLConfig:
    """Section 3.3: Reinforcement Learning Algorithm (GRPO + DAPO + GSPO)."""

    base_model: str = "Qwen/Qwen3-8B"
    smoke_test_model: str = "Qwen/Qwen3-0.6B"   # [INFERRED] tiny dev model
    group_size: int = 8                          # [INFERRED] rollouts per prompt
    clip_epsilon_low: float = 0.2                # [INFERRED] DAPO lower bound
    clip_epsilon_high: float = 0.28              # [INFERRED] DAPO upper bound
    entropy_bonus_C_H: float = 1e-3              # Section 3.3
    entropy_suppression_eta: float = 1e-3        # Section 3.3
    entropy_terms_start_step: int = 50           # Section 3.3
    learning_rate: float = 1e-6                  # [INFERRED]
    weight_decay: float = 0.0                    # [INFERRED]
    max_grad_norm: float = 1.0                   # [INFERRED]
    optimizer: str = "adamw"                     # [INFERRED]
    lr_schedule: str = "cosine"                  # [INFERRED]
    warmup_steps: int = 0                        # [INFERRED]
    max_new_tokens: int = 2048                   # [INFERRED]
    temperature: float = 0.8                     # [INFERRED]
    seed: int = 42


@dataclass
class TrainingConfig:
    """Section 3.5: Multi-stage Training Framework."""

    sft_warmup_fraction: float = 0.03            # Section 3.5
    sft_epochs: int = 1                          # [INFERRED]
    sft_learning_rate: float = 2e-4              # [INFERRED] LoRA-appropriate (was 1e-5)
    stage_termination_step: int = 200            # Section 3.5
    num_rl_stages: int = 3
    passk_retain_min: int = 3                    # Section 3.5
    passk_retain_max: int = 7                    # Section 3.5
    passk_k: int = 8                             # [INFERRED]
    stage3_threshold_score: float = 5.0          # Section 3.5 (below 5)
    stage1_general_ratio: float = 0.8            # [INFERRED] "dominated by"
    stage2_table_ratio: float = 0.8              # [INFERRED] "significantly increases"


@dataclass
class ExecConfig:
    """Code execution environment (Section 3.2.1 / 3.4.2)."""

    timeout_seconds: int = 30
    max_output_chars: int = 4000
    allow_network: bool = False
    use_subprocess: bool = True
    error_style: str = "full"                    # "minimal" | "full"


@dataclass
class EvalConfig:
    """Section 4: Evaluation."""

    table_benchmarks: List[str] = field(
        default_factory=lambda: [
            "spider", "bird", "tablebench", "realhitbench",
            "infiagent-da", "internal",
        ]
    )
    general_benchmarks: List[str] = field(
        default_factory=lambda: [
            "humaneval", "mbpp", "gsm8k", "math", "aime",
            "cmmlu-5", "gpqa-diamond",
        ]
    )
    internal_input_modes: List[str] = field(
        default_factory=lambda: ["table_info", "table_path"]
    )
    max_new_tokens: int = 1024


@dataclass
class QuantConfig:
    """Optional parameter-efficient training (QLoRA / LoRA).

    Not specified in the paper (which uses full fine-tuning); these are
    [INFERRED] defaults enabling the recipe to run on consumer/Kaggle GPUs.
    All HuggingFace / peft / bitsandbytes imports stay lazy so the CPU-only
    test suite never depends on them.
    """

    enabled: bool = False
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"          # [INFERRED] QLoRA NF4
    bnb_4bit_use_double_quant: bool = True    # [INFERRED]
    bnb_4bit_compute_dtype: str = "auto"      # "auto" -> bf16 if GPU supports, else fp16
    lora_r: int = 16                          # [INFERRED]
    lora_alpha: int = 32                      # [INFERRED]
    lora_dropout: float = 0.05                # [INFERRED]
    target_modules: str = "auto"              # "auto" | "all-linear" | CSV list
    gradient_checkpointing: bool = True       # [INFERRED]
    use_paged_optimizer: bool = True          # [INFERRED] bitsandbytes paged AdamW


@dataclass
class Config:
    """Root config aggregating all sections."""

    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    exec: ExecConfig = field(default_factory=ExecConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)

    output_dir: str = "outputs"
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            data=DataConfig(**d.get("data", {})),
            reward=RewardConfig(**d.get("reward", {})),
            rl=RLConfig(**d.get("rl", {})),
            training=TrainingConfig(**d.get("training", {})),
            exec=ExecConfig(**d.get("exec", {})),
            eval=EvalConfig(**d.get("eval", {})),
            quant=QuantConfig(**d.get("quant", {})),
            output_dir=d.get("output_dir", "outputs"),
            seed=d.get("seed", 42),
        )


DEFAULT_CONFIG = Config()
