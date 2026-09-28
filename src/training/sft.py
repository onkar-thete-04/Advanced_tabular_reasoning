"""Supervised fine-tuning warm-up (Section 3.5).

Direct RL on the base model often produces a high rate of syntactically invalid
rollouts. An SFT warm-up on ~3% of the full dataset reinforces consistent
adherence to the target output format (reasoning steps, structured code
execution blocks, formatted final answers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from config import TrainingConfig
from src.data.schema import Sample, Trajectory


@dataclass
class SFTExample:
    prompt: str
    completion: str


def _collate(batch, tokenizer):
    """Pad a batch of tokenized examples to equal length."""
    import torch

    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(list(b["input_ids"]) + [tokenizer.pad_token_id] * pad)
        attn.append(list(b["attention_mask"]) + [0] * pad)
        labels.append(list(b["labels"]) + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn),
        "labels": torch.tensor(labels),
    }


def trajectory_to_sft_example(traj: Trajectory) -> SFTExample:
    """Convert a validated trajectory into a (prompt, completion) pair."""
    prompt = f"{traj.question}\n{traj.table_ref}"
    completion = "\n".join(s.to_text() for s in traj.steps)
    return SFTExample(prompt=prompt, completion=completion)


def mask_prompt_labels(
    input_ids: Sequence[int], prompt_len: int, ignore_index: int = -100
) -> List[int]:
    """Copy ``input_ids`` but replace the first ``prompt_len`` ids with
    ``ignore_index`` so the loss is computed on the completion only."""
    labels = list(input_ids)
    n = min(max(int(prompt_len), 0), len(labels))
    for i in range(n):
        labels[i] = ignore_index
    return labels


def encode_sft_example(example: SFTExample, tokenizer, max_length: int = 8192) -> dict:
    """Tokenize an example with prompt labels masked out (completion-only loss)."""
    text = f"{example.prompt}\n{example.completion}{tokenizer.eos_token}"
    enc = dict(tokenizer(text, truncation=True, max_length=max_length))
    prefix = tokenizer(example.prompt + "\n", truncation=True, max_length=max_length)
    enc["labels"] = mask_prompt_labels(enc["input_ids"], len(prefix["input_ids"]))
    return enc


class SFTWarmup:
    """Builds and (optionally) runs the SFT warm-up."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.examples: List[SFTExample] = []

    def build_dataset(self, trajectories: Sequence[Trajectory]) -> List[SFTExample]:
        self.examples = [trajectory_to_sft_example(t) for t in trajectories]
        return self.examples

    def train(
        self,
        trajectories: Sequence[Trajectory],
        model_name: Optional[str] = None,
        quant=None,
        output_dir: Optional[str] = None,
        model=None,
        tokenizer=None,
    ) -> None:
        """Run HF supervised fine-tuning.

        Imported lazily; if transformers is unavailable this raises a clear
        error rather than silently doing nothing. When ``quant`` is an enabled
        ``QuantConfig`` the model is loaded in 4-bit with LoRA adapters and only
        the adapter is saved at the end (survives the Kaggle 12 h cutoff).

        Pass an existing ``model``/``tokenizer`` to warm up a policy in place
        (used by the Kaggle demo so SFT feeds RL without reloading the model).
        """
        self.build_dataset(trajectories)
        if not self.examples:
            return
        try:
            from transformers import Trainer, TrainingArguments
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "transformers is required for SFT warm-up. Install it or use a "
                "custom sft_fn in MultiStageTrainer."
            ) from exc

        import torch
        from torch.utils.data import Dataset
        from src.training.quant import is_peft_model, preferred_amp_dtype

        model_id = model_name or "Qwen/Qwen3-8B"
        out_dir = output_dir or "outputs/sft_warmup"

        if model is None:
            use_qlora = bool(quant is not None and getattr(quant, "enabled", False))
            if use_qlora:
                from src.training.quant import load_causal_lm

                model, tokenizer = load_causal_lm(model_id, qcfg=quant, trainable=True)
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(model_id)
        elif tokenizer is None:
            raise ValueError("tokenizer must be provided with an existing model")
        else:
            use_qlora = is_peft_model(model)

        class _DS(Dataset):
            def __init__(self, rows: List[SFTExample]):
                self.rows = rows

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, idx: int):
                return encode_sft_example(self.rows[idx], tokenizer)

        args_kwargs = dict(
            output_dir=out_dir,
            num_train_epochs=self.config.sft_epochs,
            learning_rate=self.config.sft_learning_rate,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            logging_steps=1,
            report_to=[],
        )
        if use_qlora:
            amp = preferred_amp_dtype()          # bf16 on Ampere+, fp16 on T4
            on_cuda = bool(torch.cuda.is_available())
            args_kwargs.update(
                save_strategy="steps",
                save_steps=50,
                save_total_limit=1,
                gradient_checkpointing=bool(getattr(quant, "gradient_checkpointing", True)),
                bf16=on_cuda and amp == "bfloat16",
                fp16=on_cuda and amp == "float16",
                optim=(
                    "paged_adamw_8bit"
                    if getattr(quant, "use_paged_optimizer", True)
                    else "adamw_torch"
                ),
            )
        else:
            args_kwargs.update(save_strategy="no")

        args = TrainingArguments(**args_kwargs)
        trainer = Trainer(model=model, args=args, train_dataset=_DS(self.examples),
                          data_collator=lambda batch: _collate(batch, tokenizer))
        trainer.train()
        if use_qlora and is_peft_model(model):
            model.save_pretrained(out_dir)
