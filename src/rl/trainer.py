"""RL training loop (Section 3.3).

Implements the rollout -> reward -> group-advantage -> hybrid-objective update
cycle. HuggingFace is imported lazily so the objective/advantage modules remain
usable without a model installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from config import RLConfig
from src.rl.advantage import group_advantage
from src.rl.objective import HybridRLObjective, RLObjectiveOutput


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # (N, T)
    attention_mask: torch.Tensor     # (N, T)
    response_mask: torch.Tensor      # (N, T) 1 for response tokens
    responses: List[str] = field(default_factory=list)
    metas: List[dict] = field(default_factory=list)
    group_size: int = 1

    @property
    def num_sequences(self) -> int:
        return self.input_ids.shape[0]


class HFPolicy:
    """Thin wrapper around a HuggingFace causal LM.

    Passing ``quant`` (a ``QuantConfig`` with ``enabled=True``) loads the model
    in 4-bit with LoRA adapters and gradient checkpointing, so only the adapters
    are trainable. ``RLTrainer`` transparently picks up the adapter parameters
    because it optimises only ``requires_grad`` tensors.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trainable: bool = True,
        quant=None,
        gradient_checkpointing: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_peft = False
        self.gradient_checkpointing = bool(gradient_checkpointing)

        if quant is not None and getattr(quant, "enabled", False):
            from dataclasses import replace

            from src.training.quant import is_peft_model, load_causal_lm

            qcfg = replace(quant, gradient_checkpointing=True) if gradient_checkpointing else quant
            self.model, self.tokenizer = load_causal_lm(
                model_name, qcfg=qcfg, device=self.device, dtype=dtype, trainable=trainable
            )
            self.is_peft = is_peft_model(self.model)
            self.gradient_checkpointing = bool(qcfg.gradient_checkpointing)
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype or torch.float32
        ).to(self.device)
        if self.gradient_checkpointing:
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
        self.model.train(trainable)

    def generate_rollouts(
        self, prompt_texts: List[str], group_size: int, config: RLConfig
    ) -> RolloutBatch:
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.device)
        prompt_len = enc.input_ids.shape[1]

        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=config.temperature,
                num_return_sequences=group_size,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        self.model.train()

        pad_id = self.tokenizer.pad_token_id
        full_attention = (out != pad_id).long()
        response_mask = torch.zeros_like(out)
        response_mask[:, prompt_len:] = 1
        response_mask = response_mask * (out != pad_id).long()

        responses = self.tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        return RolloutBatch(
            input_ids=out,
            attention_mask=full_attention,
            response_mask=response_mask,
            responses=responses,
            group_size=group_size,
        )

    def token_logprobs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return per-token log-probs aligned to input positions, shape (N, T)."""
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]
        gathered = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (N, T-1)
        return torch.nn.functional.pad(gathered, (1, 0))  # (N, T)


class RLTrainer:
    """Drives RL optimization with the hybrid objective."""

    def __init__(
        self,
        policy,
        reward_fn: Callable[[dict, str], float],
        config: Optional[RLConfig] = None,
        objective: Optional[HybridRLObjective] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.policy = policy
        self.reward_fn = reward_fn
        self.config = config or RLConfig()
        self.objective = objective or HybridRLObjective(self.config)
        self.step = 0
        params = (
            [p for p in policy.model.parameters() if p.requires_grad]
            if hasattr(policy, "model")
            else []
        )
        self.optimizer = optimizer or (
            torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            if params
            else None
        )

    def _group_advantages(self, rewards: List[float], group_size: int) -> torch.Tensor:
        t = torch.tensor(rewards, dtype=torch.float32)
        n_groups = len(rewards) // group_size
        reshaped = t.view(n_groups, group_size)
        adv = group_advantage(reshaped)
        return adv.view(-1)

    def train_step(
        self, prompt_texts: List[str], metas: List[dict]
    ) -> Dict[str, float]:
        group_size = self.config.group_size
        rollout = self.policy.generate_rollouts(prompt_texts, group_size, self.config)

        # Rewards: generate returns responses in prompt-major order.
        rewards = [
            float(self.reward_fn(meta, resp))
            for meta, resp in zip(
                [m for m in metas for _ in range(group_size)], rollout.responses
            )
        ]
        advantages = self._group_advantages(rewards, group_size).to(rollout.input_ids.device)

        with torch.no_grad():
            old_logprobs = self.policy.token_logprobs(
                rollout.input_ids, rollout.attention_mask
            )
        logprobs = self.policy.token_logprobs(rollout.input_ids, rollout.attention_mask)

        out: RLObjectiveOutput = self.objective(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=rollout.response_mask,
            step=self.step,
        )

        if self.optimizer is not None:
            self.optimizer.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                self.config.max_grad_norm,
            )
            self.optimizer.step()

        self.step += 1
        return {
            "step": self.step,
            "loss": float(out.loss.detach()),
            "policy_loss": float(out.policy_loss.detach()),
            "mean_reward": float(sum(rewards) / max(1, len(rewards))),
            "mean_ratio": float(out.mean_ratio.detach()),
            "mean_entropy": float(out.mean_entropy.detach()),
            "entropy_active": float(out.entropy_active),
        }
