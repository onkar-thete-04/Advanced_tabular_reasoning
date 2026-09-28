"""Hybrid RL optimization objective (Section 3.3).

Adapts Group Relative Policy Optimization (GRPO) with improvements from DAPO
(asymmetric/decoupled clipping) and GSPO (sentence-likelihood importance ratio),
plus two entropy regularization terms:

    * high-entropy exploration bonus        C_H * E_t[H(pi_theta)]
    * entropy-decay suppression term        eta * E_t[max(0, H(pi_theta) - H(pi_theta_old))]

Both entropy terms are deactivated for the first 50 steps and activated
thereafter with C_H = eta = 1e-3.

Importance ratio (GSPO, sentence likelihood):

    s^(i)(theta) = exp( (1/|o_i|) * sum_t log( pi_theta / pi_theta_old ) )

NOTE: The exact equations are image-only in the source and were reconstructed
from the prose description; see paper_workspace/01_algorithm_extraction.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import RLConfig


@dataclass
class RLObjectiveOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    entropy_bonus: torch.Tensor
    entropy_suppression: torch.Tensor
    mean_ratio: torch.Tensor
    mean_entropy: torch.Tensor
    entropy_active: bool


def sentence_importance_ratio(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """GSPO-style sentence-level importance ratio, shape (G,).

    logprobs / old_logprobs: (G, T) token log-probabilities.
    mask: (G, T) 1 for valid response tokens, 0 for padding.
    """
    diff = (logprobs - old_logprobs) * mask
    lengths = mask.sum(dim=-1).clamp(min=1.0)
    return torch.exp(diff.sum(dim=-1) / (lengths + eps))


def asymmetric_clip(ratio: torch.Tensor, eps_low: float, eps_high: float) -> torch.Tensor:
    """DAPO decoupled clipping: clip to [1 - eps_low, 1 + eps_high]."""
    return torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Exact entropy from logits of shape (..., V). Returns (...)."""
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def sampled_entropy_proxy(logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Entropy proxy from sampled token log-probs (when logits are unavailable).

    Uses mean(-log pi) over valid tokens as a monotone proxy for policy entropy.
    """
    lengths = mask.sum().clamp(min=1.0)
    return (-(logprobs * mask).sum()) / lengths


class HybridRLObjective(nn.Module):
    """Computes the TableGPT-R1 RL loss."""

    def __init__(self, config: Optional[RLConfig] = None):
        super().__init__()
        self.config = config or RLConfig()

    def forward(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        step: int = 0,
        logits: Optional[torch.Tensor] = None,
        old_logits: Optional[torch.Tensor] = None,
    ) -> RLObjectiveOutput:
        """Compute the loss.

        Args:
            logprobs: (G, T) current-policy token log-probs.
            old_logprobs: (G, T) behavior-policy token log-probs.
            advantages: (G,) group-normalized advantages.
            mask: (G, T) valid-token mask.
            step: current optimization step (controls entropy activation).
            logits / old_logits: (G, T, V) optional, for exact entropy.
        """
        cfg = self.config
        ratio = sentence_importance_ratio(logprobs, old_logprobs, mask)
        clipped = asymmetric_clip(ratio, cfg.clip_epsilon_low, cfg.clip_epsilon_high)

        # Per-response clipped surrogate (GSPO uses a single sequence ratio).
        surrogate = torch.minimum(ratio * advantages, clipped * advantages)
        policy_loss = -surrogate.mean()

        # Entropy estimation (exact if logits given, else sampled proxy).
        if logits is not None:
            ent_tokens = token_entropy(logits)             # (G, T)
            mean_entropy = (ent_tokens * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            mean_entropy = sampled_entropy_proxy(logprobs, mask)

        if old_logits is not None:
            old_ent_tokens = token_entropy(old_logits)
            old_entropy = (old_ent_tokens * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            old_entropy = sampled_entropy_proxy(old_logprobs, mask)

        entropy_active = step >= cfg.entropy_terms_start_step
        zero = torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)

        if entropy_active:
            # Exploration bonus: maximize entropy -> subtract in a min objective.
            entropy_bonus = cfg.entropy_bonus_C_H * mean_entropy
            # Entropy-decay suppression term, implemented exactly as written in
            # Section 3.3: eta * max(0, H(pi_theta) - H(pi_theta_old)).
            entropy_suppression = cfg.entropy_suppression_eta * torch.clamp(
                mean_entropy - old_entropy, min=0.0
            )
            loss = policy_loss - entropy_bonus + entropy_suppression
        else:
            entropy_bonus = zero
            entropy_suppression = zero
            loss = policy_loss

        return RLObjectiveOutput(
            loss=loss,
            policy_loss=policy_loss,
            entropy_bonus=entropy_bonus,
            entropy_suppression=entropy_suppression,
            mean_ratio=ratio.mean(),
            mean_entropy=mean_entropy,
            entropy_active=entropy_active,
        )
