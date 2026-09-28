"""Group-relative advantage estimation (Section 3.3).

The advantage of the i-th response is computed by normalizing the group-level
reward:

    Â^(i) = (R^(i) - mean({R^(j)}_{j=1}^{G})) / std({R^(j)}_{j=1}^{G})

Shared by GRPO/DAPO/GSPO (Section 3.3, GSPO Eq. 6).
"""

from __future__ import annotations

from typing import Sequence

import torch


def group_advantage(
    rewards: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Normalize a group of rewards (shape (G,) or (G, ...)).

    Normalization is computed over the group dimension (dim=0). When the group
    has zero variance, advantages are set to zero (avoids div-by-zero).
    """
    if rewards.numel() == 0:
        return rewards
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if float(std) < eps:
        return torch.zeros_like(rewards)
    return (rewards - mean) / (std + eps)


def group_advantage_list(rewards: Sequence[float], eps: float = 1e-8) -> list:
    t = torch.tensor(list(rewards), dtype=torch.float32)
    return group_advantage(t, eps=eps).tolist()
