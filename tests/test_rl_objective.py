"""Tests for the hybrid RL objective (Section 3.3)."""

import math
import torch

from config import RLConfig
from src.rl.advantage import group_advantage, group_advantage_list
from src.rl.objective import (
    HybridRLObjective, sentence_importance_ratio, asymmetric_clip, token_entropy,
)


def test_group_advantage_normalization():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv = group_advantage(rewards)
    assert abs(float(adv.mean())) < 1e-6
    assert abs(float(adv.std(unbiased=False)) - 1.0) < 1e-6


def test_group_advantage_zero_variance():
    adv = group_advantage(torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(adv, torch.zeros(3))


def test_group_advantage_list():
    adv = group_advantage_list([1.0, 0.0])
    assert abs(adv[0] - 1.0) < 1e-6 and abs(adv[1] + 1.0) < 1e-6


def test_sentence_ratio_is_one_when_policies_match():
    lp = torch.randn(3, 5)
    mask = torch.ones(3, 5)
    ratio = sentence_importance_ratio(lp, lp.clone(), mask)
    assert torch.allclose(ratio, torch.ones(3), atol=1e-6)


def test_sentence_ratio_geometric_mean():
    # Single token: ratio = exp(logp - old_logp)
    lp = torch.tensor([[math.log(1.5)]])
    old = torch.tensor([[0.0]])
    ratio = sentence_importance_ratio(lp, old, torch.ones(1, 1))
    assert abs(float(ratio) - 1.5) < 1e-6


def test_asymmetric_clip_bounds():
    r = torch.tensor([0.5, 1.0, 2.0])
    clipped = asymmetric_clip(r, 0.2, 0.28)
    assert abs(float(clipped[0]) - 0.8) < 1e-6
    assert abs(float(clipped[2]) - 1.28) < 1e-6


def test_token_entropy_uniform():
    logits = torch.zeros(1, 1, 4)  # uniform over 4 classes
    ent = token_entropy(logits)
    assert abs(float(ent) - math.log(4)) < 1e-6


def test_entropy_terms_inactive_before_start_step():
    cfg = RLConfig()
    obj = HybridRLObjective(cfg)
    G, T = 2, 2
    lp = torch.zeros(G, T)
    out = obj(lp, lp.clone(), torch.tensor([1.0, -1.0]), torch.ones(G, T), step=0)
    assert not out.entropy_active
    assert float(out.entropy_bonus) == 0.0 and float(out.entropy_suppression) == 0.0


def test_entropy_terms_active_after_start_step():
    cfg = RLConfig()
    obj = HybridRLObjective(cfg)
    G, T = 1, 1
    lp = torch.zeros(G, T)
    logits = torch.zeros(G, T, 4)  # entropy = ln4
    out = obj(lp, lp.clone(), torch.tensor([1.0]), torch.ones(G, T),
              step=cfg.entropy_terms_start_step, logits=logits, old_logits=logits.clone())
    assert out.entropy_active
    assert abs(float(out.entropy_bonus) - cfg.entropy_bonus_C_H * math.log(4)) < 1e-9
    assert float(out.entropy_suppression) == 0.0  # H == H_old


def test_loss_matches_hand_computation_with_clipping():
    cfg = RLConfig()
    obj = HybridRLObjective(cfg)
    G, T = 2, 1
    lp = torch.full((G, T), math.log(1.5))
    old = torch.zeros(G, T)
    advantages = torch.tensor([1.0, 1.0])
    mask = torch.ones(G, T)
    out = obj(lp, old, advantages, mask, step=0)
    # ratio=1.5 clipped to 1.28; surrogate=1.28 -> loss=-1.28
    expected = -1.28
    assert abs(float(out.loss) - expected) < 1e-6


def test_policy_loss_zero_for_balanced_advantage():
    cfg = RLConfig()
    obj = HybridRLObjective(cfg)
    lp = torch.zeros(2, 2)
    out = obj(lp, lp.clone(), torch.tensor([1.0, -1.0]), torch.ones(2, 2), step=0)
    assert abs(float(out.policy_loss)) < 1e-6
