"""Tests for the optional QLoRA / LoRA utilities.

These run on a CPU-only machine with no transformers/peft/bitsandbytes: the
config and helper paths are always exercised, and optional-dependency paths
assert they fail with a clear error rather than an opaque ImportError.
"""

import importlib

import pytest
import torch
import torch.nn as nn

from config import Config, QuantConfig
from src.training import quant as Q


def _has(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def test_quant_config_defaults():
    q = QuantConfig()
    assert q.enabled is False
    assert q.load_in_4bit is True
    assert q.bnb_4bit_quant_type == "nf4"
    assert q.bnb_4bit_use_double_quant is True
    assert q.lora_r == 16
    assert q.lora_alpha == 32
    assert q.lora_dropout == pytest.approx(0.05)
    assert q.gradient_checkpointing is True


def test_config_wires_quant_roundtrip():
    cfg = Config()
    assert isinstance(cfg.quant, QuantConfig)
    d = cfg.to_dict()
    assert d["quant"]["lora_r"] == 16
    assert Config.from_dict(d).quant == cfg.quant


def test_torch_dtype_mapping():
    assert Q._torch_dtype("bf16") is torch.bfloat16
    assert Q._torch_dtype("float16") is torch.float16
    assert Q._torch_dtype("fp32") is torch.float32
    with pytest.raises(ValueError):
        Q._torch_dtype("int3")


def test_resolve_target_modules_auto():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)
            self.v_proj = nn.Linear(4, 4)
            self.other = nn.Linear(4, 4)

    mods = Q._resolve_target_modules(M(), "auto")
    assert set(mods) == {"q_proj", "v_proj"}


def test_resolve_target_modules_passthrough():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

    assert Q._resolve_target_modules(M(), "all-linear") == "all-linear"
    assert Q._resolve_target_modules(M(), "q_proj, k_proj") == ["q_proj", "k_proj"]


def test_build_bnb_config_none_when_fp():
    assert Q.build_bnb_config(QuantConfig(load_in_4bit=False)) is None


def test_build_bnb_config_graceful_without_transformers():
    if _has("transformers"):
        cfg = Q.build_bnb_config(QuantConfig())
        assert cfg is not None and cfg.load_in_4bit is True
    else:
        with pytest.raises(RuntimeError):
            Q.build_bnb_config(QuantConfig())


def test_apply_lora_graceful_without_peft():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)

    if _has("peft"):
        pytest.skip("peft installed; wrapper path covered on GPU runs")
    with pytest.raises(RuntimeError):
        Q.apply_lora(M(), QuantConfig(gradient_checkpointing=False))


def test_is_peft_model_false_for_plain_module():
    assert Q.is_peft_model(nn.Linear(2, 2)) is False


def test_qlora_requires_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    with pytest.raises(RuntimeError, match="CUDA"):
        Q.load_causal_lm("dummy-model", QuantConfig(enabled=True))


def test_trainable_parameter_names():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    for p in model[1].parameters():
        p.requires_grad_(False)
    names = Q.trainable_parameter_names(model)
    assert names and all(n.startswith("0.") for n in names)


def test_preferred_amp_dtype_follows_hardware():
    assert Q.preferred_amp_dtype(bf16_supported=True) == "bfloat16"
    assert Q.preferred_amp_dtype(bf16_supported=False) == "float16"


def test_preferred_amp_dtype_default_picks_fp16_without_bf16_gpu():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        assert Q.preferred_amp_dtype() == "bfloat16"
    else:
        assert Q.preferred_amp_dtype() == "float16"


def test_build_bnb_config_resolves_auto_compute_dtype():
    if not _has("transformers"):
        pytest.skip("transformers not installed")
    cfg = Q.build_bnb_config(QuantConfig(bnb_4bit_compute_dtype="auto"))
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        assert cfg.bnb_4bit_compute_dtype is torch.bfloat16
    else:
        assert cfg.bnb_4bit_compute_dtype is torch.float16
