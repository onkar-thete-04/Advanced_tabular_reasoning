"""Optional QLoRA / LoRA loading utilities.

Centralises 4-bit quantization, LoRA adapter injection and gradient
checkpointing so both the RL policy (``HFPolicy``) and the SFT warm-up share
one implementation. The paper uses full fine-tuning; this path exists so the
same recipe can run on consumer / Kaggle GPUs.

Every heavy import (transformers, peft, bitsandbytes) is performed lazily so the
CPU-only test suite never depends on them. Missing optional deps raise a clear
``RuntimeError`` instead of an opaque ImportError.
"""

from __future__ import annotations

from typing import Optional, Tuple

from config import QuantConfig

_COMPUTE_DTYPES = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "float32": "float32",
    "fp32": "float32",
}

_AUTO_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def _torch_dtype(name: str):
    import torch

    attr = _COMPUTE_DTYPES.get(str(name).lower())
    if attr is None:
        raise ValueError(f"unsupported compute dtype: {name!r}")
    return getattr(torch, attr)


def _bf16_supported() -> bool:
    import torch

    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def preferred_amp_dtype(bf16_supported: Optional[bool] = None) -> str:
    """Pick the training math format for the current GPU.

    Turing GPUs (T4, GTX 16-series) have no native bfloat16, so the emulated
    path is slow and lossy -> use ``float16``. Ampere+ (A100/L4/40-series) does
    support bfloat16 -> prefer it for numerical stability.

    ``bf16_supported`` is injectable so the decision is unit-testable on CPU.
    """
    if bf16_supported is None:
        bf16_supported = _bf16_supported()
    return "bfloat16" if bf16_supported else "float16"


def _resolve_compute_dtype(name: str) -> str:
    """Resolve ``"auto"``/``""`` to the hardware-appropriate compute dtype."""
    if str(name).strip().lower() in ("", "auto"):
        return preferred_amp_dtype()
    return name


def build_bnb_config(qcfg: QuantConfig):
    """Build a ``BitsAndBytesConfig`` from ``qcfg`` (or ``None`` if fp)."""
    if not qcfg.load_in_4bit:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "transformers is required for 4-bit quantization. Install the "
            "'train' extra: pip install -e .[train]"
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qcfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=qcfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_torch_dtype(
            _resolve_compute_dtype(qcfg.bnb_4bit_compute_dtype)
        ),
    )


def _resolve_target_modules(model, spec: str):
    if spec in (None, "", "all-linear"):
        return "all-linear"
    if spec == "auto":
        names = set()
        for name, _module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in _AUTO_TARGET_SUFFIXES:
                names.add(leaf)
        return sorted(names) or "all-linear"
    return [s.strip() for s in spec.split(",") if s.strip()]


def apply_lora(model, qcfg: QuantConfig):
    """Wrap ``model`` with LoRA adapters per ``qcfg``."""
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "peft is required for LoRA/QLoRA. Install the 'train' extra: "
            "pip install -e .[train]"
        ) from exc

    lora_config = LoraConfig(
        r=qcfg.lora_r,
        lora_alpha=qcfg.lora_alpha,
        lora_dropout=qcfg.lora_dropout,
        target_modules=_resolve_target_modules(model, qcfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if qcfg.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    return model


def is_peft_model(model) -> bool:
    """True if ``model`` is a PEFT wrapper (never raises if peft is absent)."""
    try:
        from peft import PeftModel
    except Exception:
        return False
    return isinstance(model, PeftModel)


def load_adapter(model, adapter_path: str, adapter_name: str = "adapted"):
    """Attach a saved LoRA adapter to an existing PEFT model and activate it."""
    if not is_peft_model(model):
        raise RuntimeError(
            "model is not a PEFT model; load it with qlora enabled to attach an adapter"
        )
    model.load_adapter(adapter_path, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    return model


def trainable_parameter_names(model) -> list:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def load_causal_lm(
    model_id: str,
    qcfg: Optional[QuantConfig] = None,
    device: Optional[str] = None,
    dtype=None,
    trainable: bool = True,
) -> Tuple[object, object]:
    """Load a causal LM + tokenizer, optionally with 4-bit QLoRA.

    Returns ``(model, tokenizer)`` on ``device``. When ``qcfg.enabled`` is
    False this reproduces the previous full-precision behaviour exactly.
    """
    import torch

    qcfg = qcfg or QuantConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if qcfg.enabled and not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA requires a CUDA GPU. Set --no-qlora to run on CPU."
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if qcfg.enabled:
        if qcfg.load_in_4bit:
            try:
                from peft import prepare_model_for_kbit_training
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("peft is required for QLoRA.") from exc
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=build_bnb_config(qcfg),
                torch_dtype=_torch_dtype(_resolve_compute_dtype(qcfg.bnb_4bit_compute_dtype)),
            )
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=qcfg.gradient_checkpointing
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype or torch.float32
            )
        model = apply_lora(model, qcfg)
        model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype or torch.float32
        )

    if not trainable:
        for p in model.parameters():
            p.requires_grad_(False)

    model.to(device)
    model.train(trainable)
    return model, tokenizer
