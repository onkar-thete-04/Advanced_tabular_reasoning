"""Data augmentation strategies (Section 3.2.2).

Key focus: Table Input Diversity. Tables are stochastically presented either in
conventional static formats or exclusively as file paths. When only a path is
provided, the model must autonomously load the table, inspect its structure,
and determine what to retrieve.

Additional augmentations:
    (1) System Prompt Variation
    (2) Special Token Alternation (<function_call> <-> <tool_call>)
    (3) Error Message Diversity (minimal exceptions vs full stack traces)
"""

from __future__ import annotations

import random
from typing import List, Optional

from config import DataConfig
from src.constants import (
    TOOL_CALL_OPEN, TOOL_CALL_CLOSE, FUNCTION_CALL_OPEN, FUNCTION_CALL_CLOSE,
)
from src.data.schema import Sample
from src.execution.error_format import minimal_error, full_error

# Static table representations (Figure 4).
STATIC_FORMATS = ("df.head()", "df.columns()", "df.dtypes", "df.head().to_string()", "schema")

SYSTEM_PROMPT_TEMPLATES = (
    "You are a helpful data analysis assistant.",
    "You are an expert tabular reasoning agent. Use code when needed.",
    "Answer the user's question about the table accurately and concisely.",
    "You are an autonomous data analyst with a Python interpreter.",
)


def present_table(
    sample: Sample,
    static_table_text: str,
    rng: Optional[random.Random] = None,
    force_mode: Optional[str] = None,
) -> Sample:
    """Present the table as static info or as a file path (Table Input Diversity)."""
    rng = rng or random.Random()
    mode = force_mode or rng.choice(["info", "path"])
    if mode == "path":
        sample.table_ref = sample.metadata.get("table_path", sample.table_ref)
        sample.metadata["input_format"] = "path"
    else:
        fmt = rng.choice(STATIC_FORMATS)
        sample.table_ref = static_table_text
        sample.metadata["input_format"] = f"static:{fmt}"
    return sample


def vary_system_prompt(rng: Optional[random.Random] = None) -> str:
    rng = rng or random.Random()
    return rng.choice(SYSTEM_PROMPT_TEMPLATES)


def alternate_special_tokens(text: str, rng: Optional[random.Random] = None) -> str:
    """Rotate equivalent syntactic markers to prevent token-pattern overfitting."""
    rng = rng or random.Random()
    if rng.random() < 0.5:
        return text
    return (
        text.replace(TOOL_CALL_OPEN, FUNCTION_CALL_OPEN)
        .replace(TOOL_CALL_CLOSE, FUNCTION_CALL_CLOSE)
    )


def diverse_error(exc_type: str, message: str, rng: Optional[random.Random] = None) -> str:
    """Inject heterogeneous error formats."""
    rng = rng or random.Random()
    if rng.random() < 0.5:
        return minimal_error(exc_type, message)
    return full_error(exc_type, message)


class DataAugmenter:
    """Applies the augmentation suite controlled by DataConfig toggles."""

    def __init__(self, config: Optional[DataConfig] = None, seed: int = 42):
        self.config = config or DataConfig()
        self.rng = random.Random(seed)

    def augment(self, sample: Sample, static_table_text: str = "") -> Sample:
        if self.config.augment_table_input_diversity:
            present_table(sample, static_table_text, self.rng)
        if self.config.augment_system_prompt_variation:
            sample.metadata["system_prompt"] = vary_system_prompt(self.rng)
        return sample

    def augment_text(self, text: str) -> str:
        if self.config.augment_special_token_alternation:
            text = alternate_special_tokens(text, self.rng)
        return text
