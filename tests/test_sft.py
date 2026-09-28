"""Tests for the SFT warm-up data encoding.

The warm-up must train on the *completion only*: if the prompt (question +
whole CSV table) is left in the labels, the loss is dominated by copying the
table and the model never learns the ``<answer>`` format. These tests pin that
behaviour without needing a real tokenizer.
"""

from src.training.sft import SFTExample, encode_sft_example, mask_prompt_labels


class FakeTokenizer:
    """Whitespace tokenizer: token id = length of the word."""

    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=8192):
        ids = [1 + len(w) for w in text.split()]
        return {"input_ids": ids[:max_length], "attention_mask": [1] * len(ids[:max_length])}


def test_mask_prompt_labels_masks_prefix():
    assert mask_prompt_labels([1, 2, 3, 4, 5], 3) == [-100, -100, -100, 4, 5]


def test_mask_prompt_labels_edges():
    assert mask_prompt_labels([1, 2, 3], 0) == [1, 2, 3]
    assert mask_prompt_labels([1, 2, 3], -5) == [1, 2, 3]
    assert mask_prompt_labels([1, 2, 3], 99) == [-100, -100, -100]


def test_encode_sft_example_trains_on_completion_only():
    tok = FakeTokenizer()
    ex = SFTExample(prompt="What is the sum", completion="<answer>3</answer>")
    enc = encode_sft_example(ex, tok)

    # prompt is "What is the sum" + "\n" -> 4 tokens masked out.
    assert enc["labels"][:4] == [-100, -100, -100, -100]
    # the completion (and eos) must still be trained on.
    assert all(v != -100 for v in enc["labels"][4:])
    assert sum(v != -100 for v in enc["labels"]) >= 1
