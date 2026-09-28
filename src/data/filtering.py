"""Two-stage data filtering pipeline (Section 2.2).

Stage A (input-based):
    (1) Length Filtering      : <= 8192 tokens
    (2) Deduplication         : MinHash near-duplicate removal
    (3) Rule-based Cleaning   : garbled/harmful/identity-leak removal
    (4) Language Alignment    : drop mismatched input-output languages

Stage B (output-based):
    (5) quality gate          : keep reference responses scored >= 0.7 (0-1)
    (6) difficulty estimation : 5 rollouts -> m_bar
    (7) difficulty classes    : high m_bar<0.3, medium 0.3<=m_bar<=0.7, low m_bar>0.7
    (8) resample to 20% high / 60% medium / 20% low
"""

from __future__ import annotations

import re
import random
from typing import Callable, Dict, List, Optional, Sequence

from config import DataConfig
from src.constants import DIFFICULTY_HIGH, DIFFICULTY_MEDIUM, DIFFICULTY_LOW
from src.data.schema import Sample

# ---------------------------------------------------------------------------
# Token length (plug in a HF tokenizer via set_tokenizer for exact counts)
# ---------------------------------------------------------------------------
_TOKENIZER = None


def set_tokenizer(tokenizer) -> None:
    global _TOKENIZER
    _TOKENIZER = tokenizer


def token_len(text: str) -> int:
    """Approximate token length; uses HF tokenizer when available."""
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text))
    # Whitespace approximation (~1.3 tokens per word for English).
    return int(len(text.split()) * 1.3) + 1


# ---------------------------------------------------------------------------
# Stage A: input-based filtering
# ---------------------------------------------------------------------------
def length_filter(samples: Sequence[Sample], max_tokens: int = 8192) -> List[Sample]:
    """(1) Length Filtering: keep samples within the context window."""
    kept = []
    for s in samples:
        text = f"{s.question}\n{s.table_ref}\n{s.reference_answer}"
        if token_len(text) <= max_tokens:
            kept.append(s)
    return kept


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def dedup(samples: Sequence[Sample], config: DataConfig) -> List[Sample]:
    """(2) Deduplication via MinHash (falls back to normalized exact dedup)."""
    try:
        from datasketch import MinHash  # type: ignore
    except Exception:
        seen, out = set(), []
        for s in samples:
            key = _normalize(s.question)
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    kept_keys: List[str] = []
    minhashes = []
    out = []
    for s in samples:
        norm = _normalize(s.question)
        m = MinHash(num_perm=config.minhash_num_perm)
        for shingle in {norm[i:i + 5] for i in range(max(1, len(norm) - 4))}:
            m.update(shingle.encode("utf8"))
        is_dup = False
        for prev in minhashes:
            if m.jaccard(prev) >= config.minhash_threshold:
                is_dup = True
                break
        if not is_dup:
            minhashes.append(m)
            kept_keys.append(norm)
            out.append(s)
    return out


_GARBLED_RE = re.compile(r"[\ufffd]|(?:[^\x00-\x7f\u3000-\u9fff]{8,})")
_HARMFUL_RE = re.compile(
    r"(how to (?:make|build) (?:a )?(?:bomb|weapon)|child (?:porn|abuse))",
    re.IGNORECASE,
)
_IDENTITY_RE = re.compile(
    r"(i am (?:gpt|chatgpt|claude|an ai (?:assistant|language model)))", re.IGNORECASE
)


def rule_clean(samples: Sequence[Sample]) -> List[Sample]:
    """(3) Rule-based Cleaning: remove garbled/harmful/identity-leak samples."""
    out = []
    for s in samples:
        blob = f"{s.question} {s.reference_answer}"
        if _GARBLED_RE.search(blob):
            continue
        if _HARMFUL_RE.search(blob):
            continue
        if _IDENTITY_RE.search(blob):
            continue
        out.append(s)
    return out


def language_align(samples: Sequence[Sample]) -> List[Sample]:
    """(4) Language Alignment: drop mismatched input/output language pairs."""
    return [s for s in samples if s.language == s.output_language]


# ---------------------------------------------------------------------------
# Stage B: output-based filtering
# ---------------------------------------------------------------------------
def output_quality_filter(
    samples: Sequence[Sample],
    scorer: Callable[[Sample], float],
    threshold: float = 0.7,
) -> List[Sample]:
    """(5) Keep samples whose reference response scores >= threshold (0-1)."""
    out = []
    for s in samples:
        score = scorer(s)
        s.quality_score = score
        if score >= threshold:
            out.append(s)
    return out


def estimate_difficulty(
    sample: Sample,
    rollout_fn: Callable[[Sample], str],
    scorer: Callable[[Sample, str], float],
    num_rollouts: int = 5,
) -> float:
    """(6) m_bar = (1/5) * sum_{i=1}^{5} m^(i)."""
    scores = []
    for _ in range(num_rollouts):
        response = rollout_fn(sample)
        scores.append(scorer(sample, response))
    m_bar = sum(scores) / len(scores) if scores else 0.0
    sample.difficulty_score = m_bar
    return m_bar


def classify_difficulty(m_bar: float, config: DataConfig) -> str:
    """(7) Bucket m_bar into high / medium / low (Section 2.2)."""
    if m_bar < config.difficulty_high_max:
        return DIFFICULTY_HIGH
    if m_bar <= config.difficulty_medium_max:
        return DIFFICULTY_MEDIUM
    return DIFFICULTY_LOW


def assign_difficulty(
    samples: Sequence[Sample],
    rollout_fn: Callable[[Sample], str],
    scorer: Callable[[Sample, str], float],
    config: DataConfig,
) -> List[Sample]:
    for s in samples:
        m_bar = estimate_difficulty(s, rollout_fn, scorer, config.difficulty_num_rollouts)
        s.difficulty = classify_difficulty(m_bar, config)
    return list(samples)


def resample_difficulty(
    samples: Sequence[Sample], config: DataConfig, seed: int = 42
) -> List[Sample]:
    """(8) Resample to the target 20/60/20 distribution."""
    rng = random.Random(seed)
    buckets: Dict[str, List[Sample]] = {
        DIFFICULTY_HIGH: [],
        DIFFICULTY_MEDIUM: [],
        DIFFICULTY_LOW: [],
    }
    for s in samples:
        buckets.setdefault(s.difficulty or DIFFICULTY_MEDIUM, []).append(s)

    total = len(samples)
    targets = {
        DIFFICULTY_HIGH: int(round(total * config.target_high)),
        DIFFICULTY_MEDIUM: int(round(total * config.target_medium)),
        DIFFICULTY_LOW: int(round(total * config.target_low)),
    }

    out: List[Sample] = []
    for bucket_name, target in targets.items():
        pool = buckets.get(bucket_name, [])
        if not pool:
            continue
        if len(pool) >= target:
            out.extend(rng.sample(pool, target))
        else:
            # With replacement to reach the target when a bucket is scarce.
            out.extend(pool)
            out.extend(rng.choices(pool, k=target - len(pool)))
    rng.shuffle(out)
    return out


def run_input_filtering(samples: Sequence[Sample], config: DataConfig) -> List[Sample]:
    """Apply Stage A filters in order."""
    out = length_filter(samples, config.max_seq_length_tokens)
    if config.dedup_use_minhash:
        out = dedup(out, config)
    out = rule_clean(out)
    out = language_align(out)
    return out
