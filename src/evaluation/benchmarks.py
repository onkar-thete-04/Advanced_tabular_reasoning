"""Benchmark registry and runners (Section 4.1).

Table benchmarks : Spider, BIRD, TableBench, RealHitBench, InfiAgent-DABench,
                   Internal Benchmark (59 tables / 620 QA pairs).
General          : HumanEval, MBPP, GSM8K, MATH, AIME, CMMLU-5, GPQA-Diamond.

The internal benchmark supports two input configurations:
    (1) Table Info : first N rows of the table as context.
    (2) Table Path : the full table's file path (agentic loading).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from src.evaluation.metrics import compute_metrics


@dataclass
class Benchmark:
    name: str
    category: str                       # "table" | "general"
    metrics: List[str]
    description: str = ""
    subtasks: List[str] = field(default_factory=list)


TABLE_BENCHMARKS = {
    "internal": Benchmark("Internal Benchmark", "table", ["Acc"],
                          "59 tables, 620 QA pairs; Table Info & Table Path",
                          subtasks=["table_info", "table_path"]),
    "spider": Benchmark("Spider", "table", ["EX"], "NL-to-SQL"),
    "bird": Benchmark("BIRD", "table", ["EX"], "NL-to-SQL"),
    "tablebench": Benchmark("TableBench", "table", ["Rge"],
                            "Holistic table evaluation",
                            subtasks=["DP", "PoT", "SCoT", "TCoT"]),
    "realhitbench": Benchmark("RealHitBench", "table", ["EM", "GPT", "ECR"],
                              "Irregular/hierarchical tables",
                              subtasks=["FC", "NR", "SC", "DA", "CG"]),
    "infiagent-da": Benchmark("InfiAgent-DABench", "table", ["Acc"],
                              "Agent-based data analysis"),
}

GENERAL_BENCHMARKS = {
    "humaneval": Benchmark("HumanEval", "general", ["Acc"], "Code generation"),
    "mbpp": Benchmark("MBPP", "general", ["Acc"], "Code generation"),
    "gsm8k": Benchmark("GSM8K", "general", ["Acc"], "Math word problems"),
    "math": Benchmark("MATH", "general", ["Acc"], "Competition math"),
    "aime": Benchmark("AIME", "general", ["Acc"], "Competition math"),
    "cmmlu-5": Benchmark("CMMLU-5", "general", ["Acc"], "Chinese knowledge"),
    "gpqa-diamond": Benchmark("GPQA-Diamond", "general", ["Acc"], "Graduate QA"),
}

BENCHMARK_REGISTRY: Dict[str, Benchmark] = {**TABLE_BENCHMARKS, **GENERAL_BENCHMARKS}

# Reference targets from Table 2 / Table 3 (TableGPT-R1-8B).
REFERENCE_TARGETS = {
    "internal_table_info": 80.00,
    "internal_table_path": 82.70,
    "spider": 86.73,
    "bird": 63.17,
    "humaneval": 95.73,
    "gsm8k": 95.60,
    "math": 93.30,
    "aime": 50.00,
}


def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(f"Unknown benchmark '{name}'. Valid: {list(BENCHMARK_REGISTRY)}")
    return BENCHMARK_REGISTRY[name]


def evaluate_policy(
    policy_fn: Callable[[str, str], str],
    samples: Sequence[dict],
    benchmark_name: str,
    input_mode: str = "table_info",
    judge_fn: Optional[Callable[[str], float]] = None,
) -> Dict[str, float]:
    """Evaluate a policy over samples.

    Args:
        policy_fn: (question, table_ref) -> prediction text.
        samples: list of dicts with keys question, table_ref, reference.
        benchmark_name: key in BENCHMARK_REGISTRY.
        input_mode: "table_info" or "table_path".
        judge_fn: optional LLM judge for GPT metric.

    Returns:
        metric name -> value.
    """
    bench = get_benchmark(benchmark_name)
    predictions, references = [], []
    for s in samples:
        table_ref = s["table_ref"]
        if input_mode == "table_path":
            table_ref = s.get("table_path", table_ref)
        predictions.append(policy_fn(s["question"], table_ref))
        references.append(s["reference"])

    results = compute_metrics(bench.metrics, predictions, references, judge_fn=judge_fn)
    return {r.name: r.value for r in results}
