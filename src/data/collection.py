"""Data collection registry (Section 2.1).

Aggregates heterogeneous raw data from diverse sources: general instruction
corpora, table-specific benchmarks, and agent interaction logs, spanning
government, academia, manufacturing, finance, education, and healthcare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.constants import DATA_COMPOSITION
from src.data.schema import Sample


@dataclass
class DataSource:
    """A registered data source."""

    name: str
    category: str                       # key into DATA_COMPOSITION
    loader: Callable[[], List[Sample]]
    description: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


class DatasetRegistry:
    """Registry mapping categories to pluggable data loaders."""

    def __init__(self) -> None:
        self._sources: Dict[str, DataSource] = {}

    def register(self, source: DataSource) -> None:
        if source.category not in DATA_COMPOSITION:
            raise ValueError(
                f"Unknown category '{source.category}'. "
                f"Valid: {list(DATA_COMPOSITION)}"
            )
        self._sources[source.name] = source

    def sources(self) -> List[DataSource]:
        return list(self._sources.values())

    def collect(self, name: Optional[str] = None) -> List[Sample]:
        """Collect samples from one source (by name) or all sources."""
        if name is not None:
            return self._sources[name].loader()
        out: List[Sample] = []
        for src in self._sources.values():
            out.extend(src.loader())
        return out

    def collect_by_category(self) -> Dict[str, List[Sample]]:
        """Return samples grouped by composition category."""
        grouped: Dict[str, List[Sample]] = {k: [] for k in DATA_COMPOSITION}
        for src in self._sources.values():
            grouped[src.category].extend(src.loader())
        return grouped


def default_registry() -> DatasetRegistry:
    """Register placeholder loaders for the public sources named in the paper.

    The actual dataset files are not shipped; each loader returns an empty list
    until the user wires in a real path (see README).
    """
    registry = DatasetRegistry()
    public = [
        ("spider", "sql", "Spider NL-to-SQL benchmark"),
        ("bird", "sql", "BIRD NL-to-SQL benchmark"),
        ("tablebench", "table_task", "TableBench holistic evaluation"),
        ("realhitbench", "table_task", "RealHitBench irregular tables"),
        ("infiagent-da", "table_task", "InfiAgent-DABench agent analysis"),
        ("gsm8k", "math_logic", "GSM8K math word problems"),
        ("math", "math_logic", "MATH dataset"),
        ("humaneval", "code", "HumanEval code generation"),
        ("mbpp", "code", "MBPP code generation"),
        ("open_qa", "general", "Open-domain QA"),
        ("agentic_synth", "table_agent", "Synthetic agentic trajectories"),
        ("qa_with_answer", "qa_with_answer", "Labeled QA pairs"),
    ]
    for name, category, desc in public:
        registry.register(
            DataSource(name=name, category=category, loader=lambda: [], description=desc)
        )
    return registry
