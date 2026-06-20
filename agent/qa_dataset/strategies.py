"""Strategy catalog: load, validate, and resolve the profile's selection.

The catalog (strategies/catalog.yaml) is the shipped, versioned menu of Q&A
diversification strategies. The profile picks ids from it; this module merges
profile-level param/weight overrides onto the catalog defaults and hands the
pipeline ready-to-use Strategy objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_PKG_DIR = Path(__file__).resolve().parent

QUESTION_TYPES = [
    "factual",
    "boolean",
    "list",
    "comparative",
    "causal",
    "multi_hop",
    "counterfactual",
    "unanswerable",
]


def shipped_catalog_path() -> Path:
    return _PKG_DIR / "strategies" / "catalog.yaml"


@dataclass
class Strategy:
    id: str
    category: str
    mode: str = "single_call"
    enabled_by_default: bool = False
    applies_to: list[str] = field(default_factory=lambda: ["*"])
    description: str = ""
    notes: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    prompt_template: str = ""
    output_schema: dict[str, Any] = field(default_factory=dict)
    weight: float | None = None

    def handles_type(self, qtype: str) -> bool:
        return "*" in self.applies_to or qtype in self.applies_to


@dataclass
class Catalog:
    version: int
    categories: dict[str, str]
    strategies: dict[str, Strategy]

    def get(self, sid: str) -> Strategy | None:
        return self.strategies.get(sid)

    def by_category(self, category: str) -> list[Strategy]:
        return [s for s in self.strategies.values() if s.category == category]

    def defaults(self) -> list[str]:
        return [s.id for s in self.strategies.values() if s.enabled_by_default]


def _strategy_from_dict(d: dict[str, Any]) -> Strategy:
    applies = d.get("applies_to", ["*"])
    if isinstance(applies, str):
        applies = [applies]
    return Strategy(
        id=d["id"],
        category=d.get("category", "question_type"),
        mode=d.get("mode", "single_call"),
        enabled_by_default=bool(d.get("enabled_by_default", False)),
        applies_to=list(applies),
        description=str(d.get("description", "")).strip(),
        notes=str(d.get("notes", "")).strip(),
        params=dict(d.get("params") or {}),
        prompt_template=d.get("prompt_template", ""),
        output_schema=dict(d.get("output_schema") or {}),
    )


def load_catalog(path: str | Path | None = None) -> Catalog:
    p = Path(path) if path else shipped_catalog_path()
    data = yaml.safe_load(p.read_text()) or {}
    strategies: dict[str, Strategy] = {}
    for entry in data.get("strategies", []):
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        s = _strategy_from_dict(entry)
        if s.id in strategies:
            raise ValueError(f"duplicate strategy id in catalog: {s.id}")
        strategies[s.id] = s
    return Catalog(
        version=int(data.get("catalog_version", 0)),
        categories=dict(data.get("categories") or {}),
        strategies=strategies,
    )


def validate_selection(profile, catalog: Catalog) -> list[str]:
    """Return a list of problems (empty == OK). Does not raise."""
    problems: list[str] = []
    if profile.catalog_version != catalog.version:
        problems.append(
            f"profile catalog_version={profile.catalog_version} but catalog is "
            f"version {catalog.version}"
        )
    for sid in profile.enabled_strategy_ids():
        if sid not in catalog.strategies:
            problems.append(f"enabled strategy '{sid}' not found in catalog")
    # Every question type with positive share must have an enabled generator.
    enabled = [
        catalog.strategies[s]
        for s in profile.enabled_strategy_ids()
        if s in catalog.strategies
    ]
    qt_generators = [s for s in enabled if s.category == "question_type"]
    for qtype, share in (profile.question_types or {}).items():
        if share <= 0:
            continue
        if not any(s.handles_type(qtype) for s in qt_generators):
            problems.append(
                f"question type '{qtype}' has share {share} but no enabled generator handles it"
            )
    return problems


def resolve_enabled_strategies(profile, catalog: Catalog) -> list[Strategy]:
    """Return enabled Strategy objects with profile param/weight overrides merged."""
    resolved: list[Strategy] = []
    for ref in profile.strategies.enabled:
        base = catalog.get(ref.id)
        if base is None:
            continue
        merged_params = {**base.params, **(ref.params or {})}
        resolved.append(
            Strategy(
                id=base.id,
                category=base.category,
                mode=base.mode,
                enabled_by_default=base.enabled_by_default,
                applies_to=list(base.applies_to),
                description=base.description,
                notes=base.notes,
                params=merged_params,
                prompt_template=base.prompt_template,
                output_schema=base.output_schema,
                weight=ref.weight,
            )
        )
    return resolved


def generators_for_type(strategies: list[Strategy], qtype: str) -> list[Strategy]:
    return [
        s
        for s in strategies
        if s.category == "question_type" and s.handles_type(qtype)
    ]
