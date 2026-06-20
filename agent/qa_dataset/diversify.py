"""Diversity, coverage, and balance measurement + acceptance gates.

Measures lexical diversity (distinct-n), structural histograms (type / topic /
difficulty / persona), and source coverage (fraction of eligible chunks that
produced at least one kept item). The gate checks tell the pipeline whether to
rebalance or generate more before packaging.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

_WS = re.compile(r"\s+")


def _toks(s: str) -> list[str]:
    return _WS.sub(" ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).split()


def distinct_n(texts: Iterable[str], n: int = 2) -> float:
    """Distinct n-grams / total n-grams across texts (0..1). Higher = more diverse."""
    total = 0
    seen = set()
    for t in texts:
        toks = _toks(t)
        if len(toks) < n:
            grams = [" ".join(toks)] if toks else []
        else:
            grams = [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        seen.update(grams)
    return (len(seen) / total) if total else 0.0


def histogram(items, attr: str) -> dict[str, int]:
    c: Counter = Counter()
    for it in items:
        v = getattr(it, attr, None)
        c[str(v)] += 1
    return dict(c)


def type_shares(items) -> dict[str, float]:
    n = len(items)
    if not n:
        return {}
    h = histogram(items, "question_type")
    return {k: v / n for k, v in h.items()}


def source_coverage(items, eligible_chunk_ids: Iterable[str]) -> float:
    eligible = set(eligible_chunk_ids)
    if not eligible:
        return 0.0
    covered = set()
    for it in items:
        for cid in getattr(it, "supporting_chunk_ids", []) or []:
            if cid in eligible:
                covered.add(cid)
    return len(covered) / len(eligible)


def measure(items, eligible_chunk_ids: Optional[Iterable[str]] = None) -> dict:
    questions = [it.question for it in items]
    return {
        "n": len(items),
        "distinct_2": round(distinct_n(questions, 2), 3),
        "distinct_3": round(distinct_n(questions, 3), 3),
        "type_shares": {k: round(v, 3) for k, v in type_shares(items).items()},
        "by_difficulty": histogram(items, "difficulty_bucket"),
        "by_persona": histogram(items, "persona_id"),
        "coverage": round(source_coverage(items, eligible_chunk_ids or []), 3),
    }


def gate_failures(
    items, profile, eligible_chunk_ids: Optional[Iterable[str]] = None
) -> list[str]:
    """Return human-readable gate failures (empty == passes)."""
    g = profile.diversity_gates
    fails: list[str] = []
    shares = type_shares(items)
    over = {k: v for k, v in shares.items() if v > g.max_single_type_share}
    if over:
        worst = max(over, key=over.get)
        fails.append(
            f"question type '{worst}' is {over[worst]:.0%} of items "
            f"(> {g.max_single_type_share:.0%}) — rebalance"
        )
    d2 = distinct_n([it.question for it in items], 2)
    if items and d2 < g.min_distinct_2:
        fails.append(
            f"distinct-2 = {d2:.2f} (< {g.min_distinct_2}) — questions too repetitive"
        )
    if eligible_chunk_ids is not None:
        cov = source_coverage(items, eligible_chunk_ids)
        if cov < profile.coverage.min_chunk_coverage:
            fails.append(
                f"source coverage = {cov:.0%} (< {profile.coverage.min_chunk_coverage:.0%}) "
                f"— generate over more chunks"
            )
    return fails
