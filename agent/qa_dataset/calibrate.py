"""Difficulty calibration via base-model solve-rate.

Runs the (pre-fine-tune) base model over a sample of items, measures how often
it answers correctly, computes a solve-rate per evolution_depth, maps depth ->
difficulty bucket, then labels every item by its depth. This makes "hard" mean
"the base model can't already do it" rather than a guess.
"""

from __future__ import annotations

import re
from typing import Optional

from agent.qa_dataset.question_gen import QAItem

_WS = re.compile(r"\s+")


def _norm(s: Optional[str]) -> str:
    return _WS.sub(" ", (s or "").strip().lower())


def _bucket_from_solve_rate(rate: float) -> str:
    if rate >= 0.75:
        return "easy"
    if rate >= 0.4:
        return "medium"
    return "hard"


def _solved(base_answer: str, item: QAItem) -> bool:
    """Cheap correctness proxy: gold span/answer tokens present in base answer."""
    gold = _norm(item.answer_span or item.answer)
    ba = _norm(base_answer)
    if not gold or not ba:
        return False
    if gold in ba or ba in gold:
        return True
    gt = set(gold.split())
    bt = set(ba.split())
    if not gt:
        return False
    return len(gt & bt) / len(gt) >= 0.6


def _source_hint(item: QAItem) -> str:
    # Calibration answers closed-book-ish; provide the evidence we have.
    return " ".join(item.evidence_spans) or item.answer_span or ""


async def measure_solve_rates(
    items: list[QAItem], base_generator, *, sample_size: int = 200, seed: int = 42
) -> dict[int, float]:
    """Return {evolution_depth: solve_rate} measured on a sample of answerable items."""
    import random

    pool = [it for it in items if it.answerable and (it.answer_span or it.answer)]
    rng = random.Random(seed)
    if len(pool) > sample_size:
        pool = rng.sample(pool, sample_size)
    by_depth: dict[int, list[bool]] = {}
    for it in pool:
        prompt = (
            f"SOURCE:\n{_source_hint(it)}\n\nQUESTION: {it.question}\n"
            'Answer concisely. Return JSON: {"answer": "..."}'
        )
        obj = await base_generator.generate_json(prompt, temperature=0.0)
        ba = (obj or {}).get("answer", "") if isinstance(obj, dict) else ""
        by_depth.setdefault(it.difficulty, []).append(_solved(ba, it))
    return {d: (sum(v) / len(v) if v else 0.0) for d, v in by_depth.items()}


def assign_difficulty(
    items: list[QAItem], solve_by_depth: dict[int, float]
) -> dict[int, str]:
    """Label every item's difficulty_bucket from its depth's solve rate."""
    depth_bucket = {d: _bucket_from_solve_rate(r) for d, r in solve_by_depth.items()}
    default = "medium"
    for it in items:
        it.difficulty_bucket = depth_bucket.get(it.difficulty, default)
    return depth_bucket


async def calibrate_difficulty(
    items: list[QAItem], base_generator, *, sample_size: int = 200, seed: int = 42
) -> dict:
    """Full stage: measure -> assign. Returns a report dict."""
    solve = await measure_solve_rates(
        items, base_generator, sample_size=sample_size, seed=seed
    )
    mapping = assign_difficulty(items, solve)
    return {"solve_rate_by_depth": solve, "depth_to_bucket": mapping}
