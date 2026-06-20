"""Filtering: dedup, decontamination, per-chunk cap, balance, target sizing.

Turns verified items into a clean dataset: removes exact and near-duplicates,
drops anything overlapping the held-out evaluation text, caps items per source
chunk, and balances toward the profile's question-type targets up to the
resolved target size.
"""

from __future__ import annotations

import random
import re
from typing import Iterable, Optional

_WS = re.compile(r"\s+")


def _norm(s: Optional[str]) -> str:
    return _WS.sub(" ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def exact_dedup(items: list) -> list:
    seen = set()
    out = []
    for it in items:
        key = (_norm(it.question), _norm(it.answer))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _shingles(text: str, k: int = 3) -> set:
    toks = _norm(text).split()
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_dedup(items: list, *, jaccard_threshold: float = 0.7) -> list:
    """Greedy near-duplicate removal on question shingles (O(n*kept)).

    For very large datasets, swap this for datasketch MinHash+LSH; the threshold
    semantics are the same.
    """
    kept: list = []
    kept_shingles: list[set] = []
    for it in items:
        sh = _shingles(it.question)
        if any(_jaccard(sh, ks) >= jaccard_threshold for ks in kept_shingles):
            continue
        kept.append(it)
        kept_shingles.append(sh)
    return kept


def _word_ngrams(text: str, n: int) -> set:
    toks = _norm(text).split()
    if len(toks) < n:
        return set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def decontaminate(
    items: list, holdout_texts: Iterable[str], *, ngram: int = 13
) -> tuple[list, int]:
    """Drop items overlapping held-out text (n-gram match or question substring)."""
    holdout_grams: set = set()
    holdout_norm: list[str] = []
    for t in holdout_texts:
        holdout_grams |= _word_ngrams(t, ngram)
        holdout_norm.append(_norm(t))
    kept = []
    dropped = 0
    for it in items:
        text = f"{it.question} {it.answer or ''}"
        grams = _word_ngrams(text, ngram)
        q = _norm(it.question)
        leaked = bool(grams & holdout_grams) or any(q and q in hn for hn in holdout_norm)
        if leaked:
            dropped += 1
            continue
        kept.append(it)
    return kept, dropped


def cap_per_chunk(items: list, max_items_per_chunk: int) -> list:
    """Keep at most N items per primary supporting chunk (preserves order)."""
    counts: dict[str, int] = {}
    out = []
    for it in items:
        cid = (it.supporting_chunk_ids or ["<none>"])[0]
        if counts.get(cid, 0) >= max_items_per_chunk:
            continue
        counts[cid] = counts.get(cid, 0) + 1
        out.append(it)
    return out


def resolve_target_size(profile, eligible_chunk_count: int) -> int:
    ts = profile.output.target_size
    if isinstance(ts, int):
        return ts
    # "auto"
    return max(1, eligible_chunk_count) * profile.coverage.max_items_per_chunk


def balance(
    items: list, profile, *, target_size: Optional[int] = None, seed: int = 42
) -> list:
    """Downsample toward question_types proportions up to target_size.

    Never upsamples (generation handles volume). Caps each type to its target
    share of target_size; keeps everything if a type is under target.
    """
    if not items:
        return items
    rng = random.Random(seed)
    n_target = target_size or len(items)
    targets = profile.question_types or {}
    by_type: dict[str, list] = {}
    for it in items:
        by_type.setdefault(it.question_type, []).append(it)

    out: list = []
    for qtype, bucket in by_type.items():
        share = targets.get(qtype)
        if share is None:
            cap = len(bucket)
        else:
            cap = max(1, round(share * n_target))
        if len(bucket) > cap:
            bucket = rng.sample(bucket, cap)
        out.extend(bucket)
    rng.shuffle(out)
    return out
