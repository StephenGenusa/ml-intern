"""Packaging: chat formatting, provenance, leak-free splits, datasheet.

Converts verified+filtered items into SFT chat rows (with the fixed system
prompt and the abstention answer for unanswerable items), splits by source so
nothing leaks between train and validation, and writes a README/datasheet. The
completion_only_loss flag is recorded for the trainer (loss on the assistant
answer only).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional


def finalize_ids(items: list, prefix: str = "qa") -> list:
    for i, it in enumerate(items, 1):
        it.id = f"{prefix}-{i:05d}"
    return items


def to_chat_row(item, profile) -> dict:
    out = profile.output
    if not item.answerable:
        assistant = out.abstention_text
    else:
        assistant = item.answer or ""
    messages = [
        {"role": "system", "content": out.system_prompt},
        {"role": "user", "content": item.question},
        {"role": "assistant", "content": assistant},
    ]
    meta = {
        "id": item.id,
        "source_id": item.source_id,
        "question_type": item.question_type,
        "answerable": item.answerable,
        "difficulty_bucket": item.difficulty_bucket,
        "difficulty": item.difficulty,
        "supporting_chunk_ids": item.supporting_chunk_ids,
        "persona_id": item.persona_id,
        "strategy_id": item.strategy_id,
        "evidence_spans": item.evidence_spans,
        "verification": item.verification,
        "provenance": item.provenance,
    }
    return {"messages": messages, "meta": meta}


def split_by_source(
    items: list, *, val_fraction: float = 0.1, seed: int = 42
) -> tuple[list, list]:
    """Assign whole sources to train/val so no source straddles the split."""
    by_source: dict[str, list] = {}
    for it in items:
        by_source.setdefault(it.source_id, []).append(it)
    # Degenerate case: a single eligible source can't be split by source without
    # putting everything on one side. Fall back to an item-level split (some
    # paraphrase leakage is possible; unavoidable with one source).
    if len(by_source) < 2:
        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        n_val = int(val_fraction * len(shuffled)) if len(shuffled) > 1 else 0
        return shuffled[n_val:], shuffled[:n_val]
    sources = sorted(by_source)
    rng = random.Random(seed)
    rng.shuffle(sources)
    n_total = len(items)
    val: list = []
    val_sources: list[str] = []
    for s in sources:
        if len(val) >= val_fraction * n_total:
            break
        val.extend(by_source[s])
        val_sources.append(s)
    train = [it for it in items if it.source_id not in set(val_sources)]
    return train, val


def build_readme(
    name: str,
    *,
    profile,
    generated: int,
    kept: int,
    train_n: int,
    val_n: int,
    holdout_sources: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
) -> str:
    pct = (kept / generated * 100) if generated else 0.0
    lines = [
        f"# {name}",
        "",
        "## Generation",
        f"- Generator: {profile.generation.generator_model}",
        f"- Strategies: {', '.join(profile.enabled_strategy_ids())}",
        f"- Verification: faithfulness + self-consistency (k={profile.verification.self_consistency_k}) "
        f"+ answer-relevance; repair retries={profile.verification.max_repair_retries}",
        f"- Total generated: {generated} / kept: {kept} ({pct:.1f}%)",
        "",
        "## Splits (leak-free, by source)",
        f"- Train: {train_n} items",
        f"- Validation: {val_n} items",
    ]
    if holdout_sources:
        lines.append(
            f"- Held-out sources (excluded from generation): {', '.join(holdout_sources)}"
        )
    if sources:
        lines += ["", "## Sources", *[f"- {s}" for s in sources]]
    lines += [
        "",
        "## Training notes",
        f"- Format: {profile.output.format}; chat template applied at load time.",
        f"- completion_only_loss: {profile.output.completion_only_loss} "
        "(train loss on the assistant answer only).",
        "- Unanswerable items teach abstention via a fixed refusal answer.",
        "",
        "## Known limitations",
        "- Domain limited to the listed sources; no generalization beyond them.",
        "- No human review; expect residual noise even after verification.",
    ]
    return "\n".join(lines)


def write_jsonl(rows: list[dict], path: str | Path) -> int:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def package(
    items: list,
    out_dir: str | Path,
    profile,
    *,
    name: str = "my-books-qa",
    generated_count: Optional[int] = None,
    holdout_sources: Optional[list[str]] = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """Finalize ids, split, write train/val JSONL + README. Returns a summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    finalize_ids(items)
    train, val = split_by_source(items, val_fraction=val_fraction, seed=seed)
    train_rows = [to_chat_row(it, profile) for it in train]
    val_rows = [to_chat_row(it, profile) for it in val]
    write_jsonl(train_rows, out / "train.jsonl")
    write_jsonl(val_rows, out / "val.jsonl")
    readme = build_readme(
        name,
        profile=profile,
        generated=generated_count or len(items),
        kept=len(items),
        train_n=len(train),
        val_n=len(val),
        holdout_sources=holdout_sources,
        sources=sorted({it.source_id for it in items}),
    )
    (out / "README.md").write_text(readme, encoding="utf-8")
    return {
        "out_dir": str(out),
        "train": len(train),
        "val": len(val),
        "kept": len(items),
    }
