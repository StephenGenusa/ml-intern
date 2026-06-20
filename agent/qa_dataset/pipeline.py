"""End-to-end Q&A dataset pipeline orchestrator.

chunk -> hold out eval sources -> generate (typed, persona-conditioned,
difficulty-laddered) -> verify (faithfulness/relevance/self-consistency/repair)
-> calibrate difficulty -> measure diversity/coverage -> filter (dedup,
decontaminate, cap, balance) -> package (chat rows, leak-free split, README).

The generator/judge/base models are decoupled from the interactive agent and are
resolved from the profile. They are passed in as GeneratorClient objects so the
pipeline is fully testable with injected fakes.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

from agent.qa_dataset import answer_gen, calibrate, diversify, filter_dedup, package
from agent.qa_dataset import question_gen as QG
from agent.qa_dataset.chunking import Chunk, chunk_sources
from agent.qa_dataset.generator_client import GeneratorClient
from agent.qa_dataset.personas import load_personas, sample_personas
from agent.qa_dataset.profile import QADatasetProfile, shipped_personas_path
from agent.qa_dataset.strategies import (
    Catalog,
    generators_for_type,
    load_catalog,
    resolve_enabled_strategies,
)
from agent.qa_dataset.verify import verify_item

Progress = Optional[Callable[[str], None]]


def _log(progress: Progress, msg: str) -> None:
    if progress:
        progress(msg)


def resolve_generator(
    profile: QADatasetProfile,
    *,
    session_model: Optional[str] = None,
    hf_token: Optional[str] = None,
    completion_fn=None,
) -> GeneratorClient:
    mid = profile.generation.generator_model
    if mid == "session":
        mid = session_model or mid
    return GeneratorClient(
        model=mid,
        max_new_tokens=profile.generation.max_new_tokens,
        hf_token=hf_token,
        completion_fn=completion_fn,
    )


def resolve_judge(
    profile: QADatasetProfile,
    generator: GeneratorClient,
    *,
    hf_token=None,
    completion_fn=None,
) -> GeneratorClient:
    jm = profile.verification.open_ended_rubric.judge_model
    if jm in ("generator", "", None):
        return generator
    if jm == "session":
        jm = generator.model
    return GeneratorClient(
        model=jm,
        hf_token=hf_token,
        completion_fn=completion_fn or generator.completion_fn,
    )


def resolve_base(
    profile: QADatasetProfile, *, target_model=None, hf_token=None, completion_fn=None
) -> Optional[GeneratorClient]:
    if not profile.difficulty_calibration.enabled:
        return None
    bm = profile.difficulty_calibration.base_model
    if bm == "target":
        bm = target_model
    if not bm:
        return None
    return GeneratorClient(model=bm, hf_token=hf_token, completion_fn=completion_fn)


def _holdout_split(chunks: list[Chunk], fraction: float, seed: int):
    """Return (eligible, holdout_chunks, holdout_sources) split by whole source."""
    by_source: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_source.setdefault(c.source_id, []).append(c)
    sources = sorted(by_source)
    rng = random.Random(seed)
    rng.shuffle(sources)
    holdout_sources: list[str] = []
    held: list[Chunk] = []
    # Only hold out if there's more than one source (else there'd be no train data).
    if len(sources) > 1:
        for s in sources:
            if len(held) >= fraction * len(chunks):
                break
            held.extend(by_source[s])
            holdout_sources.append(s)
    held_ids = set(holdout_sources)
    eligible = [c for c in chunks if c.source_id not in held_ids]
    return eligible, held, holdout_sources


async def build_dataset(
    profile: QADatasetProfile,
    *,
    generator: GeneratorClient,
    judge: Optional[GeneratorClient] = None,
    base: Optional[GeneratorClient] = None,
    out_dir: str | Path,
    chunks: Optional[list[Chunk]] = None,
    catalog: Optional[Catalog] = None,
    personas: Optional[list[dict]] = None,
    name: str = "my-books-qa",
    progress: Progress = None,
) -> dict:
    seed = profile.seed
    rng = random.Random(seed)
    judge = judge or generator
    catalog = catalog or load_catalog()

    # 1. Chunks
    if chunks is None:
        chunks = chunk_sources(
            profile.sources.input_dir,
            chunk_tokens=profile.sources.chunk_tokens,
            overlap=profile.sources.chunk_overlap,
            min_chunk_chars=profile.sources.min_chunk_chars,
            preserve_tables=profile.sources.preserve_tables,
            keep_figure_captions=profile.sources.keep_figure_captions,
        )
    _log(progress, f"chunked: {len(chunks)} chunks")

    # 2. Holdout (eval) split — excluded from generation
    eligible, held, holdout_sources = _holdout_split(
        chunks, profile.holdout.fraction, seed
    )
    holdout_texts = [c.text for c in held]
    eligible_ids = [c.chunk_id for c in eligible]
    _log(progress, f"eligible: {len(eligible)} | held out sources: {holdout_sources}")

    # 3. Personas + strategies
    if personas is None:
        try:
            personas = load_personas(shipped_personas_path())
        except Exception:
            personas = []
    strategies = resolve_enabled_strategies(profile, catalog)
    has_persona = any(s.id == "persona_conditioning" for s in strategies)
    ladder = next((s for s in strategies if s.id == "difficulty_ladder"), None)
    best_of_n = next(
        (int(s.params.get("n", 1)) for s in strategies if s.id == "best_of_n"), 1
    )
    qtypes = [t for t, share in (profile.question_types or {}).items() if share > 0]

    # 4. Generate (over-generate ~2x target, then filter down)
    raw: list[tuple] = []  # (item, chunk_text)
    target = filter_dedup.resolve_target_size(profile, len(eligible))
    for i, chunk in enumerate(eligible):
        if len(raw) >= target * 2:
            break
        for qtype in qtypes:
            gens = generators_for_type(strategies, qtype)
            if not gens:
                continue
            strat = gens[0]
            persona = (
                sample_personas(personas, 1, seed=rng.randint(0, 1_000_000))[0]
                if (has_persona and personas)
                else None
            )
            item_seed = rng.randint(0, 1_000_000)
            if qtype == "multi_hop":
                if i + 1 >= len(eligible):
                    continue
                item = await QG.generate_question(
                    strat,
                    chunk=chunk,
                    chunk_b=eligible[i + 1],
                    generator=generator,
                    question_type="multi_hop",
                    seed=item_seed,
                )
            else:
                item = await QG.generate_question(
                    strat,
                    chunk=chunk,
                    generator=generator,
                    question_type=qtype,
                    persona=persona,
                    seed=item_seed,
                )
            if not item:
                continue
            await answer_gen.answer_question(
                item,
                chunk_text=chunk.text,
                generator=generator,
                k=best_of_n,
                seed=item_seed,
            )
            if (
                ladder
                and item.answerable
                and qtype not in ("unanswerable", "multi_hop")
                and rng.random() < 0.3
            ):
                item = await QG.apply_difficulty_ladder(
                    item, ladder, chunk=chunk, generator=generator, seed=item_seed
                )
            raw.append((item, chunk.text))
    _log(progress, f"generated: {len(raw)} raw items")

    # 5. Verify
    kept = []
    for item, ctext in raw:
        verdict = await verify_item(
            item, chunk_text=ctext, judge=judge, generator=generator, profile=profile
        )
        if verdict.kept:
            kept.append(item)
    _log(progress, f"verified: {len(kept)} kept ({len(raw) - len(kept)} discarded)")

    # 6. Calibrate difficulty
    calib = None
    if base is not None:
        calib = await calibrate.calibrate_difficulty(
            kept, base, sample_size=profile.difficulty_calibration.sample_size, seed=seed
        )
        _log(progress, f"calibrated difficulty: {calib['depth_to_bucket']}")

    # 7. Diversity / coverage (pre-filter snapshot)
    pre_report = diversify.measure(kept, eligible_ids)
    gate_fails = diversify.gate_failures(kept, profile, eligible_ids)

    # 8. Filter
    items = list(kept)
    items = filter_dedup.exact_dedup(items)
    items = filter_dedup.near_dedup(
        items, jaccard_threshold=profile.dedup.minhash_jaccard
    )
    items, decon = filter_dedup.decontaminate(
        items, holdout_texts, ngram=profile.decontamination.ngram
    )
    items = filter_dedup.cap_per_chunk(items, profile.coverage.max_items_per_chunk)
    items = filter_dedup.balance(items, profile, target_size=target, seed=seed)
    _log(progress, f"filtered: {len(items)} final (decontam dropped {decon})")

    # 9. Package
    summary = package.package(
        items,
        out_dir,
        profile,
        name=name,
        generated_count=len(raw),
        holdout_sources=holdout_sources,
        seed=seed,
    )
    summary.update(
        {
            "chunks": len(chunks),
            "eligible_chunks": len(eligible),
            "generated": len(raw),
            "verified": len(kept),
            "final": len(items),
            "target_size": target,
            "decontaminated": decon,
            "diversity": pre_report,
            "gate_failures": gate_fails,
            "calibration": calib,
            "profile_warnings": profile.warnings(),
        }
    )
    _log(progress, f"done: {summary['final']} items -> {summary['out_dir']}")
    return summary
