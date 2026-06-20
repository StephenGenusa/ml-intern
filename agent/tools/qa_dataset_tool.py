"""qa_dataset tool — first-class Q&A dataset generation for ml-intern.

Wraps agent.qa_dataset.* behind a single multi-operation tool so the agent can
initialize a project, browse/adjust the strategy profile collaboratively, chunk
sources, propose personas, and build a grounded, verified SFT dataset — all
driven by qa_dataset_profile.yaml. The generation model is decoupled from the
interactive agent (profile.generation.generator_model).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

QA_DATASET_TOOL_SPEC: dict[str, Any] = {
    "name": "qa_dataset",
    "description": (
        "Build a grounded, diverse, verified question-answer dataset from local "
        "documents to fine-tune a model. Operations: 'init' (seed "
        "qa_dataset_profile.yaml + qa_personas.yaml with best-practice defaults), "
        "'show_catalog' (list available diversification strategies), 'show_profile' "
        "(current settings + warnings), 'update_profile' (adjust settings during a "
        "discussion), 'chunk' (ingest + chunk source docs), 'propose_personas' "
        "(suggest reader personas for approval), 'build' (run the full pipeline: "
        "chunk -> generate -> verify+retry -> calibrate -> dedup/decontaminate -> "
        "package as SFT chat data with leak-free splits). The generation model is "
        "set by the profile, independent of the interactive agent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "init",
                    "show_catalog",
                    "show_profile",
                    "update_profile",
                    "chunk",
                    "propose_personas",
                    "build",
                ],
            },
            "working_dir": {
                "type": "string",
                "description": "Project directory holding the profile/personas (default: current dir).",
            },
            "input_dir": {
                "type": "string",
                "description": "Override the source documents directory for chunk/build.",
            },
            "out_dir": {
                "type": "string",
                "description": "Output directory for the built dataset (default: <working_dir>/qa_dataset_out).",
            },
            "updates": {
                "type": "object",
                "description": "For update_profile: top-level profile keys to merge.",
            },
            "personas": {
                "type": "array",
                "description": "For propose_personas with approve=true: the approved persona list to freeze.",
                "items": {"type": "object"},
            },
            "approve": {
                "type": "boolean",
                "description": "For propose_personas: freeze the provided personas into qa_personas.yaml.",
            },
            "topic_hint": {"type": "string"},
            "count": {"type": "integer"},
            "name": {"type": "string", "description": "Dataset name for build output."},
        },
        "required": ["operation"],
    },
}


def _session_model(session) -> Optional[str]:
    cfg = getattr(session, "config", None)
    return getattr(cfg, "model_name", None) if cfg else None


def _hf_token(session) -> Optional[str]:
    return getattr(session, "hf_token", None)


async def qa_dataset_handler(arguments: dict, session=None) -> tuple[str, bool]:
    from agent.qa_dataset import profile as P
    from agent.qa_dataset import strategies as S

    op = (arguments or {}).get("operation")
    working_dir = (arguments or {}).get("working_dir") or "."

    try:
        if op == "init":
            created = P.init_project(working_dir)
            if not created:
                return (
                    f"Profile and personas already exist in {working_dir}; nothing to do.",
                    True,
                )
            lines = [f"Initialized Q&A project in {working_dir}:"]
            for k, v in created.items():
                lines.append(f"  - {k}: {v}")
            lines.append(
                "Edit qa_dataset_profile.yaml (or ask me to) to tune strategies/mixes, "
                "then run operation 'build'."
            )
            return "\n".join(lines), True

        if op == "show_catalog":
            cat = S.load_catalog()
            lines = [f"Strategy catalog v{cat.version}:"]
            for s in cat.strategies.values():
                flag = "on " if s.enabled_by_default else "off"
                first = s.description.splitlines()[0] if s.description else ""
                lines.append(f"  [{flag}] {s.id} ({s.category}) — {first}")
            return "\n".join(lines), True

        if op == "show_profile":
            prof, path = P.resolve_profile(working_dir)
            cat = S.load_catalog()
            problems = S.validate_selection(prof, cat)
            payload = {
                "profile_path": str(path),
                "generator_model": prof.generation.generator_model,
                "enabled_strategies": prof.enabled_strategy_ids(),
                "question_types": prof.question_types,
                "difficulty_mix": prof.difficulty_mix,
                "target_size": prof.output.target_size,
                "verification": {
                    "faithfulness": prof.verification.faithfulness,
                    "answer_relevance": prof.verification.answer_relevance,
                    "self_consistency_k": prof.verification.self_consistency_k,
                    "max_repair_retries": prof.verification.max_repair_retries,
                },
                "warnings": prof.warnings(),
                "selection_problems": problems,
            }
            return json.dumps(payload, indent=2), True

        if op == "update_profile":
            updates = (arguments or {}).get("updates") or {}
            _, path = P.resolve_profile(working_dir)
            if not Path(path).name.endswith("qa_dataset_profile.yaml"):
                # Resolved to the shipped default; seed a project copy first.
                P.init_project(working_dir)
                _, path = P.resolve_profile(working_dir)
            prof = P.patch_profile(path, updates)
            return f"Updated {path}. Warnings: {prof.warnings() or 'none'}.", True

        if op == "chunk":
            from agent.qa_dataset.chunking import chunk_sources, write_chunks

            prof, _ = P.resolve_profile(working_dir)
            input_dir = (arguments or {}).get("input_dir") or prof.sources.input_dir
            chunks = chunk_sources(
                input_dir,
                chunk_tokens=prof.sources.chunk_tokens,
                overlap=prof.sources.chunk_overlap,
                min_chunk_chars=prof.sources.min_chunk_chars,
                preserve_tables=prof.sources.preserve_tables,
                keep_figure_captions=prof.sources.keep_figure_captions,
            )
            out = Path(working_dir) / "chunks.jsonl"
            write_chunks(chunks, out)
            by_source: dict[str, int] = {}
            tables = 0
            for c in chunks:
                by_source[c.source_id] = by_source.get(c.source_id, 0) + 1
                tables += 1 if c.has_table else 0
            return (
                f"Chunked {len(chunks)} chunks from {input_dir} -> {out}\n"
                f"  per source: {by_source}\n  table/figure chunks: {tables}",
                True,
            )

        if op == "propose_personas":
            from agent.qa_dataset.generator_client import GeneratorClient
            from agent.qa_dataset.personas import (
                freeze_personas,
                load_personas,
                propose_personas,
            )

            prof, _ = P.resolve_profile(working_dir)
            if (arguments or {}).get("approve"):
                given = (arguments or {}).get("personas") or []
                dst = Path(working_dir) / P.DEFAULT_PERSONAS_FILENAME
                n = freeze_personas(given, dst, version=prof.personas.version)
                return f"Froze {n} personas -> {dst}", True
            model = prof.generation.generator_model
            if model == "session":
                model = _session_model(session) or model
            gen = GeneratorClient(model=model, hf_token=_hf_token(session))
            fallback: list[dict] = []
            try:
                fallback = load_personas(P.shipped_personas_path())
            except Exception:
                pass
            proposed = await propose_personas(
                gen,
                count=int((arguments or {}).get("count") or 30),
                topic_hint=(arguments or {}).get("topic_hint") or "",
                fallback=fallback,
            )
            return (
                "Proposed personas (review, then call propose_personas with "
                "approve=true and an edited list):\n" + json.dumps(proposed, indent=2),
                True,
            )

        if op == "build":
            from agent.qa_dataset import pipeline as PL
            from agent.qa_dataset.personas import load_personas

            prof, _ = P.resolve_profile(working_dir)
            cat = S.load_catalog()
            problems = S.validate_selection(prof, cat)
            if problems:
                return (
                    "Profile has problems that would break generation:\n  - "
                    + "\n  - ".join(problems),
                    False,
                )
            session_model = _session_model(session)
            hf_token = _hf_token(session)
            generator = PL.resolve_generator(
                prof, session_model=session_model, hf_token=hf_token
            )
            judge = PL.resolve_judge(prof, generator, hf_token=hf_token)
            base = PL.resolve_base(prof, target_model=session_model, hf_token=hf_token)
            personas_path = Path(working_dir) / prof.personas.file
            personas = (
                load_personas(personas_path)
                if personas_path.exists()
                else load_personas(P.shipped_personas_path())
            )
            out_dir = (arguments or {}).get("out_dir") or str(
                Path(working_dir) / "qa_dataset_out"
            )
            logs: list[str] = []
            summary = await PL.build_dataset(
                prof,
                generator=generator,
                judge=judge,
                base=base,
                out_dir=out_dir,
                catalog=cat,
                personas=personas,
                name=(arguments or {}).get("name") or "my-books-qa",
                progress=logs.append,
            )
            report = {
                k: summary[k]
                for k in (
                    "chunks",
                    "eligible_chunks",
                    "generated",
                    "verified",
                    "final",
                    "train",
                    "val",
                    "decontaminated",
                    "out_dir",
                    "gate_failures",
                )
                if k in summary
            }
            return (
                "Build complete.\n"
                + "\n".join(logs)
                + "\n\nSummary:\n"
                + json.dumps(report, indent=2),
                True,
            )

        return f"Unknown operation: {op}", False

    except Exception as e:  # surface errors to the agent, don't crash the loop
        import traceback

        return f"qa_dataset {op} failed: {e}\n{traceback.format_exc()}", False
