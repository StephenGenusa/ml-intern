"""Verification: turn raw generated items into trustworthy ones (or discard them).

Pipeline per item:
  schema -> verbatim span (skipped for counterfactual; uses grounding_quote)
  -> faithfulness (answer entailed by evidence) -> answer-relevance (answer
  addresses the question) -> self-consistency (majority over best-of-N) ->
  optional open-ended rubric. On a fixable failure, repair-and-retry up to
  max_repair_retries, then discard. Unanswerable items skip answer checks and
  are kept as abstention examples.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from agent.qa_dataset.question_gen import QAItem

CLOSED_FORM = {"factual", "boolean", "list"}
_WS = re.compile(r"\s+")


def _norm(s: Optional[str]) -> str:
    return _WS.sub(" ", (s or "").strip().lower())


def _norm_answer(s: Optional[str]) -> str:
    """Stronger normalization for answer agreement: drop punctuation."""
    return _WS.sub(" ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


@dataclass
class Verdict:
    kept: bool
    reasons: list[str]


def check_schema(item: QAItem) -> tuple[bool, str]:
    if not item.question.strip():
        return False, "empty question"
    if item.answerable and not (item.answer or "").strip():
        return False, "answerable item has no answer"
    if not item.answerable and (item.answer or "").strip():
        return False, "unanswerable item carries an answer"
    return True, ""


def check_verbatim(item: QAItem, chunk_text: str) -> tuple[bool, str]:
    """Grounded span must appear verbatim in source (whitespace-normalized)."""
    hay = _norm(chunk_text)
    if item.strategy_id == "counterfactual_qg":
        q = _norm(item.grounding_quote)
        if not q:
            return False, "counterfactual missing grounding_quote"
        return (q in hay, "" if q in hay else "grounding_quote not found in source")
    needles = [s for s in (item.evidence_spans or []) if _norm(s)]
    if item.answer_span:
        needles.append(item.answer_span)
    if not needles:
        return True, ""  # nothing to check (rely on faithfulness judge)
    for n in needles:
        if _norm(n) in hay:
            return True, ""
    return False, "no evidence span found verbatim in source"


def self_consistency(item: QAItem) -> tuple[float, Optional[str]]:
    """Agreement rate over candidates; majority answer for closed-form types."""
    cands = [c for c in (item.candidates or []) if c]
    if len(cands) < 2:
        return (1.0 if cands else 0.0), (cands[0] if cands else None)
    counts = Counter(_norm_answer(c) for c in cands)
    norm_majority, n = counts.most_common(1)[0]
    rate = n / len(cands)
    # map normalized majority back to an original surface form
    surface = next((c for c in cands if _norm_answer(c) == norm_majority), cands[0])
    return rate, surface


async def judge_faithfulness(item: QAItem, chunk_text: str, judge) -> str:
    evidence = " ".join(item.evidence_spans) or item.grounding_quote or chunk_text
    prompt = (
        f"EVIDENCE:\n{evidence}\n\nQUESTION: {item.question}\nANSWER: {item.answer}\n\n"
        'Does the EVIDENCE fully support the ANSWER? Return JSON: '
        '{"verdict": "supported|partial|unsupported"}'
    )
    obj = await judge.generate_json(prompt, temperature=0.0)
    return (obj or {}).get("verdict", "unsupported")


async def judge_relevance(item: QAItem, judge) -> bool:
    prompt = (
        f"QUESTION: {item.question}\nANSWER: {item.answer}\n\n"
        'Does the ANSWER directly address the QUESTION? Return JSON: {"relevant": true|false}'
    )
    obj = await judge.generate_json(prompt, temperature=0.0)
    return bool((obj or {}).get("relevant", False))


async def judge_rubric(
    item: QAItem, chunk_text: str, judge, dimensions: list[str], min_score: int
) -> tuple[bool, dict]:
    dims = ", ".join(dimensions)
    prompt = (
        f"SOURCE:\n{chunk_text}\n\nQUESTION: {item.question}\nANSWER: {item.answer}\n\n"
        f"Score the ANSWER 1-5 on each of: {dims}. Return JSON: "
        '{"scores": {"<dim>": int, ...}}'
    )
    obj = await judge.generate_json(prompt, temperature=0.0)
    scores = (obj or {}).get("scores") or {}
    ok = all(int(scores.get(d, 0)) >= min_score for d in dimensions) if scores else False
    return ok, scores


async def _repair_answer(item: QAItem, chunk_text: str, generator, defect: str) -> None:
    prompt = (
        f"SOURCE:\n{chunk_text}\n\nQUESTION: {item.question}\n"
        f"Your previous answer had this problem: {defect}. Answer again using ONLY "
        'the SOURCE and directly address the question. Return JSON: '
        '{"answerable": bool, "answer": "...", "evidence_spans": ["..."]}'
    )
    obj = await generator.generate_json(prompt, temperature=0.0)
    if obj and obj.get("answer"):
        item.answer = str(obj["answer"]).strip()
        item.evidence_spans = [
            s for s in (obj.get("evidence_spans") or []) if isinstance(s, str)
        ]
        if obj.get("answerable") is False:
            item.answerable = False
            item.answer = None


async def verify_item(
    item: QAItem,
    *,
    chunk_text: str,
    judge,
    generator,
    profile,
) -> Verdict:
    """Run the full QC chain with bounded repair. Mutates item.verification."""
    v = profile.verification
    reasons: list[str] = []
    item.verification = {}

    ok, why = check_schema(item)
    if not ok:
        return Verdict(False, [why])

    # Unanswerable -> abstention example; skip answer-grounding checks, keep.
    if not item.answerable:
        item.verification = {"kind": "unanswerable", "kept": True}
        return Verdict(True, [])

    # Self-consistency: pick majority answer for closed-form types.
    rate, majority = self_consistency(item)
    if item.question_type in CLOSED_FORM and majority:
        item.answer = majority
    item.verification["agreement_rate"] = round(rate, 3)

    max_retries = int(getattr(v, "max_repair_retries", 2))
    for attempt in range(max_retries + 1):
        ok_span, span_why = (True, "")
        if _truthy(v.verbatim_span_check):
            ok_span, span_why = check_verbatim(item, chunk_text)

        faith = "supported"
        if _truthy(v.faithfulness):
            faith = await judge_faithfulness(item, chunk_text, judge)

        relevant = True
        if _truthy(v.answer_relevance):
            relevant = await judge_relevance(item, judge)

        item.verification.update(
            {"faithfulness": faith, "relevant": relevant, "verbatim_ok": ok_span}
        )

        hard_fail = (faith == "unsupported") or (not ok_span)
        fixable = (faith == "partial") or (not relevant)

        if not hard_fail and not fixable:
            break  # passed
        if attempt >= max_retries:
            reasons.append(
                f"failed QC (faithfulness={faith}, relevant={relevant}, verbatim={ok_span})"
            )
            return Verdict(False, reasons)
        defect = (
            "answer not fully supported by the source"
            if faith != "supported"
            else "answer did not address the question"
        )
        if hard_fail and faith == "unsupported":
            defect = "answer is not supported by the source at all"
        await _repair_answer(item, chunk_text, generator, defect)
        if not item.answerable:
            item.verification["kind"] = "unanswerable"
            return Verdict(True, [])

    # Optional rubric for open-ended answers.
    rub = v.open_ended_rubric
    if getattr(rub, "enabled", False) and item.question_type not in CLOSED_FORM:
        passed, scores = await judge_rubric(
            item, chunk_text, judge, list(rub.dimensions), int(rub.min_score)
        )
        item.verification["rubric"] = scores
        if not passed:
            return Verdict(False, ["failed open-ended rubric"])

    item.verification["kept"] = True
    return Verdict(True, [])


def _truthy(val) -> bool:
    return val is True or (
        isinstance(val, str) and val.lower() in {"required", "true", "yes", "on"}
    )
