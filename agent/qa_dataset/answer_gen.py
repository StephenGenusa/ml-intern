"""Grounded answer generation.

Answers a question using ONLY the cited chunk(s), attaching verbatim evidence
spans and abstaining when unanswerable. Strategies that already produce an
answer (counterfactual_qg, table_qa) are passed through. Best-of-N sampling
collects candidates; selection happens in verification (self-consistency).
"""

from __future__ import annotations

from typing import Optional

from agent.qa_dataset.question_gen import QAItem

ANSWER_SYSTEM = (
    "You answer questions using ONLY the SOURCE provided. If the SOURCE lacks the "
    "answer, set answerable=false and leave answer empty. Never use outside knowledge."
)


def _answer_prompt(question: str, source: str) -> str:
    return (
        f"SOURCE:\n{source}\n\nQUESTION: {question}\n\n"
        'Return JSON: {"answerable": bool, "answer": "...", '
        '"evidence_spans": ["verbatim quote", ...], "reasoning": "..."}'
    )


async def answer_question(
    item: QAItem,
    *,
    chunk_text: str,
    generator,
    k: int = 3,
    seed: Optional[int] = None,
) -> QAItem:
    """Populate item.answer / evidence / candidates. Honors pre-existing answers."""
    # Unanswerable items: the gold answer is abstention; nothing to generate.
    if item.strategy_id == "unanswerable_qg" or item.answerable is False:
        item.answerable = False
        item.answer = None
        return item

    # Strategies that already returned an answer (counterfactual, table_qa).
    if item.answer:
        item.candidates = [item.answer]
        return item

    prompt = _answer_prompt(item.question, chunk_text)
    candidates: list[str] = []
    chosen: Optional[dict] = None
    for i in range(max(1, k)):
        obj = await generator.generate_json(
            prompt,
            system=ANSWER_SYSTEM,
            temperature=0.7 if k > 1 else 0.3,
            seed=(seed + i) if seed is not None else None,
        )
        if not obj:
            continue
        if obj.get("answerable") is False:
            chosen = {"answerable": False}
            break
        ans = (obj.get("answer") or "").strip()
        if ans:
            candidates.append(ans)
            if chosen is None:
                chosen = obj

    if chosen is None:
        # Could not get a usable answer; mark unanswerable so verify can drop it.
        item.answerable = False
        item.answer = None
        return item
    if chosen.get("answerable") is False:
        item.answerable = False
        item.answer = None
        return item

    item.answer = (chosen.get("answer") or "").strip()
    item.evidence_spans = [
        s for s in (chosen.get("evidence_spans") or []) if isinstance(s, str)
    ]
    item.reasoning = chosen.get("reasoning")
    item.candidates = candidates
    return item
