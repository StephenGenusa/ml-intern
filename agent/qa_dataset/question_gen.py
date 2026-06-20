"""Question generation across the enabled question-type strategies.

Renders each strategy's catalog prompt template against a chunk (and persona /
answer hint / second chunk as applicable), calls the generator for schema'd
JSON, and builds QAItem rows. Also hosts the answer-first span extraction and
the difficulty-ladder evolution step.
"""

from __future__ import annotations

import string
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from agent.qa_dataset.strategies import Strategy


@dataclass
class QAItem:
    question: str = ""
    question_type: str = "factual"
    answer: Optional[str] = None
    answerable: bool = True
    answer_span: Optional[str] = None
    evidence_spans: list[str] = field(default_factory=list)
    grounding_quote: Optional[str] = None
    supporting_chunk_ids: list[str] = field(default_factory=list)
    source_id: str = ""
    persona_id: Optional[str] = None
    strategy_id: str = ""
    difficulty: int = 0  # evolution_depth
    difficulty_bucket: Optional[str] = None
    reasoning: Optional[str] = None
    candidates: list[str] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QAItem":
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)


class _Blank(dict):
    def __missing__(self, key):  # noqa: D401
        return ""


def render_prompt(template: str, mapping: dict[str, Any]) -> str:
    """Format a catalog prompt template, tolerating missing keys (-> '')."""
    return string.Formatter().vformat(template, (), _Blank(mapping))


async def extract_answer_spans(
    chunk,
    strategy: Strategy,
    generator,
    *,
    temperature: float = 0.3,
    seed: Optional[int] = None,
) -> list[str]:
    """answer_first_qg step 1: pull salient candidate answer spans from a chunk."""
    prompt = render_prompt(
        strategy.prompt_template,
        {
            "passage": chunk.text,
            "max_spans_per_chunk": strategy.params.get("max_spans_per_chunk", 5),
        },
    )
    obj = await generator.generate_json(prompt, temperature=temperature, seed=seed)
    spans = (obj or {}).get("candidate_spans") if isinstance(obj, dict) else None
    return [s for s in (spans or []) if isinstance(s, str) and s.strip()]


async def generate_question(
    strategy: Strategy,
    *,
    chunk,
    generator,
    question_type: Optional[str] = None,
    persona: Optional[dict] = None,
    answer_hint: Optional[str] = None,
    chunk_b=None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: Optional[int] = None,
) -> Optional[QAItem]:
    """Run one question-type strategy on a chunk; return a QAItem or None."""
    qtype = question_type or (
        strategy.applies_to[0]
        if strategy.applies_to and strategy.applies_to[0] != "*"
        else "factual"
    )
    mapping = {
        "passage": chunk.text,
        "question_type": qtype,
        "answer_hint": answer_hint or "",
        "persona": (persona or {}).get("description", ""),
        "roles": ", ".join(strategy.params.get("roles", [])),
        "num_distractors": strategy.params.get("num_distractors", 3),
    }
    if chunk_b is not None:
        mapping.update(
            {
                "passage_a": chunk.text,
                "passage_b": chunk_b.text,
                "chunk_id_a": chunk.chunk_id,
                "chunk_id_b": chunk_b.chunk_id,
            }
        )
    prompt = render_prompt(strategy.prompt_template, mapping)
    obj = await generator.generate_json(
        prompt, temperature=temperature, top_p=top_p, seed=seed
    )
    if not obj or not obj.get("question"):
        return None

    supporting = obj.get("supporting_chunk_ids") or [chunk.chunk_id]
    if chunk_b is not None and chunk_b.chunk_id not in supporting:
        supporting = [chunk.chunk_id, chunk_b.chunk_id]

    item = QAItem(
        question=str(obj["question"]).strip(),
        question_type=str(obj.get("question_type") or qtype),
        answer=obj.get("answer"),
        answerable=bool(obj.get("answerable", True)),
        answer_span=obj.get("answer_span"),
        grounding_quote=obj.get("grounding_quote"),
        supporting_chunk_ids=list(supporting),
        source_id=chunk.source_id,
        persona_id=(persona or {}).get("id"),
        strategy_id=strategy.id,
        provenance={
            "generator_model": getattr(generator, "model", None),
            "strategy": strategy.id,
            "seed": seed,
            "temperature": temperature,
            "requires_outside_knowledge": bool(
                obj.get("requires_outside_knowledge", False)
            ),
        },
    )
    # Unanswerable strategy: force the flag regardless of model drift.
    if strategy.id == "unanswerable_qg":
        item.answerable = False
        item.answer = None
    return item


async def apply_difficulty_ladder(
    item: QAItem,
    strategy: Strategy,
    *,
    chunk,
    generator,
    seed: Optional[int] = None,
) -> QAItem:
    """Evolve a question harder via one operator per step; re-verify answerability."""
    operators = strategy.params.get("operators", [])
    max_depth = int(strategy.params.get("max_depth", 2))
    current = item
    for depth in range(min(max_depth, len(operators))):
        op = operators[depth]
        prompt = render_prompt(
            strategy.prompt_template,
            {"operator": op, "question": current.question, "passage": chunk.text},
        )
        obj = await generator.generate_json(prompt, temperature=0.4, seed=seed)
        if not obj or not obj.get("evolved_question"):
            break
        if not obj.get("still_answerable", True):
            break  # discard the evolution, keep the last good one
        current = QAItem(
            **{
                **current.to_dict(),
                "id": uuid.uuid4().hex[:12],
                "question": str(obj["evolved_question"]).strip(),
                "answer_span": obj.get("new_answer_span") or current.answer_span,
                "difficulty": current.difficulty + 1,
            }
        )
    return current
