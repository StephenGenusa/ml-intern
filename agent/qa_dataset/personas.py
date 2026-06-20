"""Reader-persona generation, curation, and sampling.

Personas live in their own file (qa_personas.yaml), seeded from the shipped
default_personas.yaml. The collaborative step proposes candidates the user can
trim/add/approve; the frozen set is then sampled (deterministically, by seed) to
condition question generation via the persona_conditioning strategy.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import yaml


def load_personas(path: str | Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return list(data.get("personas") or [])


def freeze_personas(personas: list[dict], path: str | Path, *, version: int = 1) -> int:
    """Write the approved persona set to qa_personas.yaml. Returns count."""
    payload = {"version": version, "personas": personas}
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return len(personas)


_PROPOSE_SYSTEM = (
    "You design reader personas for generating questions about a body of text. "
    "Each persona is a distinct reader with a comprehension level, a purpose, and "
    "a disposition. Avoid national, ethnic, or cultural stereotypes."
)


async def propose_personas(
    generator,
    *,
    count: int = 30,
    topic_hint: str = "",
    fallback: Optional[list[dict]] = None,
) -> list[dict]:
    """Propose persona candidates via the generator.

    Returns a list of {id, description, tags}. Falls back to ``fallback`` (the
    shipped defaults) if the model call yields nothing usable.
    """
    prompt = (
        f"Propose {count} distinct reader personas for a Q&A dataset"
        + (f" about: {topic_hint}." if topic_hint else ".")
        + " Vary comprehension level, purpose, and disposition. Return JSON: "
        '{"personas": [{"id": "p-01-...", "description": "...", "tags": ["..."]}]}'
    )
    obj = await generator.generate_json(prompt, system=_PROPOSE_SYSTEM, temperature=0.8)
    personas = (obj or {}).get("personas") if isinstance(obj, dict) else None
    if not personas:
        return list(fallback or [])
    cleaned: list[dict] = []
    for i, p in enumerate(personas):
        if not isinstance(p, dict) or not p.get("description"):
            continue
        cleaned.append(
            {
                "id": p.get("id") or f"p-{i + 1:02d}",
                "description": str(p["description"]).strip(),
                "tags": list(p.get("tags") or []),
            }
        )
    return cleaned or list(fallback or [])


def sample_personas(personas: list[dict], k: int, *, seed: int = 42) -> list[dict]:
    """Deterministic sample of k personas (with replacement if k > len)."""
    if not personas:
        return []
    rng = random.Random(seed)
    if k <= len(personas):
        return rng.sample(personas, k)
    return [rng.choice(personas) for _ in range(k)]
