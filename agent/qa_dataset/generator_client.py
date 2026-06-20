"""Generation LLM wrapper for the Q&A pipeline.

Decoupled from the interactive agent: the generator model is whatever the
profile's generation.generator_model resolves to (a local vllm/ollama id, an HF
Router id, etc.). Calls go through litellm (lazy import). A ``completion_fn`` can
be injected for tests so no live model is needed.

Constrained decoding (Outlines/xgrammar) is a configurable hook; the always-on
fallback is "ask for JSON + robustly extract + bounded repair retries", which
works with any backend.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

CompletionFn = Callable[..., Awaitable[str]]


def extract_json(text: str) -> Optional[dict]:
    """Best-effort parse of a JSON object from model text.

    Handles code fences and leading/trailing prose by scanning for the first
    balanced ``{...}`` block. Returns None if nothing parses.
    """
    if not text:
        return None
    s = text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Scan for the first balanced object.
    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = s.find("{", start + 1)
    return None


@dataclass
class GeneratorClient:
    """Thin async wrapper around a chat-completion backend.

    model: resolved litellm model id (never the literal "session"/"generator" —
           the caller resolves those to a concrete id first).
    """

    model: str
    max_new_tokens: int = 512
    hf_token: Optional[str] = None
    completion_fn: Optional[CompletionFn] = None  # injected for tests

    async def _default_completion(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: Optional[int],
    ) -> str:
        import litellm  # lazy

        try:
            from agent.core.llm_params import _resolve_llm_params  # reuse routing

            params = _resolve_llm_params(self.model, self.hf_token, reasoning_effort=None)
        except Exception:
            params = {"model": self.model}
        resp = await litellm.acompletion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            **params,
        )
        return resp.choices[0].message.content or ""

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        fn = self.completion_fn or self._default_completion
        return await fn(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or self.max_new_tokens,
            seed=seed,
        )

    async def generate_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        seed: Optional[int] = None,
        max_repairs: int = 2,
    ) -> Optional[dict]:
        """Return a parsed JSON object, with bounded repair retries.

        On a parse miss, re-asks the model to emit ONLY valid JSON. Returns None
        if it still can't be parsed after max_repairs.
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_repairs + 1):
            text = await self.chat(
                messages,
                temperature=temperature if attempt == 0 else 0.0,
                top_p=top_p,
                seed=seed,
            )
            obj = extract_json(text)
            if obj is not None:
                return obj
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": "That was not valid JSON. Reply with ONLY a single "
                    "valid JSON object and nothing else.",
                }
            )
        return None
