"""Deterministic provider for offline tests. No network, no model."""

from __future__ import annotations

import json
from typing import Callable, Sequence

from .base import Message, ProviderHealth


class StubProvider:
    """Returns canned output and records every call.

    `responder` may be supplied to script behaviour per call; otherwise a plan
    request yields a JSON plan echoing the question and an answer request yields
    a citation of the first source it was shown.
    """

    name = "stub"

    def __init__(self, responder: Callable[[list[Message], bool], str] | None = None) -> None:
        self.responder = responder
        self.calls: list[dict] = []
        self.model = "stub"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        think: bool | None = None,
    ) -> str:
        msgs = list(messages)
        self.calls.append({"messages": msgs, "json_mode": json_mode})
        if self.responder is not None:
            return self.responder(msgs, json_mode)
        user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        if json_mode:
            question = user.strip().splitlines()[-1] if user.strip() else ""
            return json.dumps(
                {
                    "intent": "lookup",
                    "sub_questions": [question],
                    "queries": [question],
                    "needed_facts": [],
                }
            )
        if "[S1]" in user:
            return "Stub answer grounded in [S1]."
        return "Stub answer with no sources."

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Deterministic bag-of-chars vector; adequate for exercising the code path.
        out = []
        for text in texts:
            vec = [0.0] * 16
            for ch in text.lower():
                vec[ord(ch) % 16] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    def health(self) -> ProviderHealth:
        return ProviderHealth(True, "stub provider", ["stub"])
