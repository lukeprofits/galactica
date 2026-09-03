"""Model-provider boundary. The knowledge system never imports anything below this."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": ...}


@dataclass
class ProviderHealth:
    ok: bool
    detail: str
    models: list[str] = field(default_factory=list)


class ProviderError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def health(self) -> ProviderHealth: ...
