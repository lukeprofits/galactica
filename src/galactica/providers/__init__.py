from __future__ import annotations

from ..config import Config
from .base import LLMProvider, Message, ProviderError, ProviderHealth
from .ollama import OllamaProvider, strip_reasoning
from .stub import StubProvider

__all__ = [
    "LLMProvider",
    "Message",
    "ProviderError",
    "ProviderHealth",
    "OllamaProvider",
    "StubProvider",
    "strip_reasoning",
    "build_provider",
]


def build_provider(cfg: Config) -> LLMProvider:
    if cfg.provider == "ollama":
        return OllamaProvider(
            model=cfg.model,
            base_url=cfg.ollama_base_url,
            embed_model=cfg.embed_model,
            num_ctx=cfg.effective_num_ctx(),
        )
    if cfg.provider == "stub":
        return StubProvider()
    raise ProviderError(f"unknown provider '{cfg.provider}' (expected: ollama, stub)")
