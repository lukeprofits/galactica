"""Ollama adapter. Stdlib HTTP only."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Sequence

from .base import Message, ProviderError, ProviderHealth

# The single model-specific concession in the codebase: qwen3.x emits reasoning in
# <think> blocks, which must never leak into answers or JSON parsing.
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    text = _THINK.sub("", text)
    text = _OPEN_THINK.sub("", text)  # unterminated block (truncated output)
    return text.strip()


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        embed_model: str | None = None,
        timeout: float = 600.0,
        num_ctx: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.timeout = timeout
        # Without this, ollama reserves the model's maximum context as KV cache,
        # which can dwarf the weights themselves.
        self.num_ctx = num_ctx

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            raise ProviderError(f"ollama {path} HTTP {exc.code}: {exc.read().decode()[:400]}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise ProviderError(f"ollama unreachable at {self.base_url}: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            # A model that keeps generating past the socket timeout surfaced as
            # a raw TimeoutError and killed a 43-call eval run outright.
            raise ProviderError(
                f"ollama timed out after {timeout or self.timeout:.0f}s "
                f"(model may be generating without an output cap)"
            ) from exc

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        think: bool | None = None,
    ) -> str:
        options: dict = {"temperature": temperature}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        if max_tokens:
            options["num_predict"] = max_tokens
        payload: dict = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
        if think is not None:
            payload["think"] = think
        data = self._post("/api/chat", payload)
        content = (data.get("message") or {}).get("content", "")
        return strip_reasoning(content)

    def _with_context(self, payload: dict) -> dict:
        if not self.num_ctx:
            return payload
        options = {**(payload.get("options") or {})}
        options.setdefault("num_ctx", self.num_ctx)
        return {**payload, "options": options}

    def chat(self, payload: dict) -> dict:
        """Raw /api/chat call. Used by the gateway, which needs tool traffic and
        option passthrough that `complete()` deliberately hides."""
        return self._post("/api/chat", self._with_context(payload))

    def chat_stream(self, payload: dict):
        """Yield /api/chat response objects as they arrive."""
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps({**self._with_context(payload), "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for line in response:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line.decode())
                    except json.JSONDecodeError:
                        continue
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            raise ProviderError(
                f"ollama stream HTTP {exc.code}: {exc.read().decode()[:400]}"
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise ProviderError(
                f"ollama unreachable at {self.base_url}: {exc.reason}"
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise ProviderError(f"ollama stream timed out after {self.timeout:.0f}s") from exc

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.embed_model:
            raise ProviderError(
                "no embedding model configured; set GALACTICA_EMBED_MODEL and pull it in ollama"
            )
        data = self._post("/api/embed", {"model": self.embed_model, "input": list(texts)})
        vectors = data.get("embeddings")
        if not vectors:
            raise ProviderError(f"ollama /api/embed returned no embeddings: {str(data)[:200]}")
        return [[float(x) for x in vec] for vec in vectors]

    def health(self) -> ProviderHealth:
        req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            return ProviderHealth(False, f"unreachable at {self.base_url}: {exc}")
        models = [m.get("name", "") for m in data.get("models", [])]
        if self.model not in models:
            return ProviderHealth(
                False, f"model '{self.model}' not installed (ollama pull {self.model})", models
            )
        return ProviderHealth(True, f"ollama ok, model '{self.model}' present", models)
