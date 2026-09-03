"""Environment-driven configuration. Every field is overridable by a CLI flag."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

DEFAULT_MODEL = "qwen3.6:35b-mlx"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Written by `galactica setup`, so a machine keeps its chosen model, corpus
# location and budgets without the user exporting environment variables.
CONFIG_ENV = "GALACTICA_CONFIG"
DEFAULT_CONFIG_PATH = Path("~/.config/galactica/config.toml")


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV, str(DEFAULT_CONFIG_PATH))).expanduser()


def read_config_file(path: Path | None = None) -> dict:
    """Stored settings, or {} when there is no config file yet."""
    target = path or config_path()
    if not target.exists():
        return {}
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data.get("galactica", data)


def write_config_file(values: dict, path: Path | None = None) -> Path:
    """Persist settings. Only known fields are written, so typos cannot creep in."""
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    known = {f.name for f in fields(Config)}
    lines = ["# Written by `galactica setup`. Environment variables and CLI",
             "# flags still override anything here.", "[galactica]"]
    for key, value in sorted(values.items()):
        if key not in known or value is None:
            continue
        if isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f'{key} = "{value}"')
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class _Layers:
    """Setting lookup: environment variable, then config file, then default."""

    def __init__(self, stored: dict) -> None:
        self.stored = stored

    def text(self, env: str, key: str, default):
        raw = os.environ.get(env)
        if raw not in (None, ""):
            return raw
        return self.stored.get(key, default)

    def number(self, env: str, key: str, default: int) -> int:
        raw = os.environ.get(env)
        if raw:
            try:
                return int(raw)
            except ValueError:
                pass
        value = self.stored.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def flag(self, env: str, key: str, default: bool) -> bool:
        raw = os.environ.get(env)
        if raw not in (None, ""):
            return raw.lower() not in ("0", "false", "no")
        value = self.stored.get(key, default)
        return bool(value)


@dataclass(frozen=True)
class Config:
    provider: str = "ollama"
    model: str = DEFAULT_MODEL
    embed_model: str | None = None
    ollama_base_url: str = DEFAULT_OLLAMA_URL
    data_dir: Path = Path("./data")
    context_budget: int = 16000
    top_k: int = 24
    mode: str = "cortex"
    # augmented: the corpus improves the answer and gaps are filled from the
    # model's own knowledge, labelled. strict: answer only from the corpus.
    grounding: str = "augmented"
    # retrieval / packing knobs
    expand: int = 1
    expand_top: int = 3
    per_doc_fraction: float = 0.35
    # Budget alone is a weak filter on a small corpus: without a cap, every
    # weakly-matching document gets a citable slot.
    max_sources: int = 8
    # Source slots reserved for documents not already represented (corroboration).
    diversity_slots: int = 2
    hops: int = 1
    temperature: float = 0.0
    # Reasoning stays on: it is capability, not overhead. A client's max_tokens
    # governs the visible answer, so thinking gets its own allowance on top of
    # it rather than eating it.
    think: bool = True
    think_reserve: int = 2048
    # Ollama otherwise allocates the model's maximum context as KV cache: a
    # 2.5 GB qwen3:4b reserved 42 GB for its 256K window. We never send more
    # than the context budget plus reasoning room, so cap it.
    # None means "derive it from what we actually send" (see effective_num_ctx).
    # An explicit value always wins.
    num_ctx: int | None = None
    # Without a cap, a small thinking model can generate until the socket times
    # out. Reasoning is funded on top of this, as in the gateway.
    max_answer_tokens: int = 2048
    # Room for the prompt scaffolding we add ourselves: system prompt, question,
    # source headers, and slack for the 4-chars-per-token estimate being wrong.
    prompt_overhead: int = 2048
    # A gateway client (Claude Code) sends its own system prompt, tool schemas
    # and conversation history, none of which fit inside our context budget.
    client_reserve: int = 16384

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Config":
        """Settings from the config file, overridden by the environment."""
        layers = _Layers(read_config_file(config_path))
        explicit_ctx = layers.number("GALACTICA_NUM_CTX", "num_ctx", 0)
        return cls(
            provider=layers.text("GALACTICA_PROVIDER", "provider", "ollama"),
            model=layers.text("GALACTICA_MODEL", "model", DEFAULT_MODEL),
            embed_model=layers.text("GALACTICA_EMBED_MODEL", "embed_model", None) or None,
            ollama_base_url=layers.text("OLLAMA_BASE_URL", "ollama_base_url", DEFAULT_OLLAMA_URL),
            data_dir=Path(layers.text("GALACTICA_DATA_DIR", "data_dir", "./data")).expanduser(),
            context_budget=layers.number("GALACTICA_CONTEXT_BUDGET", "context_budget", 16000),
            top_k=layers.number("GALACTICA_TOP_K", "top_k", 24),
            max_sources=layers.number("GALACTICA_MAX_SOURCES", "max_sources", 8),
            mode=layers.text("GALACTICA_MODE", "mode", "cortex"),
            grounding=layers.text("GALACTICA_GROUNDING", "grounding", "augmented"),
            think=layers.flag("GALACTICA_THINK", "think", True),
            think_reserve=layers.number("GALACTICA_THINK_RESERVE", "think_reserve", 2048),
            num_ctx=explicit_ctx or None,
            max_answer_tokens=layers.number("GALACTICA_MAX_ANSWER_TOKENS", "max_answer_tokens", 2048),
            prompt_overhead=layers.number("GALACTICA_PROMPT_OVERHEAD", "prompt_overhead", 2048),
            client_reserve=layers.number("GALACTICA_CLIENT_RESERVE", "client_reserve", 16384),
        )

    @classmethod
    def from_env(cls) -> "Config":
        return cls.load()

    @classmethod
    def _env_only(cls) -> "Config":
        return cls(
            provider=os.environ.get("GALACTICA_PROVIDER", "ollama"),
            model=os.environ.get("GALACTICA_MODEL", DEFAULT_MODEL),
            embed_model=os.environ.get("GALACTICA_EMBED_MODEL") or None,
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
            data_dir=Path(os.environ.get("GALACTICA_DATA_DIR", "./data")),
            context_budget=_int("GALACTICA_CONTEXT_BUDGET", 16000),
            top_k=_int("GALACTICA_TOP_K", 24),
            max_sources=_int("GALACTICA_MAX_SOURCES", 8),
            mode=os.environ.get("GALACTICA_MODE", "cortex"),
            grounding=os.environ.get("GALACTICA_GROUNDING", "augmented"),
            think=os.environ.get("GALACTICA_THINK", "1").lower() not in ("0", "false", "no"),
            think_reserve=_int("GALACTICA_THINK_RESERVE", 2048),
            num_ctx=(_int("GALACTICA_NUM_CTX", 0) or None),
            max_answer_tokens=_int("GALACTICA_MAX_ANSWER_TOKENS", 2048),
            prompt_overhead=_int("GALACTICA_PROMPT_OVERHEAD", 2048),
            client_reserve=_int("GALACTICA_CLIENT_RESERVE", 16384),
        )

    def override(self, **kwargs) -> "Config":
        """Apply non-None overrides (CLI flags win over env)."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean) if clean else self

    MIN_NUM_CTX = 4096

    def effective_num_ctx(self, *, client_reserve: int = 0) -> int:
        """KV cache size to ask Ollama for.

        Ollama reserves the model's *maximum* context unless told otherwise, and
        that reservation usually dwarfs the weights: `qwen3:4b` is 2.5 GB and
        asked for 42 GB at its native 256K window. So the size is derived from
        what actually gets sent -- retrieved context, the answer, reasoning, our
        own scaffolding, plus a client allowance when serving a gateway -- rather
        than fixed. Lowering GALACTICA_CONTEXT_BUDGET for a small card therefore
        lowers the KV cache with it.
        """
        if self.num_ctx:
            return self.num_ctx
        needed = (
            self.context_budget
            + self.max_answer_tokens
            + (self.think_reserve if self.think else 0)
            + self.prompt_overhead
            + client_reserve
        )
        rounded = -(-needed // 1024) * 1024  # next whole 1024
        return max(self.MIN_NUM_CTX, rounded)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "galactica.db"
