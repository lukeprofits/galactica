"""Model and corpus registries, loaded from data files.

Both are TOML in `galactica/data/`, so adding a newer local model or a newly
available public corpus is a data edit rather than a code change.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .hardware import Hardware


@dataclass(frozen=True)
class ModelOption:
    tag: str
    tier: int
    weights_gb: float
    min_memory_gb: float
    notes: str = ""
    apple_only: bool = False

    def fits(self, hardware: Hardware) -> bool:
        if self.apple_only and not hardware.apple_silicon:
            return False
        return hardware.usable_gb >= self.min_memory_gb


@dataclass(frozen=True)
class SourceOption:
    name: str
    profile: str
    title: str
    license: str
    documents: int = 0
    download_gb: float = 0.0
    db_gb: float = 0.0
    peak_gb: float = 0.0
    hf_repo: str | None = None
    hf_file: str | None = None
    url: str | None = None
    default: bool = False
    notes: str = ""

    def db_gb_for(self, documents: int | None) -> float:
        """Database size for a document count, scaled from the full-corpus figure."""
        if not documents or not self.documents:
            return self.db_gb
        return self.db_gb * min(1.0, documents / self.documents)


def _load(filename: str) -> dict:
    override = Path("galactica") / filename  # allows a local override while developing
    if override.exists():
        return tomllib.loads(override.read_text(encoding="utf-8"))
    with resources.files("galactica.data").joinpath(filename).open("rb") as handle:
        return tomllib.load(handle)


def models() -> list[ModelOption]:
    entries = _load("models.toml").get("model", [])
    out = [ModelOption(**entry) for entry in entries]
    return sorted(out, key=lambda m: m.tier)


def sources() -> list[SourceOption]:
    entries = _load("sources.toml").get("source", [])
    return [SourceOption(**entry) for entry in entries]


def source_named(name: str) -> SourceOption:
    for source in sources():
        if source.name == name:
            return source
    known = ", ".join(s.name for s in sources())
    raise KeyError(f"unknown source '{name}' (known: {known})")


def recommend_model(hardware: Hardware, installed: list[str] | None = None) -> ModelOption | None:
    """Best model this machine can run, preferring one already pulled.

    Preferring an installed model avoids a multi-gigabyte download when a
    perfectly good option is already on disk.
    """
    fitting = [m for m in models() if m.fits(hardware)]
    if not fitting:
        return None
    # A model that only fits with a tiny context is worse than a smaller one
    # that can hold real retrieved material, so prefer options that still
    # afford a usable budget.
    roomy = [m for m in fitting if context_budget_for(hardware, m) >= 8000]
    fitting = roomy or fitting
    if installed:
        have = {tag.split(":")[0] + ":" + tag.split(":")[1] if ":" in tag else tag for tag in installed}
        already = [m for m in fitting if m.tag in have]
        if already:
            return max(already, key=lambda m: m.tier)
    return max(fitting, key=lambda m: m.tier)


def context_budget_for(hardware: Hardware, model: ModelOption) -> int:
    """A context budget that leaves room for the KV cache on this machine.

    Headroom above the weights has to hold the cache, and the cache grows with
    the budget. A 6 GB card running a 2.5 GB model has ~3.5 GB to work with,
    which is roughly a 4000-token budget once reasoning and the answer are
    funded.
    """
    headroom = hardware.usable_gb - model.weights_gb
    if headroom >= 12:
        return 16000
    if headroom >= 7:
        return 12000
    if headroom >= 4:
        return 8000
    if headroom >= 2:
        return 4000
    return 2000
