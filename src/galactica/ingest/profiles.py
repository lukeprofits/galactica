"""Field-mapping profiles: alternate dump column names never require code changes."""

from __future__ import annotations

from dataclasses import dataclass, field

# Logical fields the rest of the system needs from any record.
LOGICAL_FIELDS = ("id", "title", "text", "uri", "lang")


@dataclass(frozen=True)
class Profile:
    name: str
    family: str  # markdown | jsonl | wikipedia
    doc_type: str = "article"
    collection: str | None = None
    license: str | None = None
    uri_base: str | None = None
    candidates: dict[str, tuple[str, ...]] = field(default_factory=dict)


PROFILES: dict[str, Profile] = {
    "markdown": Profile(
        name="markdown",
        family="markdown",
        doc_type="document",
        collection="local",
    ),
    "generic": Profile(
        name="generic",
        family="jsonl",
        candidates={
            "id": ("id", "doc_id", "_id", "slug", "native_id"),
            "title": ("title", "name", "heading"),
            "text": ("text", "content", "body", "markdown"),
            "uri": ("uri", "url", "link"),
            "lang": ("lang", "language"),
        },
    ),
    # Hugging Face Grokipedia dump (htriedman/grokipedia-v0.1-dump): nested
    # article records in .ndjson / .jsonl(.gz) / .parquet. Flattened by
    # ingest/grokipedia.py rather than by column mapping.
    "grokipedia": Profile(
        name="grokipedia",
        family="grokipedia",
        doc_type="article",
        collection="grokipedia",
        uri_base="https://grokipedia.com/page/",
        license="CC BY-SA 4.0 (Wikipedia-derived pages) / xAI Community License",
    ),
    "wikipedia": Profile(
        name="wikipedia",
        family="wikipedia",
        doc_type="article",
        collection="wikipedia",
        license="CC BY-SA 4.0",
        uri_base="https://en.wikipedia.org/wiki/",
    ),
}


class MappingError(RuntimeError):
    pass


def parse_map_arg(raw: str | None) -> dict[str, str]:
    """`--map title=page_title,text=content` -> {'title': 'page_title', ...}"""
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise MappingError(f"bad --map entry '{part}' (expected logical=actual)")
        logical, actual = (s.strip() for s in part.split("=", 1))
        if logical not in LOGICAL_FIELDS:
            raise MappingError(
                f"unknown --map field '{logical}' (known: {', '.join(LOGICAL_FIELDS)})"
            )
        out[logical] = actual
    return out


def resolve_mapping(
    profile: Profile, columns: list[str], overrides: dict[str, str] | None = None
) -> dict[str, str]:
    """Pick the actual column for each logical field. Explicit overrides always win."""
    overrides = overrides or {}
    lower = {c.lower(): c for c in columns}
    mapping: dict[str, str] = {}
    for logical in LOGICAL_FIELDS:
        if logical in overrides:
            actual = overrides[logical]
            if actual not in columns:
                raise MappingError(
                    f"--map {logical}={actual} but record has no such field. "
                    f"Available fields: {', '.join(sorted(columns))}"
                )
            mapping[logical] = actual
            continue
        for candidate in profile.candidates.get(logical, ()):
            if candidate.lower() in lower:
                mapping[logical] = lower[candidate.lower()]
                break
    for required in ("title", "text"):
        if required not in mapping:
            raise MappingError(
                f"profile '{profile.name}' could not find a '{required}' field. "
                f"Available fields: {', '.join(sorted(columns))}. "
                f"Fix with --map {required}=<field>."
            )
    return mapping
