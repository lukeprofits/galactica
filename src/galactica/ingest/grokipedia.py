"""Grokipedia Hugging Face dump: nested article records -> markdown.

Record shape (htriedman/grokipedia-v0.1-dump, grokipedia_scrape.ndjson):

    {"title": "<slug>", "url": "...", "scraped_at": "ISO8601",
     "data": {"main_title": "...", "sections": [...], "paragraphs": [...],
              "tables": [...], "references": [...], "metadata": {...}}}

Sections carry `level` ("h1".."h3"), `title` and `content` blocks of type
`paragraph` (text), `list` / `ordered_list` (items). `paragraphs` is a flat
union of the section text, used only as a fallback. `tables` are not anchored
to a section, so they are appended under a trailing heading.
"""

from __future__ import annotations

import re
from typing import Any

# Inline reference markers ("...from 1983 to 1993.[1]") point at a reference
# list this loader does not ingest, so they are dropped.
_REF_MARKER = re.compile(r"\[\d{1,3}\]")
# The scrape drops the space where inline markup ended: "TheYamaha Venture".
# Only casing-visible seams are repairable; lowercase-lowercase glue
# ("Royaleis a") is not detectable and is left alone.
_GLUED_CASE = re.compile(r"([a-z])([A-Z])")
_GLUED_SENTENCE = re.compile(r"([a-z0-9])([.,;:])([A-Z])")
# "(73.1 cu in)V4 engine" - a closing bracket against a letter is a lost space.
# Digits are excluded so chemical formulae like Ca(OH)2 survive.
_GLUED_BRACKET = re.compile(r"([)\]])([A-Za-z])")
_WS = re.compile(r"[ \t]+")
_LEVEL = re.compile(r"h([1-6])", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = _REF_MARKER.sub("", text)
    text = _GLUED_SENTENCE.sub(r"\1\2 \3", text)
    text = _GLUED_BRACKET.sub(r"\1 \2", text)
    text = _GLUED_CASE.sub(r"\1 \2", text)
    return _WS.sub(" ", text).strip()


def _heading(level: Any, title: str) -> str:
    match = _LEVEL.fullmatch(str(level or "h2").strip())
    depth = int(match.group(1)) if match else 2
    return "#" * depth + " " + clean_text(title)


def _render_block(block: Any) -> str:
    if not isinstance(block, dict):
        return clean_text(str(block)) if block else ""
    kind = block.get("type")
    if kind == "paragraph":
        return clean_text(str(block.get("text") or ""))
    if kind in ("list", "ordered_list"):
        items = [clean_text(str(i)) for i in (block.get("items") or [])]
        items = [i for i in items if i]
        if not items:
            return ""
        if kind == "ordered_list":
            return "\n".join(f"{n}. {item}" for n, item in enumerate(items, start=1))
        return "\n".join(f"- {item}" for item in items)
    # Unknown block type: keep any text it carries rather than dropping content.
    return clean_text(str(block.get("text") or ""))


def _render_table(table: Any) -> str:
    if not isinstance(table, dict):
        return ""
    headers = [clean_text(str(h)) for h in (table.get("headers") or [])]
    rows = table.get("rows") or []
    if not headers or not rows:
        return ""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        cells = [clean_text(str(c)) for c in row][: len(headers)]
        cells += [""] * (len(headers) - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) if len(lines) > 2 else ""


def to_markdown(data: dict, *, title: str, include_tables: bool = True) -> str:
    """Rebuild an article as markdown so heading paths survive into chunks."""
    parts: list[str] = []
    for section in data.get("sections") or []:
        if not isinstance(section, dict):
            continue
        heading = _heading(section.get("level"), str(section.get("title") or ""))
        # Empty sections keep their heading: it preserves the hierarchy for
        # nested subsections, and the chunker emits nothing for a bare heading.
        blocks = [_render_block(b) for b in (section.get("content") or [])]
        blocks = [b for b in blocks if b]
        parts.append(heading if not blocks else heading + "\n\n" + "\n\n".join(blocks))

    body = "\n\n".join(parts).strip()
    if not body:
        # No usable sections: fall back to the flat paragraph union.
        paragraphs = [clean_text(str(p)) for p in (data.get("paragraphs") or [])]
        paragraphs = [p for p in paragraphs if p]
        if paragraphs:
            body = f"# {title}\n\n" + "\n\n".join(paragraphs)

    if include_tables:
        tables = [_render_table(t) for t in (data.get("tables") or [])]
        tables = [t for t in tables if t]
        if tables:
            body = (body + "\n\n## Tables\n\n" + "\n\n".join(tables)).strip()
    return body


def flatten(record: dict, *, include_tables: bool = True) -> dict | None:
    """Nested dump record -> flat fields. Returns None if unusable."""
    if not isinstance(record, dict):
        return None
    data = record.get("data")
    if not isinstance(data, dict):
        return None
    slug = str(record.get("title") or data.get("title") or "").strip()
    title = clean_text(str(data.get("main_title") or "")) or slug.replace("_", " ")
    if not title:
        return None
    uri = str(record.get("url") or data.get("url") or "").strip() or None
    scraped = str(record.get("scraped_at") or "").strip()
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    meta = {
        "reference_count": len(data.get("references") or []),
        "table_count": len(data.get("tables") or []),
        "section_count": len(data.get("sections") or []),
    }
    if metadata.get("has_edits") is not None:
        meta["has_edits"] = bool(metadata.get("has_edits"))
    return {
        "native_id": slug or title,
        "title": title,
        "text": to_markdown(data, title=title, include_tables=include_tables),
        "uri": uri,
        # Per-document provenance: the scrape date this article was captured on.
        "source_version": scraped[:10] or None,
        "meta": meta,
    }
