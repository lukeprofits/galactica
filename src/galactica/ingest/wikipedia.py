"""Streaming Wikipedia XML dump reader. Stdlib only, flat memory on huge dumps."""

from __future__ import annotations

import bz2
import re
from pathlib import Path
from typing import IO, Iterator
from xml.etree import ElementTree as ET

_DATE = re.compile(r"(\d{8})")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_PAIR = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_REF_SELF = re.compile(r"<ref[^>]*/\s*>", re.IGNORECASE)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_FILE_LINK = re.compile(r"\[\[(?:File|Image|Category)\s*:[^\]]*\]\]", re.IGNORECASE)
_LINK_PIPED = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
_LINK_PLAIN = re.compile(r"\[\[([^\]|]*)\]\]")
_EXT_LINK = re.compile(r"\[https?://\S+\s+([^\]]*)\]")
_EXT_BARE = re.compile(r"\[https?://\S+\]")
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
_QUOTES = re.compile(r"'{2,5}")
_BLANKS = re.compile(r"\n{3,}")
_LIST = re.compile(r"^[*:;#]+\s*", re.MULTILINE)


def dump_version(path: Path) -> str | None:
    m = _DATE.search(path.name)
    return m.group(1) if m else None


def _strip_nested(text: str, open_tok: str, close_tok: str, limit: int = 12) -> str:
    """Remove nested {{templates}} / {|tables|} by repeated innermost removal."""
    pattern = re.compile(
        re.escape(open_tok) + r"(?:(?!" + re.escape(open_tok) + r"|" + re.escape(close_tok) + r").)*"
        + re.escape(close_tok),
        re.DOTALL,
    )
    for _ in range(limit):
        new = pattern.sub(" ", text)
        if new == text:
            break
        text = new
    return text


def clean_wikitext(text: str) -> str:
    """Lossy but adequate wikitext -> markdown-ish plain text."""
    text = _COMMENT.sub(" ", text)
    text = _REF_PAIR.sub(" ", text)
    text = _REF_SELF.sub(" ", text)
    text = _strip_nested(text, "{{", "}}")
    text = _strip_nested(text, "{|", "|}")
    text = _FILE_LINK.sub(" ", text)
    text = _LINK_PIPED.sub(r"\1", text)
    text = _LINK_PLAIN.sub(r"\1", text)
    text = _EXT_LINK.sub(r"\1", text)
    text = _EXT_BARE.sub(" ", text)
    text = _TAG.sub(" ", text)
    text = _QUOTES.sub("", text)
    # List markers first: wikitext numbered lists use '#', which would otherwise
    # collide with the markdown headings produced on the next line.
    text = _LIST.sub("- ", text)
    # == Section == -> ## Section (dump level 2 is the article's top section)
    text = _HEADING.sub(lambda m: "#" * len(m.group(1)) + " " + m.group(2), text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return _BLANKS.sub("\n\n", text).strip()


def _open(path: Path) -> IO[bytes]:
    if path.suffix == ".bz2":
        return bz2.open(path, "rb")
    return path.open("rb")


def iter_pages(
    path: Path, *, skip_redirects: bool = True, max_documents: int | None = None
) -> Iterator[dict]:
    """Yield {'native_id', 'title', 'text'} for namespace-0 pages, streaming."""
    emitted = 0
    with _open(path) as handle:
        context = ET.iterparse(handle, events=("end",))
        for _, elem in context:
            tag = elem.tag.rpartition("}")[2]
            if tag != "page":
                continue
            page = {c.tag.rpartition("}")[2]: c for c in elem}
            ns = page.get("ns")
            title_el = page.get("title")
            revision = page.get("revision")
            redirect = page.get("redirect")
            text = ""
            if revision is not None:
                for child in revision:
                    if child.tag.rpartition("}")[2] == "text":
                        text = child.text or ""
                        break
            keep = (
                (ns is None or (ns.text or "0") == "0")
                and title_el is not None
                and (redirect is None or not skip_redirects)
                and text.strip()
            )
            if keep:
                pid = page.get("id")
                yield {
                    "native_id": (pid.text if pid is not None else title_el.text) or "",
                    "title": title_el.text or "",
                    "text": clean_wikitext(text),
                }
                emitted += 1
            elem.clear()
            if max_documents and emitted >= max_documents:
                return
