"""Heading-aware chunker. Pure text in, chunk drafts out."""

from __future__ import annotations

import re
from dataclasses import dataclass

# No tokenizer dependency: 4 chars/token is close enough for budget arithmetic.
CHARS_PER_TOKEN = 4
TARGET_TOKENS = 700
OVERLAP_TOKENS = 80
MAX_HEADING_DEPTH = 6

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class ChunkDraft:
    text: str
    heading_path: str
    char_start: int
    char_end: int

    @property
    def approx_tokens(self) -> int:
        return approx_tokens(self.text)


@dataclass
class _Piece:
    text: str
    start: int
    end: int


def _sections(text: str) -> list[tuple[str, list[_Piece]]]:
    """Split into (heading_path, paragraph pieces) preserving original offsets."""
    stack: list[str] = []
    out: list[tuple[str, list[_Piece]]] = []
    buf: list[_Piece] = []
    path = ""
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        m = _HEADING.match(stripped)
        if m:
            if buf:
                out.append((path, buf))
                buf = []
            depth = min(len(m.group(1)), MAX_HEADING_DEPTH)
            del stack[depth - 1 :]
            stack.append(m.group(2).strip())
            path = " > ".join(stack)
        elif stripped:
            buf.append(_Piece(stripped, offset, offset + len(line)))
        elif buf and buf[-1].text != "":
            buf.append(_Piece("", offset, offset))  # paragraph break marker
        offset += len(line)
    if buf:
        out.append((path, buf))
    return out


def _paragraphs(pieces: list[_Piece]) -> list[_Piece]:
    paras: list[_Piece] = []
    cur: list[_Piece] = []
    for piece in pieces:
        if piece.text == "":
            if cur:
                paras.append(_join(cur))
                cur = []
        else:
            cur.append(piece)
    if cur:
        paras.append(_join(cur))
    return paras


def _join(pieces: list[_Piece]) -> _Piece:
    return _Piece(
        "\n".join(p.text for p in pieces), pieces[0].start, pieces[-1].end
    )


def _split_oversized(para: _Piece, limit_chars: int) -> list[_Piece]:
    if len(para.text) <= limit_chars:
        return [para]
    out: list[_Piece] = []
    cursor = para.start
    buf: list[str] = []
    for sentence in _SENTENCE.split(para.text):
        candidate = (" ".join(buf + [sentence])).strip()
        if buf and len(candidate) > limit_chars:
            joined = " ".join(buf).strip()
            out.append(_Piece(joined, cursor, cursor + len(joined)))
            cursor += len(joined) + 1
            buf = [sentence]
        else:
            buf.append(sentence)
    if buf:
        joined = " ".join(buf).strip()
        out.append(_Piece(joined, cursor, min(para.end, cursor + len(joined))))
    return out


def chunk_text(
    text: str,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[ChunkDraft]:
    """Chunk within section boundaries; a chunk never spans two headings."""
    target_chars = target_tokens * CHARS_PER_TOKEN
    # Overlap must stay a small fraction of the target, or carried tails inflate
    # every chunk past the budget arithmetic that select.py depends on.
    overlap_chars = min(overlap_tokens * CHARS_PER_TOKEN, target_chars // 3)
    drafts: list[ChunkDraft] = []

    for path, pieces in _sections(text):
        paras: list[_Piece] = []
        for para in _paragraphs(pieces):
            paras.extend(_split_oversized(para, int(target_chars * 1.5)))
        buf: list[_Piece] = []
        size = 0
        for para in paras:
            if buf and size + len(para.text) > target_chars:
                drafts.append(_emit(buf, path))
                tail = buf[-1]
                buf = [tail] if overlap_chars and len(tail.text) <= overlap_chars else []
                size = sum(len(p.text) for p in buf)
            buf.append(para)
            size += len(para.text)
        if buf:
            drafts.append(_emit(buf, path))

    return [d for d in drafts if d.text.strip()]


def _emit(buf: list[_Piece], path: str) -> ChunkDraft:
    body = "\n\n".join(p.text for p in buf)
    return ChunkDraft(
        text=body,
        heading_path=path,
        char_start=buf[0].start,
        char_end=buf[-1].end,
    )
