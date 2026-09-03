"""Context assembly: budget-obeying packing with section-scoped neighbor expansion.

Every token that reaches the model passes through here, so nothing can bypass
the context budget.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Sequence

from .ingest.chunker import approx_tokens
from .store import Hit, get_neighbor


@dataclass
class SelectedSource:
    """One `[Sn]` block: a retrieved anchor plus any in-section neighbors."""

    label: str
    anchor: Hit
    neighbors: list[Hit] = field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.parts]

    @property
    def parts(self) -> list[Hit]:
        return sorted([self.anchor, *self.neighbors], key=lambda h: h.ord)

    @property
    def doc_id(self) -> str:
        return self.anchor.doc_id

    def provenance(self) -> str:
        h = self.anchor
        bits = [h.source_name or "local"]
        if h.source_version:
            bits.append(f"v{h.source_version}")
        if h.license:
            bits.append(h.license)
        if h.uri:
            bits.append(h.uri)
        return " · ".join(bits)

    def section_label(self) -> str:
        """Heading path without the redundant leading document title."""
        h = self.anchor
        path = h.heading_path or h.title
        if h.title and path.startswith(h.title):
            path = path[len(h.title) :].lstrip(" >")
        return path or h.title

    def render(self) -> str:
        h = self.anchor
        section = self.section_label()
        head = f"[{self.label}] {h.title} — {section}\n({self.provenance()})"
        body = "\n\n".join(p.text for p in self.parts)
        return f"{head}\n{body}"

    @property
    def tokens(self) -> int:
        return approx_tokens(self.render())


@dataclass
class Selection:
    sources: list[SelectedSource]
    used_tokens: int = 0
    budget: int = 0
    dropped: list[tuple[str, str]] = field(default_factory=list)

    def render(self) -> str:
        return "\n\n---\n\n".join(s.render() for s in self.sources)

    def by_label(self) -> dict[str, SelectedSource]:
        return {s.label: s for s in self.sources}

    @property
    def doc_ids(self) -> list[str]:
        return list(dict.fromkeys(s.doc_id for s in self.sources))

    @property
    def chunk_ids(self) -> list[str]:
        return [cid for s in self.sources for cid in s.chunk_ids]


def _same_section(anchor: Hit, candidate: Hit) -> bool:
    """Neighbor must live in the anchor's section (or a subsection of it)."""
    a, c = anchor.heading_path or "", candidate.heading_path or ""
    return c == a or c.startswith(a + " > ")


def _block_tokens(anchor: Hit, neighbors: Sequence[Hit]) -> int:
    return SelectedSource("S99", anchor, list(neighbors)).tokens


def select_context(
    conn: sqlite3.Connection,
    hits: Sequence[Hit],
    *,
    budget: int,
    expand: int = 1,
    expand_top: int = 3,
    per_doc_fraction: float = 0.35,
    max_sources: int | None = None,
    diversity_slots: int | None = None,
) -> Selection:
    """Pack retrieved hits into the context budget.

    Fused relevance order drives selection: the ranking already reflects every
    query the planner asked, so overriding it with a diversity round starves the
    document that actually answers the question (its chunks occupy several
    adjacent ranks). Two guards keep one document from monopolising the context:
    a per-document token cap, and a minority of source slots reserved for
    documents not yet represented. Neighbors then spend whatever budget is left.
    """
    per_doc_cap = max(1, int(budget * per_doc_fraction))
    if diversity_slots is None:
        diversity_slots = max(1, max_sources // 4) if max_sources else 0
    used = 0
    doc_used: dict[str, int] = {}
    chosen: dict[str, SelectedSource] = {}  # chunk_id -> block
    order: list[str] = []  # anchor chunk_ids in fused order
    dropped: list[tuple[str, str]] = []
    taken_chunks: set[str] = set()

    def try_add_anchor(hit: Hit) -> bool:
        nonlocal used
        cost = _block_tokens(hit, [])
        if used + cost > budget:
            dropped.append((hit.chunk_id, "context budget exhausted"))
            return False
        doc_total = doc_used.get(hit.doc_id, 0)
        if doc_total and doc_total + cost > per_doc_cap:
            dropped.append((hit.chunk_id, f"per-document cap ({per_doc_cap} tokens)"))
            return False
        block = SelectedSource(label="", anchor=hit)
        chosen[hit.chunk_id] = block
        order.append(hit.chunk_id)
        taken_chunks.add(hit.chunk_id)
        doc_used[hit.doc_id] = doc_total + cost
        used += cost
        return True

    # Pass 1 - strict fused relevance order, leaving room for the reserved slots.
    relevance_limit = (max_sources - diversity_slots) if max_sources else None
    for hit in hits:
        if relevance_limit is not None and len(chosen) >= relevance_limit:
            break
        try_add_anchor(hit)

    # Pass 2 - reserved slots: documents not yet represented, still in fused order.
    seen_docs = {block.anchor.doc_id for block in chosen.values()}
    for hit in hits:
        if max_sources and len(chosen) >= max_sources:
            break
        if hit.chunk_id in taken_chunks or hit.doc_id in seen_docs:
            continue
        seen_docs.add(hit.doc_id)
        try_add_anchor(hit)

    # Pass 3 - any capacity still free, fused order.
    for hit in hits:
        if max_sources and len(chosen) >= max_sources:
            break
        if hit.chunk_id in taken_chunks:
            continue
        try_add_anchor(hit)

    # Pass 4 - neighbor expansion, only on top-ranked anchors, leftover budget only.
    if expand > 0:
        for rank, chunk_id in enumerate(order):
            if rank >= expand_top:
                break
            block = chosen[chunk_id]
            anchor = block.anchor
            for offset in _offsets(expand):
                neighbor = get_neighbor(conn, anchor.doc_id, anchor.ord + offset)
                if neighbor is None or neighbor.chunk_id in taken_chunks:
                    continue
                if not _same_section(anchor, neighbor):
                    dropped.append((neighbor.chunk_id, "neighbor outside anchor section"))
                    continue
                before = block.tokens
                after = _block_tokens(anchor, [*block.neighbors, neighbor])
                delta = after - before
                if used + delta > budget:
                    dropped.append((neighbor.chunk_id, "no budget left for neighbor"))
                    continue
                if doc_used.get(anchor.doc_id, 0) + delta > per_doc_cap:
                    dropped.append((neighbor.chunk_id, f"per-document cap ({per_doc_cap} tokens)"))
                    continue
                block.neighbors.append(neighbor)
                taken_chunks.add(neighbor.chunk_id)
                doc_used[anchor.doc_id] = doc_used.get(anchor.doc_id, 0) + delta
                used += delta

    sources = [chosen[cid] for cid in order]
    for idx, block in enumerate(sources, start=1):
        block.label = f"S{idx}"
    return Selection(
        sources=sources,
        used_tokens=sum(s.tokens for s in sources),
        budget=budget,
        dropped=dropped,
    )


def _offsets(expand: int) -> list[int]:
    """Nearest-first: -1, +1, -2, +2 ... capped at `expand` per side."""
    out: list[int] = []
    for step in range(1, expand + 1):
        out.extend([-step, step])
    return out
