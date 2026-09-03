"""Retrieval: BM25, optional vector search, Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from typing import Sequence

from .providers.base import LLMProvider, ProviderError
from .store import (
    Hit,
    get_hits_by_ids,
    iter_embeddings,
    search_bm25,
    search_proximity,
    stem_terms,
)

RRF_K = 60


@dataclass
class RetrievalResult:
    hits: list[Hit]  # fused, best first
    per_query: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def rrf_fuse(ranked_lists: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: rank-based, so no score normalization to tune."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# How much a chunk's term coverage may reorder the fused ranking. A chunk that
# contains four distinct query terms is worth more than one that repeats a
# single common term four times, which is how "religious contexts of ruminants"
# reached the context for a pasture stocking-rate question.
COVERAGE_WEIGHT = 1.0


def term_coverage(hit: Hit, stems: Sequence[str]) -> float:
    """Fraction of distinct query stems present in this chunk."""
    if not stems:
        return 1.0
    haystack = f"{hit.title} {hit.heading_path} {hit.text}".lower()
    return sum(1 for stem in stems if stem in haystack) / len(stems)


def rerank_by_coverage(
    conn: sqlite3.Connection, hits: Sequence[Hit], queries: Sequence[str]
) -> list[Hit]:
    """Reorder fused hits by how much of the query each one actually covers.

    Free: no model call, no embeddings. The fused score still dominates; coverage
    breaks the many near-ties that a lexical index produces.
    """
    stems = stem_terms(conn, " ".join(queries))
    if not stems or not hits:
        return list(hits)
    scored = []
    for hit in hits:
        coverage = term_coverage(hit, stems)
        scored.append((hit.score * (1.0 + COVERAGE_WEIGHT * coverage), coverage, hit))
    scored.sort(key=lambda row: (-row[0], row[2].chunk_id))
    out = []
    for score, coverage, hit in scored:
        hit.score = score
        out.append(hit)
    return out


def interleave(ranked_lists: Sequence[Sequence[Hit]]) -> list[Hit]:
    """Round-robin across per-sub-question results so each one gets slots.

    A multi-part question ("what do I do, what must I not do, how dangerous is
    it") is several lookups sharing one budget. Fusing everything into a single
    ranking lets the best-served part crowd the others out entirely, which is how
    "the sources do not specify what not to do" happened while the answer for
    that part sat unretrieved.
    """
    out: list[Hit] = []
    seen: set[str] = set()
    for rank in range(max((len(lst) for lst in ranked_lists), default=0)):
        for lst in ranked_lists:
            if rank >= len(lst):
                continue
            hit = lst[rank]
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            out.append(hit)
    return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def vector_search(
    conn: sqlite3.Connection,
    provider: LLMProvider,
    embed_model: str,
    query: str,
    limit: int,
) -> list[tuple[str, float]]:
    """Brute-force cosine over stored vectors. Fine at proof-of-concept scale."""
    qvec = provider.embed([query])[0]
    scored = [
        (chunk_id, cosine(qvec, vec)) for chunk_id, vec in iter_embeddings(conn, embed_model)
    ]
    scored.sort(key=lambda kv: -kv[1])
    return scored[:limit]


def search(
    conn: sqlite3.Connection,
    queries: Sequence[str],
    *,
    top_k: int = 24,
    hybrid: bool = False,
    provider: LLMProvider | None = None,
    embed_model: str | None = None,
    rerank: bool = True,
) -> RetrievalResult:
    queries = [q for q in (q.strip() for q in queries) if q]
    if not queries:
        return RetrievalResult(hits=[])

    ranked_lists: list[list[str]] = []
    per_query: dict[str, list[str]] = {}
    warnings: list[str] = []
    # Per-query depth: fetch deeper than top_k so fusion has material to work with.
    depth = max(top_k, 10) * 2

    for query in queries:
        lexical = search_bm25(conn, query, limit=depth)
        ids = [h.chunk_id for h in lexical]
        ranked_lists.append(ids)
        per_query[query] = ids[:top_k]

        # Word-adjacency pass, fused as its own ranked list. Bag-of-words
        # scoring alone confuses "Katana Zero" with "DA20 Katana".
        nearby = search_proximity(conn, query, limit=depth)
        if nearby:
            ranked_lists.append([h.chunk_id for h in nearby])

        if hybrid:
            if provider is None or not embed_model:
                warnings.append("hybrid requested but no embedding model configured; BM25 only")
                hybrid = False
            else:
                try:
                    vector = vector_search(conn, provider, embed_model, query, depth)
                except ProviderError as exc:
                    warnings.append(f"vector search unavailable ({exc}); BM25 only")
                    hybrid = False
                else:
                    if not vector:
                        warnings.append(
                            f"no embeddings stored for model '{embed_model}'; "
                            "run: galactica ingest <path> --embed"
                        )
                        hybrid = False
                    else:
                        ranked_lists.append([cid for cid, _ in vector])

    fused = rrf_fuse(ranked_lists)[:top_k]
    lookup = get_hits_by_ids(conn, [cid for cid, _ in fused])
    hits: list[Hit] = []
    for chunk_id, score in fused:
        hit = lookup.get(chunk_id)
        if hit is None:
            continue
        hit.score = score
        hits.append(hit)
    if rerank:
        hits = rerank_by_coverage(conn, hits, queries)
    return RetrievalResult(hits=hits, per_query=per_query, warnings=sorted(set(warnings)))


def search_grouped(
    conn: sqlite3.Connection,
    groups: Sequence[Sequence[str]],
    *,
    top_k: int = 24,
    hybrid: bool = False,
    provider: LLMProvider | None = None,
    embed_model: str | None = None,
) -> RetrievalResult:
    """Retrieve each group of queries separately, then interleave the results.

    One group per sub-question, so every part of a multi-part question is
    represented in what reaches the context budget.
    """
    groups = [list(g) for g in groups if any(q.strip() for q in g)]
    if not groups:
        return RetrievalResult(hits=[])
    if len(groups) == 1:
        return search(
            conn, groups[0], top_k=top_k, hybrid=hybrid, provider=provider,
            embed_model=embed_model,
        )

    per_group: list[list[Hit]] = []
    per_query: dict[str, list[str]] = {}
    warnings: list[str] = []
    for group in groups:
        result = search(
            conn, group, top_k=top_k, hybrid=hybrid, provider=provider,
            embed_model=embed_model, rerank=True,
        )
        per_group.append(result.hits)
        per_query.update(result.per_query)
        warnings.extend(result.warnings)
    return RetrievalResult(
        hits=interleave(per_group)[: top_k * 2],
        per_query=per_query,
        warnings=sorted(set(warnings)),
    )
