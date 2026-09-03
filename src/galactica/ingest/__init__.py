"""Ingest runner: stream RawDocs -> provenance-complete documents + indexed chunks."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..providers.base import LLMProvider
from ..store import (
    Chunk,
    Document,
    checksum_of,
    chunk_id_for,
    chunks_without_embeddings,
    doc_id_for,
    document_checksum,
    replace_chunks,
    store_embeddings,
    upsert_document,
    upsert_source,
)
from .chunker import TARGET_TOKENS, chunk_text
from .loaders import LoaderError, RawDoc, load
from .profiles import PROFILES, MappingError, Profile, parse_map_arg

__all__ = [
    "IngestReport",
    "ingest_path",
    "embed_missing",
    "LoaderError",
    "MappingError",
    "PROFILES",
    "RawDoc",
]

EMBED_BATCH = 32
COMMIT_EVERY = 200


@dataclass
class IngestReport:
    source: str
    profile: str
    documents_seen: int = 0
    documents_written: int = 0
    documents_unchanged: int = 0
    documents_empty: int = 0
    chunks_written: int = 0
    embeddings_written: int = 0
    warnings: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_source_name(path: Path, profile: Profile) -> str:
    if profile.family in ("wikipedia",) or profile.collection in ("grokipedia",):
        return profile.name
    return (path.name if path.is_dir() else path.stem) or profile.name


def ingest_path(
    conn: sqlite3.Connection,
    path: Path,
    *,
    profile_name: str = "markdown",
    source: str | None = None,
    collection: str | None = None,
    license: str | None = None,
    source_version: str | None = None,
    map_arg: str | None = None,
    max_documents: int | None = None,
    sample: int | None = None,
    resume: bool = False,
    skip_redirects: bool = True,
    target_tokens: int = TARGET_TOKENS,
    provider: LLMProvider | None = None,
    embed: bool = False,
    embed_model: str | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> IngestReport:
    if profile_name not in PROFILES:
        raise LoaderError(
            f"unknown profile '{profile_name}' (known: {', '.join(sorted(PROFILES))})"
        )
    profile = PROFILES[profile_name]
    overrides = parse_map_arg(map_arg)
    source_name = source or _default_source_name(path, profile)
    ingested_at = _now()

    source_id = upsert_source(
        conn,
        source_name,
        kind=profile.family,
        source_version=source_version,
        license=license or profile.license,
        uri_base=profile.uri_base,
        loader_profile=profile.name,
        ingested_at=ingested_at,
    )
    report = IngestReport(source=source_name, profile=profile.name)

    for raw in load(
        path, profile, overrides=overrides, skip_redirects=skip_redirects, sample=sample
    ):
        report.documents_seen += 1
        _ingest_one(
            conn,
            raw,
            source_id=source_id,
            source_name=source_name,
            profile=profile,
            collection=collection,
            license=license,
            source_version=source_version,
            target_tokens=target_tokens,
            resume=resume,
            ingested_at=ingested_at,
            report=report,
        )
        if progress and report.documents_seen % 100 == 0:
            progress(report.documents_seen, raw.title)
        if report.documents_seen % COMMIT_EVERY == 0:
            conn.commit()
        if max_documents and report.documents_seen >= max_documents:
            break

    conn.commit()

    if embed:
        model = embed_model or getattr(provider, "embed_model", None)
        if provider is None or not model:
            report.warnings.append(
                "--embed requested but no embedding model configured "
                "(set GALACTICA_EMBED_MODEL and pull it in ollama); skipped"
            )
        else:
            report.embeddings_written = embed_missing(conn, provider, model)

    return report


def _ingest_one(
    conn: sqlite3.Connection,
    raw: RawDoc,
    *,
    source_id: str,
    source_name: str,
    profile: Profile,
    collection: str | None,
    license: str | None,
    source_version: str | None,
    target_tokens: int,
    resume: bool,
    ingested_at: str,
    report: IngestReport,
) -> None:
    body = raw.text or ""
    if not body.strip():
        report.documents_empty += 1
        return

    doc_id = doc_id_for(source_name, raw.native_id or raw.title)
    checksum = checksum_of(body)
    if document_checksum(conn, doc_id) == checksum:
        report.documents_unchanged += 1
        if resume:
            return  # untouched: skip re-chunking entirely

    doc = Document(
        doc_id=doc_id,
        source_id=source_id,
        title=raw.title,
        checksum=checksum,
        collection=collection or raw.collection or profile.collection,
        doc_type=raw.doc_type or profile.doc_type,
        uri=raw.uri,
        license=license or raw.license or profile.license,
        source_version=source_version or raw.source_version,
        lang=raw.lang,
        native_id=raw.native_id,
        meta=raw.meta,
    )
    upsert_document(conn, doc, ingested_at=ingested_at)

    drafts = chunk_text(body, target_tokens=target_tokens)
    chunks = [
        Chunk(
            chunk_id=chunk_id_for(doc_id, i),
            doc_id=doc_id,
            ord=i,
            text=d.text,
            heading_path=d.heading_path or raw.title,
            title=raw.title,
            char_start=d.char_start,
            char_end=d.char_end,
            approx_tokens=d.approx_tokens,
            checksum=checksum_of(d.text),
        )
        for i, d in enumerate(drafts)
    ]
    replace_chunks(conn, doc_id, chunks)
    report.documents_written += 1
    report.chunks_written += len(chunks)


def embed_missing(
    conn: sqlite3.Connection, provider: LLMProvider, model: str, batch: int = EMBED_BATCH
) -> int:
    """Embed every chunk lacking a vector for `model`. Used by --embed / --hybrid setup."""
    written = 0
    while True:
        pending = chunks_without_embeddings(conn, model, limit=batch)
        if not pending:
            break
        ids = [cid for cid, _ in pending]
        vectors = provider.embed([text for _, text in pending])
        written += store_embeddings(conn, model, zip(ids, vectors))
        conn.commit()
    return written
