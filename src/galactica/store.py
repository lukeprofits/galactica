"""SQLite storage: provenance-complete documents/chunks, FTS5 BM25 index, embeddings.

This module owns all SQL. Nothing else in the package writes queries.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

SCHEMA_VERSION = 2

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id      TEXT PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    kind           TEXT NOT NULL,
    source_version TEXT,
    license        TEXT,
    uri_base       TEXT,
    loader_profile TEXT,
    ingested_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id         TEXT PRIMARY KEY,
    source_id      TEXT NOT NULL REFERENCES sources(source_id),
    collection     TEXT,
    doc_type       TEXT,
    title          TEXT NOT NULL,
    uri            TEXT,
    license        TEXT,
    source_version TEXT,
    lang           TEXT,
    native_id      TEXT,
    checksum       TEXT NOT NULL,
    meta_json      TEXT,
    ingested_at    TEXT NOT NULL,
    -- Denormalised chunk rollups. Aggregating 19M chunk rows to answer `stats`
    -- takes minutes; these are written when the document's chunks are written.
    chunk_count    INTEGER,
    approx_tokens  INTEGER
);
CREATE INDEX IF NOT EXISTS documents_source ON documents(source_id);

CREATE TABLE IF NOT EXISTS chunks (
    rowid         INTEGER PRIMARY KEY,
    chunk_id      TEXT NOT NULL UNIQUE,
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ord           INTEGER NOT NULL,
    heading_path  TEXT NOT NULL DEFAULT '',
    title         TEXT NOT NULL DEFAULT '',
    char_start    INTEGER NOT NULL DEFAULT 0,
    char_end      INTEGER NOT NULL DEFAULT 0,
    text          TEXT NOT NULL,
    approx_tokens INTEGER NOT NULL DEFAULT 0,
    checksum      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc_ord ON chunks(doc_id, ord);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, heading_path, title,
    content='chunks', content_rowid='rowid', tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, heading_path, title)
    VALUES (new.rowid, new.text, new.heading_path, new.title);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path, title)
    VALUES ('delete', old.rowid, old.text, old.heading_path, old.title);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path, title)
    VALUES ('delete', old.rowid, old.text, old.heading_path, old.title);
    INSERT INTO chunks_fts(rowid, text, heading_path, title)
    VALUES (new.rowid, new.text, new.heading_path, new.title);
END;

-- Term document-frequencies, read straight off the FTS index. Used to order
-- query terms rarest-first so a query never OR-scans millions of rows.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vocab USING fts5vocab(chunks_fts, 'row');

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL
);
"""


# --------------------------------------------------------------------------- ids


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Whitespace-normalized form used for checksums (stable across reformatting)."""
    return re.sub(r"\s+", " ", text).strip()


def checksum_of(text: str) -> str:
    return sha256_hex(normalize_text(text))


def doc_id_for(source_name: str, native_id: str) -> str:
    """Stable, deterministic document id: same dump record -> same id, forever."""
    return sha256_hex(f"{source_name}\x00{native_id}")[:16]


def chunk_id_for(doc_id: str, ord_: int) -> str:
    return f"{doc_id}#{ord_:04d}"


def source_id_for(name: str) -> str:
    return sha256_hex(f"source\x00{name}")[:16]


# ------------------------------------------------------------------------ models


@dataclass
class Document:
    doc_id: str
    source_id: str
    title: str
    checksum: str
    collection: str | None = None
    doc_type: str | None = None
    uri: str | None = None
    license: str | None = None
    source_version: str | None = None
    lang: str | None = None
    native_id: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ord: int
    text: str
    heading_path: str = ""
    title: str = ""
    char_start: int = 0
    char_end: int = 0
    approx_tokens: int = 0
    checksum: str = ""


@dataclass
class Hit:
    """A retrieved chunk with its full provenance, ready to cite."""

    chunk_id: str
    doc_id: str
    ord: int
    heading_path: str
    title: str
    text: str
    approx_tokens: int
    score: float = 0.0
    source_name: str = ""
    source_version: str | None = None
    license: str | None = None
    uri: str | None = None
    collection: str | None = None
    doc_type: str | None = None


_HIT_COLUMNS = """
    c.chunk_id, c.doc_id, c.ord, c.heading_path, c.title, c.text, c.approx_tokens,
    s.name, d.source_version, d.license, d.uri, d.collection, d.doc_type
"""


def _row_to_hit(row: sqlite3.Row | tuple, score: float = 0.0) -> Hit:
    return Hit(
        chunk_id=row[0],
        doc_id=row[1],
        ord=row[2],
        heading_path=row[3],
        title=row[4],
        text=row[5],
        approx_tokens=row[6],
        score=score,
        source_name=row[7],
        source_version=row[8],
        license=row[9],
        uri=row[10],
        collection=row[11],
        doc_type=row[12],
    )


# --------------------------------------------------------------------------- open


def connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys=ON")
    # A long write (a big ingest, a counts refresh) must not make readers fail
    # outright; wait for the lock instead.
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
    for column in ("chunk_count", "approx_tokens"):
        if column not in existing:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {column} INTEGER")


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def schema_is_current(conn: sqlite3.Connection) -> bool:
    """True when the database is already at this schema version."""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.DatabaseError:
        return False
    if not row or row[0] != str(SCHEMA_VERSION):
        return False
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }
    return {"sources", "documents", "chunks", "chunks_fts", "chunks_vocab"} <= tables


def open_db(db_path: Path | str) -> sqlite3.Connection:
    """Open the corpus. Only takes a write lock when the schema needs work, so
    read-only commands keep working during a long ingest or counts refresh."""
    conn = connect(db_path)
    if not schema_is_current(conn):
        init_db(conn)
    return conn


# ------------------------------------------------------------------------- writes


def upsert_source(
    conn: sqlite3.Connection,
    name: str,
    *,
    kind: str,
    source_version: str | None = None,
    license: str | None = None,
    uri_base: str | None = None,
    loader_profile: str | None = None,
    ingested_at: str,
) -> str:
    sid = source_id_for(name)
    conn.execute(
        """
        INSERT INTO sources(source_id, name, kind, source_version, license, uri_base,
                            loader_profile, ingested_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(name) DO UPDATE SET
            kind=excluded.kind,
            source_version=COALESCE(excluded.source_version, sources.source_version),
            license=COALESCE(excluded.license, sources.license),
            uri_base=COALESCE(excluded.uri_base, sources.uri_base),
            loader_profile=excluded.loader_profile,
            ingested_at=excluded.ingested_at
        """,
        (sid, name, kind, source_version, license, uri_base, loader_profile, ingested_at),
    )
    return sid


def document_checksum(conn: sqlite3.Connection, doc_id: str) -> str | None:
    row = conn.execute("SELECT checksum FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    return row[0] if row else None


def upsert_document(conn: sqlite3.Connection, doc: Document, *, ingested_at: str) -> None:
    conn.execute(
        """
        INSERT INTO documents(doc_id, source_id, collection, doc_type, title, uri, license,
                              source_version, lang, native_id, checksum, meta_json, ingested_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(doc_id) DO UPDATE SET
            source_id=excluded.source_id, collection=excluded.collection,
            doc_type=excluded.doc_type, title=excluded.title, uri=excluded.uri,
            license=excluded.license, source_version=excluded.source_version,
            lang=excluded.lang, native_id=excluded.native_id, checksum=excluded.checksum,
            meta_json=excluded.meta_json, ingested_at=excluded.ingested_at
        """,
        (
            doc.doc_id,
            doc.source_id,
            doc.collection,
            doc.doc_type,
            doc.title,
            doc.uri,
            doc.license,
            doc.source_version,
            doc.lang,
            doc.native_id,
            doc.checksum,
            json.dumps(doc.meta, sort_keys=True) if doc.meta else None,
            ingested_at,
        ),
    )


def replace_chunks(conn: sqlite3.Connection, doc_id: str, chunks: Sequence[Chunk]) -> None:
    """Write a document's chunks and keep its rollup columns in step."""
    conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    conn.executemany(
        """
        INSERT INTO chunks(chunk_id, doc_id, ord, heading_path, title, char_start, char_end,
                           text, approx_tokens, checksum)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                c.chunk_id,
                c.doc_id,
                c.ord,
                c.heading_path,
                c.title,
                c.char_start,
                c.char_end,
                c.text,
                c.approx_tokens,
                c.checksum or checksum_of(c.text),
            )
            for c in chunks
        ],
    )
    conn.execute(
        "UPDATE documents SET chunk_count=?, approx_tokens=? WHERE doc_id=?",
        (len(chunks), sum(c.approx_tokens for c in chunks), doc_id),
    )


def store_embeddings(
    conn: sqlite3.Connection, model: str, pairs: Iterable[tuple[str, Sequence[float]]]
) -> int:
    rows = []
    for chunk_id, vec in pairs:
        buf = array("f", vec).tobytes()
        rows.append((chunk_id, model, len(vec), buf))
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings(chunk_id, model, dim, vec) VALUES (?,?,?,?)", rows
    )
    return len(rows)


def iter_embeddings(conn: sqlite3.Connection, model: str) -> Iterator[tuple[str, list[float]]]:
    cur = conn.execute("SELECT chunk_id, vec FROM embeddings WHERE model=?", (model,))
    for chunk_id, blob in cur:
        vec = array("f")
        vec.frombytes(blob)
        yield chunk_id, list(vec)


def chunks_without_embeddings(
    conn: sqlite3.Connection, model: str, limit: int | None = None
) -> list[tuple[str, str]]:
    sql = """
        SELECT c.chunk_id, c.title || ' ' || c.heading_path || ' ' || c.text
        FROM chunks c
        LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id AND e.model = ?
        WHERE e.chunk_id IS NULL
    """
    params: list = [model]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return list(conn.execute(sql, params))


# -------------------------------------------------------------------------- reads

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
}


def fts_terms(text: str) -> list[str]:
    """Query words, lowercased and de-duplicated, stopwords dropped."""
    words = [t for t in re.findall(r"[0-9A-Za-z_']+", text.lower()) if len(t) > 1]
    kept = [w for w in words if w not in _STOPWORDS] or words
    return list(dict.fromkeys(kept))


def _ensure_query_tokenizer(conn: sqlite3.Connection) -> None:
    """A temp FTS5 table with the index's tokenizer, so query words can be
    stemmed exactly the way the index stemmed them. No stemmer dependency."""
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS temp.galactica_q "
        "USING fts5(t, tokenize='porter unicode61')"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS temp.galactica_qv "
        "USING fts5vocab(galactica_q, 'row')"
    )


def stem_terms(conn: sqlite3.Connection, text: str) -> list[str]:
    words = fts_terms(text)
    if not words:
        return []
    _ensure_query_tokenizer(conn)
    conn.execute("DELETE FROM temp.galactica_q")
    conn.execute("INSERT INTO temp.galactica_q(t) VALUES (?)", (" ".join(words),))
    stems = [r[0] for r in conn.execute("SELECT term FROM temp.galactica_qv")]
    return stems or words


def term_frequencies(conn: sqlite3.Connection, stems: Sequence[str]) -> dict[str, int]:
    if not stems:
        return {}
    marks = ",".join("?" * len(stems))
    rows = conn.execute(
        f"SELECT term, doc FROM chunks_vocab WHERE term IN ({marks})", list(stems)
    ).fetchall()
    counts = {term: int(doc) for term, doc in rows}
    return {stem: counts.get(stem, 0) for stem in stems}


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


class _Deadline:
    """Aborts a query that exceeds its time budget via SQLite's progress handler.

    Some queries have no cheap plan at all ("what is the history of the
    service" matches millions of chunks). Rather than hang, the rung is
    abandoned and the ladder moves on.
    """

    def __init__(self, conn: sqlite3.Connection, seconds: float) -> None:
        self.conn = conn
        self.seconds = seconds

    def __enter__(self) -> "_Deadline":
        end = time.monotonic() + self.seconds
        self.conn.set_progress_handler(lambda: 1 if time.monotonic() > end else 0, 20_000)
        return self

    def __exit__(self, *exc) -> None:
        self.conn.set_progress_handler(None, 0)


def _run_match(conn: sqlite3.Connection, match: str, limit: int) -> list[Hit]:
    # Rank inside a subquery over the FTS index alone, then join provenance for
    # the surviving rows only. Joining every match before sorting costs 5-50x
    # more on a large corpus, because a common term matches tens of thousands
    # of chunks and all of them get joined just to be discarded.
    sql = f"""
        SELECT {_HIT_COLUMNS}, ranked.rank AS rank
        FROM (
            SELECT rowid AS rid, bm25(chunks_fts, 1.0, 2.0, 3.0) AS rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        ) AS ranked
        JOIN chunks c ON c.rowid = ranked.rid
        JOIN documents d ON d.doc_id = c.doc_id
        JOIN sources s ON s.source_id = d.source_id
        ORDER BY ranked.rank
    """
    try:
        with _Deadline(conn, RUNG_TIMEOUT_SECONDS):
            rows = conn.execute(sql, (match, limit)).fetchall()
    except sqlite3.OperationalError:
        return []  # rung too expensive; the caller tries a cheaper one
    # bm25() is negative-better; expose a positive relevance score.
    return [_row_to_hit(r, score=-float(r[13])) for r in rows]


# An OR query costs time in proportion to the rows it matches, so its terms are
# capped by total document frequency. AND queries are cheap at any frequency
# because FTS5 intersects doclists, which is why the ladder starts there.
# Ranking before joining made OR queries 5-50x cheaper, so the budget here is
# a latency choice rather than a hard limit: roughly one second per million
# matched chunks.
OR_COST_BUDGET = 3_000_000
RUNG_TIMEOUT_SECONDS = 6.0


PROXIMITY_SLOP = 3


def search_proximity(conn: sqlite3.Connection, query: str, limit: int = 24) -> list[Hit]:
    """Rank chunks where consecutive query words appear close together.

    A precision pass to sit alongside `search_bm25`. Bag-of-words ranking cannot
    tell "Katana Zero" (a game) from "DA20 Katana" (an aircraft), and the OR
    budget may drop "zero" as a common word -- losing the very term that
    identifies the subject. FTS5 `NEAR` keeps word adjacency, and the slop lets
    intervening stopwords through, which a strict phrase match would not
    ("Free State of Fiume" indexes "of", the query drops it).
    """
    words = fts_terms(query)
    if len(words) < 2:
        return []
    pairs = [
        f'NEAR("{a}" "{b}", {PROXIMITY_SLOP})'
        for a, b in zip(words, words[1:])
        if a != b
    ]
    if not pairs:
        return []
    return _run_match(conn, " OR ".join(pairs), limit)


def search_bm25(conn: sqlite3.Connection, query: str, limit: int = 24) -> list[Hit]:
    """Rank chunks for one query.

    Strategy: require every term (precise and fast), then relax by dropping the
    most common term until enough hits appear, and only then fall back to an OR
    of the rarest terms within a cost budget. On a 19M-chunk corpus a plain OR
    of all terms takes ~100 seconds; this returns in well under a second.
    """
    stems = stem_terms(conn, query)
    if not stems:
        return []
    freq = term_frequencies(conn, stems)
    stems.sort(key=lambda t: (freq.get(t, 0), t))  # rarest first

    # Rung 1: every term present. Cheap (~10ms) and the most precise thing
    # available, so it wins outright when it can fill the requested limit.
    strict = _run_match(conn, " AND ".join(_quote(t) for t in stems), limit)
    if len(strict) >= limit:
        return strict

    # Rung 2: OR the terms, scoring all of them, capped by total document
    # frequency to bound latency. This is the best general ranking: it weighs
    # every query term instead of betting on whichever happens to be rarest.
    # Rarity is not importance -- "gradient defence" is rarer than "adversarial
    # machine learning" and far less topical.
    or_terms: list[str] = []
    budget = 0
    for stem in stems:
        cost = freq.get(stem, 0)
        if or_terms and budget + cost > OR_COST_BUDGET:
            break
        or_terms.append(stem)
        budget += cost
    best = _run_match(conn, " OR ".join(_quote(t) for t in or_terms), limit)
    if best:
        return best

    # Rung 3: the OR was abandoned or empty. Relax the strict match instead,
    # dropping the most common term each time.
    best = strict
    for count in range(len(stems) - 1, 1, -1):
        hits = _run_match(conn, " AND ".join(_quote(t) for t in stems[:count]), limit)
        if len(hits) >= limit:
            return hits
        if len(hits) > len(best):
            best = hits
    if best:
        return best

    # Last resort: match titles only. The title column is a tiny fraction of
    # the index, so this stays fast even for all-common-word queries.
    titles = " AND ".join(f"title:{_quote(t)}" for t in stems[:4])
    return _run_match(conn, titles, limit)


def get_hits_by_ids(conn: sqlite3.Connection, chunk_ids: Sequence[str]) -> dict[str, Hit]:
    if not chunk_ids:
        return {}
    marks = ",".join("?" * len(chunk_ids))
    sql = f"""
        SELECT {_HIT_COLUMNS}
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        JOIN sources s ON s.source_id = d.source_id
        WHERE c.chunk_id IN ({marks})
    """
    return {r[0]: _row_to_hit(r) for r in conn.execute(sql, list(chunk_ids))}


def get_neighbor(conn: sqlite3.Connection, doc_id: str, ord_: int) -> Hit | None:
    sql = f"""
        SELECT {_HIT_COLUMNS}
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        JOIN sources s ON s.source_id = d.source_id
        WHERE c.doc_id = ? AND c.ord = ?
    """
    row = conn.execute(sql, (doc_id, ord_)).fetchone()
    return _row_to_hit(row) if row else None


def counts_are_complete(conn: sqlite3.Connection) -> bool:
    """False when documents predate the rollup columns and need a backfill."""
    row = conn.execute(
        "SELECT 1 FROM documents WHERE chunk_count IS NULL LIMIT 1"
    ).fetchone()
    return row is None


def refresh_document_counts(conn: sqlite3.Connection) -> int:
    """Backfill the rollup columns from the chunks table. One full pass."""
    # One grouped pass joined against documents. The obvious correlated-subquery
    # form costs two index lookups per document, which runs into many minutes on
    # a corpus of this size.
    cur = conn.execute(
        """
        UPDATE documents SET chunk_count = g.n, approx_tokens = g.t
        FROM (SELECT doc_id, COUNT(*) AS n, COALESCE(SUM(approx_tokens), 0) AS t
              FROM chunks GROUP BY doc_id) AS g
        WHERE documents.doc_id = g.doc_id
          AND (documents.chunk_count IS NULL OR documents.approx_tokens IS NULL)
        """
    )
    updated = cur.rowcount
    # Documents with no chunks at all are never reached by the join above.
    conn.execute(
        "UPDATE documents SET chunk_count=0, approx_tokens=0 "
        "WHERE chunk_count IS NULL OR approx_tokens IS NULL"
    )
    conn.commit()
    return updated


def list_sources(conn: sqlite3.Connection) -> list[dict]:
    sql = """
        SELECT s.name, s.kind,
               COALESCE(s.source_version, GROUP_CONCAT(DISTINCT d.source_version)),
               COALESCE(s.license, GROUP_CONCAT(DISTINCT d.license)),
               s.loader_profile, s.ingested_at,
               COUNT(d.doc_id), COALESCE(SUM(d.chunk_count), 0),
               COALESCE(SUM(d.approx_tokens), 0)
        FROM sources s
        LEFT JOIN documents d ON d.source_id = s.source_id
        GROUP BY s.source_id
        ORDER BY s.name
    """
    keys = (
        "name", "kind", "source_version", "license", "loader_profile", "ingested_at",
        "documents", "chunks", "approx_tokens",
    )
    return [dict(zip(keys, row)) for row in conn.execute(sql)]


def stats(conn: sqlite3.Connection) -> dict:
    """Corpus totals, read from the document rollups rather than every chunk."""
    sources, documents, chunks, tokens = conn.execute(
        """
        SELECT (SELECT COUNT(*) FROM sources),
               COUNT(*), COALESCE(SUM(chunk_count), 0), COALESCE(SUM(approx_tokens), 0)
        FROM documents
        """
    ).fetchone()
    return {
        "sources": int(sources or 0),
        "documents": int(documents or 0),
        "chunks": int(chunks or 0),
        "approx_tokens": int(tokens or 0),
        "embeddings": int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] or 0),
        "counts_complete": counts_are_complete(conn),
        "by_source": list_sources(conn),
    }
