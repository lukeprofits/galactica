from galactica import store
from galactica.ingest import ingest_path
from tests.conftest import SEED_DIR


def test_ids_are_stable_and_deterministic():
    first = store.doc_id_for("grokipedia", "krios-r7")
    assert first == store.doc_id_for("grokipedia", "krios-r7")
    assert first != store.doc_id_for("wikipedia", "krios-r7")
    assert store.chunk_id_for(first, 3) == f"{first}#0003"


def test_checksum_ignores_whitespace_reflow():
    assert store.checksum_of("a b\n c") == store.checksum_of("a  b   c")


def test_every_provenance_field_is_populated(conn):
    ingest_path(
        conn,
        SEED_DIR,
        profile_name="markdown",
        source="seed",
        source_version="2026-03-01",
    )
    row = conn.execute(
        """
        SELECT d.doc_id, s.name, d.collection, d.doc_type, d.title, d.uri, d.license,
               d.source_version, d.native_id, d.checksum
        FROM documents d JOIN sources s ON s.source_id = d.source_id
        WHERE d.title = 'Krios Relay R7 field replacement procedure'
        """
    ).fetchone()
    assert row is not None
    doc_id, source, collection, doc_type, title, uri, lic, version, native, checksum = row
    assert (source, collection, doc_type) == ("seed", "seed-procedures", "procedure")
    assert lic == "proprietary-sample"
    assert version == "2026-03-01"  # explicit flag beats front matter
    assert native.endswith("krios-relay-r7-replacement.md")
    assert uri.startswith("file://")
    assert len(checksum) == 64
    chunk = conn.execute(
        "SELECT chunk_id, heading_path, title, approx_tokens, checksum FROM chunks "
        "WHERE doc_id=? ORDER BY ord LIMIT 1",
        (doc_id,),
    ).fetchone()
    assert chunk[0].startswith(doc_id + "#")
    assert chunk[1] and chunk[2] and chunk[3] > 0 and len(chunk[4]) == 64


def test_reingest_is_idempotent_and_resume_skips(conn):
    first = ingest_path(conn, SEED_DIR, profile_name="markdown", source="seed")
    counts = store.stats(conn)
    second = ingest_path(conn, SEED_DIR, profile_name="markdown", source="seed", resume=True)
    assert second.documents_unchanged == first.documents_written
    assert second.documents_written == 0
    assert store.stats(conn)["chunks"] == counts["chunks"]


def test_fts_terms_drops_stopwords_and_dedupes():
    assert store.fts_terms("What is the torque wrench?") == ["torque", "wrench"]
    assert store.fts_terms("torque torque wrench") == ["torque", "wrench"]
    # An all-stopword question still yields terms rather than matching nothing.
    assert store.fts_terms("the of and") == ["the", "of", "and"]
    assert store.fts_terms("!!! ???") == []


def test_query_terms_are_stemmed_with_the_index_tokenizer(seeded):
    stems = store.stem_terms(seeded, "calibrating wrenches annually")
    assert "calibr" in stems and "wrench" in stems  # porter stems, as indexed


def test_term_frequencies_read_document_counts_off_the_index(seeded):
    stems = store.stem_terms(seeded, "torque wrench")
    freq = store.term_frequencies(seeded, stems)
    assert all(stem in freq for stem in stems)
    assert freq[[s for s in stems if s.startswith("wrench")][0]] > 0
    assert store.term_frequencies(seeded, ["zzzznotaterm"]) == {"zzzznotaterm": 0}


def test_rarest_terms_survive_query_relaxation(seeded):
    # "Krios" is rare and decisive; "procedure" and "order" are not. The rare
    # term must drive the result even though the query carries common words.
    hits = store.search_bm25(seeded, "what is the order of the krios procedure", limit=5)
    assert hits and "Krios" in hits[0].title


def test_deadline_aborts_a_long_running_query(conn):
    import sqlite3

    import pytest

    # A query with enough work to reach the progress handler must be aborted.
    with pytest.raises(sqlite3.OperationalError):
        with store._Deadline(conn, 0.0):
            conn.execute(
                "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000)"
                " SELECT COUNT(*) FROM n"
            ).fetchone()
    # The handler is cleared on exit, so the connection still works.
    assert conn.execute("SELECT 1").fetchone() == (1,)


def test_search_survives_a_tiny_deadline(seeded, monkeypatch):
    # Small corpora finish inside any deadline; the guard must not corrupt results.
    monkeypatch.setattr(store, "RUNG_TIMEOUT_SECONDS", 0.0)
    hits = store.search_bm25(seeded, "torque wrench calibration", limit=5)
    assert all(h.chunk_id for h in hits)


def test_bm25_search_returns_hits_with_provenance(seeded):
    hits = store.search_bm25(seeded, "krios relay breaker order", limit=5)
    assert hits
    assert any("Krios" in h.title for h in hits)
    top = hits[0]
    assert top.source_name == "seed" and top.license and top.score > 0


def test_deleting_chunks_keeps_fts_in_sync(seeded):
    doc_id = seeded.execute("SELECT doc_id FROM documents LIMIT 1").fetchone()[0]
    before = len(store.search_bm25(seeded, "torque wrench calibration", limit=50))
    seeded.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    seeded.commit()
    after = len(store.search_bm25(seeded, "torque wrench calibration", limit=50))
    assert after <= before
    ids = {h.chunk_id.split("#")[0] for h in store.search_bm25(seeded, "torque", limit=50)}
    assert doc_id not in ids


def test_ingest_populates_document_rollups(seeded):
    rows = seeded.execute(
        "SELECT chunk_count, approx_tokens FROM documents"
    ).fetchall()
    assert rows and all(c is not None and c > 0 for c, _ in rows)
    assert all(t is not None and t > 0 for _, t in rows)
    # Rollups must agree with the chunk table they summarise.
    real = seeded.execute(
        "SELECT COUNT(*), SUM(approx_tokens) FROM chunks"
    ).fetchone()
    summed = seeded.execute(
        "SELECT SUM(chunk_count), SUM(approx_tokens) FROM documents"
    ).fetchone()
    assert summed == real
    assert store.counts_are_complete(seeded)


def test_stats_uses_rollups_and_reports_completeness(seeded):
    data = store.stats(seeded)
    assert data["documents"] == 10 and data["chunks"] > 10
    assert data["counts_complete"] is True
    assert data["by_source"][0]["chunks"] == data["chunks"]


def test_stale_counts_are_detected_and_backfilled(seeded):
    seeded.execute("UPDATE documents SET chunk_count=NULL, approx_tokens=NULL")
    seeded.commit()
    assert store.counts_are_complete(seeded) is False
    assert store.stats(seeded)["chunks"] == 0  # rollups empty until refreshed

    updated = store.refresh_document_counts(seeded)
    assert updated == 10
    assert store.counts_are_complete(seeded) is True
    real = seeded.execute("SELECT COUNT(*), SUM(approx_tokens) FROM chunks").fetchone()
    assert store.stats(seeded)["chunks"] == real[0]


def test_migration_adds_rollup_columns_to_an_older_database(tmp_path):
    """A v1 database (no rollup columns) must upgrade in place."""
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE documents (doc_id TEXT PRIMARY KEY, source_id TEXT, title TEXT,
                                checksum TEXT, ingested_at TEXT);
        INSERT INTO documents VALUES ('d1', 's1', 'Old doc', 'x', 'then');
        """
    )
    conn.commit()
    conn.close()

    upgraded = store.open_db(path)
    columns = {row[1] for row in upgraded.execute("PRAGMA table_info(documents)")}
    assert {"chunk_count", "approx_tokens"} <= columns
    assert store.counts_are_complete(upgraded) is False
    upgraded.close()


def test_reads_do_not_need_a_write_lock_once_schema_is_current(tmp_path):
    """A reader must not be blocked by a concurrent long-running write."""
    import sqlite3

    import pytest

    path = tmp_path / "locked.db"
    setup = store.open_db(path)
    assert store.schema_is_current(setup)

    writer = store.connect(path)
    writer.execute("BEGIN EXCLUSIVE")  # simulate an ingest holding the write lock
    try:
        reader = store.connect(path)
        reader.execute("PRAGMA busy_timeout=100")
        assert store.schema_is_current(reader) is True  # read-only check succeeds
        assert store.search_bm25(reader, "anything", limit=3) == []
        # Writing would block, proving the lock is genuinely held.
        with pytest.raises(sqlite3.OperationalError):
            reader.execute("INSERT OR REPLACE INTO meta VALUES ('x','y')")
        reader.close()
    finally:
        writer.execute("ROLLBACK")
        writer.close()
        setup.close()


def test_strict_match_wins_when_it_fills_the_limit(seeded):
    hits = store.search_bm25(seeded, "calibrate torque wrench", limit=2)
    assert len(hits) == 2
    assert all("Torque wrench" in h.title for h in hits)


def test_topical_terms_beat_merely_rare_ones(seeded):
    """Ranking must weigh every query term, not bet on the rarest.

    On the full corpus, betting on the rarest pair ranked "DRB-HICOM Defence"
    above "Adversarial machine learning" for a question about gradient masking.
    """
    hits = store.search_bm25(
        seeded, "how should a click-type torque wrench be stored and calibrated", limit=8
    )
    assert hits and "Torque wrench" in hits[0].title


def test_or_budget_trims_the_commonest_terms(seeded, monkeypatch):
    monkeypatch.setattr(store, "OR_COST_BUDGET", 1)
    # With no budget the OR keeps only the rarest term, and must still answer.
    hits = store.search_bm25(seeded, "krios relay carrier screws", limit=5)
    assert hits and "Krios" in hits[0].title
