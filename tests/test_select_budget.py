import pytest

from galactica.retrieve import search
from galactica.select import SelectedSource, select_context
from galactica.store import Chunk, Document, chunk_id_for, doc_id_for, replace_chunks, upsert_document, upsert_source


@pytest.fixture
def sectioned(conn):
    """One document, three chunks: two in section A, one in section B."""
    sid = upsert_source(conn, "fixture", kind="markdown", ingested_at="now")
    did = doc_id_for("fixture", "doc")
    upsert_document(
        conn,
        Document(doc_id=did, source_id=sid, title="Fixture", checksum="c"),
        ingested_at="now",
    )
    texts = [
        ("Fixture > Alpha", "The alpha marker chunk mentions widgets."),
        ("Fixture > Alpha", "Alpha continues with more widget detail."),
        ("Fixture > Beta", "Beta section is about something unrelated entirely."),
    ]
    replace_chunks(
        conn,
        did,
        [
            Chunk(
                chunk_id=chunk_id_for(did, i),
                doc_id=did,
                ord=i,
                text=text,
                heading_path=path,
                title="Fixture",
                approx_tokens=max(1, len(text) // 4),
            )
            for i, (path, text) in enumerate(texts)
        ],
    )
    conn.commit()
    return conn, did


def _hits(conn, query="torque wrench calibration bolt"):
    return search(conn, [query], top_k=20).hits


def test_nothing_exceeds_the_budget(seeded):
    hits = _hits(seeded)
    for budget in (200, 800, 4000):
        sel = select_context(seeded, hits, budget=budget)
        assert sel.used_tokens <= budget
        assert sum(s.tokens for s in sel.sources) <= budget


def test_labels_are_sequential_and_unique(seeded):
    sel = select_context(seeded, _hits(seeded), budget=4000)
    assert [s.label for s in sel.sources] == [f"S{i}" for i in range(1, len(sel.sources) + 1)]


def test_relevance_order_is_preserved_for_the_top_hits(seeded):
    hits = _hits(seeded, "krios relay breaker order carrier screw torque")
    top_docs = [h.doc_id for h in hits[:3]]
    assert len(set(top_docs)) == 1  # the best hits all live in one document
    sel = select_context(seeded, hits, budget=16000, max_sources=8, expand=0)
    kept = [s.chunk_ids[0] for s in sel.sources]
    # Diversity must not displace the chunks that actually answer the question.
    assert kept[:6] == [h.chunk_id for h in hits[:6]]


def test_reserved_slots_admit_unseen_documents(seeded):
    hits = _hits(seeded, "krios relay breaker order carrier screw torque")
    sel = select_context(
        seeded, hits, budget=16000, max_sources=8, diversity_slots=2, expand=0
    )
    docs = [s.doc_id for s in sel.sources]
    assert len(set(docs)) >= 3  # corroborating documents still get in
    assert len(sel.sources) == 8


def test_zero_reserved_slots_is_pure_relevance_order(seeded):
    hits = _hits(seeded, "krios relay breaker order carrier screw torque")
    sel = select_context(
        seeded, hits, budget=16000, max_sources=4, diversity_slots=0, expand=0
    )
    assert [s.chunk_ids[0] for s in sel.sources] == [h.chunk_id for h in hits[:4]]


def test_per_document_cap_holds_across_the_corpus(seeded):
    hits = _hits(seeded, "torque")
    sel = select_context(seeded, hits, budget=3000, per_doc_fraction=0.2, expand=0)
    cap = int(3000 * 0.2)
    per_doc: dict[str, int] = {}
    blocks: dict[str, int] = {}
    for src in sel.sources:
        per_doc[src.doc_id] = per_doc.get(src.doc_id, 0) + src.tokens
        blocks[src.doc_id] = blocks.get(src.doc_id, 0) + 1
    # A document may exceed the cap only as a single indivisible block.
    assert all(total <= cap or blocks[doc] == 1 for doc, total in per_doc.items())


def test_per_document_cap_drops_extra_chunks_when_it_binds(sectioned):
    conn, did = sectioned
    from galactica.store import get_hits_by_ids

    ids = [chunk_id_for(did, i) for i in range(3)]
    lookup = get_hits_by_ids(conn, ids)
    sel = select_context(
        conn, [lookup[i] for i in ids], budget=4000, per_doc_fraction=0.005, expand=0
    )
    assert len(sel.sources) == 1  # one doc, cap admits only the first block
    assert any("per-document cap" in reason for _, reason in sel.dropped)


def test_drops_are_explained(seeded):
    sel = select_context(seeded, _hits(seeded), budget=120, expand=0)
    assert sel.dropped
    assert all(reason for _, reason in sel.dropped)


def test_render_includes_labels_and_provenance(seeded):
    sel = select_context(seeded, _hits(seeded), budget=4000)
    text = sel.render()
    assert "[S1]" in text
    assert "seed" in text  # source name in the provenance line
    assert sel.sources[0].anchor.title in text


def test_expansion_stays_inside_the_anchor_section(sectioned):
    conn, did = sectioned
    from galactica.store import get_hits_by_ids

    anchor = get_hits_by_ids(conn, [chunk_id_for(did, 1)])[chunk_id_for(did, 1)]
    sel = select_context(conn, [anchor], budget=4000, expand=1)
    kept = sel.sources[0].chunk_ids
    assert chunk_id_for(did, 0) in kept  # same section neighbor pulled in
    assert chunk_id_for(did, 2) not in kept  # different section refused
    assert any("outside anchor section" in reason for _, reason in sel.dropped)


def test_expansion_only_applies_to_top_ranked_anchors(sectioned):
    conn, did = sectioned
    from galactica.store import get_hits_by_ids

    lookup = get_hits_by_ids(conn, [chunk_id_for(did, 1)])
    anchor = lookup[chunk_id_for(did, 1)]
    sel = select_context(conn, [anchor], budget=4000, expand=1, expand_top=0)
    assert sel.sources[0].chunk_ids == [chunk_id_for(did, 1)]


def test_expansion_disabled_by_expand_zero(sectioned):
    conn, did = sectioned
    from galactica.store import get_hits_by_ids

    anchor = get_hits_by_ids(conn, [chunk_id_for(did, 1)])[chunk_id_for(did, 1)]
    sel = select_context(conn, [anchor], budget=4000, expand=0)
    assert sel.sources[0].neighbors == []


def test_expansion_cannot_break_the_budget(sectioned):
    conn, did = sectioned
    from galactica.store import get_hits_by_ids

    anchor = get_hits_by_ids(conn, [chunk_id_for(did, 1)])[chunk_id_for(did, 1)]
    # Budget with room for the anchor block and nothing more.
    # per_doc_fraction=1.0 isolates the budget rule from the per-document cap.
    budget = SelectedSource("S1", anchor).tokens + 1
    sel = select_context(conn, [anchor], budget=budget, expand=1, per_doc_fraction=1.0)
    assert sel.used_tokens <= budget
    assert sel.sources[0].neighbors == []
    assert any("no budget left for neighbor" in reason for _, reason in sel.dropped)


def test_max_sources_caps_citable_blocks(seeded):
    sel = select_context(seeded, _hits(seeded), budget=16000, max_sources=3)
    assert len(sel.sources) == 3
    assert [s.label for s in sel.sources] == ["S1", "S2", "S3"]
