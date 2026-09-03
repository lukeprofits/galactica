from galactica.providers import StubProvider
from galactica.retrieve import cosine, rrf_fuse, search, vector_search
from galactica.ingest import embed_missing


def test_rrf_rewards_agreement_across_lists():
    fused = dict(rrf_fuse([["a", "b", "c"], ["c", "a"], ["a"]]))
    assert max(fused, key=fused.get) == "a"
    assert fused["c"] > fused["b"]  # appears in two lists despite worse ranks


def test_rrf_is_deterministic_on_ties():
    assert rrf_fuse([["b", "a"]]) == rrf_fuse([["b", "a"]])


def test_search_fuses_multiple_queries(seeded):
    result = search(seeded, ["krios relay breaker", "hydraulic cabinet reset"], top_k=8)
    assert result.hits
    assert len(result.per_query) == 2
    assert any("Krios" in h.title for h in result.hits)
    assert all(h.score > 0 for h in result.hits)


def test_search_respects_top_k_and_dedupes(seeded):
    result = search(seeded, ["torque", "torque"], top_k=3)
    ids = [h.chunk_id for h in result.hits]
    assert len(ids) == len(set(ids)) <= 3


def test_empty_query_returns_nothing(seeded):
    assert search(seeded, ["   "], top_k=5).hits == []


def test_hybrid_without_embeddings_warns_and_degrades(seeded):
    result = search(
        seeded, ["torque wrench"], top_k=5, hybrid=True,
        provider=StubProvider(), embed_model="stub",
    )
    assert result.hits  # BM25 still answers
    assert any("no embeddings stored" in w for w in result.warnings)


def test_hybrid_uses_vectors_once_embedded(seeded):
    provider = StubProvider()
    written = embed_missing(seeded, provider, "stub")
    assert written > 0
    vectors = vector_search(seeded, provider, "stub", "torque wrench calibration", 5)
    assert vectors and all(-1.0 <= score <= 1.0 for _, score in vectors)
    result = search(
        seeded, ["torque wrench calibration"], top_k=5, hybrid=True,
        provider=provider, embed_model="stub",
    )
    assert result.hits and result.warnings == []


def test_cosine_edges():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([0, 0], [1, 1]) == 0.0
    assert cosine([1, 0], [1, 0, 0]) == 0.0


def test_proximity_pass_prefers_adjacent_query_words(seeded):
    from galactica.store import search_proximity

    hits = search_proximity(seeded, "click type torque wrench", limit=5)
    assert hits and "Torque wrench" in hits[0].title


def test_proximity_needs_at_least_two_words(seeded):
    from galactica.store import search_proximity

    assert search_proximity(seeded, "torque", limit=5) == []
    assert search_proximity(seeded, "the", limit=5) == []


def test_proximity_tolerates_dropped_stopwords(seeded):
    """"Free State of Fiume" indexes "of" but the query drops it; NEAR slop covers that."""
    from galactica.store import search_proximity

    hits = search_proximity(seeded, "accuracy and calibration of a torque wrench", limit=5)
    assert hits


def test_search_fuses_the_proximity_list(seeded):
    result = search(seeded, ["krios relay carrier screws"], top_k=8)
    assert result.hits and "Krios" in result.hits[0].title
