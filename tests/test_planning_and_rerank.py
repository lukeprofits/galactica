"""Per-sub-question retrieval, coverage reranking, and skipping a useless planner call."""

import json

from galactica.pipeline import ask_cortex, needs_planning, query_groups, Plan
from galactica.providers import StubProvider
from galactica.retrieve import interleave, rerank_by_coverage, search_grouped, term_coverage
from galactica.store import Hit


def _hit(chunk_id, text, title="T", heading="T", score=1.0):
    return Hit(
        chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], ord=0, heading_path=heading,
        title=title, text=text, approx_tokens=10, score=score,
    )


# ------------------------------------------------------------------ planner gate


def test_short_keyword_questions_skip_planning():
    assert needs_planning("Krios R7 breaker order?") is False
    assert needs_planning("copperhead bite") is False
    assert needs_planning("Yamaha XVZ13D displacement") is False


def test_multipart_or_long_questions_get_planned():
    assert needs_planning("What do I do for a copperhead bite, and what must I not do?") is True
    assert needs_planning("How long do I boil water; does elevation change it?") is True
    assert needs_planning(
        "What engine displacement did the Yamaha XVZ13D Venture Royale Mk2 use in 1986"
    ) is True


def test_planner_call_is_skipped_and_warned(seeded, cfg):
    provider = StubProvider(lambda msgs, jm: "Answer [S1].")
    answer = ask_cortex(seeded, provider, cfg, "krios breaker order")
    assert len(provider.calls) == 1  # answer only, no planner round trip
    assert answer.plan is None
    assert any("planner skipped" in w for w in answer.warnings)


# ----------------------------------------------------------------- query groups


def test_each_sub_question_becomes_its_own_group():
    plan = Plan(
        intent="lookup",
        sub_questions=["what to do for a bite", "what not to do for a bite"],
        queries=["copperhead first aid"],
    )
    groups = query_groups(plan, "What do I do and not do?")
    assert ["what to do for a bite"] in groups
    assert ["what not to do for a bite"] in groups
    # The raw question and planner keywords still form a group of their own.
    assert any("What do I do and not do?" in g for g in groups)


def test_groups_fall_back_to_the_question_without_a_plan():
    assert query_groups(None, "just the question") == [["just the question"]]


def test_missing_queries_join_as_an_extra_group():
    plan = Plan(sub_questions=["a"], queries=["b"])
    groups = query_groups(plan, "q", extra=["second hop query"])
    assert ["second hop query"] in groups


# -------------------------------------------------------------------- reranking


def test_term_coverage_counts_distinct_terms_not_repeats():
    many = _hit("d1#0", "alpha beta gamma delta")
    repeated = _hit("d2#0", "alpha alpha alpha alpha")
    stems = ["alpha", "beta", "gamma", "delta"]
    assert term_coverage(many, stems) == 1.0
    assert term_coverage(repeated, stems) == 0.25


def test_coverage_promotes_the_chunk_that_answers_more_of_the_query(seeded):
    # Equal fused scores; the one covering more distinct terms must win.
    partial = _hit("d1#0", "wrenches are tools " * 5, score=0.02)
    complete = _hit("d2#0", "calibrate the torque wrench annually", score=0.02)
    ordered = rerank_by_coverage(seeded, [partial, complete], ["calibrate torque wrench"])
    assert ordered[0].chunk_id == "d2#0"


def test_reranking_is_stable_with_no_terms(seeded):
    hits = [_hit("d1#0", "x"), _hit("d2#0", "y")]
    assert [h.chunk_id for h in rerank_by_coverage(seeded, hits, ["!!!"])] == ["d1#0", "d2#0"]


# ------------------------------------------------------------------ interleaving


def test_interleave_gives_every_group_a_slot():
    a = [_hit("a#0", "1"), _hit("a#1", "2"), _hit("a#2", "3")]
    b = [_hit("b#0", "1"), _hit("b#1", "2")]
    order = [h.chunk_id for h in interleave([a, b])]
    assert order[:4] == ["a#0", "b#0", "a#1", "b#1"]


def test_interleave_dedupes_across_groups():
    shared = _hit("s#0", "shared")
    order = [h.chunk_id for h in interleave([[shared], [shared]])]
    assert order == ["s#0"]


def test_grouped_search_represents_both_sub_questions(seeded):
    result = search_grouped(
        seeded,
        [["krios relay breaker order"], ["torque wrench calibration"]],
        top_k=6,
    )
    titles = {h.title for h in result.hits}
    assert any("Krios" in t for t in titles)
    assert any("Torque wrench" in t for t in titles)


def test_grouped_search_with_one_group_matches_plain_search(seeded):
    from galactica.retrieve import search

    grouped = search_grouped(seeded, [["krios relay breaker"]], top_k=5)
    plain = search(seeded, ["krios relay breaker"], top_k=5)
    assert [h.chunk_id for h in grouped.hits] == [h.chunk_id for h in plain.hits]


def test_empty_groups_return_nothing(seeded):
    assert search_grouped(seeded, [["   "], []], top_k=5).hits == []
