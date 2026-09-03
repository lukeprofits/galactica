import json
from pathlib import Path

from galactica.evaluate import (
    aggregate,
    keyword_coverage,
    load_cases,
    retrieval_hit,
    run_cases,
    run_retrieval_only,
    save_run,
    uplift,
)
from galactica.providers import StubProvider

EVAL_FILE = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"


def test_shipped_question_set_loads_and_covers_both_arms():
    cases = load_cases(EVAL_FILE)
    assert len(cases) >= 10
    assert any(c.answerable for c in cases)
    assert any(not c.answerable for c in cases)
    for case in cases:
        assert case.question and case.id
        if case.answerable:
            assert case.expect_docs, f"{case.id} needs expect_docs to score retrieval"


def test_keyword_coverage_is_case_insensitive_fraction():
    assert keyword_coverage("Open B4 then B1", ["b4", "b1"]) == 1.0
    assert keyword_coverage("Open B4", ["b4", "b1"]) == 0.5
    assert keyword_coverage("anything", []) is None


def test_retrieval_hit_matches_title_or_chunk_ids():
    sources = [{"title": "Krios Relay R7", "doc_id": "abc", "chunk_ids": ["abc#0001"]}]
    assert retrieval_hit(sources, ["Krios Relay R7"]) == 1.0
    assert retrieval_hit(sources, ["abc#0001"]) == 1.0
    assert retrieval_hit(sources, ["Photosynthesis"]) == 0.0
    assert retrieval_hit(sources, []) is None


def test_retrieval_only_scores_without_any_model(seeded, cfg):
    cases = load_cases(EVAL_FILE)
    results = run_retrieval_only(seeded, cfg, cases)
    assert len(results) == len(cases)
    answerable = [r for r in results if r.case.answerable]
    hits = [r.metrics["retrieval_hit"] for r in answerable]
    # The seed corpus must surface the expected document for most answerable cases.
    assert sum(hits) / len(hits) >= 0.75
    assert all(r.metrics["context_tokens"] <= cfg.context_budget for r in results)


def test_uplift_table_compares_the_same_model_both_ways(seeded, cfg, tmp_path):
    def respond(messages, json_mode):
        if json_mode:
            return json.dumps({"queries": ["krios relay breaker order"]})
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        if "SOURCES:" in user:
            return "Open B4 then B1 [S1]."
        return "I do not have enough information to answer this."

    cases = load_cases(EVAL_FILE)[:3]
    results = run_cases(
        seeded, StubProvider(respond), cfg, cases, modes=("baseline", "cortex")
    )
    agg = aggregate(results)
    assert set(agg) == {"baseline", "cortex"}
    assert agg["cortex"]["cited_any"] == 1.0
    assert agg["baseline"]["cited_any"] is None  # baseline has no sources to cite
    delta = uplift(agg)
    assert delta["keyword_coverage"] is not None
    assert delta["keyword_coverage"] > 0  # the cortex arm is the one with the facts

    run_path = save_run(results, cfg, runs_dir=tmp_path / "runs")
    lines = [json.loads(line) for line in run_path.read_text().splitlines()]
    header, rows = lines[0], lines[1:]
    assert header["record"] == "run_header"
    assert header["model"] == cfg.model and "uplift" in header
    assert len(rows) == len(results)
    assert {r["mode"] for r in rows} == {"baseline", "cortex"}
    assert all("metrics" in r and "sources" in r for r in rows)


def test_refusal_accuracy_splits_by_answerability(seeded, cfg):
    def respond(messages, json_mode):
        if json_mode:
            return json.dumps({"queries": ["q"]})
        return "The corpus does not contain enough information to answer this."

    cases = load_cases(EVAL_FILE)
    results = run_cases(seeded, StubProvider(respond), cfg, cases, modes=("cortex",))
    agg = aggregate(results)["cortex"]
    # A model that always declines: perfect on absent topics, zero on answerable ones.
    assert agg["refused_when_absent"] == 1.0
    assert agg["answered_correctly_flagged"] == 0.0


def test_unsupported_claim_rate_counts_uncited_sentences(seeded, cfg):
    def respond(messages, json_mode):
        if json_mode:
            return json.dumps({"queries": ["krios"]})
        return "First claim [S1]. Second claim has no citation. Third also bare."

    results = run_cases(
        seeded, StubProvider(respond), cfg, load_cases(EVAL_FILE)[:1], modes=("cortex",)
    )
    assert results[0].metrics["unsupported_claim_rate"] == round(2 / 3, 10) or abs(
        results[0].metrics["unsupported_claim_rate"] - 2 / 3
    ) < 1e-9


def test_a_provider_failure_does_not_destroy_the_run(seeded, cfg):
    """One slow question killed a 43-call run outright; it must not any more."""
    from galactica.providers.base import ProviderError

    class FlakyProvider(StubProvider):
        def complete(self, messages, **kwargs):
            self.calls.append({"messages": list(messages), "json_mode": False})
            if len(self.calls) == 2:
                raise ProviderError("ollama timed out after 600s")
            return "Answer [S1]."

    cases = load_cases(EVAL_FILE)[:3]
    results = run_cases(seeded, FlakyProvider(), cfg, cases, modes=("cortex",))
    assert len(results) == len(cases)  # every case still reported
    failed = [r for r in results if r.metrics.get("failed")]
    assert len(failed) == 1
    assert any("timed out" in w for w in failed[0].answer.warnings)
    # Aggregation still works with a failure mixed in.
    assert aggregate(results)["cortex"]["cases"] == 3.0


def test_answers_are_capped_so_a_loop_cannot_run_forever(seeded, cfg):
    from galactica.pipeline import answer_budget, ask_cortex

    seen = {}

    class Recorder(StubProvider):
        def complete(self, messages, **kwargs):
            seen.update(kwargs)
            return "Answer [S1]."

    ask_cortex(seeded, Recorder(), cfg, "krios breaker order")
    assert seen["max_tokens"] == answer_budget(cfg)
    assert answer_budget(cfg) == cfg.max_answer_tokens + cfg.think_reserve
    assert answer_budget(cfg.override(think=False)) == cfg.max_answer_tokens


def test_retrieval_hit_accepts_any_source_that_answers_the_question():
    """A more specific article than the one a fact was sampled from is a hit."""
    sources = [{"title": "Moscow Cathedral Mosque", "doc_id": "d1", "chunk_ids": ["d1#0"]}]
    # The fact was sampled from the list page, but the dedicated article answers it.
    assert retrieval_hit(sources, ["List of mosques in Russia", "Moscow Cathedral Mosque"]) == 1.0
    # Scored against only the sampled page, better retrieval reads as failure.
    assert retrieval_hit(sources, ["List of mosques in Russia"]) == 0.0


def test_shipped_grokipedia_cases_list_acceptable_sources():
    from pathlib import Path

    cases = load_cases(Path(__file__).resolve().parents[1] / "eval" / "questions-grokipedia.jsonl")
    assert len(cases) == 25
    assert all(c.expect_docs for c in cases if c.answerable)
