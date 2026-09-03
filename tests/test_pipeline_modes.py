import json

from galactica.pipeline import ask, ask_baseline, ask_cortex
from galactica.providers import StubProvider


def _responder(answer_text):
    def respond(messages, json_mode):
        if json_mode:
            return json.dumps({"intent": "lookup", "queries": ["krios relay breaker order"]})
        return answer_text

    return respond


def test_baseline_never_retrieves(seeded, cfg):
    provider = StubProvider(_responder("Breakers open in some order."))
    answer = ask_baseline(provider, cfg, "Krios R7 breaker order?")
    assert answer.mode == "baseline"
    assert answer.sources == [] and answer.context == "" and answer.context_tokens == 0
    # Exactly one model call, and no SOURCES block was ever built.
    assert len(provider.calls) == 1
    assert all("SOURCES:" not in m["content"] for m in provider.calls[0]["messages"])


def test_cortex_retrieves_cites_and_reports_provenance(seeded, cfg):
    provider = StubProvider(_responder("Open B4 then B1 [S1]."))
    # Long enough to warrant a planner call, so both calls are exercised.
    answer = ask_cortex(
        seeded,
        provider,
        cfg,
        "In what order must the breakers be opened on a Krios R7 relay, and what happens otherwise?",
    )
    assert answer.mode == "cortex"
    assert answer.sources and answer.citations == ["S1"]
    assert answer.invalid_citations == []
    assert 0 < answer.context_tokens <= cfg.context_budget
    src = answer.sources[0]
    assert src.source_name == "seed" and src.chunk_ids and src.title
    # The planner call happened, and the answer call saw the sources.
    assert len(provider.calls) == 2
    assert any("SOURCES:" in m["content"] for m in provider.calls[1]["messages"])


def test_no_plan_skips_the_planner_call(seeded, cfg):
    provider = StubProvider(_responder("Answer [S1]."))
    answer = ask_cortex(seeded, provider, cfg, "Krios R7 breaker order?", use_plan=False)
    assert len(provider.calls) == 1 and answer.plan is None


def test_fabricated_citation_is_flagged(seeded, cfg):
    provider = StubProvider(_responder("Nonsense [S99]."))
    answer = ask_cortex(seeded, provider, cfg, "Krios R7 breaker order?")
    assert answer.invalid_citations == ["S99"]
    assert any("unknown sources" in w for w in answer.warnings)


def test_gap_and_missing_lines_are_extracted_and_stripped(seeded, cfg):
    provider = StubProvider(
        _responder("Partial answer [S1].\nGAP: no torque value present.\nMISSING: krios torque spec")
    )
    answer = ask_cortex(seeded, provider, cfg, "Krios R7 torque?")
    assert answer.gaps == ["no torque value present."]
    assert answer.missing_queries == ["krios torque spec"]
    assert "MISSING:" not in answer.text


def test_declining_is_detected(seeded, cfg):
    provider = StubProvider(
        _responder("The corpus does not contain enough information to answer this.")
    )
    answer = ask_cortex(seeded, provider, cfg, "Bhutan policy rate?")
    assert answer.declined is True


def test_second_hop_reanswers_when_requested(seeded, cfg):
    calls = {"n": 0}

    def respond(messages, json_mode):
        if json_mode:
            return json.dumps({"queries": ["krios relay"]})
        calls["n"] += 1
        if calls["n"] == 1:
            return "Partial [S1].\nMISSING: krios carrier screw torque"
        return "Complete answer [S1]."

    two_hops = cfg.override(hops=2)
    answer = ask_cortex(seeded, StubProvider(respond), two_hops, "Krios torque?")
    assert answer.hops_used == 2 and answer.text == "Complete answer [S1]."

    one_hop = ask_cortex(seeded, StubProvider(respond), cfg, "Krios torque?")
    assert one_hop.hops_used == 1


def test_both_mode_runs_two_independent_arms(seeded, cfg):
    provider = StubProvider(_responder("Answer [S1]."))
    answers = ask(seeded, provider, cfg, "Krios R7 breaker order?", mode="both")
    assert set(answers) == {"baseline", "cortex"}
    assert answers["baseline"].sources == [] and answers["cortex"].sources
    assert answers["baseline"].question == answers["cortex"].question


def test_answer_serializes_for_run_persistence(seeded, cfg):
    provider = StubProvider(_responder("Answer [S1]."))
    answer = ask_cortex(seeded, provider, cfg, "Krios R7 breaker order?")
    blob = json.dumps(answer.to_dict())
    assert '"mode": "cortex"' in blob and "chunk_ids" in blob
