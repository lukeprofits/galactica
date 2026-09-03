"""Augmented grounding: the corpus improves answers instead of restricting them."""

import json

import pytest

from galactica.pipeline import ask_cortex
from galactica.prompts import (
    CORTEX_AUGMENTED_SYSTEM,
    CORTEX_STRICT_SYSTEM,
    cortex_system,
    gateway_suffix,
)
from galactica.providers import StubProvider
from galactica.server import Gateway


def _responder(answer):
    def respond(messages, json_mode):
        if json_mode:
            return json.dumps({"queries": ["krios relay"]})
        return answer

    return respond


def test_grounding_selects_the_prompt():
    assert cortex_system("augmented") is CORTEX_AUGMENTED_SYSTEM
    assert cortex_system("strict") is CORTEX_STRICT_SYSTEM
    assert cortex_system() is CORTEX_AUGMENTED_SYSTEM  # the default
    with pytest.raises(ValueError):
        cortex_system("nonsense")


def test_augmented_prompt_forbids_refusing_over_a_thin_corpus():
    text = CORTEX_AUGMENTED_SYSTEM.lower()
    assert "answer from your own knowledge anyway" in text
    assert "way of\n  declining" in text or "declining a question" in text
    assert "uncited:" in text


def test_strict_prompt_still_refuses_beyond_the_corpus():
    assert "only information present in the sources" in CORTEX_STRICT_SYSTEM.lower()


def test_uncited_line_is_extracted_and_stripped(seeded, cfg):
    provider = StubProvider(
        _responder(
            "Open B4 then B1 [S1]. Torque them to 12 N·m.\n"
            "UNCITED: the 12 N·m figure is from my own knowledge, not the sources."
        )
    )
    answer = ask_cortex(seeded, provider, cfg.override(grounding="augmented"), "Krios order?")
    assert answer.uncited == [
        "the 12 N·m figure is from my own knowledge, not the sources."
    ]
    assert "UNCITED:" not in answer.text
    assert answer.citations == ["S1"]  # corpus-backed part still validated


def test_no_uncited_line_when_everything_was_corpus_backed(seeded, cfg):
    provider = StubProvider(_responder("Open B4 then B1 [S1]."))
    answer = ask_cortex(seeded, provider, cfg, "Krios order?")
    assert answer.uncited == []


def test_grounding_choice_reaches_the_model(seeded, cfg):
    for grounding, expected in (("strict", CORTEX_STRICT_SYSTEM), ("augmented", CORTEX_AUGMENTED_SYSTEM)):
        provider = StubProvider(_responder("answer [S1]"))
        ask_cortex(seeded, provider, cfg.override(grounding=grounding), "Krios order?")
        systems = [
            m["content"]
            for call in provider.calls
            for m in call["messages"]
            if m["role"] == "system"
        ]
        assert expected in systems


def test_gateway_suffix_follows_the_grounding_mode(seeded, cfg):
    augmented = gateway_suffix("augmented")
    strict = gateway_suffix("strict")
    assert "never to limit it" in augmented
    assert "Answer only from those excerpts" in strict

    class Fake:
        def __init__(self):
            self.payloads = []

        def chat(self, payload):
            self.payloads.append(payload)
            return {"message": {"content": "ok"}}

    for grounding, expected in (("augmented", "never to limit it"), ("strict", "Answer only from")):
        fake = Fake()
        gateway = Gateway(cfg.override(grounding=grounding), mode="always", provider=fake)
        gateway._local.conn = seeded
        gateway.complete({"messages": [{"role": "user", "content": "what torque for krios screws?"}]})
        assert expected in fake.payloads[0]["messages"][0]["content"]


def test_default_config_is_augmented():
    from galactica.config import Config

    assert Config().grounding == "augmented"
    assert Config.from_env().grounding == "augmented"


def test_eval_tracks_whether_uncited_parts_were_labelled(seeded, cfg):
    from galactica.evaluate import EvalCase, run_cases

    provider = StubProvider(
        _responder("Partial [S1]. Rest from memory.\nUNCITED: the rest.")
    )
    case = EvalCase(question="Krios order?", expect_docs=["Krios"], answerable=True, id="k")
    results = run_cases(seeded, provider, cfg, [case], modes=("cortex",))
    assert results[0].metrics["labelled_uncited"] == 1.0


def test_refusal_detection_covers_augmented_phrasings():
    """Strict mode has one refusal sentence; augmented mode has many."""
    from galactica.pipeline import looks_like_refusal

    refusals = [
        "The corpus does not contain enough information to answer this.",
        "I do not have enough information to answer this.",
        "The 2026 Winter Olympics have not yet taken place; no final medal table exists.",
        "The 2026 FIFA World Cup has not yet been played, so the winner is currently unknown.",
        "Anthropic has not announced or released a Claude Opus 5.",
        "I do not have access to real-time market data.",
        "I cannot answer that from the material available.",
        "No information is available on this topic.",
    ]
    for text in refusals:
        assert looks_like_refusal(text), text

    answers = [
        "The Yamaha XVZ13D used a 1,294 cc engine [S1].",
        "Netflix published the Android and iOS ports [S2].",
        "Open breaker B4, then B1 [S1]. Torque the screws to 12 N·m [S2].",
    ]
    for text in answers:
        assert not looks_like_refusal(text), text


def test_augmented_refusal_is_scored_as_a_correct_refusal(seeded, cfg):
    from galactica.evaluate import EvalCase, score_answer
    from galactica.pipeline import ask_cortex

    provider = StubProvider(
        _responder(
            "The provided sources do not contain Nvidia's closing price for that date. "
            "The most recent documented is $186.26 on October 24, 2025 [S1]. "
            "I do not have access to real-time market data.\nUNCITED: none."
        )
    )
    answer = ask_cortex(seeded, provider, cfg, "Nvidia close on 15 August 2026?")
    assert answer.declined is True
    case = EvalCase(question="x", answerable=False, id="absent")
    assert score_answer(answer, case)["refusal_correct"] == 1.0


def test_refusal_detection_covers_source_absence_phrasings():
    """Each new model phrases "not in the corpus" its own way."""
    from galactica.pipeline import looks_like_refusal

    for text in [
        "The context window is not mentioned in the provided SOURCES.",
        "No SOURCES provide data for August 15, 2026.",
        "No information about a Claude Opus 5 variant exists in the corpus.",
        "That figure does not appear in the corpus.",
        "The exact date is not specified in the material available.",
    ]:
        assert looks_like_refusal(text), text
    assert not looks_like_refusal("The Mk2 used a 1,294 cc engine [S1].")


def test_empty_answer_retries_without_reasoning(seeded, cfg):
    """The output cap can be consumed entirely by reasoning; don't return blank."""
    calls = []

    class Starving(StubProvider):
        def complete(self, messages, **kwargs):
            calls.append(kwargs.get("think"))
            if kwargs.get("json_mode"):
                return json.dumps({"queries": ["krios"]})
            return "" if kwargs.get("think") else "Recovered answer [S1]."

    answer = ask_cortex(seeded, Starving(), cfg, "krios breaker order")
    assert answer.text == "Recovered answer [S1]."
    assert calls == [True, False]
    assert any("retried without it" in w for w in answer.warnings)
