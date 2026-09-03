from galactica.config import Config
from galactica.pipeline import make_plan, parse_plan
from galactica.providers import StubProvider
from galactica.providers.ollama import strip_reasoning


def test_clean_json_plan():
    plan = parse_plan(
        '{"intent":"lookup","sub_questions":["a"],"queries":["x","y"],"needed_facts":["f"]}', "Q"
    )
    assert plan.queries == ["x", "y"] and plan.intent == "lookup" and not plan.fallback


def test_plan_inside_prose_and_code_fence():
    raw = 'Sure, here you go:\n```json\n{"queries": ["alpha", "beta"]}\n```\nhope that helps'
    assert parse_plan(raw, "Q").queries == ["alpha", "beta"]


def test_trailing_comma_repair():
    assert parse_plan('{"queries": ["a", "b",],}', "Q").queries == ["a", "b"]


def test_braces_inside_strings_do_not_confuse_extraction():
    plan = parse_plan('{"intent": "find {this} thing", "queries": ["a"]}', "Q")
    assert plan.intent == "find {this} thing" and plan.queries == ["a"]


def test_unparseable_falls_back_to_the_question():
    plan = parse_plan("I cannot do that", "How do I calibrate?")
    assert plan.queries == ["How do I calibrate?"] and plan.fallback


def test_string_instead_of_list_is_tolerated():
    assert parse_plan('{"queries": "single query"}', "Q").queries == ["single query"]


def test_missing_queries_uses_sub_questions():
    plan = parse_plan('{"sub_questions": ["what is x", "what is y"]}', "Q")
    assert plan.queries == ["what is x", "what is y"]


def test_queries_capped_and_question_always_included():
    provider = StubProvider(
        responder=lambda msgs, json_mode: '{"queries": ["a","b","c","d","e","f","g"]}'
    )
    plan, warnings = make_plan(provider, Config(), "the original question")
    assert len(plan.queries) <= 6
    assert "the original question" in plan.queries
    assert warnings == []


def test_planner_failure_is_warned_not_fatal():
    provider = StubProvider(responder=lambda msgs, json_mode: "no json at all")
    plan, warnings = make_plan(provider, Config(), "Q")
    assert plan.queries == ["Q"] and warnings


def test_reasoning_blocks_are_stripped_before_parsing():
    raw = '<think>let me think about this</think>\n{"queries": ["a"]}'
    assert parse_plan(strip_reasoning(raw), "Q").queries == ["a"]


def test_unterminated_reasoning_block_is_stripped():
    assert strip_reasoning("answer text\n<think>truncated...") == "answer text"
