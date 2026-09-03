import json
import threading
import urllib.request

import pytest

from galactica.providers.base import ProviderHealth
from galactica.server import (
    Gateway,
    build_server,
    count_tokens,
    latest_user_text,
    should_retrieve,
    sse_events,
    to_anthropic_message,
    to_ollama_messages,
    to_ollama_tools,
)


class FakeOllama:
    """Stands in for the local model. Records the payload it was handed."""

    def __init__(self, response=None):
        self.payloads = []
        self.response = response or {
            "message": {"content": "A grounded answer [S1]."},
            "prompt_eval_count": 120,
            "eval_count": 7,
        }

    def chat(self, payload):
        self.payloads.append(payload)
        return self.response

    def chat_stream(self, payload):
        """Emit the canned reply a few characters at a time, like the real thing."""
        self.payloads.append(payload)
        message = self.response.get("message", {})
        text = str(message.get("content") or "")
        for start in range(0, len(text), 8):
            yield {"message": {"content": text[start : start + 8]}}
        yield {
            "message": {"content": "", "tool_calls": message.get("tool_calls") or []},
            "done": True,
            "done_reason": self.response.get("done_reason", "stop"),
            "eval_count": self.response.get("eval_count", 0),
        }

    def embed(self, texts):
        return [[0.0] * 4 for _ in texts]

    def health(self):
        return ProviderHealth(True, "fake")


# ------------------------------------------------------------------ translation


def test_latest_user_text_reads_the_last_user_turn():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]
    assert latest_user_text(messages) == "second"


def test_system_and_corpus_context_are_merged_into_one_system_message():
    out = to_ollama_messages("Base rules.", [{"role": "user", "content": "hi"}], "CORPUS")
    assert out[0]["role"] == "system"
    assert "Base rules." in out[0]["content"] and "CORPUS" in out[0]["content"]
    assert out[1] == {"role": "user", "content": "hi"}


def test_system_as_block_list_is_flattened():
    out = to_ollama_messages([{"type": "text", "text": "block system"}], [])
    assert out[0]["content"] == "block system"


def test_tool_use_and_tool_result_survive_translation():
    messages = [
        {"role": "user", "content": "read the file"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.py"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file body"}],
        },
    ]
    out = to_ollama_messages(None, messages)
    assistant = next(m for m in out if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["function"] == {"name": "Read", "arguments": {"path": "a.py"}}
    assert {"role": "tool", "content": "file body"} in out


def test_tool_schemas_translate_to_ollama_functions():
    tools = [{"name": "Read", "description": "read a file", "input_schema": {"type": "object"}}]
    assert to_ollama_tools(tools) == [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "read a file",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert to_ollama_tools(None) == []


def test_response_translation_strips_reasoning_and_maps_stop_reason():
    msg = to_anthropic_message(
        {"message": {"content": "<think>hmm</think>Answer."}, "eval_count": 3}, "m"
    )
    assert msg["content"] == [{"type": "text", "text": "Answer."}]
    assert msg["stop_reason"] == "end_turn"
    assert msg["usage"]["output_tokens"] == 3
    assert msg["role"] == "assistant" and msg["type"] == "message"


def test_tool_calls_become_tool_use_blocks_with_stop_reason():
    msg = to_anthropic_message(
        {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "Read", "arguments": {"path": "x"}}}],
            }
        },
        "m",
    )
    block = msg["content"][0]
    assert block["type"] == "tool_use" and block["name"] == "Read"
    assert block["input"] == {"path": "x"} and block["id"].startswith("toolu_")
    assert msg["stop_reason"] == "tool_use"


def test_string_encoded_tool_arguments_are_parsed():
    msg = to_anthropic_message(
        {"message": {"tool_calls": [{"function": {"name": "R", "arguments": '{"a": 1}'}}]}}, "m"
    )
    assert msg["content"][0]["input"] == {"a": 1}


def test_length_truncation_maps_to_max_tokens():
    msg = to_anthropic_message(
        {"message": {"content": "cut"}, "done_reason": "length"}, "m"
    )
    assert msg["stop_reason"] == "max_tokens"


# ------------------------------------------------------------------------- gate


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Who founded the Free State of Fiume?", True),
        ("What torque do the carrier screws take?", True),
        ("explain photorespiration", True),
        ("run pytest and fix the failure", False),
        ("refactor src/galactica/store.py", False),
        ("```python\nprint(1)\n```", False),
        ("ok", False),
    ],
)
def test_gate_retrieves_for_questions_not_code_work(text, expected):
    got, _ = should_retrieve([{"role": "user", "content": text}], "auto")
    assert got is expected


def test_gate_skips_mid_agent_loop_tool_results():
    messages = [
        {"role": "user", "content": "What is a torque wrench?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "R", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "out"}]},
    ]
    got, reason = should_retrieve(messages, "auto")
    assert got is False and "tool result" in reason


def test_gate_modes_override_the_heuristic():
    code = [{"role": "user", "content": "refactor store.py"}]
    question = [{"role": "user", "content": "who is Rufus Wainwright?"}]
    assert should_retrieve(code, "always")[0] is True
    assert should_retrieve(question, "off")[0] is False


# -------------------------------------------------------------------- streaming


def test_sse_sequence_is_well_formed():
    msg = to_anthropic_message({"message": {"content": "hello"}}, "m")
    events = list(sse_events(msg))
    names = [name for name, _ in events]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = events[0][1]["message"]
    assert start["content"] == [] and start["stop_reason"] is None
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "hello"}
    assert events[-2][1]["delta"]["stop_reason"] == "end_turn"


def test_sse_streams_tool_input_as_json_delta():
    msg = to_anthropic_message(
        {"message": {"tool_calls": [{"function": {"name": "R", "arguments": {"a": 1}}}]}}, "m"
    )
    deltas = [d for name, d in sse_events(msg) if name == "content_block_delta"]
    assert deltas[0]["delta"]["type"] == "input_json_delta"
    assert json.loads(deltas[0]["delta"]["partial_json"]) == {"a": 1}


def test_count_tokens_covers_system_messages_and_tools():
    request = {
        "system": "a" * 40,
        "messages": [{"role": "user", "content": "b" * 40}],
        "tools": [{"name": "R", "input_schema": {"type": "object"}}],
    }
    assert count_tokens(request) > 20


# -------------------------------------------------------------------- gateway


def test_corpus_context_is_injected_but_only_for_the_model(seeded, cfg, tmp_path):
    fake = FakeOllama()
    gateway = Gateway(cfg.override(data_dir=cfg.data_dir), mode="auto", provider=fake)
    gateway._local.conn = seeded  # reuse the seeded corpus connection

    request = {
        "messages": [{"role": "user", "content": "What torque do the Krios carrier screws take?"}],
        "stream": False,
    }
    message, info = gateway.complete(request)

    assert info.retrieved and info.sources > 0
    system = fake.payloads[0]["messages"][0]["content"]
    assert "CORPUS EXCERPTS" in system and "Krios" in system
    # What comes back holds the answer only: no corpus text leaks into the reply.
    assert message["content"][0]["text"] == "A grounded answer [S1]."
    assert "CORPUS EXCERPTS" not in json.dumps(message)


def test_code_requests_skip_retrieval_entirely(seeded, cfg):
    fake = FakeOllama()
    gateway = Gateway(cfg, mode="auto", provider=fake)
    gateway._local.conn = seeded
    _, info = gateway.complete(
        {"messages": [{"role": "user", "content": "run pytest and fix the failure"}]}
    )
    assert info.retrieved is False
    assert "CORPUS EXCERPTS" not in fake.payloads[0]["messages"][0].get("content", "")


def test_request_options_pass_through_to_the_model(seeded, cfg):
    fake = FakeOllama()
    gateway = Gateway(cfg, mode="off", provider=fake)
    gateway._local.conn = seeded
    gateway.complete(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "temperature": 0.3,
            "stop_sequences": ["STOP"],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        }
    )
    payload = fake.payloads[0]
    # 64 is the answer budget; reasoning is funded on top of it.
    assert payload["options"]["num_predict"] == 64 + cfg.think_reserve
    assert payload["options"]["temperature"] == 0.3
    assert payload["options"]["stop"] == ["STOP"]
    assert payload["tools"][0]["function"]["name"] == "Read"


def test_unknown_mode_is_rejected(cfg):
    with pytest.raises(ValueError):
        Gateway(cfg, mode="nonsense", provider=FakeOllama())


# ------------------------------------------------------------------ http server


@pytest.fixture
def running_server(seeded, cfg):
    fake = FakeOllama()
    httpd = build_server(cfg, host="127.0.0.1", port=0, mode="off", provider=fake, verbose=False)
    httpd.gateway._local.conn = seeded
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    yield f"http://{host}:{port}", fake
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _post(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read().decode(), response.headers


def test_messages_endpoint_returns_an_anthropic_message(running_server):
    base, _ = running_server
    status, body, _ = _post(f"{base}/v1/messages", {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 200
    message = json.loads(body)
    assert message["type"] == "message" and message["role"] == "assistant"
    assert message["content"][0]["text"] == "A grounded answer [S1]."


def test_streaming_endpoint_emits_sse(running_server):
    base, _ = running_server
    status, body, headers = _post(
        f"{base}/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    )
    assert status == 200
    assert headers["Content-Type"] == "text/event-stream"
    assert body.startswith("event: message_start")
    assert "event: message_stop" in body
    assert "data: {" in body


def test_count_tokens_endpoint(running_server):
    base, _ = running_server
    status, body, _ = _post(
        f"{base}/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "hello"}]}
    )
    assert status == 200 and json.loads(body)["input_tokens"] >= 1


def test_health_endpoint(running_server):
    base, _ = running_server
    with urllib.request.urlopen(f"{base}/health", timeout=10) as response:
        assert json.loads(response.read().decode())["status"] == "ok"


def test_unknown_route_is_an_api_error(running_server):
    base, _ = running_server
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(f"{base}/v1/nonsense", {})
    assert err.value.code == 404
    assert json.loads(err.value.read().decode())["type"] == "error"


def test_reasoning_stays_on_and_gets_its_own_token_reserve(seeded, cfg):
    fake = FakeOllama()
    gateway = Gateway(cfg, mode="off", provider=fake)
    gateway._local.conn = seeded
    gateway.complete({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 400})
    payload = fake.payloads[0]
    assert payload["think"] is True
    # The client's 400 is the answer budget; reasoning is funded on top of it.
    assert payload["options"]["num_predict"] == 400 + cfg.think_reserve


def test_reasoning_can_be_turned_off_and_then_gets_no_reserve(seeded, cfg):
    fake = FakeOllama()
    gateway = Gateway(cfg.override(think=False), mode="off", provider=fake)
    gateway._local.conn = seeded
    gateway.complete({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 400})
    assert fake.payloads[0]["think"] is False
    assert fake.payloads[0]["options"]["num_predict"] == 400


class StarvingOllama(FakeOllama):
    """Burns the whole allowance on reasoning, then answers on the retry."""

    def chat(self, payload):
        self.payloads.append(payload)
        if payload.get("think"):
            return {"message": {"content": "<think>still reasoning"}, "done_reason": "length"}
        return {"message": {"content": "Concise answer."}, "eval_count": 4}


def test_starved_answer_retries_without_reasoning(seeded, cfg):
    fake = StarvingOllama()
    gateway = Gateway(cfg, mode="off", provider=fake)
    gateway._local.conn = seeded
    message, info = gateway.complete(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50}
    )
    assert info.think_retry is True
    assert [p["think"] for p in fake.payloads] == [True, False]
    assert message["content"][0]["text"] == "Concise answer."


def test_a_tool_call_is_not_treated_as_starved(seeded, cfg):
    fake = FakeOllama(
        {
            "message": {"content": "", "tool_calls": [{"function": {"name": "R", "arguments": {}}}]},
            "done_reason": "length",
        }
    )
    gateway = Gateway(cfg, mode="off", provider=fake)
    gateway._local.conn = seeded
    message, info = gateway.complete({"messages": [{"role": "user", "content": "hi"}]})
    assert info.think_retry is False and len(fake.payloads) == 1
    assert message["stop_reason"] == "tool_use"


def test_budget_exhausted_by_reasoning_reports_instead_of_returning_empty():
    msg = to_anthropic_message(
        {"message": {"content": "<think>still reasoning and then cut off"}, "done_reason": "length"},
        "m",
    )
    text = msg["content"][0]["text"]
    assert "output limit was reached" in text
    assert msg["stop_reason"] == "max_tokens"


def test_no_content_at_all_is_reported():
    msg = to_anthropic_message({"message": {"content": ""}}, "m")
    assert "no answer returned" in msg["content"][0]["text"]


def test_context_window_is_capped_on_every_path():
    """Ollama reserves the model's max context as KV cache unless told otherwise."""
    from galactica.providers.ollama import OllamaProvider

    provider = OllamaProvider("m", num_ctx=8192)
    sent = {}
    provider._post = lambda path, payload, timeout=None: sent.update(payload) or {"message": {}}
    provider.complete([{"role": "user", "content": "hi"}])
    assert sent["options"]["num_ctx"] == 8192

    sent.clear()
    provider.chat({"model": "m", "messages": []})
    assert sent["options"]["num_ctx"] == 8192

    # An explicit num_ctx in the payload is respected rather than overridden.
    sent.clear()
    provider.chat({"model": "m", "messages": [], "options": {"num_ctx": 1024}})
    assert sent["options"]["num_ctx"] == 1024

    # No cap configured means no option sent.
    bare = OllamaProvider("m")
    seen = {}
    bare._post = lambda path, payload, timeout=None: seen.update(payload) or {"message": {}}
    bare.complete([{"role": "user", "content": "hi"}])
    assert "num_ctx" not in seen["options"]
