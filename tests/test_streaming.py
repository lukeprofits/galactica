"""Real token streaming through the gateway, and the reasoning filter it needs."""

import json
import threading
import urllib.request

import pytest

from galactica.providers.base import ProviderError, ProviderHealth
from galactica.server import ReasoningFilter, build_server, stream_events


class StreamingFake:
    def __init__(self, chunks, done_reason="stop", tool_calls=None):
        self.chunks = chunks
        self.done_reason = done_reason
        self.tool_calls = tool_calls or []
        self.payloads = []

    def chat_stream(self, payload):
        self.payloads.append(payload)
        for piece in self.chunks:
            yield {"message": {"content": piece}}
        yield {
            "message": {"content": "", "tool_calls": self.tool_calls},
            "done": True,
            "done_reason": self.done_reason,
            "eval_count": 11,
        }

    def chat(self, payload):
        self.payloads.append(payload)
        return {"message": {"content": "".join(self.chunks)}}

    def health(self):
        return ProviderHealth(True, "fake")


# --------------------------------------------------------------- reasoning filter


def test_filter_strips_a_whole_reasoning_block():
    f = ReasoningFilter()
    out = f.feed("<think>deliberating</think>The answer.") + f.flush()
    assert out == "The answer."


def test_filter_handles_a_tag_split_across_chunks():
    f = ReasoningFilter()
    out = "".join(f.feed(piece) for piece in ["<thi", "nk>hid", "den</thi", "nk>vis", "ible"])
    assert (out + f.flush()) == "visible"


def test_filter_holds_back_a_possible_partial_tag():
    f = ReasoningFilter()
    # "<" could begin "<think>", so it must not be emitted yet.
    assert f.feed("answer <") == "answer "
    assert f.feed("no") == "<no"  # not a think tag after all
    assert f.flush() == ""


def test_filter_drops_unterminated_reasoning():
    f = ReasoningFilter()
    assert f.feed("<think>never closed") == ""
    assert f.flush() == ""
    assert f.emitted_any is False


def test_filter_reports_whether_anything_visible_was_emitted():
    f = ReasoningFilter()
    f.feed("<think>x</think>")
    assert f.emitted_any is False
    f.feed("real text")
    assert f.emitted_any is True


# -------------------------------------------------------------------- event order


def _events(cfg, conn, provider, request=None):
    from galactica.server import Gateway

    gateway = Gateway(cfg, mode="off", provider=provider)
    gateway._local.conn = conn
    return list(stream_events(gateway, request or {"messages": [{"role": "user", "content": "hi"}]}))


def test_stream_emits_a_valid_anthropic_sequence(seeded, cfg):
    events = _events(cfg, seeded, StreamingFake(["Hello ", "world."]))
    names = [name for name, _ in events]
    assert names[0] == "message_start"
    assert names[1] == "content_block_start"
    assert names[-1] == "message_stop"
    assert names[-2] == "message_delta"
    text = "".join(
        d["delta"]["text"] for n, d in events if n == "content_block_delta" and "text" in d["delta"]
    )
    assert text == "Hello world."


def test_stream_arrives_in_pieces_not_one_lump(seeded, cfg):
    events = _events(cfg, seeded, StreamingFake(["one ", "two ", "three"]))
    deltas = [d for n, d in events if n == "content_block_delta"]
    assert len(deltas) >= 3  # the point of streaming


def test_stream_strips_reasoning_tokens(seeded, cfg):
    fake = StreamingFake(["<think>plan", "ning</think>", "Answer only."])
    events = _events(cfg, seeded, fake)
    text = "".join(
        d["delta"]["text"] for n, d in events if n == "content_block_delta" and "text" in d["delta"]
    )
    assert text == "Answer only."
    assert "think" not in text


def test_stream_reports_when_reasoning_ate_the_whole_budget(seeded, cfg):
    fake = StreamingFake(["<think>still going"], done_reason="length")
    events = _events(cfg, seeded, fake)
    text = "".join(
        d["delta"]["text"] for n, d in events if n == "content_block_delta" and "text" in d["delta"]
    )
    assert "output limit was reached" in text
    stop = [d for n, d in events if n == "message_delta"][0]
    assert stop["delta"]["stop_reason"] == "max_tokens"


def test_stream_emits_tool_use_blocks(seeded, cfg):
    fake = StreamingFake(
        ["working"], tool_calls=[{"function": {"name": "Read", "arguments": {"path": "a.py"}}}]
    )
    events = _events(cfg, seeded, fake)
    starts = [d for n, d in events if n == "content_block_start"]
    assert starts[1]["content_block"]["type"] == "tool_use"
    assert starts[1]["content_block"]["name"] == "Read"
    payload = [
        d for n, d in events if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta"
    ][0]
    assert json.loads(payload["delta"]["partial_json"]) == {"path": "a.py"}
    assert [d for n, d in events if n == "message_delta"][0]["delta"]["stop_reason"] == "tool_use"


# ------------------------------------------------------------------- over http


@pytest.fixture
def server(seeded, cfg, request):
    provider = getattr(request, "param", None) or StreamingFake(["streamed ", "reply"])
    httpd = build_server(cfg, host="127.0.0.1", port=0, mode="off", provider=provider, verbose=False)
    httpd.gateway._local.conn = seeded
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", provider
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _post_stream(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.read().decode()


def test_http_streaming_delivers_incremental_deltas(server):
    base, _ = server
    body = _post_stream(
        f"{base}/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    )
    assert body.startswith("event: message_start")
    assert body.count("event: content_block_delta") >= 2
    assert body.rstrip().endswith('data: {"type": "message_stop"}')


class NoStreamProvider:
    """A provider that cannot stream; the gateway must still serve stream requests."""

    def chat(self, payload):
        return {"message": {"content": "blocking reply"}, "eval_count": 3}

    def health(self):
        return ProviderHealth(True, "fake")


@pytest.mark.parametrize("server", [NoStreamProvider()], indirect=True)
def test_provider_without_streaming_falls_back(server):
    base, _ = server
    body = _post_stream(
        f"{base}/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    )
    assert "event: message_start" in body and "event: message_stop" in body
    assert "blocking reply" in body


class FailingStream(StreamingFake):
    def chat_stream(self, payload):
        yield {"message": {"content": "partial answer"}}
        raise ProviderError("connection lost")


@pytest.mark.parametrize("server", [FailingStream(["x"])], indirect=True)
def test_midstream_failure_closes_the_message(server):
    base, _ = server
    body = _post_stream(
        f"{base}/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    )
    # The client is told what happened and the message is terminated properly.
    assert "partial answer" in body
    assert "stream failed" in body
    assert "event: message_stop" in body


# ---------------------------------------------------- clause-split retrieval


def test_multipart_questions_split_into_retrieval_groups():
    from galactica.server import split_question

    groups = split_question(
        "What do I do for a copperhead bite and what must I never do to the wound?"
    )
    assert len(groups) >= 3  # two clauses plus the whole question
    assert any("copperhead bite" in g[0] for g in groups)
    assert any("never do" in g[0] for g in groups)


def test_single_clause_questions_are_left_alone():
    from galactica.server import split_question

    assert split_question("who published Katana Zero") == [["who published Katana Zero"]]


def test_tiny_fragments_do_not_become_groups():
    from galactica.server import split_question

    # "and iOS" is too short to be worth its own retrieval pass.
    groups = split_question("Which publisher brought Katana Zero to Android and iOS")
    assert groups == [["Which publisher brought Katana Zero to Android and iOS"]]


def test_gateway_retrieves_per_clause(seeded, cfg):
    from galactica.server import Gateway

    fake = StreamingFake(["ok"])
    gateway = Gateway(cfg, mode="always", provider=fake)
    gateway._local.conn = seeded
    context, info = gateway.build_context(
        [
            {
                "role": "user",
                "content": "How do I calibrate a click-type torque wrench "
                "and what breaker order does the Krios R7 need?",
            }
        ]
    )
    # Both halves of the question are represented in the assembled context.
    assert "Torque wrench" in context and "Krios" in context
    assert info.retrieved and info.sources > 1
