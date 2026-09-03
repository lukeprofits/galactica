"""Anthropic Messages API gateway: Claude Code talks to this, this talks to Ollama.

Point Claude Code at it with:

    ANTHROPIC_BASE_URL=http://localhost:8787 ANTHROPIC_AUTH_TOKEN=local claude

Each turn, the latest user message is used to retrieve from the local corpus and
the excerpts are injected into the model's prompt for that single call. Only the
model's answer travels back, so the corpus never enters the client's transcript
and never accumulates across turns.
"""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

from .config import Config
from .ingest.chunker import approx_tokens
from .prompts import PROMPT_VERSION, gateway_context, gateway_suffix
from .providers.base import ProviderError
from .providers.ollama import OllamaProvider, strip_reasoning
from .retrieve import search_grouped
from .select import select_context
from .store import open_db

MODES = ("auto", "always", "off")

# Retrieval is pointless for "fix this test" and costs seconds, so the gate looks
# for a knowledge-shaped question rather than agent traffic or code work.
_QUESTION_WORDS = re.compile(
    r"\b(who|what|when|where|which|why|how|whose|whom|explain|define|describe|"
    r"compare|history|meaning|difference)\b",
    re.IGNORECASE,
)
_CODE_MARKERS = re.compile(
    r"(```|\bdef \b|\bclass \b|\bimport \b|\bnpm\b|\bgit\b|\bpytest\b|=>|::|"
    r"\brefactor\b|\bstack ?trace\b|\btraceback\b|/[\w.-]+/[\w.-]+|\.\w{1,4}\b:\d+)",
    re.IGNORECASE,
)
MAX_QUERY_CHARS = 600
# Clause splitting for the gateway, which has no planner to lean on. Cheap
# heuristics only: no model call, so a multi-part question still gets one
# retrieval group per part.
_CLAUSE_SPLIT = re.compile(r"(?<=\?)\s+|\s*;\s*|\s+\band\b\s+|\s+\bplus\b\s+", re.IGNORECASE)
MIN_CLAUSE_WORDS = 4
MAX_CLAUSE_GROUPS = 4


def split_question(text: str) -> list[list[str]]:
    """One retrieval group per clause, plus the whole question as its own group."""
    whole = text[:MAX_QUERY_CHARS]
    clauses = [
        c.strip()
        for c in _CLAUSE_SPLIT.split(whole)
        if c and len(c.split()) >= MIN_CLAUSE_WORDS
    ]
    groups = [[c] for c in clauses[:MAX_CLAUSE_GROUPS]]
    if len(groups) < 2:
        return [[whole]]
    groups.append([whole])
    return groups


# ------------------------------------------------------------------- translation


def _text_of(content: Any) -> str:
    """Flatten an Anthropic content field to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_result":
                parts.append(_text_of(block.get("content")))
        return "\n".join(p for p in parts if p)
    return ""


def latest_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _text_of(message.get("content")).strip()
    return ""


def carries_tool_result(messages: list[dict]) -> bool:
    """True mid-agent-loop: the client is feeding tool output back, not asking."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            return any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            )
        return False
    return False


def should_retrieve(messages: list[dict], mode: str) -> tuple[bool, str]:
    """Decide whether this turn gets corpus context, and say why."""
    if mode == "off":
        return False, "retrieval disabled"
    text = latest_user_text(messages)
    if not text:
        return False, "no user text"
    if mode == "always":
        return True, "mode=always"
    if carries_tool_result(messages):
        return False, "tool result continuation"
    if _CODE_MARKERS.search(text):
        return False, "looks like code work"
    if "?" in text or _QUESTION_WORDS.search(text):
        return True, "knowledge question"
    return False, "not a knowledge question"


def to_ollama_messages(system: Any, messages: list[dict], extra_system: str = "") -> list[dict]:
    """Anthropic messages -> Ollama chat messages, preserving tool traffic."""
    out: list[dict] = []
    system_text = "\n\n".join(p for p in (_text_of(system), extra_system) if p)
    if system_text:
        out.append({"role": "system", "content": system_text})

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                text_parts.append(str(block.get("text") or ""))
            elif kind == "tool_use":
                tool_calls.append(
                    {
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": block.get("input") or {},
                        }
                    }
                )
            elif kind == "tool_result":
                # Ollama expects tool output as its own message.
                out.append(
                    {
                        "role": "tool",
                        "content": _text_of(block.get("content")) or "(no output)",
                    }
                )
            elif kind == "image":
                text_parts.append("(image omitted)")

        if text_parts or tool_calls:
            entry: dict = {"role": role, "content": "\n".join(p for p in text_parts if p)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
    return out


def to_ollama_tools(tools: Any) -> list[dict]:
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object"},
                },
            }
        )
    return out


def _tool_use_id() -> str:
    return "toolu_" + secrets.token_hex(12)


def _message_id() -> str:
    return "msg_" + secrets.token_hex(12)


def to_anthropic_message(ollama: dict, model: str) -> dict:
    """Ollama chat response -> Anthropic Messages response."""
    message = ollama.get("message") or {}
    text = strip_reasoning(str(message.get("content") or ""))
    blocks: list[dict] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"value": arguments}
        blocks.append(
            {
                "type": "tool_use",
                "id": _tool_use_id(),
                "name": function.get("name", ""),
                "input": arguments or {},
            }
        )
    if not blocks:
        # A thinking model can burn the whole output budget on reasoning that
        # gets stripped. Say so rather than handing the client an empty message.
        note = (
            "(no answer returned: the output limit was reached before the model "
            "finished. Raise max_tokens, or set GALACTICA_THINK=0 to stop the "
            "model spending the budget on reasoning.)"
            if ollama.get("done_reason") == "length"
            else "(no answer returned by the model)"
        )
        blocks.append({"type": "text", "text": note})

    if any(b["type"] == "tool_use" for b in blocks):
        stop_reason = "tool_use"
    elif ollama.get("done_reason") == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    return {
        "id": _message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(ollama.get("prompt_eval_count") or 0),
            "output_tokens": int(ollama.get("eval_count") or 0),
        },
    }


def sse_events(message: dict) -> Iterator[tuple[str, dict]]:
    """Frame a finished message as the Anthropic streaming event sequence."""
    head = {k: v for k, v in message.items() if k != "content"}
    head["content"] = []
    head["stop_reason"] = None
    head["usage"] = {"input_tokens": message["usage"]["input_tokens"], "output_tokens": 0}
    yield "message_start", {"type": "message_start", "message": head}

    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            yield "content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            }
            if block["text"]:
                yield "content_block_delta", {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]},
                }
        else:
            yield "content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": block["id"],
                    "name": block["name"],
                    "input": {},
                },
            }
            yield "content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["input"]),
                },
            }
        yield "content_block_stop", {"type": "content_block_stop", "index": index}

    yield "message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": message["stop_reason"],
            "stop_sequence": message["stop_sequence"],
        },
        "usage": {"output_tokens": message["usage"]["output_tokens"]},
    }
    yield "message_stop", {"type": "message_stop"}


# ---------------------------------------------------------------------- gateway


class ReasoningFilter:
    """Removes <think> blocks from a token stream.

    The non-streaming path can strip reasoning with a regex over the whole
    reply. A stream cannot: a tag can be split across chunks, so the filter has
    to hold back anything that might be the start of one.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buffer = ""
        self.inside = False
        self.emitted_any = False

    def feed(self, text: str) -> str:
        self.buffer += text
        out: list[str] = []
        while self.buffer:
            if self.inside:
                end = self.buffer.find(self.CLOSE)
                if end < 0:
                    # Keep only enough to recognise a split closing tag.
                    self.buffer = self.buffer[-len(self.CLOSE) :]
                    break
                self.buffer = self.buffer[end + len(self.CLOSE) :]
                self.inside = False
                continue
            start = self.buffer.find(self.OPEN)
            if start >= 0:
                out.append(self.buffer[:start])
                self.buffer = self.buffer[start + len(self.OPEN) :]
                self.inside = True
                continue
            # No tag in hand: emit all but a possible partial tag at the tail.
            hold = 0
            for size in range(1, min(len(self.OPEN), len(self.buffer)) + 1):
                if self.OPEN.startswith(self.buffer[-size:]):
                    hold = size
            if hold:
                out.append(self.buffer[:-hold])
                self.buffer = self.buffer[-hold:]
            else:
                out.append(self.buffer)
                self.buffer = ""
            break
        text_out = "".join(out)
        if text_out.strip():
            self.emitted_any = True
        return text_out

    def flush(self) -> str:
        rest = "" if self.inside else self.buffer
        self.buffer = ""
        if rest.strip():
            self.emitted_any = True
        return rest


@dataclass
class TurnInfo:
    retrieved: bool
    reason: str
    sources: int
    context_tokens: int
    query: str = ""
    think_retry: bool = False


def _is_starved(raw: dict, message: dict) -> bool:
    """True when reasoning ran to the token limit and left no answer behind."""
    if raw.get("done_reason") != "length":
        return False
    if any(block.get("type") == "tool_use" for block in message.get("content", [])):
        return False
    text = "".join(
        block.get("text", "")
        for block in message.get("content", [])
        if block.get("type") == "text"
    ).strip()
    return not text or text.startswith("(no answer returned")


class Gateway:
    """Retrieval + model call for one request. Holds no conversation state."""

    def __init__(self, cfg: Config, mode: str = "auto", provider=None) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode '{mode}' (expected: {', '.join(MODES)})")
        self.cfg = cfg
        self.mode = mode
        self.provider = provider or OllamaProvider(
            model=cfg.model,
            base_url=cfg.ollama_base_url,
            embed_model=cfg.embed_model,
            num_ctx=cfg.effective_num_ctx(client_reserve=cfg.client_reserve),
        )
        self._local = threading.local()
        self.turns: list[TurnInfo] = []

    def _conn(self) -> sqlite3.Connection:
        # One connection per serving thread; sqlite objects are not shareable.
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = open_db(self.cfg.db_path)
            self._local.conn = conn
        return conn

    def build_context(self, messages: list[dict]) -> tuple[str, TurnInfo]:
        retrieve, reason = should_retrieve(messages, self.mode)
        if not retrieve:
            return "", TurnInfo(False, reason, 0, 0)
        query = latest_user_text(messages)[:MAX_QUERY_CHARS]
        result = search_grouped(self._conn(), split_question(query), top_k=self.cfg.top_k)
        selection = select_context(
            self._conn(),
            result.hits,
            budget=self.cfg.context_budget,
            expand=self.cfg.expand,
            expand_top=self.cfg.expand_top,
            per_doc_fraction=self.cfg.per_doc_fraction,
            max_sources=self.cfg.max_sources,
            diversity_slots=self.cfg.diversity_slots,
        )
        if not selection.sources:
            return "", TurnInfo(False, "nothing retrieved", 0, 0, query)
        block = gateway_context(selection.render()) + gateway_suffix(self.cfg.grounding)
        return block, TurnInfo(
            True, reason, len(selection.sources), selection.used_tokens, query
        )

    def prepare(self, request: dict) -> tuple[dict, TurnInfo]:
        """Retrieve context and build the model payload. No model call yet."""
        messages = request.get("messages") or []
        extra_system, info = self.build_context(messages)

        payload: dict = {
            "model": self.cfg.model,
            "messages": to_ollama_messages(request.get("system"), messages, extra_system),
            "stream": False,
            "options": {"temperature": float(request.get("temperature", 0.0) or 0.0)},
            "think": self.cfg.think,
        }
        if request.get("max_tokens"):
            # The client's max_tokens is a budget for the *answer*. A thinking
            # model also needs room to reason, so reasoning gets its own
            # allowance on top; only the post-reasoning answer is returned.
            answer_budget = int(request["max_tokens"])
            payload["options"]["num_predict"] = (
                answer_budget + self.cfg.think_reserve if self.cfg.think else answer_budget
            )
        if request.get("stop_sequences"):
            payload["options"]["stop"] = list(request["stop_sequences"])
        tools = to_ollama_tools(request.get("tools"))
        if tools:
            payload["tools"] = tools
        return payload, info

    def complete(self, request: dict) -> tuple[dict, TurnInfo]:
        payload, info = self.prepare(request)
        self.turns.append(info)
        raw = self.provider.chat(payload)
        message = to_anthropic_message(raw, self.cfg.model)

        # If reasoning still consumed the whole allowance, retry once without it
        # rather than hand back an empty answer. Reasoning is preferred, not
        # mandatory.
        if self.cfg.think and _is_starved(raw, message):
            message = to_anthropic_message(
                self.provider.chat(dict(payload, think=False)), self.cfg.model
            )
            info.think_retry = True
        return message, info


def stream_events(gateway: "Gateway", request: dict) -> Iterator[tuple[str, dict]]:
    """Run a request against the model and yield Anthropic SSE events live."""
    payload, info = gateway.prepare(request)
    gateway.turns.append(info)
    message_id = _message_id()
    model = gateway.cfg.model

    head = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    yield "message_start", {"type": "message_start", "message": head}
    yield "content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }

    reasoning = ReasoningFilter()
    tool_calls: list[dict] = []
    output_tokens = 0
    done_reason = None

    for chunk in gateway.provider.chat_stream(payload):
        message = chunk.get("message") or {}
        piece = reasoning.feed(str(message.get("content") or ""))
        if piece:
            yield "content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": piece},
            }
        tool_calls.extend(message.get("tool_calls") or [])
        if chunk.get("done"):
            output_tokens = int(chunk.get("eval_count") or 0)
            done_reason = chunk.get("done_reason")

    tail = reasoning.flush()
    if tail:
        yield "content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": tail},
        }
    if not reasoning.emitted_any and not tool_calls:
        # Reasoning consumed the whole allowance. Say so instead of streaming
        # an empty message; a mid-stream retry cannot be taken back.
        note = (
            "(no answer returned: the output limit was reached before the model "
            "finished. Raise max_tokens, or set GALACTICA_THINK=0.)"
            if done_reason == "length"
            else "(no answer returned by the model)"
        )
        yield "content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": note},
        }
    yield "content_block_stop", {"type": "content_block_stop", "index": 0}

    index = 1
    for call in tool_calls:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"value": arguments}
        yield "content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": _tool_use_id(),
                "name": function.get("name", ""),
                "input": {},
            },
        }
        yield "content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(arguments or {})},
        }
        yield "content_block_stop", {"type": "content_block_stop", "index": index}
        index += 1

    if tool_calls:
        stop_reason = "tool_use"
    elif done_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    yield "message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    yield "message_stop", {"type": "message_stop"}


def count_tokens(request: dict) -> int:
    """Cheap estimate for /v1/messages/count_tokens (no tokenizer dependency)."""
    text = _text_of(request.get("system"))
    for message in request.get("messages") or []:
        text += "\n" + _text_of(message.get("content"))
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                text += "\n" + json.dumps(block.get("input") or {})
    for tool in request.get("tools") or []:
        if isinstance(tool, dict):
            text += "\n" + json.dumps(tool.get("input_schema") or {})
    return approx_tokens(text)


# ------------------------------------------------------------------ http plumbing


class _Handler(BaseHTTPRequestHandler):
    server_version = "galactica"
    gateway: Gateway
    verbose: bool = True

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        if self.verbose:
            super().log_message(fmt, *args)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _begin_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _write_event(self, event: str, data: dict) -> None:
        self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def _send_error_payload(self, status: int, message: str) -> None:
        self._send_json(
            status, {"type": "error", "error": {"type": "api_error", "message": message}}
        )

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._send_json(200, {"status": "ok", "model": self.gateway.cfg.model})
        else:
            self._send_error_payload(404, f"no route {self.path}")

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?")[0].rstrip("/")
        request = self._read_json()
        if route == "/v1/messages/count_tokens":
            self._send_json(200, {"input_tokens": count_tokens(request)})
            return
        if route != "/v1/messages":
            self._send_error_payload(404, f"no route {self.path}")
            return

        if request.get("stream"):
            self._stream(request)
            return

        try:
            message, info = self.gateway.complete(request)
        except ProviderError as exc:
            self._send_error_payload(502, str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            self._send_error_payload(500, f"{type(exc).__name__}: {exc}")
            return

        if self.verbose:
            note = (
                f"corpus: {info.sources} sources, {info.context_tokens} tokens"
                if info.retrieved
                else f"corpus: skipped ({info.reason})"
            )
            if info.think_retry:
                note += " | reasoning hit the token limit, retried without it"
            print(f"  [galactica] {note}", flush=True)

        self._send_json(200, message)

    def _stream(self, request: dict) -> None:
        """Stream the answer token by token as it is generated."""
        # A provider without streaming support still has to work: fall back to
        # one blocking call re-framed as the same event sequence.
        if not hasattr(self.gateway.provider, "chat_stream"):
            try:
                message, _ = self.gateway.complete(request)
            except ProviderError as exc:
                self._send_error_payload(502, str(exc))
                return
            self._begin_stream()
            for event, data in sse_events(message):
                self._write_event(event, data)
            return

        started = False
        try:
            for event, data in stream_events(self.gateway, request):
                if not started:
                    self._begin_stream()
                    started = True
                self._write_event(event, data)
        except (ProviderError, Exception) as exc:
            if not started:
                status = 502 if isinstance(exc, ProviderError) else 500
                self._send_error_payload(status, f"{type(exc).__name__}: {exc}")
                return
            # Mid-stream failure: close the message properly so the client is
            # told what happened instead of being left waiting on a truncated
            # event stream.
            print(f"  [galactica] stream failed: {type(exc).__name__}: {exc}", flush=True)
            self._write_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": f"\n\n[stream failed: {exc}]"},
                },
            )
            self._write_event("content_block_stop", {"type": "content_block_stop", "index": 0})
            self._write_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                },
            )
            self._write_event("message_stop", {"type": "message_stop"})
            return
        if self.verbose and self.gateway.turns:
            info = self.gateway.turns[-1]
            note = (
                f"corpus: {info.sources} sources, {info.context_tokens} tokens"
                if info.retrieved
                else f"corpus: skipped ({info.reason})"
            )
            print(f"  [galactica] {note} | streamed", flush=True)


def build_server(
    cfg: Config, *, host: str = "127.0.0.1", port: int = 8787, mode: str = "auto",
    provider=None, verbose: bool = True,
) -> ThreadingHTTPServer:
    gateway = Gateway(cfg, mode=mode, provider=provider)
    handler = type("_BoundHandler", (_Handler,), {"gateway": gateway, "verbose": verbose})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.gateway = gateway  # type: ignore[attr-defined]
    return httpd


def serve(
    cfg: Config, *, host: str = "127.0.0.1", port: int = 8787, mode: str = "auto"
) -> None:
    httpd = build_server(cfg, host=host, port=port, mode=mode)
    print(f"galactica gateway on http://{host}:{port}")
    print(f"  model      {cfg.model} via {cfg.ollama_base_url}")
    print(f"  corpus     {cfg.db_path}")
    print(f"  retrieval  mode={mode}, budget={cfg.context_budget} tokens, prompts={PROMPT_VERSION}")
    reasoning = (
        f"on (+{cfg.think_reserve} token reserve)"
        if cfg.think
        else "off (GALACTICA_THINK=1 to enable)"
    )
    print(f"  reasoning  {reasoning}")
    print(f"  grounding  {cfg.grounding}")
    print("\nPoint Claude Code at it:")
    print(f"  ANTHROPIC_BASE_URL=http://{host}:{port} ANTHROPIC_AUTH_TOKEN=local claude")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        time.sleep(0)
