"""The navigator loop and its control: baseline (no cortex) vs cortex (retrieval)."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Sequence

from .config import Config
from .prompts import (
    BASELINE_SYSTEM,
    PLANNER_SYSTEM,
    PROMPT_VERSION,
    baseline_user,
    cortex_system,
    cortex_user,
    planner_user,
)
from .providers.base import LLMProvider, ProviderError
from .retrieve import search_grouped
from .select import Selection, select_context

MODES = ("cortex", "baseline", "both")
CITATION = re.compile(r"\[S(\d+)\]")
MISSING_LINE = re.compile(r"^\s*MISSING:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
UNCITED_LINE = re.compile(r"^\s*UNCITED:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
GAP_LINE = re.compile(r"^\s*GAP:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
NO_INFO = "the corpus does not contain enough information"
NO_INFO_BASELINE = "i do not have enough information"

# Augmented grounding never uses the strict refusal sentence, so recognising a
# refusal by that phrase alone scored four correct refusals as failures. These
# are the shapes an honest non-answer actually takes. It stays a proxy: a
# refusal is a judgement, and no pattern list captures every phrasing.
REFUSAL_PATTERNS = re.compile(
    r"("
    r"corpus does not contain enough information"
    r"|do(?: not|n't) have enough information"
    r"|(?:has|have) not (?:yet )?(?:taken place|been played|been held|occurred|happened)"
    r"|(?:has|have) not (?:yet )?(?:been )?(?:announced|released|introduced)"
    r"|do(?: not|n't) have access to (?:real-?time|market|current)"
    r"|no (?:final )?(?:medal table|result|winner|data|record|such model)\s+(?:exists|is available)"
    r"|(?:is|are) (?:currently )?unknown"
    r"|cannot (?:answer|be answered|determine)"
    r"|unable to (?:answer|determine|provide)"
    r"|no information (?:is )?available"
    r"|not (?:mentioned|documented|present|found|listed|specified|covered) in the"
    r"|no (?:sources?|excerpts?) (?:provide|contain|mention)"
    r"|(?:does|do) not (?:appear|exist) in the (?:corpus|sources|material)"
    r"|no information about .{0,60} exists"
    r")",
    re.IGNORECASE,
)


def looks_like_refusal(text: str) -> bool:
    """Did the answer decline to assert a fact, in any of its usual phrasings?"""
    return bool(REFUSAL_PATTERNS.search(text or ""))
MAX_MISSING = 3


# A planner call costs a full model round trip. It earns that on a multi-part
# question and wastes it on one that is already keyword-shaped.
_MULTIPART = re.compile(r"(\band\b|\bthen\b|;|\?.+\?|\balso\b|\bversus\b|\bvs\b|,)", re.IGNORECASE)
PLAN_WORD_THRESHOLD = 12


def needs_planning(question: str) -> bool:
    """Would planning add anything for this question?"""
    words = question.split()
    if len(words) > PLAN_WORD_THRESHOLD:
        return True
    return bool(_MULTIPART.search(question))


@dataclass
class Plan:
    intent: str = ""
    sub_questions: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    needed_facts: list[str] = field(default_factory=list)
    fallback: bool = False


@dataclass
class SourceRef:
    label: str
    chunk_ids: list[str]
    doc_id: str
    title: str
    heading_path: str
    source_name: str
    source_version: str | None
    license: str | None
    uri: str | None


@dataclass
class Answer:
    mode: str
    question: str
    text: str
    model: str
    prompt_version: str = PROMPT_VERSION
    sources: list[SourceRef] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    missing_queries: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    # Augmented grounding: what the model answered from its own knowledge.
    uncited: list[str] = field(default_factory=list)
    declined: bool = False
    plan: Plan | None = None
    hops_used: int = 0
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)
    context_tokens: int = 0
    context_budget: int = 0
    context: str = ""
    latency_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["plan"] = asdict(self.plan) if self.plan else None
        return data


# ------------------------------------------------------------------- plan parsing


def _balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _as_str_list(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def parse_plan(raw: str, question: str) -> Plan:
    """Tolerant parse. A planner failure must never cost us the answer."""
    blob = _balanced_json(raw or "")
    data = None
    if blob:
        for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
    if not isinstance(data, dict):
        return Plan(intent="lookup", queries=[question], fallback=True)
    queries = _as_str_list(data.get("queries"))
    sub = _as_str_list(data.get("sub_questions"))
    if not queries:
        queries = sub or [question]
    return Plan(
        intent=str(data.get("intent") or "").strip(),
        sub_questions=sub,
        queries=queries[:5],
        needed_facts=_as_str_list(data.get("needed_facts")),
        fallback=False,
    )


def make_plan(provider: LLMProvider, cfg: Config, question: str) -> tuple[Plan, list[str]]:
    try:
        raw = provider.complete(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": planner_user(question)},
            ],
            temperature=cfg.temperature,
            json_mode=True,
        )
    except ProviderError as exc:
        return Plan(intent="lookup", queries=[question], fallback=True), [f"planner failed: {exc}"]
    plan = parse_plan(raw, question)
    warnings = ["planner output unparseable; fell back to the raw question"] if plan.fallback else []
    if question not in plan.queries:
        plan.queries = [*plan.queries, question][:6]
    return plan, warnings


# ---------------------------------------------------------------------- answering


def _source_refs(selection: Selection) -> list[SourceRef]:
    refs = []
    for src in selection.sources:
        a = src.anchor
        refs.append(
            SourceRef(
                label=src.label,
                chunk_ids=src.chunk_ids,
                doc_id=a.doc_id,
                title=a.title,
                heading_path=a.heading_path,
                source_name=a.source_name,
                source_version=a.source_version,
                license=a.license,
                uri=a.uri,
            )
        )
    return refs


def validate_citations(text: str, labels: Sequence[str]) -> tuple[list[str], list[str]]:
    """Returns (valid, invalid) cited labels in first-appearance order."""
    known = set(labels)
    seen: list[str] = []
    for num in CITATION.findall(text or ""):
        label = f"S{num}"
        if label not in seen:
            seen.append(label)
    valid = [s for s in seen if s in known]
    invalid = [s for s in seen if s not in known]
    return valid, invalid


def ask_baseline(provider: LLMProvider, cfg: Config, question: str) -> Answer:
    started = time.monotonic()
    text = provider.complete(
        [
            {"role": "system", "content": BASELINE_SYSTEM},
            {"role": "user", "content": baseline_user(question)},
        ],
        temperature=cfg.temperature,
        max_tokens=answer_budget(cfg),
    )
    # Baseline has no sources, so every source label it emits is fabricated.
    _, invalid = validate_citations(text, [])
    return Answer(
        mode="baseline",
        question=question,
        text=text.strip(),
        model=cfg.model,
        invalid_citations=invalid,
        declined=looks_like_refusal(text),
        latency_s=round(time.monotonic() - started, 3),
    )


def ask_cortex(
    conn: sqlite3.Connection,
    provider: LLMProvider,
    cfg: Config,
    question: str,
    *,
    hybrid: bool = False,
    use_plan: bool = True,
) -> Answer:
    started = time.monotonic()
    warnings: list[str] = []
    plan: Plan | None = None
    if use_plan and needs_planning(question):
        plan, plan_warnings = make_plan(provider, cfg, question)
        warnings.extend(plan_warnings)
    elif use_plan:
        warnings.append("planner skipped: question is already a usable query")

    groups = query_groups(plan, question)
    selection, retrieved, hop_warnings = _retrieve_and_select(
        conn, provider, cfg, groups, hybrid=hybrid
    )
    warnings.extend(hop_warnings)

    answer_text = _answer_from(provider, cfg, question, selection)
    if not answer_text and cfg.think:
        # Reasoning consumed the whole output cap. Retry once without it rather
        # than return an empty answer, as the gateway does.
        answer_text = _answer_from(provider, cfg.override(think=False), question, selection)
        warnings.append("reasoning exhausted the output budget; retried without it")
    hops_used = 1
    missing = [m.strip() for m in MISSING_LINE.findall(answer_text)][:MAX_MISSING]

    # Second hop: re-search what the model said it was missing, then re-answer once.
    if missing and cfg.hops > 1:
        extra_selection, extra_retrieved, more_warnings = _retrieve_and_select(
            conn, provider, cfg, query_groups(plan, question, extra=missing), hybrid=hybrid
        )
        warnings.extend(more_warnings)
        if extra_selection.sources:
            selection = extra_selection
            retrieved = extra_retrieved
            answer_text = _answer_from(provider, cfg, question, selection)
            hops_used = 2
            missing = [m.strip() for m in MISSING_LINE.findall(answer_text)][:MAX_MISSING]

    refs = _source_refs(selection)
    valid, invalid = validate_citations(answer_text, [r.label for r in refs])
    if invalid:
        warnings.append(f"answer cited unknown sources: {', '.join(invalid)}")
    if retrieved and not selection.sources:
        warnings.append(
            f"context budget ({cfg.context_budget} tokens) too small for any retrieved chunk"
        )

    clean = UNCITED_LINE.sub("", MISSING_LINE.sub("", answer_text)).strip()
    return Answer(
        mode="cortex",
        question=question,
        text=clean,
        model=cfg.model,
        sources=refs,
        citations=valid,
        invalid_citations=invalid,
        missing_queries=missing,
        gaps=[g.strip() for g in GAP_LINE.findall(answer_text)],
        uncited=[u.strip() for u in UNCITED_LINE.findall(answer_text)],
        declined=looks_like_refusal(answer_text),
        plan=plan,
        hops_used=hops_used,
        retrieved_chunk_ids=retrieved,
        dropped=selection.dropped,
        context_tokens=selection.used_tokens,
        context_budget=cfg.context_budget,
        context=selection.render(),
        latency_s=round(time.monotonic() - started, 3),
        warnings=warnings,
    )


def query_groups(plan: Plan | None, question: str, extra: Sequence[str] = ()) -> list[list[str]]:
    """One retrieval group per sub-question, so no part goes unsearched."""
    if plan is None or not plan.sub_questions:
        return [[question, *plan.queries] if plan else [question], *([list(extra)] if extra else [])]
    groups = [[sub] for sub in plan.sub_questions]
    # The planner's own keyword queries and the raw question form one more group.
    groups.append([question, *plan.queries])
    if extra:
        groups.append(list(extra))
    return groups


def _retrieve_and_select(
    conn: sqlite3.Connection,
    provider: LLMProvider,
    cfg: Config,
    groups: Sequence[Sequence[str]],
    *,
    hybrid: bool,
) -> tuple[Selection, list[str], list[str]]:
    result = search_grouped(
        conn,
        groups,
        top_k=cfg.top_k,
        hybrid=hybrid,
        provider=provider,
        embed_model=cfg.embed_model,
    )
    selection = select_context(
        conn,
        result.hits,
        budget=cfg.context_budget,
        expand=cfg.expand,
        expand_top=cfg.expand_top,
        per_doc_fraction=cfg.per_doc_fraction,
        max_sources=cfg.max_sources,
        diversity_slots=cfg.diversity_slots,
    )
    return selection, [h.chunk_id for h in result.hits], list(result.warnings)


def answer_budget(cfg: Config) -> int:
    """Output cap for one answer, with room for reasoning on top."""
    return cfg.max_answer_tokens + (cfg.think_reserve if cfg.think else 0)


def _answer_from(
    provider: LLMProvider, cfg: Config, question: str, selection: Selection
) -> str:
    return provider.complete(
        [
            {"role": "system", "content": cortex_system(cfg.grounding)},
            {"role": "user", "content": cortex_user(question, selection.render())},
        ],
        temperature=cfg.temperature,
        max_tokens=answer_budget(cfg),
        think=cfg.think,
    ).strip()


def ask(
    conn: sqlite3.Connection,
    provider: LLMProvider,
    cfg: Config,
    question: str,
    *,
    mode: str | None = None,
    hybrid: bool = False,
    use_plan: bool = True,
) -> dict[str, Answer]:
    """Run the requested arm(s). Returns {mode_name: Answer} for uplift comparison."""
    mode = mode or cfg.mode
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}' (expected: {', '.join(MODES)})")
    out: dict[str, Answer] = {}
    if mode in ("baseline", "both"):
        out["baseline"] = ask_baseline(provider, cfg, question)
    if mode in ("cortex", "both"):
        out["cortex"] = ask_cortex(conn, provider, cfg, question, hybrid=hybrid, use_plan=use_plan)
    return out
