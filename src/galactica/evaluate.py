"""A/B harness: does the external cortex actually lift this model? Uplift, not scores."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import Config
from .pipeline import Answer, ask_baseline, ask_cortex
from .providers.base import LLMProvider, ProviderError

from .retrieve import search
from .select import select_context

DEFAULT_RUNS_DIR = Path("eval/runs")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_GAP_OR_MISSING = re.compile(r"^\s*(GAP|MISSING)\s*:", re.IGNORECASE)


@dataclass
class EvalCase:
    question: str
    expect_keywords: list[str] = field(default_factory=list)
    expect_docs: list[str] = field(default_factory=list)
    answerable: bool = True
    id: str | None = None
    note: str | None = None


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rec = json.loads(line)
        cases.append(
            EvalCase(
                question=rec["question"],
                expect_keywords=[str(k) for k in rec.get("expect_keywords", [])],
                expect_docs=[str(d) for d in rec.get("expect_docs", [])],
                answerable=bool(rec.get("answerable", True)),
                id=rec.get("id"),
                note=rec.get("note"),
            )
        )
    return cases


# ------------------------------------------------------------------------ metrics


def keyword_coverage(text: str, keywords: Sequence[str]) -> float | None:
    if not keywords:
        return None
    low = (text or "").lower()
    hit = sum(1 for k in keywords if k.lower() in low)
    return hit / len(keywords)


def retrieval_hit(answer_sources: Sequence[dict], expect_docs: Sequence[str]) -> float | None:
    """Did retrieval surface material that can answer the question?

    `expect_docs` is a list of ACCEPTABLE sources, not the one article a fact was
    sampled from. Scoring it as a single expected title punished models that
    found the dedicated article ("Moscow Cathedral Mosque") instead of the list
    page the fact was copied out of ("List of mosques in Russia") -- which is
    better retrieval, scored as failure. When adding cases, list every source
    that would genuinely answer the question.
    """
    if not expect_docs:
        return None
    haystack = " ".join(
        " ".join(
            str(v)
            for k, v in src.items()
            if k in ("title", "doc_id", "uri", "heading_path") and v
        )
        + " "
        + " ".join(src.get("chunk_ids", []))
        for src in answer_sources
    ).lower()
    return 1.0 if any(d.lower() in haystack for d in expect_docs) else 0.0


def unsupported_claim_rate(answer: Answer) -> float | None:
    """Fraction of substantive sentences carrying no [Sn] citation (cortex only)."""
    if answer.mode != "cortex":
        return None
    body = [
        s.strip()
        for s in _SENTENCE.split(answer.text)
        if s.strip() and not _GAP_OR_MISSING.match(s.strip())
    ]
    if not body or answer.declined:
        return None
    uncited = sum(1 for s in body if "[S" not in s)
    return uncited / len(body)


def score_answer(answer: Answer, case: EvalCase) -> dict:
    sources = [asdict(s) for s in answer.sources]
    if not answer.text and answer.warnings:
        # A failed call is not a refusal and must not be scored as one.
        return {"failed": 1.0, "latency_s": answer.latency_s}
    metrics: dict = {
        "keyword_coverage": keyword_coverage(answer.text, case.expect_keywords),
        "retrieval_hit": retrieval_hit(sources, case.expect_docs) if answer.mode == "cortex" else None,
        "valid_citations": len(answer.citations) if answer.mode == "cortex" else None,
        "cited_any": (1.0 if answer.citations else 0.0) if answer.mode == "cortex" else None,
        "fabricated_citations": len(answer.invalid_citations),
        "refusal_correct": 1.0 if answer.declined == (not case.answerable) else 0.0,
        # Augmented grounding answers beyond the corpus; the honesty test is
        # whether it labels those parts rather than whether it refuses.
        "labelled_uncited": (1.0 if answer.uncited else 0.0) if answer.mode == "cortex" else None,
        "unsupported_claim_rate": unsupported_claim_rate(answer),
        "latency_s": answer.latency_s,
        "context_tokens": answer.context_tokens if answer.mode == "cortex" else None,
    }
    return metrics


# ---------------------------------------------------------------------- execution


@dataclass
class CaseResult:
    case: EvalCase
    mode: str
    metrics: dict
    answer: Answer | None = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    source_labels: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)

    def row(self, cfg: Config) -> dict:
        return {
            "id": self.case.id,
            "question": self.case.question,
            "mode": self.mode,
            "answerable": self.case.answerable,
            "answer": self.answer.text if self.answer else None,
            "sources": [asdict(s) for s in self.answer.sources] if self.answer else self.sources,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "metrics": self.metrics,
            "model": cfg.model,
            "prompt_version": self.answer.prompt_version if self.answer else None,
            "context_budget": cfg.context_budget,
            "top_k": cfg.top_k,
        }


def run_retrieval_only(
    conn: sqlite3.Connection, cfg: Config, cases: Sequence[EvalCase], *, hybrid: bool = False
) -> list[CaseResult]:
    """Zero LLM calls: does the corpus surface the expected material at all?"""
    out: list[CaseResult] = []
    for case in cases:
        result = search(conn, [case.question], top_k=cfg.top_k, hybrid=hybrid)
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
        sources = [
            {
                "title": s.anchor.title,
                "doc_id": s.doc_id,
                "uri": s.anchor.uri or "",
                "heading_path": s.anchor.heading_path,
                "chunk_ids": s.chunk_ids,
            }
            for s in selection.sources
        ]
        out.append(
            CaseResult(
                case=case,
                mode="retrieval",
                metrics={
                    "retrieval_hit": retrieval_hit(sources, case.expect_docs),
                    "sources_selected": len(sources),
                    "context_tokens": selection.used_tokens,
                },
                retrieved_chunk_ids=[h.chunk_id for h in result.hits],
                source_labels=[s.label for s in selection.sources],
                sources=sources,
            )
        )
    return out


def run_cases(
    conn: sqlite3.Connection,
    provider: LLMProvider,
    cfg: Config,
    cases: Sequence[EvalCase],
    *,
    modes: Sequence[str] = ("baseline", "cortex"),
    hybrid: bool = False,
    use_plan: bool = True,
    on_case=None,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        for mode in modes:
            try:
                if mode == "baseline":
                    answer = ask_baseline(provider, cfg, case.question)
                else:
                    answer = ask_cortex(
                        conn, provider, cfg, case.question, hybrid=hybrid, use_plan=use_plan
                    )
            except ProviderError as exc:
                # Record the failure and keep going: losing every completed case
                # to one slow question wastes the whole run.
                answer = Answer(
                    mode=mode,
                    question=case.question,
                    text="",
                    model=cfg.model,
                    warnings=[f"failed: {exc}"],
                )
            results.append(
                CaseResult(
                    case=case,
                    mode=mode,
                    metrics=score_answer(answer, case),
                    answer=answer,
                    retrieved_chunk_ids=answer.retrieved_chunk_ids,
                    source_labels=[s.label for s in answer.sources],
                )
            )
            if on_case:
                on_case(results[-1])
    return results


# --------------------------------------------------------------------- aggregation

AGGREGATE_KEYS = (
    "keyword_coverage",
    "labelled_uncited",
    "retrieval_hit",
    "cited_any",
    "fabricated_citations",
    "refusal_correct",
    "unsupported_claim_rate",
    "latency_s",
    "context_tokens",
)


def aggregate(results: Sequence[CaseResult]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    modes = list(dict.fromkeys(r.mode for r in results))
    for mode in modes:
        rows = [r for r in results if r.mode == mode]
        agg: dict[str, float | None] = {"cases": float(len(rows))}
        for key in AGGREGATE_KEYS:
            values = [
                r.metrics[key] for r in rows if r.metrics.get(key) is not None
            ]
            agg[key] = round(sum(values) / len(values), 4) if values else None
        # Refusal accuracy split: the corpus-gap behaviour is the interesting half.
        for label, answerable in (("answered_correctly_flagged", True), ("refused_when_absent", False)):
            subset = [
                r.metrics["refusal_correct"]
                for r in rows
                if r.case.answerable is answerable and r.metrics.get("refusal_correct") is not None
            ]
            agg[label] = round(sum(subset) / len(subset), 4) if subset else None
        out[mode] = agg
    return out


def uplift(agg: dict[str, dict[str, float | None]]) -> dict[str, float | None]:
    """cortex minus baseline, for the metrics both arms produce."""
    base, cortex = agg.get("baseline"), agg.get("cortex")
    if not base or not cortex:
        return {}
    delta: dict[str, float | None] = {}
    for key in AGGREGATE_KEYS + ("answered_correctly_flagged", "refused_when_absent"):
        b, c = base.get(key), cortex.get(key)
        delta[key] = round(c - b, 4) if isinstance(b, (int, float)) and isinstance(c, (int, float)) else None
    return delta


def save_run(
    results: Sequence[CaseResult],
    cfg: Config,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    extra: dict | None = None,
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = runs_dir / f"{stamp}.jsonl"
    agg = aggregate(results)
    header = {
        "record": "run_header",
        "timestamp": stamp,
        "model": cfg.model,
        "provider": cfg.provider,
        "context_budget": cfg.context_budget,
        "top_k": cfg.top_k,
        "aggregate": agg,
        "uplift": uplift(agg),
        **(extra or {}),
    }
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for result in results:
            fh.write(json.dumps({"record": "case", **result.row(cfg)}) + "\n")
    return path
