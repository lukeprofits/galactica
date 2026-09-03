"""galactica CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Config, read_config_file, write_config_file
from .evaluate import (
    DEFAULT_RUNS_DIR,
    aggregate,
    load_cases,
    run_cases,
    run_retrieval_only,
    save_run,
    uplift,
)
from .ingest import LoaderError, MappingError, PROFILES, ingest_path
from .pipeline import MODES, ask
from .providers import ProviderError, build_provider
from .hardware import detect as detect_hardware
from .registry import context_budget_for, models as model_registry, recommend_model, source_named, sources as source_registry
from .retrieve import search
from .server import MODES as SERVE_MODES, serve
from . import setup as setup_mod
from .store import (
    counts_are_complete,
    list_sources,
    open_db,
    refresh_document_counts,
    stats,
)

RAW_DIRS = {
    "grokipedia": "Hugging Face Grokipedia dump: *.jsonl, *.jsonl.gz or *.parquet files.\n"
    "Ingest with:\n"
    "  galactica ingest corpus/raw/grokipedia --profile grokipedia --max-documents 5000\n",
    "wikipedia": "Wikipedia XML dump: enwiki-YYYYMMDD-pages-articles.xml.bz2 (no need to extract).\n"
    "Ingest with:\n"
    "  galactica ingest corpus/raw/wikipedia --profile wikipedia --max-documents 5000\n",
}


# ----------------------------------------------------------------------- plumbing


def _shutil_which(command: str) -> str | None:
    import shutil

    return shutil.which(command)


def _nudge(cfg: Config) -> None:
    """One line, on any command, while setup is incomplete."""
    try:
        step = setup_mod.check(cfg).next_step()
    except Exception:  # pragma: no cover - never let a hint break a command
        return
    if step:
        print(f"note: {step}", file=sys.stderr)


def cmd_setup(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    hardware = detect_hardware()
    print(f"hardware   {hardware.describe()}")
    print(f"usable     {hardware.usable_gb:.0f} GB for model weights and context\n")

    if not setup_mod.ollama_running(cfg):
        import shutil as _shutil

        if not _shutil.which("ollama") and not setup_mod.install_ollama(args.yes):
            print("\nerror: Ollama is required to run a local model", file=sys.stderr)
            return 2
        if not setup_mod.start_ollama(cfg):
            print(f"\nerror: ollama installed but not reachable at {cfg.ollama_base_url}",
                  file=sys.stderr)
            print("Start it (`ollama serve`) and run galactica setup again", file=sys.stderr)
            return 2

    have = setup_mod.installed_models(cfg)
    chosen = args.model or None
    if not chosen:
        recommended = recommend_model(hardware, installed=have)
        if recommended is None:
            print("error: no known model fits this machine", file=sys.stderr)
            return 2
        budget = context_budget_for(hardware, recommended)
        print(f"model      {recommended.tag} ({recommended.weights_gb:.1f} GB weights)")
        if recommended.notes:
            print(f"           {recommended.notes}")
        print(f"budget     {budget} context tokens on this hardware")
        if recommended.tag in have:
            print("           already installed")
        if not setup_mod.ask_yes("\nUse this model?", True, args.yes):
            print("Pick one with: galactica setup --model <tag>   (see: galactica models)")
            return 0
        chosen = recommended.tag
        settings = {"model": chosen, "context_budget": budget}
    else:
        settings = {"model": chosen}

    if chosen not in have and not setup_mod.pull_model(chosen):
        print("error: could not pull the model", file=sys.stderr)
        return 2

    # Corpus
    source = source_named(args.source) if args.source else next(
        (s for s in source_registry() if s.default), None
    )
    if source:
        drop = Path("corpus/raw") / source.name
        available = setup_mod.free_gb(drop)
        limit, explanation = setup_mod.plan_corpus(source, available)
        print(f"\ncorpus     {source.title}")
        print(f"           {source.download_gb:.0f} GB download, {available:.0f} GB free here")
        print(f"           plan: {explanation}")
        print(f"           licence: {source.license}")
        # No prompt: the corpus is the point of the tool, and anyone running
        # setup wants a working install. Only disk space and the absence of a
        # terminal can stop it.
        if limit == 0:
            print("           skipping: free up space, then run galactica setup again")
        elif not (setup_mod.interactive() or args.yes):
            print("           skipping: no terminal attached (use --yes for unattended installs)")
        else:
            print()
            path = setup_mod.fetch_source(source, drop)
            if path:
                conn = open_db(cfg.db_path)
                report = ingest_path(
                    conn,
                    path,
                    profile_name=source.profile,
                    source=source.name,
                    license=source.license,
                    sample=limit,
                    progress=lambda n, title: print(f"  ... {n} documents ({title[:50]})"),
                )
                conn.close()
                print(f"  ingested {report.documents_written} documents, "
                      f"{report.chunks_written} chunks")
                settings["data_dir"] = str(cfg.data_dir.resolve())

    setup_mod.offer_claude_code(args.yes)

    written = write_config_file({**read_config_file(), **settings})
    print(f"\nsaved      {written}")
    state = setup_mod.check(_cfg(args))
    if state.ready:
        print('ready      yes — try: galactica ask "..."')
        if _shutil_which("claude"):
            print('           or use it inside Claude Code: claude-lookup')
    else:
        print("ready      " + (state.next_step() or ""))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    hardware = detect_hardware()
    have = setup_mod.installed_models(cfg)
    rows = []
    for option in model_registry():
        rows.append({
            "tag": option.tag,
            "weights_gb": option.weights_gb,
            "needs_gb": option.min_memory_gb,
            "fits": option.fits(hardware),
            "installed": option.tag in have,
            "context_budget": context_budget_for(hardware, option),
            "notes": option.notes,
        })
    if args.json:
        _emit(args, {"hardware": hardware.describe(), "models": rows})
        return 0
    print(f"{hardware.describe()} — {hardware.usable_gb:.0f} GB usable\n")
    print(f"{'model':20}{'weights':>9}{'needs':>7}{'fits':>6}{'have':>6}{'budget':>8}")
    for row in rows:
        print(
            f"{row['tag']:20}{row['weights_gb']:>8.1f}G{row['needs_gb']:>6.0f}G"
            f"{'yes' if row['fits'] else 'no':>6}{'yes' if row['installed'] else '-':>6}"
            f"{row['context_budget'] if row['fits'] else 0:>8}"
        )
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    if not args.name:
        rows = [
            {
                "name": s.name,
                "title": s.title,
                "documents": s.documents,
                "download_gb": s.download_gb,
                "db_gb": s.db_gb,
                "license": s.license,
            }
            for s in source_registry()
        ]
        if args.json:
            _emit(args, {"sources": rows})
            return 0
        print("available corpora:\n")
        for row in rows:
            print(f"  {row['name']:12} {row['title']}")
            print(f"  {'':12} {row['documents']:,} docs · {row['download_gb']:.0f} GB download "
                  f"· ~{row['db_gb']:.0f} GB indexed")
            print(f"  {'':12} {row['license']}\n")
        print("fetch one with: galactica fetch <name>")
        return 0

    source = source_named(args.name)
    drop = Path("corpus/raw") / source.name
    limit = args.max_documents or args.sample
    if limit is None:
        limit, explanation = setup_mod.plan_corpus(source, setup_mod.free_gb(drop))
        print(f"plan: {explanation}")
        if limit == 0:
            return 2
    path = setup_mod.fetch_source(source, drop)
    if path is None:
        return 2
    conn = open_db(cfg.db_path)
    report = ingest_path(
        conn,
        path,
        profile_name=source.profile,
        source=source.name,
        license=source.license,
        sample=limit if args.sample or limit else None,
        max_documents=args.max_documents,
        progress=None if args.json else (lambda n, t: print(f"  ... {n} documents ({t[:50]})")),
    )
    conn.close()
    if args.json:
        _emit(args, report.__dict__)
    else:
        print(f"ingested {report.documents_written} documents, {report.chunks_written} chunks")
    return 0


def _cfg(args: argparse.Namespace) -> Config:
    cfg = Config.from_env().override(
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        embed_model=getattr(args, "embed_model", None),
        data_dir=Path(args.data_dir) if getattr(args, "data_dir", None) else None,
        context_budget=getattr(args, "budget", None),
        top_k=getattr(args, "top_k", None),
        max_sources=getattr(args, "max_sources", None),
        grounding=getattr(args, "grounding", None),
        hops=getattr(args, "hops", None),
        expand=getattr(args, "expand", None),
    )
    return cfg


def _emit(args: argparse.Namespace, payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _globals(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", help="corpus database location (GALACTICA_DATA_DIR)")
    parser.add_argument("--model", help="navigator/answer model tag (GALACTICA_MODEL)")
    parser.add_argument("--provider", choices=("ollama", "stub"), help="model provider")
    parser.add_argument("--embed-model", help="embedding model for --hybrid / --embed")
    parser.add_argument("--json", action="store_true", help="machine-readable output")


# ----------------------------------------------------------------------- commands


def cmd_init(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = open_db(cfg.db_path)
    conn.close()
    made = [str(cfg.data_dir)]
    for name, blurb in RAW_DIRS.items():
        target = Path("corpus/raw") / name
        target.mkdir(parents=True, exist_ok=True)
        readme = target / "README.txt"
        if not readme.exists():
            readme.write_text(blurb, encoding="utf-8")
        made.append(str(target))
    DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    made.append(str(DEFAULT_RUNS_DIR))
    if args.json:
        _emit(args, {"db": str(cfg.db_path), "created": made})
    else:
        print(f"database: {cfg.db_path}")
        for path in made:
            print(f"ready:    {path}")
        print("\nDrop large dumps into corpus/raw/<source>/ then run galactica ingest (see README).")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    provider = build_provider(cfg)
    health = provider.health()
    db_exists = cfg.db_path.exists()
    counts = {}
    if db_exists:
        conn = open_db(cfg.db_path)
        counts = {k: v for k, v in stats(conn).items() if k != "by_source"}
        conn.close()
    try:
        import pyarrow  # type: ignore  # noqa: F401

        parquet = True
    except ImportError:
        parquet = False
    embed_ready = bool(cfg.embed_model) and cfg.embed_model in (health.models or [])
    payload = {
        "version": __version__,
        "provider": cfg.provider,
        "model": cfg.model,
        "provider_ok": health.ok,
        "provider_detail": health.detail,
        "db_path": str(cfg.db_path),
        "db_exists": db_exists,
        "counts": counts,
        "embed_model": cfg.embed_model,
        "embed_model_installed": embed_ready,
        "parquet_support": parquet,
        "context_budget": cfg.context_budget,
        "num_ctx": cfg.effective_num_ctx(),
        "num_ctx_gateway": cfg.effective_num_ctx(client_reserve=cfg.client_reserve),
        "num_ctx_explicit": cfg.num_ctx,
        "max_answer_tokens": cfg.max_answer_tokens,
        "grounding": cfg.grounding,
        "reasoning": cfg.think,
    }
    if args.json:
        _emit(args, payload)
    else:
        ok = "ok" if health.ok else "FAIL"
        print(f"galactica {__version__}")
        print(f"provider   {cfg.provider} [{ok}] {health.detail}")
        print(f"model      {cfg.model}")
        print(f"database   {cfg.db_path} {'(missing - run galactica init)' if not db_exists else ''}")
        if counts:
            print(
                f"corpus     {counts['documents']} docs · {counts['chunks']} chunks · "
                f"{counts['approx_tokens'] or 0} approx tokens · {counts['embeddings']} embeddings"
            )
        print(f"budget     {cfg.context_budget} context tokens, {cfg.max_answer_tokens} answer tokens")
        # num_ctx decides the KV cache ollama reserves, which is often larger
        # than the weights and is what actually determines whether a model fits.
        derived = "set explicitly" if cfg.num_ctx else "derived from budgets"
        print(
            f"num_ctx    {cfg.effective_num_ctx()} for ask/eval, "
            f"{cfg.effective_num_ctx(client_reserve=cfg.client_reserve)} for serve ({derived})"
        )
        print(f"grounding  {cfg.grounding}, reasoning {'on' if cfg.think else 'off'}")
        print(
            "embeddings "
            + (
                f"{cfg.embed_model} {'installed' if embed_ready else 'NOT installed (ollama pull ' + str(cfg.embed_model) + ')'}"
                if cfg.embed_model
                else "not configured (set GALACTICA_EMBED_MODEL for --hybrid)"
            )
        )
        print(f"parquet    {'available' if parquet else 'unavailable (pip install -e \".[parquet]\")'}")
    return 0 if health.ok else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = open_db(cfg.db_path)
    provider = None
    if args.embed:
        provider = build_provider(cfg)
    try:
        report = ingest_path(
            conn,
            Path(args.path),
            profile_name=args.profile,
            source=args.source,
            collection=args.collection,
            license=args.license,
            source_version=args.source_version,
            map_arg=args.map,
            max_documents=args.max_documents,
            sample=args.sample,
            resume=args.resume,
            skip_redirects=not args.keep_redirects,
            provider=provider,
            embed=args.embed,
            embed_model=cfg.embed_model,
            progress=None if args.json else (lambda n, title: print(f"  ... {n} docs ({title[:60]})")),
        )
    except (LoaderError, MappingError, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    if args.json:
        _emit(args, report.__dict__)
    else:
        print(
            f"source '{report.source}' via profile '{report.profile}': "
            f"{report.documents_written} documents written, "
            f"{report.chunks_written} chunks indexed"
        )
        if report.documents_unchanged:
            print(f"  {report.documents_unchanged} unchanged (same checksum)")
        if report.documents_empty:
            print(f"  {report.documents_empty} skipped (empty text)")
        if report.embeddings_written:
            print(f"  {report.embeddings_written} embeddings written")
        for warning in report.warnings:
            print(f"  warning: {warning}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = open_db(cfg.db_path)
    provider = build_provider(cfg) if args.hybrid else None
    result = search(
        conn,
        [args.question],
        top_k=args.k or cfg.top_k,
        hybrid=args.hybrid,
        provider=provider,
        embed_model=cfg.embed_model,
    )
    conn.close()
    if args.json:
        _emit(args, {"hits": [h.__dict__ for h in result.hits], "warnings": result.warnings})
        return 0
    for warning in result.warnings:
        print(f"warning: {warning}")
    if not result.hits:
        print("no matches")
        return 0
    for i, hit in enumerate(result.hits, start=1):
        print(f"{i:2}. [{hit.score:.4f}] {hit.title} — {hit.heading_path}")
        print(f"    {hit.source_name}{' v' + hit.source_version if hit.source_version else ''} · {hit.chunk_id}")
        snippet = " ".join(hit.text.split())[:200]
        print(f"    {snippet}")
    return 0


def _print_answer(answer, show_context: bool) -> None:
    bar = "─" * 12
    print(f"\n{bar} {answer.mode} {bar}")
    print(answer.text or "(empty answer)")
    if answer.sources:
        print("\nsources:")
        for src in answer.sources:
            version = f" v{src.source_version}" if src.source_version else ""
            print(f"  [{src.label}] {src.title} — {src.heading_path}")
            print(
                f"        {src.source_name}{version}"
                f"{' · ' + src.license if src.license else ''}"
                f"{' · ' + src.uri if src.uri else ''}"
            )
            print(f"        chunks: {', '.join(src.chunk_ids)}")
    if answer.mode == "cortex":
        cited = ", ".join(answer.citations) or "none"
        print(f"\ncitations: {cited}")
    if answer.invalid_citations:
        print(f"FABRICATED citations: {', '.join(answer.invalid_citations)}")
    if answer.gaps:
        for gap in answer.gaps:
            print(f"gap: {gap}")
    if answer.uncited:
        for note in answer.uncited:
            print(f"not corpus-backed: {note}")
    if answer.missing_queries:
        print(f"missing lookups: {'; '.join(answer.missing_queries)}")
    if answer.mode == "cortex":
        print(
            f"context: {answer.context_tokens}/{answer.context_budget} tokens · "
            f"{len(answer.sources)} sources · hops {answer.hops_used} · {answer.latency_s}s"
        )
    else:
        print(f"latency: {answer.latency_s}s")
    for warning in answer.warnings:
        print(f"warning: {warning}")
    if show_context and answer.mode == "cortex":
        print("\n--- assembled context ---")
        print(answer.context or "(empty)")
        if answer.dropped:
            print("\n--- dropped ---")
            for chunk_id, reason in answer.dropped:
                print(f"  {chunk_id}: {reason}")


def cmd_ask(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = open_db(cfg.db_path)
    provider = build_provider(cfg)
    try:
        answers = ask(
            conn,
            provider,
            cfg,
            args.question,
            mode=args.mode,
            hybrid=args.hybrid,
            use_plan=not args.no_plan,
        )
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    if args.json:
        _emit(args, {mode: a.to_dict() for mode, a in answers.items()})
        return 0
    for answer in answers.values():
        _print_answer(answer, args.show_context)
    if len(answers) > 1:
        base, cortex = answers["baseline"], answers["cortex"]
        print("\n" + "─" * 12 + " comparison " + "─" * 12)
        print(f"baseline: {len(base.text.split())} words · declined={base.declined} · {base.latency_s}s")
        print(
            f"cortex:   {len(cortex.text.split())} words · declined={cortex.declined} · "
            f"{len(cortex.citations)} valid citations · {cortex.latency_s}s"
        )
    return 0


def _mixed(value: str | None) -> str:
    """Per-document versions/licenses can differ within one source; keep it readable."""
    if not value:
        return "-"
    parts = [p for p in dict.fromkeys(value.split(",")) if p]
    return parts[0] if len(parts) == 1 else f"mixed ({len(parts)} values)"


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = open_db(cfg.db_path)
    cases = load_cases(Path(args.file))
    if args.limit:
        cases = cases[: args.limit]

    if args.retrieval_only:
        results = run_retrieval_only(conn, cfg, cases, hybrid=args.hybrid)
    else:
        modes = ("baseline", "cortex") if (args.compare or args.mode == "both") else (args.mode,)
        provider = build_provider(cfg)
        results = run_cases(
            conn,
            provider,
            cfg,
            cases,
            modes=modes,
            hybrid=args.hybrid,
            use_plan=not args.no_plan,
            on_case=None
            if args.json
            else (lambda r: print(f"  {r.mode:8} {r.case.question[:64]}")),
        )
    conn.close()

    agg = aggregate(results)
    delta = uplift(agg)
    run_path = save_run(results, cfg, runs_dir=Path(args.runs_dir), extra={"file": args.file})

    if args.json:
        _emit(args, {"aggregate": agg, "uplift": delta, "run": str(run_path)})
        return 0

    modes = list(agg)
    width = max(len(k) for k in ("metric", *agg[modes[0]])) + 2
    print(f"\ncases: {len(cases)}   model: {cfg.model}   budget: {cfg.context_budget}")
    header = "metric".ljust(width) + "".join(m.rjust(12) for m in modes)
    if delta:
        header += "uplift".rjust(12)
    print("\n" + header)
    print("-" * len(header))
    keys = list(agg[modes[0]].keys())
    for key in keys:
        row = key.ljust(width) + "".join(_fmt(agg[m].get(key)).rjust(12) for m in modes)
        if delta:
            row += _fmt(delta.get(key)).rjust(12)
        print(row)
    print(f"\nrun saved: {run_path}")
    return 0


_STALE_COUNTS = (
    "document counts are stale (corpus ingested before rollups existed); "
    "run: galactica stats --refresh"
)


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    if not cfg.db_path.exists():
        print(f"error: no corpus at {cfg.db_path} (run galactica init && galactica ingest)",
              file=sys.stderr)
        return 2
    serve(cfg, host=args.host, port=args.port, mode=args.mode)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = open_db(cfg.db_path)
    if args.refresh:
        updated = refresh_document_counts(conn)
        if not args.json:
            print(f"refreshed counts for {updated} documents")
    data = stats(conn)
    conn.close()
    if not data["counts_complete"] and not args.json:
        print(f"warning: {_STALE_COUNTS}")
    if args.json:
        _emit(args, data)
        return 0
    print(
        f"{data['documents']} documents · {data['chunks']} chunks · "
        f"{data['approx_tokens'] or 0} approx tokens · {data['embeddings']} embeddings"
    )
    for src in data["by_source"]:
        print(
            f"  {src['name']:<16} {src['documents']:>7} docs {src['chunks']:>8} chunks  "
            f"{_mixed(src['source_version'])}  {_mixed(src['license'])}"
        )
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = open_db(cfg.db_path)
    rows = list_sources(conn)
    complete = counts_are_complete(conn)
    conn.close()
    if not complete and not args.json:
        print(f"warning: {_STALE_COUNTS}")
    if args.json:
        _emit(args, {"sources": rows})
        return 0
    if not rows:
        print("no sources ingested yet")
        return 0
    for row in rows:
        print(f"{row['name']}")
        print(f"  profile        {row['loader_profile']} ({row['kind']})")
        print(f"  version        {_mixed(row['source_version'])}")
        print(f"  license        {_mixed(row['license'])}")
        print(f"  ingested       {row['ingested_at']}")
        print(f"  documents      {row['documents']}")
        print(f"  chunks         {row['chunks']} ({row['approx_tokens'] or 0} approx tokens)")
    return 0


# ------------------------------------------------------------------------- parsing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="galactica",
        description="Encyclopedia Galactica: precomputed intelligence as data, "
        "a cheap local model as its navigator.",
    )
    parser.add_argument("--version", action="version", version=f"galactica {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create the database and corpus drop directories")
    _globals(p_init)
    p_init.set_defaults(func=cmd_init)

    p_doctor = sub.add_parser("doctor", help="check provider, model, database, extras")
    _globals(p_doctor)
    p_doctor.set_defaults(func=cmd_doctor)

    p_ing = sub.add_parser("ingest", help="ingest a corpus path into the knowledge base")
    _globals(p_ing)
    p_ing.add_argument("path")
    p_ing.add_argument(
        "--profile", default="markdown", choices=sorted(PROFILES), help="loader/field profile"
    )
    p_ing.add_argument("--source", help="source name (default: derived from path/profile)")
    p_ing.add_argument("--collection", help="override collection label")
    p_ing.add_argument("--license", help="license for these documents")
    p_ing.add_argument("--source-version", help="dump date or revision (provenance)")
    p_ing.add_argument("--map", help="field mapping, e.g. title=page_title,text=content")
    p_ing.add_argument("--max-documents", type=int, help="stop after N documents (from the start)")
    p_ing.add_argument(
        "--sample",
        type=int,
        help="ingest ~N documents from random offsets (topically spread; "
        "uncompressed .ndjson/.jsonl only)",
    )
    p_ing.add_argument("--resume", action="store_true", help="skip documents already at the same checksum")
    p_ing.add_argument("--keep-redirects", action="store_true", help="wikipedia: keep redirect pages")
    p_ing.add_argument("--embed", action="store_true", help="also compute embeddings (needs embed model)")
    p_ing.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="raw retrieval, no LLM")
    _globals(p_search)
    p_search.add_argument("question")
    p_search.add_argument("-k", type=int, help="number of hits")
    p_search.add_argument("--hybrid", action="store_true", help="fuse BM25 with vector search")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="answer a question")
    _globals(p_ask)
    p_ask.add_argument("question")
    p_ask.add_argument("--mode", choices=MODES, help="cortex (default), baseline, or both")
    p_ask.add_argument("--hops", type=int, help="1 (default) or 2 retrieval hops")
    p_ask.add_argument("--expand", type=int, help="neighbor chunks per side (default 1, 0 disables)")
    p_ask.add_argument("--budget", type=int, help="context budget in tokens")
    p_ask.add_argument("--top-k", type=int, help="retrieval depth")
    p_ask.add_argument("--max-sources", type=int, help="max citable source blocks (default 8)")
    p_ask.add_argument("--hybrid", action="store_true", help="fuse BM25 with vector search")
    p_ask.add_argument(
        "--grounding",
        choices=("augmented", "strict"),
        help="augmented (default): corpus improves the answer, gaps filled from model knowledge and labelled; strict: corpus only",
    )
    p_ask.add_argument("--no-plan", action="store_true", help="skip the planner LLM call")
    p_ask.add_argument("--show-context", action="store_true", help="print assembled context and drops")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="score baseline vs cortex uplift")
    _globals(p_eval)
    p_eval.add_argument("file")
    p_eval.add_argument("--compare", action="store_true", help="run both arms (uplift table)")
    p_eval.add_argument("--mode", default="cortex", choices=MODES, help="which arm(s) to run")
    p_eval.add_argument("--retrieval-only", action="store_true", help="no LLM calls at all")
    p_eval.add_argument("--hybrid", action="store_true")
    p_eval.add_argument(
        "--grounding",
        choices=("augmented", "strict"),
        help="augmented (default): corpus improves the answer, gaps filled from model knowledge and labelled; strict: corpus only",
    )
    p_eval.add_argument("--no-plan", action="store_true")
    p_eval.add_argument("--limit", type=int, help="only the first N cases")
    p_eval.add_argument("--budget", type=int)
    p_eval.add_argument("--top-k", type=int)
    p_eval.add_argument("--max-sources", type=int)
    p_eval.add_argument("--hops", type=int)
    p_eval.add_argument("--expand", type=int)
    p_eval.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    p_eval.set_defaults(func=cmd_eval)

    p_serve = sub.add_parser(
        "serve", help="run an Anthropic-API gateway so Claude Code can use this corpus"
    )
    _globals(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8787)
    p_serve.add_argument(
        "--mode",
        default="auto",
        choices=SERVE_MODES,
        help="auto: retrieve for knowledge questions only (default); "
        "always: every turn; off: passthrough",
    )
    p_serve.add_argument("--budget", type=int)
    p_serve.add_argument(
        "--grounding",
        choices=("augmented", "strict"),
        help="augmented (default): corpus improves the answer, gaps filled from model knowledge and labelled; strict: corpus only",
    )
    p_serve.add_argument("--top-k", type=int)
    p_serve.add_argument("--max-sources", type=int)
    p_serve.set_defaults(func=cmd_serve)

    p_setup = sub.add_parser(
        "setup", help="pick a model for this machine, fetch a corpus, save the config"
    )
    _globals(p_setup)  # provides --model, used here to override the recommendation
    p_setup.add_argument("--source", help="corpus to fetch (default: the registry default)")
    p_setup.add_argument("--yes", "-y", action="store_true", help="accept every prompt")
    p_setup.set_defaults(func=cmd_setup)

    p_models = sub.add_parser("models", help="local models known to work, and what fits here")
    _globals(p_models)
    p_models.set_defaults(func=cmd_models)

    p_fetch = sub.add_parser("fetch", help="download and ingest a registered corpus")
    _globals(p_fetch)
    p_fetch.add_argument("name", nargs="?", help="corpus name (omit to list what is available)")
    p_fetch.add_argument("--sample", type=int, help="ingest ~N documents from random offsets")
    p_fetch.add_argument("--max-documents", type=int, help="ingest the first N documents")
    p_fetch.set_defaults(func=cmd_fetch)

    p_stats = sub.add_parser("stats", help="corpus counts per source")
    _globals(p_stats)
    p_stats.add_argument(
        "--refresh",
        action="store_true",
        help="recompute per-document chunk rollups (needed once after upgrading)",
    )
    p_stats.set_defaults(func=cmd_stats)

    p_sources = sub.add_parser("sources", help="provenance inventory")
    _globals(p_sources)
    p_sources.set_defaults(func=cmd_sources)

    return parser


NUDGE_COMMANDS = {"ask", "search", "eval", "serve"}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in NUDGE_COMMANDS and not getattr(args, "json", False):
            _nudge(_cfg(args))
        return args.func(args)
    except ProviderError as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        return 2
    except (LoaderError, MappingError) as exc:
        print(f"ingest error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
