"""Loaders: filesystem/dump records -> RawDoc. One shape for the whole pipeline."""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .grokipedia import flatten as flatten_grokipedia
from .profiles import Profile, resolve_mapping
from .wikipedia import dump_version, iter_pages

MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdown", ".txt"}
JSONL_SUFFIXES = {".jsonl", ".ndjson"}


@dataclass
class RawDoc:
    native_id: str
    title: str
    text: str
    uri: str | None = None
    lang: str | None = None
    # Per-document provenance overrides (front matter, dump metadata).
    license: str | None = None
    source_version: str | None = None
    collection: str | None = None
    doc_type: str | None = None
    meta: dict = field(default_factory=dict)


class LoaderError(RuntimeError):
    pass


# ------------------------------------------------------------------- front matter

_FM_KEYS = {"title", "license", "version", "source_version", "collection", "doc_type", "uri", "lang"}


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Minimal `---` delimited `key: value` front matter. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    if lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() in ("---", "..."):
            return meta, "".join(lines[idx + 1 :]).lstrip("\n")
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip().strip("'\"")
    return {}, text  # unterminated block: treat as body


def _from_front_matter(meta: dict[str, str]) -> dict:
    known = {k: v for k, v in meta.items() if k in _FM_KEYS}
    if "version" in known and "source_version" not in known:
        known["source_version"] = known.pop("version")
    else:
        known.pop("version", None)
    extra = {k: v for k, v in meta.items() if k not in _FM_KEYS}
    return {"known": known, "extra": extra}


# ---------------------------------------------------------------------- markdown


def iter_markdown(root: Path) -> Iterator[RawDoc]:
    paths = (
        sorted(p for p in root.rglob("*") if p.suffix.lower() in MARKDOWN_SUFFIXES)
        if root.is_dir()
        else [root]
    )
    if not paths:
        raise LoaderError(f"no markdown/text files under {root}")
    for path in paths:
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_front_matter(raw)
        parsed = _from_front_matter(meta)
        known = parsed["known"]
        rel = path.name if not root.is_dir() else str(path.relative_to(root))
        title = known.get("title") or _first_heading(body) or path.stem.replace("_", " ")
        yield RawDoc(
            native_id=rel,
            title=title,
            text=body,
            uri=known.get("uri") or path.resolve().as_uri(),
            lang=known.get("lang"),
            license=known.get("license"),
            source_version=known.get("source_version"),
            collection=known.get("collection"),
            doc_type=known.get("doc_type"),
            meta=parsed["extra"] | {"path": rel},
        )


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


# ------------------------------------------------------------------- jsonl/parquet


def _jsonl_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    out = [
        p
        for p in sorted(root.rglob("*"))
        if p.suffix.lower() in JSONL_SUFFIXES
        or (p.suffix.lower() == ".gz" and Path(p.stem).suffix.lower() in JSONL_SUFFIXES)
    ]
    return out


def _parquet_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".parquet" else []
    return [p for p in sorted(root.rglob("*.parquet"))]


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _record_to_rawdoc(rec: dict, mapping: dict[str, str], profile: Profile) -> RawDoc | None:
    text = rec.get(mapping["text"]) or ""
    title = rec.get(mapping["title"]) or ""
    if not str(title).strip():
        return None  # untitled records cannot be cited
    native = str(rec.get(mapping.get("id", ""), "") or title)
    uri = rec.get(mapping["uri"]) if "uri" in mapping else None
    uri = str(uri) if uri else None
    if uri and not uri.startswith(("http://", "https://", "file://")) and profile.uri_base:
        uri = profile.uri_base + uri.lstrip("/")
    known = {"id", "title", "text", "uri", "lang"}
    extra = {
        k: v
        for k, v in rec.items()
        if k not in {mapping.get(f) for f in known} and isinstance(v, (str, int, float, bool))
    }
    return RawDoc(
        native_id=native,
        title=str(title),
        text=str(text),
        uri=uri,
        lang=str(rec.get(mapping["lang"])) if "lang" in mapping and rec.get(mapping["lang"]) else None,
        collection=profile.collection,
        doc_type=profile.doc_type,
        license=profile.license,
        meta=extra,
    )


def iter_jsonl(root: Path, profile: Profile, overrides: dict[str, str]) -> Iterator[RawDoc]:
    files = _jsonl_files(root)
    parquet = _parquet_files(root)
    if not files and parquet:
        yield from iter_parquet(root, profile, overrides)
        return
    if not files:
        raise LoaderError(
            f"no .jsonl/.jsonl.gz/.parquet files under {root} "
            "(place the dump files there, see README)"
        )
    mapping: dict[str, str] | None = None
    for path in files:
        with _open_text(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if mapping is None:
                    mapping = resolve_mapping(profile, list(rec.keys()), overrides)
                doc = _record_to_rawdoc(rec, mapping, profile)
                if doc:
                    yield doc


def iter_parquet(root: Path, profile: Profile, overrides: dict[str, str]) -> Iterator[RawDoc]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional extra
        raise LoaderError(
            "parquet support needs the optional extra: pip install -e '.[parquet]' "
            "(or convert the dump to .jsonl)"
        ) from exc
    files = _parquet_files(root)
    if not files:
        raise LoaderError(f"no .parquet files under {root}")
    mapping: dict[str, str] | None = None
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256):
            records = batch.to_pylist()
            for rec in records:
                if mapping is None:
                    mapping = resolve_mapping(profile, list(rec.keys()), overrides)
                doc = _record_to_rawdoc(rec, mapping, profile)
                if doc:
                    yield doc


# --------------------------------------------------------------------- grokipedia


def iter_grokipedia(
    root: Path, profile: Profile, *, sample: int | None = None
) -> Iterator[RawDoc]:
    """Nested Grokipedia dump records, streamed line by line."""
    files = _jsonl_files(root)
    parquet = _parquet_files(root)
    if not files and not parquet:
        raise LoaderError(
            f"no .ndjson/.jsonl/.jsonl.gz/.parquet dump under {root} "
            "(place grokipedia_scrape.ndjson there, see README)"
        )
    records = (
        _iter_sampled_records(files, sample) if sample else _iter_records(files, parquet)
    )
    for record in records:
        flat = flatten_grokipedia(record)
        if flat is None:
            continue
        yield RawDoc(
            native_id=flat["native_id"],
            title=flat["title"],
            text=flat["text"],
            uri=flat["uri"],
            collection=profile.collection,
            doc_type=profile.doc_type,
            license=profile.license,
            source_version=flat["source_version"],
            meta=flat["meta"],
        )


def _iter_records(files: list[Path], parquet: list[Path]) -> Iterator[dict]:
    for path in files:
        with _open_text(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record
    for path in parquet:
        for record in _iter_parquet_rows(path):
            yield record


def _iter_sampled_records(files: list[Path], sample: int) -> Iterator[dict]:
    """Random-offset sample, without reading the whole file.

    The Grokipedia dump is alphabetically clustered in shuffled blocks, so the
    first N records are a narrow topical neighbourhood. Seeking to random byte
    offsets and taking the next complete line gives a topically spread sample in
    O(sample) reads instead of parsing every record. Longer records are
    marginally more likely to be picked, which is acceptable for sampling.
    """
    plain = [p for p in files if p.suffix.lower() != ".gz"]
    if not plain:
        raise LoaderError(
            "--sample needs an uncompressed .ndjson/.jsonl file (gzip is not seekable); "
            "use --max-documents instead, or decompress the dump"
        )
    seen: set[str] = set()
    per_file = max(1, sample // len(plain))
    for path in plain:
        size = path.stat().st_size
        rng = random.Random(f"{path.name}:{size}:{sample}")
        emitted = attempts = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            # Budget the attempts so duplicate hits cannot loop forever.
            while emitted < per_file and attempts < per_file * 8:
                attempts += 1
                handle.seek(rng.randrange(0, max(1, size)))
                handle.readline()  # discard the partial line
                line = handle.readline().strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                key = str(record.get("title") or record.get("id") or line[:80])
                if key in seen:
                    continue
                seen.add(key)
                emitted += 1
                yield record


def _iter_parquet_rows(path: Path) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional extra
        raise LoaderError(
            "parquet support needs the optional extra: pip install -e '.[parquet]'"
        ) from exc
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64):
        yield from batch.to_pylist()


# ---------------------------------------------------------------------- wikipedia


def iter_wikipedia(root: Path, profile: Profile, *, skip_redirects: bool) -> Iterator[RawDoc]:
    if root.is_dir():
        files = [
            p
            for p in sorted(root.rglob("*"))
            if p.name.endswith((".xml", ".xml.bz2"))
        ]
    else:
        files = [root]
    if not files:
        raise LoaderError(
            f"no *.xml / *.xml.bz2 dump under {root} "
            "(place e.g. enwiki-YYYYMMDD-pages-articles.xml.bz2 there, see README)"
        )
    for path in files:
        version = dump_version(path)
        for page in iter_pages(path, skip_redirects=skip_redirects):
            yield RawDoc(
                native_id=page["native_id"],
                title=page["title"],
                text=page["text"],
                uri=(profile.uri_base or "") + page["title"].replace(" ", "_"),
                collection=profile.collection,
                doc_type=profile.doc_type,
                license=profile.license,
                source_version=version,
                meta={"dump_file": path.name},
            )


# ----------------------------------------------------------------------- dispatch


def load(
    path: Path,
    profile: Profile,
    *,
    overrides: dict[str, str] | None = None,
    skip_redirects: bool = True,
    sample: int | None = None,
) -> Iterator[RawDoc]:
    overrides = overrides or {}
    if not path.exists():
        raise LoaderError(f"path does not exist: {path}")
    if profile.family == "markdown":
        return iter_markdown(path)
    if profile.family == "jsonl":
        return iter_jsonl(path, profile, overrides)
    if profile.family == "grokipedia":
        return iter_grokipedia(path, profile, sample=sample)
    if profile.family == "wikipedia":
        return iter_wikipedia(path, profile, skip_redirects=skip_redirects)
    raise LoaderError(f"unknown loader family '{profile.family}'")
