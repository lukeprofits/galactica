import json

import pytest

from galactica.ingest import ingest_path
from galactica.ingest.loaders import LoaderError, parse_front_matter
from galactica.ingest.profiles import (
    PROFILES,
    MappingError,
    parse_map_arg,
    resolve_mapping,
)
from galactica.store import stats


def test_generic_profile_resolves_common_column_names():
    mapping = resolve_mapping(PROFILES["generic"], ["slug", "name", "content", "url"])
    assert mapping == {"id": "slug", "title": "name", "text": "content", "uri": "url"}


def test_alternate_column_names_resolve_too():
    mapping = resolve_mapping(PROFILES["generic"], ["doc_id", "title", "markdown"])
    assert mapping["id"] == "doc_id" and mapping["text"] == "markdown"


def test_explicit_map_overrides_candidates():
    columns = ["slug", "title", "content", "body", "url"]
    mapping = resolve_mapping(PROFILES["generic"], columns, {"text": "body"})
    assert mapping["text"] == "body"


def test_unmappable_dump_reports_actual_columns():
    with pytest.raises(MappingError) as err:
        resolve_mapping(PROFILES["generic"], ["foo", "bar"])
    message = str(err.value)
    assert "foo" in message and "bar" in message and "--map title=" in message


def test_map_arg_validation():
    assert parse_map_arg("title=page_title, text=content") == {
        "title": "page_title",
        "text": "content",
    }
    assert parse_map_arg(None) == {}
    with pytest.raises(MappingError):
        parse_map_arg("nonsense")
    with pytest.raises(MappingError):
        parse_map_arg("bogus=x")


def test_max_documents_samples_a_dump(conn, grokipedia_dump):
    report = ingest_path(conn, grokipedia_dump, profile_name="grokipedia", max_documents=1)
    assert report.documents_seen == 1 and report.documents_written == 1


def test_generic_profile_with_explicit_mapping(conn, tmp_path):
    path = tmp_path / "custom.jsonl"
    path.write_text(
        json.dumps({"pk": "1", "headline": "Widget", "prose": "Widgets spin."}) + "\n",
        encoding="utf-8",
    )
    report = ingest_path(
        conn,
        path,
        profile_name="generic",
        map_arg="id=pk,title=headline,text=prose",
        source="custom",
    )
    assert report.documents_written == 1


def test_missing_dump_directory_is_a_clear_error(conn, tmp_path):
    empty = tmp_path / "grokipedia_empty"
    empty.mkdir()
    with pytest.raises(LoaderError) as err:
        ingest_path(conn, empty, profile_name="grokipedia")
    assert "grokipedia_scrape.ndjson" in str(err.value)


def test_unknown_profile_rejected(conn, tmp_path):
    with pytest.raises(LoaderError):
        ingest_path(conn, tmp_path, profile_name="nope")


def test_front_matter_parsing():
    meta, body = parse_front_matter("---\ntitle: X\nlicense: CC0\nversion: 7\n---\nbody\n")
    assert meta == {"title": "X", "license": "CC0", "version": "7"}
    assert body == "body\n"
    assert parse_front_matter("no front matter") == ({}, "no front matter")
    # Unterminated block is treated as body, not silently swallowed.
    assert parse_front_matter("---\ntitle: X\n")[1].startswith("---")
