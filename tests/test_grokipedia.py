from galactica.ingest import ingest_path
from galactica.ingest.grokipedia import clean_text, flatten, to_markdown
from galactica.store import search_bm25


def test_reference_markers_are_dropped():
    assert clean_text("built from 1983 to 1993.[1] Later[12] revised.") == (
        "built from 1983 to 1993. Later revised."
    )


def test_glued_words_repaired_where_casing_makes_it_visible():
    assert clean_text("TheYamaha Venture is large.") == "The Yamaha Venture is large."
    assert clean_text("1,198 cc (73.1 cu in)V4 engine") == "1,198 cc (73.1 cu in) V4 engine"
    assert clean_text("ended in 1993.The next model") == "ended in 1993. The next model"


def test_repair_leaves_undetectable_and_risky_cases_alone():
    # lowercase-lowercase glue carries no signal to split on
    assert "Royaleis" in clean_text("The Venture Royaleis a tourer")
    # digits after a bracket are left intact (chemical formulae)
    assert clean_text("Ca(OH)2 precipitate") == "Ca(OH)2 precipitate"


def test_sections_become_markdown_headings_at_their_level():
    data = {
        "sections": [
            {"level": "h1", "title": "Top", "content": [{"type": "paragraph", "text": "Lead."}]},
            {"level": "h3", "title": "Deep", "content": [{"type": "paragraph", "text": "Detail."}]},
        ]
    }
    md = to_markdown(data, title="Top")
    assert "# Top" in md and "### Deep" in md and "Lead." in md


def test_empty_sections_keep_their_heading_for_hierarchy():
    data = {
        "sections": [
            {"level": "h1", "title": "Parent", "content": []},
            {"level": "h2", "title": "Child", "content": [{"type": "paragraph", "text": "x"}]},
        ]
    }
    md = to_markdown(data, title="Parent")
    assert "# Parent" in md and "## Child" in md


def test_lists_and_ordered_lists_render():
    data = {
        "sections": [
            {
                "level": "h2",
                "title": "Awards",
                "content": [
                    {"type": "list", "items": ["Best Spiker", "MVP"]},
                    {"type": "ordered_list", "items": ["First", "Second"]},
                ],
            }
        ]
    }
    md = to_markdown(data, title="T")
    assert "- Best Spiker" in md and "1. First" in md and "2. Second" in md


def test_tables_render_as_markdown_under_a_trailing_heading():
    data = {
        "sections": [{"level": "h1", "title": "T", "content": [{"type": "paragraph", "text": "x"}]}],
        "tables": [{"headers": ["Years", "Teams"], "rows": [["2014-2017", "Sunbirds"]]}],
    }
    md = to_markdown(data, title="T")
    assert "## Tables" in md
    assert "| Years | Teams |" in md and "| 2014-2017 | Sunbirds |" in md


def test_ragged_table_rows_are_padded_not_dropped():
    data = {"tables": [{"headers": ["a", "b", "c"], "rows": [["1"], ["1", "2", "3", "4"]]}]}
    md = to_markdown(data, title="T")
    assert "| 1 |  |  |" in md and "| 1 | 2 | 3 |" in md


def test_paragraph_fallback_when_no_sections():
    data = {"sections": [], "paragraphs": ["Only paragraphs here."]}
    assert to_markdown(data, title="Fallback") == "# Fallback\n\nOnly paragraphs here."


def test_flatten_carries_provenance():
    record = {
        "title": "Yamaha_Venture_Royale",
        "url": "https://grokipedia.com/page/Yamaha_Venture_Royale",
        "scraped_at": "2025-10-29T19:33:13.650809",
        "data": {
            "main_title": "Yamaha Venture Royale",
            "sections": [{"level": "h1", "title": "T", "content": [{"type": "paragraph", "text": "x"}]}],
            "paragraphs": [],
            "tables": [],
            "references": [{"text": "r", "link": "u"}],
            "metadata": {"has_edits": True},
        },
    }
    flat = flatten(record)
    assert flat["native_id"] == "Yamaha_Venture_Royale"
    assert flat["title"] == "Yamaha Venture Royale"
    assert flat["uri"].endswith("Yamaha_Venture_Royale")
    assert flat["source_version"] == "2025-10-29"
    assert flat["meta"]["reference_count"] == 1 and flat["meta"]["has_edits"] is True


def test_flatten_rejects_records_without_data():
    assert flatten({"title": "x"}) is None
    assert flatten({"title": "x", "data": "not a dict"}) is None
    assert flatten("not a record") is None


def test_missing_main_title_falls_back_to_slug():
    flat = flatten({"title": "Some_Slug", "data": {"sections": [], "paragraphs": ["body"]}})
    assert flat["title"] == "Some Slug"


def test_ingest_nested_dump_end_to_end(conn, grokipedia_dump):
    report = ingest_path(conn, grokipedia_dump, profile_name="grokipedia")
    assert report.documents_written == 2  # third record has no text at all
    assert report.documents_empty == 1

    row = conn.execute(
        "SELECT collection, doc_type, uri, license, source_version, native_id "
        "FROM documents WHERE title='Gwaii Passage tidal array'"
    ).fetchone()
    collection, doc_type, uri, license_, version, native = row
    assert (collection, doc_type, native) == ("grokipedia", "article", "Gwaii_Passage_tidal_array")
    assert uri == "https://grokipedia.com/page/Gwaii_Passage_tidal_array"
    assert "CC BY-SA 4.0" in license_ and "xAI" in license_
    assert version == "2025-10-29"

    # Heading hierarchy from the nested sections survived into chunk paths.
    paths = [r[0] for r in conn.execute("SELECT heading_path FROM chunks")]
    assert any("Design" in p for p in paths)
    assert any("Tables" in p for p in paths)

    # Repaired text is searchable: the glued "arrayhas" no longer hides "array".
    hits = search_bm25(conn, "Gwaii Passage array turbines", limit=5)
    assert hits and "Gwaii" in hits[0].title


def test_random_sample_is_spread_and_deduped(conn, tmp_path):
    """--sample must not return the same record repeatedly, and must spread out."""
    import json as _json

    path = tmp_path / "dump.ndjson"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(500):
            fh.write(
                _json.dumps(
                    {
                        "title": f"Article_{i:04d}",
                        "url": f"https://grokipedia.com/page/Article_{i:04d}",
                        "scraped_at": "2025-10-29T00:00:00",
                        "data": {
                            "main_title": f"Article {i:04d}",
                            "sections": [
                                {
                                    "level": "h1",
                                    "title": f"Article {i:04d}",
                                    "content": [
                                        {"type": "paragraph", "text": "Body text " * 40}
                                    ],
                                }
                            ],
                            "paragraphs": [],
                        },
                    }
                )
                + "\n"
            )
    report = ingest_path(conn, path, profile_name="grokipedia", sample=40, source="grok")
    titles = [r[0] for r in conn.execute("SELECT title FROM documents")]
    assert len(titles) == len(set(titles)) == report.documents_written
    assert 20 <= len(titles) <= 40
    # A spread sample reaches well beyond the first records of the file.
    indices = sorted(int(t.split()[1]) for t in titles)
    assert indices[-1] > 300 and indices[0] < 200


def test_sample_rejects_gzip_with_a_clear_error(conn, grokipedia_dump):
    import pytest

    from galactica.ingest.loaders import LoaderError

    with pytest.raises(LoaderError) as err:
        ingest_path(conn, grokipedia_dump, profile_name="grokipedia", sample=5)
    assert "--sample needs an uncompressed" in str(err.value)
