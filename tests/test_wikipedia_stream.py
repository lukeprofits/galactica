from galactica.ingest import ingest_path
from galactica.ingest.wikipedia import clean_wikitext, dump_version, iter_pages


def test_dump_version_from_filename(wikipedia_dump):
    assert dump_version(wikipedia_dump) == "20260101"


def test_streams_only_article_namespace_and_skips_redirects(wikipedia_dump):
    pages = list(iter_pages(wikipedia_dump))
    assert [p["title"] for p in pages] == ["Torque wrench"]


def test_redirects_can_be_kept(wikipedia_dump):
    titles = [p["title"] for p in iter_pages(wikipedia_dump, skip_redirects=False)]
    assert titles == ["Torque wrench", "Torque spanner"]


def test_max_documents_stops_early(wikipedia_dump):
    assert len(list(iter_pages(wikipedia_dump, skip_redirects=False, max_documents=1))) == 1


def test_wikitext_cleanup():
    text = clean_wikitext(
        "{{Infobox|a={{nested|b}}}}A '''bold''' [[link|word]] and [[Plain]].<ref>x</ref>\n"
        "== Section ==\n* item\n{| class=\"wikitable\"\n! h\n|}\n"
        "[[File:pic.jpg|thumb]]\nSee [http://e.com Example] and [http://bare.com]."
    )
    assert "Infobox" not in text and "nested" not in text
    assert "bold" in text and "word" in text and "Plain" in text
    assert "<ref>" not in text and "wikitable" not in text and "File:" not in text
    assert "## Section" in text  # dump level 2 becomes a markdown h2
    assert "- item" in text
    assert "Example" in text and "http://bare.com" not in text


def test_headings_survive_list_conversion():
    # Wikitext numbered lists use '#', which must not be read as a heading marker.
    out = clean_wikitext("== Steps ==\n# first\n# second")
    assert "## Steps" in out and "- first" in out


def test_ingest_wikipedia_dump_end_to_end(conn, wikipedia_dump):
    report = ingest_path(conn, wikipedia_dump, profile_name="wikipedia")
    assert report.documents_written == 1
    row = conn.execute(
        "SELECT source_version, license, uri, collection, native_id FROM documents"
    ).fetchone()
    assert row == (
        "20260101",
        "CC BY-SA 4.0",
        "https://en.wikipedia.org/wiki/Torque_wrench",
        "wikipedia",
        "42",
    )
    paths = [
        r[0]
        for r in conn.execute("SELECT heading_path FROM chunks ORDER BY ord")
    ]
    assert any("Calibration" in p for p in paths)
