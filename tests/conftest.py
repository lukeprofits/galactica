from __future__ import annotations

import bz2
import gzip
import html
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # allow running tests without an editable install
    sys.path.insert(0, str(SRC))

from galactica import store  # noqa: E402
from galactica.config import Config  # noqa: E402
from galactica.ingest import ingest_path  # noqa: E402

SEED_DIR = Path(__file__).resolve().parents[1] / "corpus" / "seed"


@pytest.fixture
def conn(tmp_path):
    connection = store.open_db(tmp_path / "test.db")
    yield connection
    connection.close()


@pytest.fixture
def cfg(tmp_path):
    return Config(
        provider="stub",
        model="stub",
        data_dir=tmp_path,
        context_budget=4000,
        top_k=12,
    )


@pytest.fixture
def seeded(conn):
    ingest_path(conn, SEED_DIR, profile_name="markdown", source="seed")
    return conn


@pytest.fixture
def grokipedia_dump(tmp_path):
    """Nested Grokipedia HF dump records, as in grokipedia_scrape.ndjson."""
    target = tmp_path / "grokipedia"
    target.mkdir()
    records = [
        {
            "title": "Gwaii_Passage_tidal_array",
            "url": "https://grokipedia.com/page/Gwaii_Passage_tidal_array",
            "scraped_at": "2025-10-29T19:33:13.650809",
            "data": {
                "main_title": "Gwaii Passage tidal array",
                "sections": [
                    {
                        "level": "h1",
                        "id": "gwaii",
                        "title": "Gwaii Passage tidal array",
                        "content": [
                            {
                                "type": "paragraph",
                                "text": "TheGwaii Passage arrayhas 14 turbines rated 1.8 MW"
                                " each.[1] Rated flow is 3.1 m/s.[12]",
                            }
                        ],
                    },
                    {"level": "h1", "id": "info", "title": "Array Information", "content": []},
                    {
                        "level": "h2",
                        "id": "design",
                        "title": "Design",
                        "content": [
                            {"type": "list", "items": ["Two-bladed rotor", "Passive yaw"]},
                            {"type": "ordered_list", "items": ["Survey", "Install"]},
                        ],
                    },
                ],
                "paragraphs": ["TheGwaii Passage arrayhas 14 turbines rated 1.8 MW each.[1]"],
                "tables": [
                    {"headers": ["Year", "Capacity factor"], "rows": [["2025", "38%"]]}
                ],
                "references": [{"text": "ref one", "link": "http://x"}],
                "metadata": {"has_edits": False, "fact_check_timestamp": None},
            },
        },
        {
            "title": "Paragraphs_Only",
            "url": "https://grokipedia.com/page/Paragraphs_Only",
            "scraped_at": "2025-11-01T00:00:00",
            "data": {
                "main_title": "Paragraphs Only",
                "sections": [],
                "paragraphs": ["Fallback body text about widgets."],
                "tables": [],
                "references": [],
            },
        },
        {
            "title": "Empty_Article",
            "url": "https://grokipedia.com/page/Empty_Article",
            "scraped_at": "2025-11-01T00:00:00",
            "data": {"main_title": "Empty Article", "sections": [], "paragraphs": []},
        },
    ]
    with gzip.open(target / "part-000.jsonl.gz", "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return target


@pytest.fixture
def wikipedia_dump(tmp_path):
    """Tiny valid pages-articles dump: article, non-article namespace, redirect."""
    wikitext = (
        "{{Infobox tool|name={{nowrap|torque}}}}A '''torque wrench''' is a "
        "[[tool|hand tool]].<ref name=a>cite</ref>\n\n"
        "== Calibration ==\n"
        "* Calibrate every 5,000 cycles.\n"
        "{| class=\"wikitable\"\n! a !! b\n|}\n"
        "See [http://example.com Example]."
    )
    xml = (
        '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'
        "<siteinfo><sitename>Wikipedia</sitename></siteinfo>"
        "<page><title>Torque wrench</title><ns>0</ns><id>42</id>"
        f"<revision><text>{html.escape(wikitext)}</text></revision></page>"
        "<page><title>Talk:Torque wrench</title><ns>1</ns><id>43</id>"
        "<revision><text>chatter</text></revision></page>"
        '<page><title>Torque spanner</title><ns>0</ns><id>44</id>'
        '<redirect title="Torque wrench" />'
        "<revision><text>#REDIRECT [[Torque wrench]]</text></revision></page>"
        "</mediawiki>"
    ).encode()
    path = tmp_path / "enwiki-20260101-pages-articles.xml.bz2"
    path.write_bytes(bz2.compress(xml))
    return path
