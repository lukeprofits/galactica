from galactica.ingest.chunker import CHARS_PER_TOKEN, approx_tokens, chunk_text

DOC = """# Widget

Opening paragraph.

## Maintenance

Calibrate annually.

Store it dry.

## History

Invented in 1918.
"""


def test_chunks_never_span_headings():
    chunks = chunk_text(DOC, target_tokens=6, overlap_tokens=0)
    paths = [c.heading_path for c in chunks]
    assert paths == [
        "Widget",
        "Widget > Maintenance",
        "Widget > Maintenance",
        "Widget > History",
    ]
    assert all("Invented" not in c.text for c in chunks if "Calibrate" in c.text)


def test_heading_path_nests_and_pops():
    doc = "# A\n\ntext\n\n## B\n\ntext\n\n### C\n\ntext\n\n## D\n\ntext\n"
    paths = [c.heading_path for c in chunk_text(doc)]
    assert paths == ["A", "A > B", "A > B > C", "A > D"]


def test_offsets_point_into_original_text():
    for chunk in chunk_text(DOC, target_tokens=6, overlap_tokens=0):
        window = DOC[chunk.char_start : chunk.char_end]
        first_line = chunk.text.splitlines()[0]
        assert first_line in window


def test_target_size_respected_and_oversized_paragraph_split():
    body = "# T\n\n" + ("This is a sentence about widgets. " * 200)
    chunks = chunk_text(body, target_tokens=50)
    assert len(chunks) > 1
    assert max(c.approx_tokens for c in chunks) <= 50 * 2


def test_overlap_repeats_trailing_paragraph():
    doc = "# T\n\n" + "\n\n".join(f"Paragraph number {i} of text." for i in range(20))
    chunks = chunk_text(doc, target_tokens=60, overlap_tokens=20)
    assert len(chunks) > 1
    tail = chunks[0].text.splitlines()[-1]
    assert tail in chunks[1].text


def test_approx_tokens_uses_chars_per_token():
    assert approx_tokens("x" * (CHARS_PER_TOKEN * 5)) == 5
