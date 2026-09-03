from galactica.pipeline import validate_citations


def test_valid_and_invalid_split():
    valid, invalid = validate_citations("Claim [S1] and claim [S9].", ["S1", "S2"])
    assert valid == ["S1"] and invalid == ["S9"]


def test_first_appearance_order_and_dedup():
    valid, _ = validate_citations("[S2] then [S1] then [S2] again", ["S1", "S2"])
    assert valid == ["S2", "S1"]


def test_no_citations():
    assert validate_citations("no citations here", ["S1"]) == ([], [])


def test_baseline_labels_are_all_fabricated():
    valid, invalid = validate_citations("Per [S1] the answer is x", [])
    assert valid == [] and invalid == ["S1"]
