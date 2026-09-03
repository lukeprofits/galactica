import json

import pytest

from galactica.cli import main
from tests.conftest import SEED_DIR


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("GALACTICA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GALACTICA_PROVIDER", "stub")
    monkeypatch.setenv("GALACTICA_MODEL", "stub")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _json_out(capsys):
    return json.loads(capsys.readouterr().out)


def test_init_creates_db_and_dump_drop_dirs(env, capsys):
    assert main(["init", "--json"]) == 0
    payload = _json_out(capsys)
    assert (env / "corpus/raw/grokipedia/README.txt").exists()
    assert (env / "corpus/raw/wikipedia/README.txt").exists()
    assert (env / "eval/runs").is_dir()
    assert payload["db"].endswith("galactica.db")


def test_doctor_reports_state(env, capsys):
    main(["init", "--json"])
    capsys.readouterr()
    assert main(["doctor", "--json"]) == 0
    payload = _json_out(capsys)
    assert payload["provider_ok"] and payload["db_exists"]
    assert "parquet_support" in payload


def test_ingest_then_search_then_stats_and_sources(env, capsys):
    main(["init", "--json"])
    capsys.readouterr()
    assert main(["ingest", str(SEED_DIR), "--source", "seed", "--json"]) == 0
    report = _json_out(capsys)
    assert report["documents_written"] == 10 and report["chunks_written"] > 10

    assert main(["search", "krios relay breaker order", "-k", "3", "--json"]) == 0
    hits = _json_out(capsys)["hits"]
    assert hits and any("Krios" in h["title"] for h in hits)

    assert main(["stats", "--json"]) == 0
    assert _json_out(capsys)["documents"] == 10

    assert main(["sources", "--json"]) == 0
    assert _json_out(capsys)["sources"][0]["name"] == "seed"


def test_ask_both_modes_via_cli(env, capsys):
    main(["init", "--json"])
    main(["ingest", str(SEED_DIR), "--source", "seed", "--json"])
    capsys.readouterr()
    assert main(["ask", "What torque do the Krios carrier screws take?", "--mode", "both", "--json"]) == 0
    payload = _json_out(capsys)
    assert set(payload) == {"baseline", "cortex"}
    assert payload["cortex"]["sources"] and not payload["baseline"]["sources"]


def test_eval_retrieval_only_needs_no_model(env, capsys):
    main(["init", "--json"])
    main(["ingest", str(SEED_DIR), "--source", "seed", "--json"])
    capsys.readouterr()
    questions = env / "q.jsonl"
    questions.write_text(
        json.dumps(
            {
                "id": "k",
                "question": "Krios R7 breaker order",
                "expect_docs": ["Krios Relay R7"],
                "answerable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["eval", str(questions), "--retrieval-only", "--json"]) == 0
    payload = _json_out(capsys)
    assert payload["aggregate"]["retrieval"]["retrieval_hit"] == 1.0
    assert (env / payload["run"]).exists() or __import__("pathlib").Path(payload["run"]).exists()


def test_ingest_bad_profile_exits_nonzero(env, capsys):
    main(["init", "--json"])
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["ingest", str(SEED_DIR), "--profile", "does-not-exist"])


def test_ask_show_context_prints_context_and_drops(env, capsys):
    main(["init", "--json"])
    main(["ingest", str(SEED_DIR), "--source", "seed", "--json"])
    capsys.readouterr()
    assert main(["ask", "krios breaker order", "--show-context", "--budget", "600"]) == 0
    out = capsys.readouterr().out
    assert "assembled context" in out and "[S1]" in out
    assert "dropped" in out  # a 600-token budget cannot fit everything


def test_search_hybrid_without_embeddings_warns(env, capsys):
    main(["init", "--json"])
    main(["ingest", str(SEED_DIR), "--source", "seed", "--json"])
    capsys.readouterr()
    assert main(["search", "torque", "--hybrid", "--embed-model", "stub"]) == 0
    assert "warning:" in capsys.readouterr().out
