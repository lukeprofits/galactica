"""Release surface: registries, hardware sizing, config file, wizard, launcher."""

from __future__ import annotations

import pytest

from galactica.config import Config, read_config_file, write_config_file
from galactica.hardware import Hardware, detect
from galactica.launcher import claude_environment
from galactica.registry import (
    context_budget_for,
    models,
    recommend_model,
    source_named,
    sources,
)
from galactica.setup import Readiness, ask_yes, plan_corpus

LAPTOP = Hardware("Linux", False, 16.0, 6.0, "GTX 1060")   # a 6 GB card, the tight case
DESKTOP = Hardware("Linux", False, 32.0, 12.0, "RTX 3060")
MAC = Hardware("Darwin", True, 64.0, None, None)
POTATO = Hardware("Linux", False, 8.0, None, None)


# ------------------------------------------------------------------- registries


def test_model_registry_loads_and_is_ordered_by_capability():
    entries = models()
    assert len(entries) >= 5
    assert [m.tier for m in entries] == sorted(m.tier for m in entries)
    assert all(m.weights_gb < m.min_memory_gb for m in entries)  # room for KV cache


def test_source_registry_carries_the_numbers_a_user_needs_first():
    grokipedia = source_named("grokipedia")
    assert grokipedia.profile == "grokipedia"
    assert grokipedia.download_gb > 0 and grokipedia.db_gb > 0
    assert grokipedia.peak_gb > grokipedia.download_gb  # both files coexist during ingest
    assert "CC BY-SA" in grokipedia.license
    assert any(s.default for s in sources())
    with pytest.raises(KeyError):
        source_named("not-a-corpus")


def test_source_size_scales_with_a_sample():
    grokipedia = source_named("grokipedia")
    assert grokipedia.db_gb_for(grokipedia.documents // 2) == pytest.approx(grokipedia.db_gb / 2, rel=0.01)
    assert grokipedia.db_gb_for(None) == grokipedia.db_gb


# ---------------------------------------------------------------- hardware fit


def test_apple_silicon_shares_memory_and_a_discrete_card_does_not():
    assert MAC.usable_gb == pytest.approx(48.0)
    assert LAPTOP.usable_gb == pytest.approx(5.0)  # 6 GB VRAM minus desktop overhead
    assert POTATO.usable_gb == pytest.approx(4.0)  # CPU inference, half of RAM


def test_a_six_gigabyte_card_gets_the_model_measured_on_one():
    """A 6 GB card runs qwen3:4b: 0.881 factual coverage with the corpus."""
    assert recommend_model(LAPTOP).tag == "qwen3:4b"
    assert context_budget_for(LAPTOP, recommend_model(LAPTOP)) == 4000


def test_bigger_hardware_gets_bigger_models():
    assert recommend_model(DESKTOP).tag == "qwen3:8b"
    assert recommend_model(MAC).tag.startswith("qwen3.6:35b")


def test_a_model_that_only_fits_with_no_context_is_not_recommended():
    """14b technically fits 11 GB but leaves no room for retrieved material."""
    twelve_gb = Hardware("Linux", False, 32.0, 12.0, "RTX 3060")
    assert recommend_model(twelve_gb).tag != "qwen3:14b"


def test_an_already_installed_model_is_preferred_over_a_download():
    chosen = recommend_model(MAC, installed=["qwen3:8b"])
    assert chosen.tag == "qwen3:8b"


def test_nothing_fits_returns_none():
    assert recommend_model(Hardware("Linux", False, 1.0, 0.5, "old")) is None


def test_apple_only_builds_are_not_offered_elsewhere():
    mlx = next(m for m in models() if m.tag.endswith("-mlx"))
    assert mlx.fits(MAC)
    assert not mlx.fits(Hardware("Linux", False, 128.0, 80.0, "A100"))


def test_detect_reports_something_coherent_about_this_machine():
    hardware = detect()
    assert hardware.total_ram_gb > 0
    assert hardware.usable_gb > 0
    assert hardware.describe()


# --------------------------------------------------------------- config file


def test_config_file_is_read_and_the_environment_still_wins(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    written = write_config_file({"model": "qwen3:4b", "context_budget": 4000, "think": False}, path)
    assert read_config_file(written)["model"] == "qwen3:4b"

    monkeypatch.setenv("GALACTICA_CONFIG", str(path))
    monkeypatch.delenv("GALACTICA_MODEL", raising=False)
    cfg = Config.load()
    assert cfg.model == "qwen3:4b" and cfg.context_budget == 4000 and cfg.think is False
    # Derived num_ctx follows the stored budget without anyone setting it.
    assert cfg.effective_num_ctx() == 8192

    monkeypatch.setenv("GALACTICA_MODEL", "qwen3:8b")
    assert Config.load().model == "qwen3:8b"


def test_unknown_keys_are_not_written(tmp_path):
    path = write_config_file({"model": "m", "not_a_setting": "x"}, tmp_path / "c.toml")
    assert "not_a_setting" not in path.read_text()


def test_missing_or_corrupt_config_is_not_fatal(tmp_path):
    assert read_config_file(tmp_path / "absent.toml") == {}
    broken = tmp_path / "broken.toml"
    broken.write_text("this is not = = toml", encoding="utf-8")
    assert read_config_file(broken) == {}


# -------------------------------------------------------------------- wizard


def test_corpus_plan_scales_to_the_disk_available():
    grokipedia = source_named("grokipedia")
    full, why = plan_corpus(grokipedia, 200.0)
    assert full is None and "full corpus" in why

    sample, why = plan_corpus(grokipedia, 120.0)
    assert sample and sample < grokipedia.documents and "sample" in why

    none, why = plan_corpus(grokipedia, 10.0)
    assert none == 0 and "not enough space" in why


def test_readiness_names_the_one_next_step():
    assert Readiness(False, False, 0, False).next_step().startswith("Ollama is not running")
    assert "No model" in Readiness(True, False, 0, False).next_step()
    assert "No corpus" in Readiness(True, True, 0, False).next_step()
    done = Readiness(True, True, 100, True)
    assert done.ready and done.next_step() is None


# ------------------------------------------------------------------ launcher


def test_launcher_pins_every_model_alias_to_the_gateway():
    env = claude_environment("http://127.0.0.1:8787", "galactica")
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    # An unpinned alias could resolve to a hosted model this endpoint cannot serve.
    for key in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "ANTHROPIC_SMALL_FAST_MODEL"):
        assert env[key] == "galactica"
    # Background calls would queue behind real turns on one local model.
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


# ------------------------------------------------------- downloads and installs


def test_corpus_urls_need_no_third_party_client():
    """A user should not have to install huggingface_hub to get a corpus."""
    from galactica.setup import source_url

    grokipedia = source_url(source_named("grokipedia"))
    assert grokipedia == (
        "https://huggingface.co/datasets/htriedman/grokipedia-v0.1-dump/"
        "resolve/main/grokipedia_scrape.ndjson"
    )
    assert source_url(source_named("wikipedia")).startswith("https://dumps.wikimedia.org/")


def test_download_resumes_a_partial_file(tmp_path):
    """A 68 GB download will be interrupted; it must continue, not restart."""
    from galactica.setup import download

    destination = tmp_path / "dump.ndjson"
    destination.write_bytes(b"already-here")
    seen = {}

    class Response:
        status = 206
        headers = {"Content-Length": "5"}

        def read(self, _size):
            if seen.get("read"):
                return b""
            seen["read"] = True
            return b"-rest"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def opener(request, timeout=None):
        seen["range"] = request.get_header("Range")
        return Response()

    assert download("https://example/dump", destination, opener=opener) == destination
    assert seen["range"] == f"bytes={len('already-here')}-"
    assert destination.read_bytes() == b"already-here-rest"


def test_download_restarts_when_the_server_ignores_the_range(tmp_path):
    from galactica.setup import download

    destination = tmp_path / "dump.ndjson"
    destination.write_bytes(b"stale-partial")
    body = [b"fresh-content", b""]

    class Response:
        status = 200  # range ignored
        headers = {"Content-Length": "13"}

        def read(self, _size):
            return body.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    assert download("https://example/d", destination, opener=lambda r, timeout=None: Response())
    assert destination.read_bytes() == b"fresh-content"


def test_download_treats_416_as_already_complete(tmp_path):
    import urllib.error

    from galactica.setup import download

    destination = tmp_path / "dump.ndjson"
    destination.write_bytes(b"complete")

    def opener(request, timeout=None):
        raise urllib.error.HTTPError("u", 416, "Range Not Satisfiable", {}, None)

    assert download("https://example/d", destination, opener=opener) == destination


def test_download_failure_keeps_the_partial_file_for_resume(tmp_path):
    import urllib.error

    from galactica.setup import download

    destination = tmp_path / "dump.ndjson"
    destination.write_bytes(b"partial")

    def opener(request, timeout=None):
        raise urllib.error.URLError("connection reset")

    assert download("https://example/d", destination, opener=opener) is None
    assert destination.read_bytes() == b"partial"  # not truncated


def test_ollama_is_installed_automatically_where_possible():
    """Ollama is a hard requirement, so setup installs it rather than asking."""
    from galactica.setup import ollama_install_command

    # The official script covers Linux and macOS.
    for system in ("Linux", "Darwin"):
        command = ollama_install_command(system)
        assert command[:2] == ["sh", "-c"]
        assert "ollama.com/install.sh" in command[2]
    # Windows ships an installer app: nothing safe to run unattended.
    assert ollama_install_command("Windows") is None


def test_claude_code_install_is_optional_and_npm_based():
    from galactica.setup import claude_install_command

    assert claude_install_command(True) == ["npm", "install", "-g", "@anthropic-ai/claude-code"]
    assert claude_install_command(False) is None


def test_prompts_refuse_rather_than_assume_without_a_terminal(monkeypatch, capsys):
    """`curl | sh` leaves stdin on the script: a default of yes would start a
    68 GB download nobody agreed to."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert ask_yes("Download 68 GB?", default=True) is False
    assert "skipped" in capsys.readouterr().out
    # --yes is the deliberate opt-in.
    assert ask_yes("Download 68 GB?", default=True, assume_yes=True) is True


def test_prompts_use_the_default_on_an_empty_answer(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert ask_yes("proceed?", default=True) is True
    assert ask_yes("proceed?", default=False) is False


def test_installer_runs_setup_with_a_real_terminal():
    """The one-liner should be enough; the wizard needs /dev/tty to prompt."""
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "install.sh").read_text()
    assert "galactica" in script and "setup < /dev/tty" in script
    assert "GALACTICA_NO_SETUP" in script  # escape hatch for scripted installs


def test_a_terminal_means_a_person_who_wants_the_corpus(monkeypatch):
    """Setup distinguishes a person from a Dockerfile: only the latter skips
    the corpus, since a 68 GB download is never what CI intended."""
    from galactica.setup import interactive

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert interactive() is True
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert interactive() is False


def test_setup_downloads_the_corpus_without_asking():
    """The corpus is the product; confirming it is friction with one answer."""
    from pathlib import Path

    cli = (Path(__file__).resolve().parents[1] / "src/galactica/cli.py").read_text()
    corpus_step = cli[cli.index("corpus     {source.title}") : cli.index("written = write_config_file")]
    assert "ask_yes" not in corpus_step  # no prompt
    assert "free up space" in corpus_step  # but disk is still checked
    assert "no terminal attached" in corpus_step  # and CI still opts out
