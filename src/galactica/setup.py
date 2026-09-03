"""First-run setup: pick a model for this machine, fetch a corpus, record both.

Everything here is optional and resumable. Each step reports what it will cost
in download size, disk and time before doing it, because the corpus is measured
in tens of gigabytes and nobody should discover that halfway through.
"""

from __future__ import annotations

import platform
import shutil
import sys
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import Config, config_path, read_config_file, write_config_file
from .hardware import Hardware, detect
from .registry import (
    ModelOption,
    SourceOption,
    context_budget_for,
    models,
    recommend_model,
    source_named,
    sources,
)
from .store import open_db, stats


@dataclass
class Readiness:
    """What is missing before Galactica can answer anything."""

    ollama_running: bool
    model_installed: bool
    corpus_documents: int
    configured: bool

    @property
    def ready(self) -> bool:
        return self.ollama_running and self.model_installed and self.corpus_documents > 0

    def next_step(self) -> str | None:
        if not self.ollama_running:
            return "Ollama is not running. Start it, then run: galactica setup"
        if not self.model_installed:
            return "No model installed yet. Run: galactica setup"
        if self.corpus_documents == 0:
            return "No corpus ingested yet. Run: galactica setup"
        return None


def installed_models(cfg: Config) -> list[str]:
    try:
        with urllib.request.urlopen(f"{cfg.ollama_base_url}/api/tags", timeout=5) as response:
            import json

            return [m.get("name", "") for m in json.loads(response.read().decode()).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def ollama_running(cfg: Config) -> bool:
    try:
        with urllib.request.urlopen(f"{cfg.ollama_base_url}/api/tags", timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False


def check(cfg: Config) -> Readiness:
    running = ollama_running(cfg)
    tags = installed_models(cfg) if running else []
    documents = 0
    if cfg.db_path.exists():
        try:
            conn = open_db(cfg.db_path)
            documents = stats(conn)["documents"]
            conn.close()
        except Exception:  # pragma: no cover - corrupt or busy database
            documents = 0
    return Readiness(
        ollama_running=running,
        model_installed=cfg.model in tags,
        corpus_documents=documents,
        configured=bool(read_config_file()),
    )


# ------------------------------------------------------------------- interaction


def ask_yes(question: str, default: bool = True, assume_yes: bool = False) -> bool:
    """Ask a yes/no question.

    With no terminal attached the answer is no, never the default. `curl | sh`
    leaves stdin pointing at the script rather than the user, and taking the
    default there would start a 68 GB download nobody agreed to. Pass --yes to
    accept everything deliberately.
    """
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(f"{question} — skipped (no terminal attached; use --yes to accept)")
        return False
    try:
        answer = input(f"{question} {'[Y/n]' if default else '[y/N]'} ").strip().lower()
    except EOFError:
        return False
    if not answer:
        return default
    return answer in ("y", "yes")


def interactive() -> bool:
    """Is a real terminal attached?

    Distinguishes a person running setup — who wants a working install, corpus
    included — from a Dockerfile or CI job, where a 68 GB download is never what
    was intended.
    """
    return sys.stdin.isatty()


def free_gb(path: Path) -> float:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / 1024**3


# ------------------------------------------------------------------------ steps


OLLAMA_INSTALLER = "https://ollama.com/install.sh"


def ollama_install_command(system: str) -> list[str] | None:
    """How to install Ollama here, or None if it cannot be automated.

    Ollama's official script covers Linux and macOS. Windows ships an installer
    application instead, so there is nothing safe to run unattended there.
    """
    if system in ("Linux", "Darwin"):
        return ["sh", "-c", f"curl -fsSL {OLLAMA_INSTALLER} | sh"]
    return None


def install_ollama(assume_yes: bool = False) -> bool:
    """Install Ollama. It is a hard requirement, so this is not optional."""
    command = ollama_install_command(platform.system())
    if command is None:
        print("  Ollama is required and cannot be installed automatically here.")
        print("     Get it from https://ollama.com/download, then run galactica setup again")
        return False
    print(f"  Ollama is required and not installed. Running: {' '.join(command)}")
    if subprocess.run(command, check=False).returncode != 0:
        print("  installation failed; see https://ollama.com/download")
        return False
    return True


def start_ollama(cfg: Config) -> bool:
    """Start the Ollama server in the background and wait for it to answer."""
    if not shutil.which("ollama"):
        return False
    print("  starting ollama")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    for _ in range(30):
        if ollama_running(cfg):
            return True
        time.sleep(1)
    return False


def claude_install_command(has_npm: bool) -> list[str] | None:
    """How to install Claude Code here, or None if it must be done by hand."""
    if has_npm:
        return ["npm", "install", "-g", "@anthropic-ai/claude-code"]
    return None


def offer_claude_code(assume_yes: bool = False) -> bool:
    """Claude Code is optional: `galactica ask` is the primary interface.

    Offered because `claude-lookup` needs it, and skipping costs the user
    nothing else.
    """
    if shutil.which("claude"):
        return True
    command = claude_install_command(bool(shutil.which("npm")))
    print("\nclaude     Claude Code is not installed (optional)")
    print("           With it, `claude-lookup` uses this corpus as the model.")
    print("           Without it, `galactica ask` works exactly the same.")
    if command is None:
        print("           To add it later: https://claude.com/product/claude-code")
        return False
    print(f"           This will run: {' '.join(command)}")
    if not ask_yes("\nInstall Claude Code too?", False, assume_yes):
        print("  skipped — add it later with: " + " ".join(command))
        return False
    if subprocess.run(command, check=False).returncode != 0:
        print("  installation failed; see https://claude.com/product/claude-code")
        return False
    return True


def pull_model(tag: str) -> bool:
    """Pull a model, streaming ollama's own progress to the terminal."""
    if not shutil.which("ollama"):
        print("  ollama command not found; install Ollama first: https://ollama.com/download")
        return False
    print(f"  pulling {tag} (this is a download; ollama will show progress)")
    result = subprocess.run(["ollama", "pull", tag], check=False)
    return result.returncode == 0


HF_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"
CHUNK = 1024 * 1024
PROGRESS_EVERY = 200 * CHUNK


def source_url(source: SourceOption) -> str | None:
    """Direct download URL for a registered corpus."""
    if source.url:
        return source.url
    if source.hf_repo and source.hf_file:
        return HF_URL.format(repo=source.hf_repo, file=source.hf_file)
    return None


def download(url: str, destination: Path, expected_gb: float = 0.0, opener=None) -> Path | None:
    """Download with resume, over plain HTTPS, with no third-party dependency.

    A 68 GB download will be interrupted sooner or later, so an existing partial
    file is continued with an HTTP range request rather than restarted. Servers
    that ignore the range answer 200 instead of 206, in which case the file is
    written from scratch.
    """
    opener = opener or urllib.request.urlopen
    destination.parent.mkdir(parents=True, exist_ok=True)
    done = destination.stat().st_size if destination.exists() else 0

    request = urllib.request.Request(url)
    if done:
        request.add_header("Range", f"bytes={done}-")
        print(f"  resuming at {done / 1024**3:.1f} GB")

    try:
        response = opener(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and done:  # already complete
            print(f"  {destination.name} already complete")
            return destination
        print(f"  download failed: HTTP {exc.code}")
        return None
    except (urllib.error.URLError, OSError) as exc:
        print(f"  download failed: {exc}")
        return None

    resuming = response.status == 206 and done > 0
    if done and not resuming:
        print("  server does not support resume; starting over")
        done = 0
    remaining = int(response.headers.get("Content-Length") or 0)
    total = done + remaining if remaining else int(expected_gb * 1024**3)

    mode = "ab" if resuming else "wb"
    written = done
    last_report = 0
    try:
        with response, destination.open(mode) as handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                written += len(block)
                if written - last_report >= PROGRESS_EVERY:
                    last_report = written
                    if total:
                        print(f"  {written / 1024**3:6.1f} / {total / 1024**3:.1f} GB "
                              f"({written * 100 // total}%)", flush=True)
                    else:
                        print(f"  {written / 1024**3:6.1f} GB", flush=True)
    except KeyboardInterrupt:
        print(f"\n  stopped at {written / 1024**3:.1f} GB — rerun to resume")
        return None
    except OSError as exc:
        print(f"  download failed after {written / 1024**3:.1f} GB: {exc}")
        print("  rerun to resume from where it stopped")
        return None

    print(f"  downloaded {written / 1024**3:.1f} GB")
    return destination


def fetch_source(source: SourceOption, target_dir: Path) -> Path | None:
    """Download a registered corpus into the drop directory."""
    url = source_url(source)
    if not url:
        print(f"  {source.name} has no download configured; place the files yourself")
        return None
    name = source.hf_file or url.rsplit("/", 1)[-1]
    destination = target_dir / name
    if destination.exists() and source.download_gb:
        size_gb = destination.stat().st_size / 1024**3
        if size_gb >= source.download_gb * 0.99:
            print(f"  {name} already present ({size_gb:.1f} GB)")
            return destination
    print(f"  downloading {name} ({source.download_gb:.0f} GB) from {url.split('/')[2]}")
    return download(url, destination, expected_gb=source.download_gb)


def plan_corpus(source: SourceOption, available_gb: float) -> tuple[int | None, str]:
    """How much of a corpus fits here: (max_documents or None for all, why)."""
    if available_gb >= source.peak_gb:
        return None, f"full corpus ({source.documents:,} documents, ~{source.db_gb:.0f} GB)"
    # Ingest needs room for the download and the database at the same time.
    for fraction in (0.5, 0.25, 0.1, 0.05):
        documents = int(source.documents * fraction)
        needed = source.download_gb + source.db_gb_for(documents) + 5
        if available_gb >= needed:
            return documents, (
                f"{documents:,} documents (~{source.db_gb_for(documents):.0f} GB) — "
                f"a sample, because the full corpus needs {source.peak_gb:.0f} GB free"
            )
    return 0, (
        f"not enough space: {available_gb:.0f} GB free, and even a small sample "
        f"needs {source.download_gb + 5:.0f} GB for the download alone"
    )
