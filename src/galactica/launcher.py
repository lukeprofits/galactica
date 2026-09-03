"""`claude-lookup`: start Claude Code against the corpus-backed local model.

The analogue of `ollama launch claude -- <model>`, except requests pass through
the Galactica gateway first, so each turn is grounded in the local corpus before
the model sees it and only the answer returns to the transcript.

A Python entry point rather than a shell script, so `pip install` puts it on
PATH on every platform instead of relying on a hand-made symlink.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .config import Config

HEALTH_TIMEOUT_S = 45.0
POLL_INTERVAL_S = 0.5


def healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _wait_for_health(base_url: str, process: subprocess.Popen | None) -> bool:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if healthy(base_url):
            return True
        if process is not None and process.poll() is not None:
            return False
        time.sleep(POLL_INTERVAL_S)
    return False


def claude_environment(base_url: str, model_tag: str) -> dict[str, str]:
    """Environment that points Claude Code at the gateway.

    Every model alias is pinned to the same tag because the gateway ignores the
    requested name and always serves the configured local model; without pinning,
    an alias can resolve to a hosted Claude model this endpoint cannot serve.
    Non-essential traffic is disabled because background calls would queue behind
    real turns on a single local model.
    """
    return {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": os.environ.get("ANTHROPIC_AUTH_TOKEN", "local"),
        "ANTHROPIC_MODEL": model_tag,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model_tag,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model_tag,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_tag,
        "ANTHROPIC_SMALL_FAST_MODEL": model_tag,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = Config.load()
    host = os.environ.get("GALACTICA_HOST", "127.0.0.1")
    port = os.environ.get("GALACTICA_PORT", "8787")
    base_url = f"http://{host}:{port}"
    claude_bin = os.environ.get("GALACTICA_CLAUDE_BIN", "claude")

    if not cfg.db_path.exists():
        print(
            f"claude-lookup: no corpus at {cfg.db_path}\n"
            "  Build one first:  galactica setup",
            file=sys.stderr,
        )
        return 2
    if not shutil.which(claude_bin):
        print(
            f"claude-lookup: '{claude_bin}' not found on PATH "
            "(set GALACTICA_CLAUDE_BIN to its location)",
            file=sys.stderr,
        )
        return 2

    gateway: subprocess.Popen | None = None
    if healthy(base_url):
        print(f"claude-lookup: reusing gateway on {base_url}")
    else:
        log_path = cfg.data_dir / "gateway.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        serve_args = os.environ.get("GALACTICA_SERVE_ARGS", "").split()
        log = log_path.open("a", encoding="utf-8")
        gateway = subprocess.Popen(
            [sys.executable, "-m", "galactica.cli", "serve", "--host", host, "--port", port,
             *serve_args],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if not _wait_for_health(base_url, gateway):
            gateway.terminate()
            print(f"claude-lookup: gateway did not start; see {log_path}", file=sys.stderr)
            return 2
        print(f"claude-lookup: gateway started on {base_url} (log: {log_path})")

    environment = {**os.environ, **claude_environment(base_url, "galactica")}
    if os.environ.get("GALACTICA_LAUNCH_DRY_RUN") == "1":
        for key, value in sorted(claude_environment(base_url, "galactica").items()):
            print(f"{key}={value}")
        print(f"claude_bin={claude_bin}")
        status = 0
    else:
        try:
            status = subprocess.run([claude_bin, *argv], env=environment, check=False).returncode
        except KeyboardInterrupt:  # pragma: no cover - interactive
            status = 130

    if gateway is not None:
        gateway.terminate()
        try:
            gateway.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            gateway.kill()
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
