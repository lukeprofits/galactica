#!/bin/sh
# Galactica installer.
#
#   curl -fsSL https://raw.githubusercontent.com/lukeprofits/galactica/main/install.sh | sh
#
# Creates an isolated virtualenv, installs Galactica into it, and links the
# `galactica` and `claude-lookup` commands onto your PATH. Nothing is installed
# system-wide and nothing is downloaded beyond the package itself: models and
# corpora are fetched later, with your consent, by `galactica setup`.

set -eu

REPO="${GALACTICA_REPO:-https://github.com/lukeprofits/galactica}"
REF="${GALACTICA_REF:-main}"
PREFIX="${GALACTICA_PREFIX:-$HOME/.local/share/galactica}"
BINDIR="${GALACTICA_BINDIR:-$HOME/.local/bin}"

say() { printf 'galactica: %s\n' "$1"; }
die() { printf 'galactica: %s\n' "$1" >&2; exit 1; }

# --- prerequisites --------------------------------------------------------
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)'; then
            PYTHON="$candidate"; break
        fi
    fi
done
[ -n "$PYTHON" ] || die "Python 3.11 or newer is required (none found on PATH)"
command -v git >/dev/null 2>&1 || die "git is required"

say "using $($PYTHON --version)"

# --- install --------------------------------------------------------------
mkdir -p "$PREFIX" "$BINDIR"
if [ -d "$PREFIX/src/.git" ]; then
    say "updating existing checkout in $PREFIX/src"
    git -C "$PREFIX/src" fetch --quiet origin "$REF"
    git -C "$PREFIX/src" checkout --quiet "$REF"
    git -C "$PREFIX/src" reset --hard --quiet "origin/$REF"
else
    say "cloning $REPO"
    rm -rf "$PREFIX/src"
    git clone --quiet --depth 1 --branch "$REF" "$REPO" "$PREFIX/src"
fi

say "creating virtualenv in $PREFIX/venv"
"$PYTHON" -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/python" -m pip install --quiet --upgrade pip
say "installing galactica"
"$PREFIX/venv/bin/python" -m pip install --quiet "$PREFIX/src"

for command in galactica claude-lookup; do
    ln -sf "$PREFIX/venv/bin/$command" "$BINDIR/$command"
done
say "linked galactica and claude-lookup into $BINDIR"

# --- next steps -----------------------------------------------------------
case ":$PATH:" in
    *":$BINDIR:"*) ;;
    *) say "NOTE: $BINDIR is not on your PATH. Add this to your shell profile:"
       printf '\n    export PATH="%s:$PATH"\n\n' "$BINDIR" ;;
esac

if ! command -v ollama >/dev/null 2>&1; then
    say "Ollama is not installed — 'galactica setup' will install it for you"
fi

# --- first-run setup -------------------------------------------------------
# Run the wizard now so a single command is enough. `curl | sh` leaves stdin
# pointing at the script, so the terminal is reattached explicitly; without a
# terminal the wizard would be answering its own prompts.
if [ "${GALACTICA_NO_SETUP:-0}" = "1" ]; then
    say "skipping setup (GALACTICA_NO_SETUP=1)"
elif [ -r /dev/tty ] && [ -w /dev/tty ]; then
    say "starting setup"
    printf '\n'
    "$PREFIX/venv/bin/galactica" setup < /dev/tty || true
    printf '\n'
    say "if setup did not finish, run it again any time: galactica setup"
    exit 0
else
    say "no terminal available for the interactive setup"
fi

cat <<'NEXT'

Installed. Next:

    galactica setup      installs Ollama if missing, picks a model for your
                         hardware, downloads and indexes a corpus, saves config
    galactica ask "..."  ask a question
    claude-lookup        use it as the model inside Claude Code

NEXT
