# Galactica

**A local corpus of knowledge that makes a small local model answer like a much larger one — fully offline, with sources cited and gaps labelled.**

[![tests](https://github.com/lukeprofits/galactica/actions/workflows/tests.yml/badge.svg)](https://github.com/lukeprofits/galactica/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/lukeprofits/galactica/main/install.sh | sh
```

**That one command does everything:** installs Galactica, installs Ollama if you
don't have it, picks a local model sized to your hardware, pulls it, then
downloads and indexes the corpus. No further setup. Expect it to run a while —
the corpus is 68 GB, and the download resumes if interrupted.

## Two ways to use it

**In the terminal** — nothing else required:

```sh
galactica ask "How did the Free State of Fiume come about?"
```

**Inside Claude Code** — how most people will want it:

```sh
claude-lookup
```

Claude Code becomes the interface, but the answers still come from the local
model reading the local corpus. Every request goes to `localhost`, so this uses
**no Anthropic tokens and no Claude subscription** — you're getting Claude Code's
chat interface, history, streaming and file context in front of a model running
on your own machine, offline.

Worth being clear about the trade: the model answering is yours, not Claude. That
makes it strong on knowledge questions and noticeably weaker on multi-step
agentic coding, where Claude itself is far better. Use `claude-lookup` when you
want grounded facts offline; use Claude normally when you want Claude.

If you don't have Claude Code, setup offers to install it — and skipping costs
you nothing, since `galactica ask` does the same work.

---

## Why

A 4B model doesn't know what displacement the Yamaha XVZ13D Venture Royale used.
Asked anyway, it says "1285 cc" with total confidence. The real answer, 1,294 cc,
is written down in an encyclopedia sitting on the same disk.

Galactica puts 882,703 Grokipedia articles into a local search index and gives
the model one job: work out what it needs, look it up, answer from what it found,
cite it, and say plainly which parts the corpus didn't cover.

```
$ galactica ask "What engine displacement did the Yamaha XVZ13D Venture Royale Mk2 use?"

The Yamaha XVZ13D (Mk2) used a 1,294 cc (79.0 cu in) engine [S1]. Its trunk and
side bags were rigidly mounted, unlike the Mk1's removable trunk and bags [S1].

sources:
  [S1] Yamaha Venture Royale — Vehicle Information > Mechanical
        grokipedia v2025-10-29 · CC BY-SA 4.0 · chunks: aab4b4667f964527#0002
```

## Results

Three model sizes, same corpus, same 25 factual questions, scored identically.
`+GAL` means the model had the corpus.

```
                                 4b    4b+GAL        8b    8b+GAL       35b   35b+GAL
-------------------------------------------------------------------------------------
keyword coverage              0.167     0.881     0.357     0.881     0.500     0.905
answered when answerable      0.286     1.000     0.381     1.000     0.714     1.000
refused when absent           0.750     0.750     1.000     1.000     1.000     1.000
retrieval hit-rate                -     1.000         -     0.952         -     1.000
answers with a citation           -     0.680         -     0.880         -     0.920
fabricated citations          0.000     0.000     0.000     0.000     0.000     0.000

weights on disk               2.5GB     2.5GB     5.2GB     5.2GB      24GB      24GB
```

**Read the top row across.** Alone, factual coverage spans 0.167 to 0.500 across
a 10× size range. With the corpus, all three land between 0.881 and 0.905 — a
2.7% spread. A 2.5 GB model reading the corpus (0.881) beats a 24 GB model
recalling from its weights (0.500).

**Retrieval is model-independent** (21/21, 20/21, 21/21). The corpus finds the
material whoever asks.

**What size still buys you.** Citation discipline scales with parameters (0.680 →
0.880 → 0.920): the small model uses retrieved facts correctly but attributes
them less reliably. Refusing when it should also scales — the 4B declined only
three of four unanswerable questions, in both arms. Recall moves into data;
judgement doesn't.

Reproduce it: `galactica eval eval/questions-grokipedia.jsonl --compare`

### Where it shows

| question | model alone | with the corpus |
|---|---|---|
| Spanish industrial socket dimensions | "4.8 mm diameter, spaced 19 mm apart" — every number invented | "10 by 2 mm, spaced 30 mm apart, earth pin 4.5 mm" [S1] |
| Who published Katana Zero's mobile ports | "Devolver Digital… require an internet connection" | "Netflix… require an active Netflix subscription" [S2] |
| Nvidia's close on 15 Aug 2026 | "I do not have enough information." | "most recent documented is $186.26 on 24 Oct 2025 [S1]. I do not have access to real-time market data." |

The first row is the failure that matters: fluent, precise, hedged in the right
places, citing a plausible standard — and wrong in every digit.

## Other ways to install

```sh
pipx install git+https://github.com/lukeprofits/galactica && galactica setup
```

Or from a clone: `python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"`.

Needs Python 3.11+. Runtime dependencies: **none** — standard library only,
SQLite FTS5 for search. [Ollama](https://ollama.com/download) is installed by
setup on Linux and macOS; on Windows it points you at the installer.
`GALACTICA_NO_SETUP=1` skips the wizard for scripted installs.

## Setup

The installer runs this for you; run it again any time to add a corpus or change
model.

```sh
galactica setup
```

Installs Ollama if missing, detects your hardware, picks the best model that
actually fits, pulls it, then downloads and indexes the corpus — printing the
download size, peak disk needed, final size and licence as it goes. If the full
corpus won't fit your disk it ingests a random-offset sample that will, and says
so. Everything lands in `~/.config/galactica/config.toml`.

Downloads resume: interrupt a 68 GB transfer at 40 GB, rerun, and it continues
from 40 GB. Claude Code is the one optional step and defaults to no —
`claude-lookup` needs it, `galactica ask` doesn't.

```sh
galactica models     # what fits this machine
galactica fetch      # available corpora, with sizes and licences
galactica doctor     # provider, model, corpus, derived context sizes
```

Ingesting the full Grokipedia dump takes ~95 minutes at ~130 articles/sec **with
no model calls** — it's pure local text processing. 882,703 articles → 19.5M
chunks → 65 GB.

## Use

```sh
galactica ask "How did the Free State of Fiume come about?"
galactica ask "..." --mode both        # with and without the corpus, side by side
galactica ask "..." --show-context     # exactly what the model was shown
galactica search "treaty of rapallo"   # retrieval only: no model, ~10ms
```

`search` is the first thing to try when an answer looks wrong — it tells you
whether retrieval failed or the model did.

## How the Claude Code integration works

Claude Code speaks the Anthropic API to whatever `ANTHROPIC_BASE_URL` points at,
so Galactica sits in that seam: `claude-lookup` starts a local gateway, points
Claude Code at it, and shuts it down when you exit. Each turn it retrieves
against your message, injects the excerpts into the local model's prompt for that
one call, and returns only the answer.

**The corpus never enters the transcript** — nothing accumulates, nothing is
re-sent next turn, and what persists is a fact you can build on.

Retrieval is gated: knowledge questions get it, code work doesn't (an
encyclopedia can't help with "fix this test"). Answers stream token by token.
Each turn costs 30–90s, since there is no prompt caching on a local model.

## How it works

```
question → plan → search 19.5M chunks → fuse → pack into a token budget
         → answer citing [S1]… → validate every citation resolves
```

Articles are rebuilt as markdown (sections → headings, tables → tables), split on
heading boundaries at ~700 tokens keeping their section path, and indexed with
SQLite FTS5. Retrieval walks a cost ladder from "all terms required" (~10ms)
through a frequency-budgeted OR to a titles-only fallback, plus a `NEAR`
proximity pass, fused with Reciprocal Rank Fusion and reranked by query-term
coverage.

Every chunk carries its source, section path, URI, licence, scrape date, stable
IDs and a checksum, so every citation resolves to auditable text.

**Grounding has two modes.** `augmented` (default) prefers the corpus for facts
and cites it, then answers from the model's own knowledge where the corpus is
silent, labelling those parts with an `UNCITED:` line. `strict` answers only from
the corpus. Strict was the original default and it was wrong: on 25 practical
homesteading and field-medicine questions it hedged 84% of answers and refused
12%, because Grokipedia has articles *about* canning but no canning tables.
**Corpus selection matters more than corpus size.**

## Adding models and corpora

Both are data files, not code:

- `src/galactica/data/models.toml` — a model is one entry with its weight size
  and the memory it needs including KV cache.
- `src/galactica/data/sources.toml` — a corpus is one entry with a Hugging Face
  repo or URL, a loader profile, sizes and licence. `galactica fetch <name>` then
  works with no new code: loaders for markdown, JSONL/NDJSON/parquet with column
  mapping, Wikipedia XML and nested Grokipedia already exist.

Wikipedia is already registered. PRs adding either are welcome.

## Hardware

What decides whether a model fits is the KV cache, not the weights: Ollama
reserves the model's *maximum* context unless told otherwise, so `qwen3:4b` —
2.5 GB of weights — asked for **42 GB** at its native 256K window. Galactica
derives the context size from your budgets instead.

| memory | model | setting | derived context |
|---|---|---|---|
| 6 GB VRAM | `qwen3:4b` | `context_budget = 4000` | 8192 |
| 12 GB | `qwen3:8b` | `context_budget = 8000` | 14336 |
| 24 GB+ / Apple unified | `qwen3.6:35b` | defaults | 22528 |

`galactica setup` picks these for you. Measured footprints at 32K context:
`qwen3:4b` 7.6 GB, `qwen3:8b` 10 GB.

## Configuration

`~/.config/galactica/config.toml`, overridden by environment variables, then
flags. The ones that matter:

| setting | default | meaning |
|---|---|---|
| `model` | chosen at setup | Ollama model tag |
| `context_budget` | 16000 | ceiling on retrieved context |
| `grounding` | `augmented` | or `strict`: corpus only |
| `num_ctx` | derived | KV cache size; unset means computed from the budgets |
| `think` | true | model reasoning; the corpus supplies facts, not judgement |
| `max_sources` | 8 | citable source blocks per answer |

## Known limits

- **Precision past rank 1.** The right chunk ranks first reliably; slots 2–8
  carry topically adjacent noise. A reranker is the next step.
- **~30% of sentences in cited answers carry no citation** — mostly connective
  prose, but that's the measured number (0.287 for the 35B, 0.485 for the 4B).
- **Verbose questions cost seconds**, keyword queries cost milliseconds.
- **Token counts are estimates** (4 chars/token), so the budget is approximate.
- **BM25 plus proximity, no embeddings by default.** `--hybrid` exists; embedding
  19.5M chunks takes hours.
- **Wikitext cleanup is lossy** — infobox tables don't survive.
- **Eval metrics are proxies.** Keyword coverage and citation validity measure
  grounding, not correctness of reasoning.

## Tests

```sh
python -m pytest      # 234 tests, fully offline, no model calls
```

Everything runs against a deterministic stub provider: no network, no model, no
downloads.

MIT licensed. The corpora it indexes are not — honouring their terms is yours.

`ARCHITECTURE.md` has the thesis, the retrieval cost design, the grounding
experiment, the measurement traps that cost the most debugging time, and the
layers this is built to grow into.
