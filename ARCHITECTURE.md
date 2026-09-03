# Architecture

## The thesis

Models do two jobs: they *know* things and they *reason* about them. Knowing
scales badly — it needs parameters, and it goes stale. Reasoning is cheap by
comparison: a 4B model can extract a fact from a paragraph perfectly well.

So move knowing out of the weights and into data, and leave reasoning alone.
What gets replaced is guessing, not thinking.

The most interesting material isn't encyclopedia prose but **compiled
intelligence**: analysis a frontier model produced once — ranked tradeoffs, a
diagnostic tree, a worked derivation — that a local model reads forever.
Expensive thinking, amortised. `corpus/seed/compiled-*.md` shows the shape.

## Measured

Three model sizes, one corpus, 25 factual questions.

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

Unaided coverage spans 0.167–0.500 across a 10× parameter range; with the corpus
all three land in 0.881–0.905. Retrieval is model-independent. **Two things the
corpus provably cannot supply:** citation discipline (0.680 → 0.880 → 0.920) and
knowing when to stop — the 4B declined only three of four unanswerable questions
in *both* arms. Recall moves into data; judgement still costs parameters.

At corpus scale, full Grokipedia dump on Apple Silicon:

| | |
|---|---|
| Documents / chunks | 882,703 / 19,554,060 (~22 per article) |
| Database | 65 GB (~79 KB per article) |
| Ingest | ~95 min, ~130 articles/sec, zero model calls |
| Retrieval | ~10 ms keyword, 3–6 s verbose natural language |
| `ask` end to end | 30–90 s, almost entirely inference |

Inference cost is flat in corpus size: the model sees at most
`context_budget` tokens whether the index holds 10 documents or 880,000.

## Baseline vs cortex

The claim isn't "retrieval works", it's *uplift* — so both arms are first-class.
`--mode baseline` runs the same model with no retrieval and the same output
contract; `--mode both` runs them side by side; `eval --compare` scores the
difference and persists every run to `eval/runs/<timestamp>.jsonl` with the model
tag and prompt version, so a corpus or prompt change is comparable rather than
arguable.

The question set is deliberately mixed: some questions the model plausibly knows
unaided (a real opponent), some corpus-only, and some answered by nothing in the
corpus, where declining correctly is scored as success.

## Retrieval and context flow

```
question → plan → search + proximity pass → RRF fuse → coverage rerank
         → pack into budget → answer citing [S1]… → validate → optional 2nd hop
```

Design decisions, each with the reason it exists:

**Rank before joining.** The single biggest performance fact. Joining
`documents` and `sources` for every FTS match and *then* sorting by BM25 pays
those joins on rows about to be discarded. Ranking in a subquery over the index
alone, then joining the surviving top-k, is **5–50× faster** (6.05 s → 0.95 s on
one measured query).

**A cost ladder, not one query.** An OR query costs roughly a second per million
matched chunks; an AND costs ~10 ms at any term frequency, because FTS5
intersects doclists. So: all terms required → budgeted OR → relax the strict
match → titles only. A plain OR of every term took **99 seconds** on 19.5M
chunks.

Ordering the OR *before* relaxation cost a correction. Relaxation bets on the
rarest terms, and **rarity is not importance**: for "gradient masking in
adversarial machine learning" the rarest pair is "gradient" and "defence", which
ranked *DRB-HICOM Defence Technologies* first and left the right article out of
the top 24. Scoring every term via OR fixed that and three like it, taking the
eval's retrieval hit-rate from 0.81 to 1.00.

**Term frequencies come from the index.** An `fts5vocab` table gives document
frequencies, and query words are stemmed by a temp FTS5 table using the index's
own tokenizer — so lookups match what was actually indexed, with no stemmer
dependency. Every rung runs under a SQLite progress-handler deadline, so no
query can hang the CLI.

**A proximity pass runs alongside.** Bag-of-words ranking cannot tell *Katana
Zero* (a game) from *DA20 Katana* (an aircraft), and the OR budget may drop
"zero" as a common word — discarding the term that identifies the subject. An
FTS5 `NEAR("a" "b", 3)` match over consecutive query words, fused as its own
list, put the right article at rank 1 in four of five test questions. Slop of 3
lets stopwords through, which a strict phrase cannot: "Free State of Fiume"
indexes "of" while the query drops it.

**One retrieval group per sub-question.** A multi-part question is several
lookups sharing one budget, and fusing them lets the best-served part crowd out
the others — which is how strict mode came to report "the sources do not specify
what not to do" about a snakebite while the answer sat unretrieved.
`search_grouped` interleaves per-sub-question results round-robin. The planner
already emitted `sub_questions`; they were being discarded. The gateway has no
planner, so it splits clauses heuristically instead.

**Coverage reranking, free.** A chunk containing four distinct query terms beats
one repeating a common term four times. Scaling the fused score by distinct-stem
coverage is what stopped "religious, cultural and historical contexts of
ruminants" reaching the context for a pasture stocking-rate question, and it
tightened one measured context from 3,053 to 2,335 tokens — which also made the
answer faster.

**Why RRF.** Fusing a BM25 score with a cosine similarity means normalising two
incomparable scales, and the constants become a tuning liability. Reciprocal
Rank Fusion uses only ranks, so nothing needs recalibrating when an arm is
added.

**Packing: relevance order, with reserved room.** An earlier version ran a
diversity round first — one chunk per document before any second chunk — and it
measurably broke answers, because the document holding the answer usually
occupies several adjacent ranks. Guards instead: no document may exceed 35% of
the budget as more than one block, and a minority of slots (`diversity_slots`,
2 of 8) is reserved for documents not yet represented.

**Neighbour expansion, restrained.** A matching chunk is often the middle of a
thought. Neighbours must be in the same document *and* section (heading-path
prefix), ±1 chunk, only for top-ranked anchors, appended to the anchor's `[Sn]`
block rather than issued as new citable sources, and only ever spending leftover
budget. All of it lives in `select.py`, so nothing bypasses the ceiling.

**Citation validation.** Every `[Sn]` is checked against the selected set; an
unknown label is reported as a fabricated citation, in the CLI and as a metric.

**The planner is skipped when it cannot help.** It's a full round trip (~20–30 s)
that turns a question into keyword queries. "krios breaker order" is already a
query, so `needs_planning` skips short single-clause questions.

## Grounding: help, don't restrict

The original rule was: answer only from the excerpts, refuse otherwise. Right for
"stop a small model inventing a torque spec", wrong in general — which only
showed up on a domain the corpus covers badly.

Twenty-five practical questions (homesteading, preservation, field medicine, in
`eval/questions-shtf.jsonl`) under strict grounding: **21 of 25 hedged with a
`GAP`, 3 refused outright, and the bare model was more useful on roughly half.**
Grokipedia has articles *about* canning and karst aquifers; it has no canning
tables, extension bulletins or first-aid protocols. Strict grounding pinned the
model to the worse of two sources and made it withhold what it knew — asked
about a copperhead bite it described antivenom accurately, then reported that the
sources did not specify contraindicated actions, dropping "do not cut, do not
suck, do not use a tourniquet."

So grounding became a mode. **`augmented`** (default) prefers the corpus for
facts and cites it, then answers from the model's own knowledge where the corpus
is silent, marking those parts with a trailing `UNCITED:` line. **`strict`** is
kept for auditing what the corpus alone supports.

| | strict | augmented |
|---|---|---|
| SHTF set: hedged with `GAP` | 84% | 0% |
| SHTF set: refused outright | 12% | 0% |
| SHTF set: citations per answer | 3.7 | 3.4 |
| Encyclopedia: keyword coverage | 0.833 | 0.905 |
| Encyclopedia: answers with a citation | 0.800 | 0.920 |
| Encyclopedia: refused when absent | 1.000 | 1.000 |

Better everywhere, worse nowhere, for ~11 s per answer. The honesty guarantee
moved rather than disappeared: strict proves it by *refusing*, augmented by
*labelling*. Citation validation is unchanged in both.

The general lesson, and the one to carry into corpus expansion: **corpus
selection matters more than corpus size.** 883k encyclopedia articles lost to
bare weights on the questions that would actually keep someone alive.

## Three measurement traps

Every disappointing number this project produced turned out to be the
measurement, not the system.

1. `refused_when_absent` fell 1.000 → 0.000 when augmented grounding shipped.
   Cause: `declined` string-matched the one refusal sentence strict mode used.
   All four answers were honest refusals.
2. The same detector then missed three further phrasings from a different model
   ("not mentioned in the provided SOURCES", "No SOURCES provide data for…").
   Each model phrases absence its own way.
3. `retrieval_hit` scored 0 when a model found the *dedicated* article ("Moscow
   Cathedral Mosque") instead of the list page a fact was sampled from — better
   retrieval, recorded as failure.

All three made the system look worse than it was, and all three would have been
believed if the number alone had been reported. **When a metric shows a cliff,
read the underlying answers before believing the cliff**, and prefer metrics that
accept every correct behaviour rather than one canonical form of it.

## The model boundary

Everything model-specific sits behind one protocol:

```python
class LLMProvider(Protocol):
    def complete(self, messages, *, temperature=0.0, max_tokens=None,
                 json_mode=False, think=None) -> str: ...
    def embed(self, texts) -> list[list[float]]: ...
    def health(self) -> ProviderHealth: ...
```

`OllamaProvider` implements it over `urllib`; `StubProvider` implements it
deterministically, which is why the whole suite runs offline with no model.
Swapping in llama.cpp, MLX or a hosted API is one file. The only model-specific
concession is stripping `<think>` blocks, because qwen3.x emits reasoning inline
and it must not leak into answers or JSON parsing.

**Reasoning is preserved and budgeted separately.** A client's `max_tokens` is
the *answer* budget; reasoning gets its own allowance on top (`think_reserve`).
Without that split, a `max_tokens` of 400 was consumed entirely by reasoning and
returned an empty answer. If reasoning still exhausts the allowance, the call
retries once without it.

**Context size is derived, not fixed.** Ollama reserves the model's *maximum*
context as KV cache unless told otherwise: `qwen3:4b` is 2.5 GB of weights and
asked for **42 GB** at its native 256K window. `effective_num_ctx()` computes it
from `context_budget + max_answer_tokens + think_reserve + prompt_overhead`, so
lowering the budget for a small card lowers the cache with it. The gateway adds
`client_reserve` because Claude Code's own prompt, tool schemas and history don't
fit inside our budget.

## Corpus and provenance

```
sources     name, kind, source_version, license, uri_base, loader_profile
documents   doc_id, source_id, collection, doc_type, title, uri, license,
            source_version, lang, native_id, checksum, chunk_count, approx_tokens
chunks      chunk_id, doc_id, ord, heading_path, title, char range, text,
            approx_tokens, checksum
chunks_fts  FTS5 external-content index over (text, heading_path, title)
chunks_vocab fts5vocab, for term document frequencies
embeddings  chunk_id, model, dim, vec
```

- **Stable IDs.** `doc_id = sha256(source + native_id)[:16]`,
  `chunk_id = <doc_id>#<ord>`. The same dump record yields the same IDs forever,
  so citations and eval runs survive re-ingests.
- **Checksums everywhere**, giving idempotent re-ingest, `--resume`, and change
  detection when a dump is refreshed.
- **Section paths, not just offsets** — what makes section-scoped expansion and
  readable citations possible, and indexed alongside the text.
- **Version and licence travel with the text**, which is the difference between a
  claim and a traceable claim.
- **Chunk rollups on the document row.** Without them `stats` aggregates 19.5M
  rows and takes minutes; with them it reads 883k and takes 3.5 s.
- **Loaders are dumb, profiles are the seam.** A loader yields
  `RawDoc(native_id, title, text, uri, meta)`; a profile maps dump-specific
  column names onto it, with `--map` overriding. A new dump format is a registry
  entry, not code.

## The gateway

`galactica serve` implements the Anthropic Messages API over stdlib
`http.server`, so any client honouring `ANTHROPIC_BASE_URL` can use the
corpus-backed model as its model. `claude-lookup` starts it, waits for health,
pins every model alias to the local model, and shuts down what it started.

- **The corpus is ephemeral, the answer is not.** Excerpts live in one prompt and
  are discarded; the client's transcript holds only the reply. Context doesn't
  grow with corpus use, and a grounded fact keeps paying off in later turns.
- **Retrieval is gated** — `auto` skips code-shaped messages and mid-agent-loop
  `tool_result` traffic; `always` and `off` override.
- **Tool traffic is translated**, not swallowed: `tool_use` → Ollama
  `tool_calls`, `tool_result` → `role: tool`, schemas → function definitions.
- **Streaming is real.** Tokens forward as generated. Stripping reasoning from a
  stream is not a regex job — a `<think>` tag can split across chunks — so
  `ReasoningFilter` holds back anything that might begin one. A provider without
  `chat_stream` falls back to one blocking call re-framed as the same events, and
  a mid-stream failure closes the message with an explanation rather than leaving
  the client on a truncated stream.

## Operational notes

- **Reads never take a write lock.** `open_db` only writes when the schema is out
  of date. An earlier version stamped a version row on every open, which made
  `eval` fail with "database is locked" during a long ingest.
- **Migrations are additive**, guarded by `SCHEMA_VERSION`, so an existing corpus
  upgrades without re-ingesting.
- **Output is capped.** Without a cap a small model looped past the socket
  timeout and killed a 43-call eval run; and one failed case no longer discards
  the whole run.

## Current limitations

- Precision past rank 1: coverage reranking cut much of the noise, but slots 2–8
  still carry topically adjacent chunks. A cross-encoder reranker is the fix.
- Brute-force vector search when `--hybrid` is on — fine at proof-of-concept
  scale, wrong for millions of chunks without an ANN index.
- Token counts are 4-chars-per-token estimates, so budgets are approximate.
- Single hop by default; `--hops 2` chases the model's own `MISSING:` queries once.
- Wikitext cleanup is lossy: infobox tables don't survive.
- Chunking is structural, not semantic — blind to topic shifts inside a section.
- Eval metrics are proxies: they measure grounding, not correctness of reasoning.

## Layers this is built to grow into

Roughly increasing cost, each slotting into an existing seam:

1. **Compiled intelligence at scale** — frontier models writing the corpus:
   derivations, diagnostic trees, ranked tradeoffs, "what usually goes wrong"
   notes. Where the thesis pays off most, since the reasoning is amortised.
2. **Semantic retrieval by default** — embed once, keep RRF as the fusion layer,
   add an ANN index so vector search scales past brute force.
3. **Reranking** — a cross-encoder or cheap LLM pass over the top ~50 fused hits.
   The interface is already a ranked list.
4. **Specialist models per domain** — the provider boundary already allows a
   different model per call; the planner and the answerer need not be the same
   model.
5. **Knowledge dependency graphs** — explicit edges (prerequisite, refines,
   contradicts, superseded-by) turn retrieval into traversal, and let an answer
   be refused because its source was superseded. The stable `doc_id` is the
   anchor such a graph needs.
6. **Community corpus packs** — signed, versioned, licensed bundles anyone can
   publish and ingest. The provenance model is already shaped for it: a pack is a
   `source` row plus its documents, and a checksum mismatch is detectable.

What holds it together is the invariant the proof of concept enforces: the model
may assert what the corpus contains, every assertion resolves to a citation, and
every citation resolves to licensed, versioned, checksummed text — with anything
outside that labelled as such.
