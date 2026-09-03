---
title: Cache invalidation strategies (compiled)
license: CC0-1.0
version: 2026-02-20
collection: compiled-intelligence
doc_type: compiled_analysis
---

# Cache invalidation strategies (compiled)

A compiled decision guide: the tradeoffs are precomputed here so a reader can
select a strategy without re-deriving the analysis.

## The four viable strategies

**TTL expiry.** Each entry carries a lifetime. Simple, needs no coordination,
and bounds staleness by construction. Cost: staleness up to the full TTL, and a
thundering herd when popular keys expire together. Mitigate with jitter of 10 to
20 percent of the TTL.

**Write-through invalidation.** The writer deletes or updates the cache entry in
the same logical operation as the database write. Staleness approaches zero for
single-writer paths. Cost: correctness depends on the write path never being
bypassed, and a failed invalidation is silent and unbounded in duration.

**Versioned keys.** The cache key embeds a version or content hash, so a write
makes old entries unreachable rather than wrong. Invalidation becomes a no-op
and races disappear. Cost: garbage accumulates and needs eviction pressure or a
sweeper, and every reader must be able to compute the current version cheaply.

**Change-data-capture fanout.** A log tail (binlog, WAL, stream) drives
invalidation asynchronously. Decouples writers from cache topology and handles
writes that bypass the application. Cost: an entire pipeline to operate, and
staleness equal to end-to-end log lag.

## Selection rule

If bounded staleness is acceptable, use TTL with jitter; it is the only option
with no coordination failure mode. If reads must reflect writes immediately and
all writes pass one service, use write-through with a versioned key as backstop.
If writes arrive from outside the application, CDC fanout is the only correct
choice. Versioned keys are the best default for immutable derived artifacts.

## The failure nobody plans for

The common production incident is not a stale entry, it is an invalidation
storm: a deploy or schema change invalidates a large key space at once and the
origin absorbs the full read load. Every strategy above needs a request
coalescing layer (single-flight) in front of the origin, independent of the
invalidation choice.
