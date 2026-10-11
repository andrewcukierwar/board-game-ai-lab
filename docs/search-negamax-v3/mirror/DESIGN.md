# Phase A: mirror-canonicalized score transposition table

Predeclared 2026-10-10, before any performance measurements.
Baseline implementation SHA: `60b99b0d42145da149f433907585b809060c946d`.
Production files remain unchanged until independent acceptance and full suite.

## Hypothesis and independent variants

A/direct: immutable starting engine loaded from Git, unchanged.
B/mirror: at every nonterminal interior node reflect the seven 7-bit column
chunks, choose the smaller full two-player packed board identity, preserve
mover and remaining depth, remap hints on storage/probe.
C/mirror-selective: the identical operation only at remaining depth >=3.
At shallower depths the direct key and direct hints apply. This is justified
before measurement: shallow entries save little descendant work and dominate
lookup count, so canonicalization overhead may exceed reuse. The threshold
is fixed, never tuned from results. No cross-depth scores/hints or preparatory
iterations. No selective pruning.

Column chunks include all six playable rows and sentinel; reflection is an
involution and preserves player identity. Canonicalization identifies only
horizontal symmetry orbits; packed fields preserve full X/O/mover/depth.
Symmetric identities choose direct orientation; center hints map to themselves.
EXACT/LOWER/UPPER and signed depth-sensitive entry encoding stay unchanged.
Every legal root gets a full window in historical center order. State must
restore in finally on recursive/root errors. The caller remains untouched.

## Positions, measurement and correctness

Reuse all 28 immutable v2 iterative manifest positions (including its post-hoc
tail, excluded from acceptance), plus four held-out nonterminal legal histories
at 7/13/21/29 plies using Random(20261012), uniform legal moves and rejection
only for terminals/duplicate boards. Freeze histories in manifest before timing.
Depths 4/6/8/10; seven paired samples, one discarded warm-up per variant and
condition, rotate variant order by condition/repetition. Constructor warm-ups
64. Measure complete choose_move wall/process CPU including conversion,
search, all roots and normal TT disposal. Diagnostics outside timing: nodes,
heuristic leaves, terminal returns, probes/hits/cutoffs, canonicalization count,
reflected hits, entries, reachable retained bytes and traced peak/current.
Two separate traced runs per condition. Report geometric A/B speedup, summed
medians, all per-condition ratios and sample dispersion. Reflect-overhead and
profiling are explanatory diagnostics only, never acceptance substitutes.

Correctness: exact ordered root vectors/moves versus immutable baseline for
all decisions; deterministic counters within each variant; independent array
minimax at depths 1/2/3 over all manifest boards plus dense draw horizons;
mirrored boards, symmetric center, both movers/depths, huge signed scores,
bound-to-full-window reuse, cached mirror move mappings, invalid/full hints,
exception rollback and inherited evidence. No bounds mislabeled as exact.

## Fixed acceptance gates

All correctness must pass. Non-tail conditions: geometric wall speedup >=1.05;
sum candidate median wall <=0.95*A and CPU <=0.97*A. For broad conditions
(depth 8/10 with >=2000 baseline nodes), geometric >=1.10 and >=70% faster.
Each condition including tail: wall <=1.20*A OR added <=0.25ms; for A>=50ms,
wall <=1.20*A AND nodes <=1.25*A. Retained/peak <=1.25*A OR added <=64KiB
per condition; broad summed retained <=1.15*A. Held-out geometric >=1.00.
Choose eligible fastest summed wall; reject all if no variant passes. Speed
claims apply only to this laptop workload, never strength or production latency.

## Budget and reproduction

Maximum Phase A measured/diagnostic compute 1200s, validation 300s. Batches
hold the shared lock for <=450s using a same-shell trap; release between
independent condition blocks. No competing heavy work used as evidence.
Commands (runner implementation will be source-frozen before measurement):

```sh
.venv/bin/python -m scripts.benchmark_negamax_v3 declare --phase mirror
# Commit and push manifest + source snapshots before run.
scripts/with_benchmark_lock.sh .venv/bin/python -m scripts.benchmark_negamax_v3 run --phase mirror --start 0 --stop 8
# Repeat independent blocks 8:16, 16:24, 24:32.
.venv/bin/python -m scripts.benchmark_negamax_v3 audit --phase mirror
.venv/bin/python -m scripts.benchmark_negamax_v3 summarize --phase mirror
```
A budget interruption leaves exclusive partial evidence labeled WIP, never
accepted. Replication uses --directory a new path containing copied DESIGN.md.
