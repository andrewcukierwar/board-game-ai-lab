# Phase B: principal variation search

Baseline: strongest independently validated Phase A winner; if none eligible,
immutable direct engine `60b99b0d42145da149f433907585b809060c946d`.
Independent B/PVS scouts every child after the first with the integer window
[-alpha-1,-alpha]. If alpha < result < beta, re-search full [-beta,-alpha].
First child (or alpha=-infinity) uses the full window. Fail-high may cut off
without re-search; fail-low remains a bound. Store against the original input
window before TT tightening. Existing key and depth-sensitive score encoding,
terminal precedence, root full windows/ties, evaluation and finally restoration
are preserved. No aspiration, iterative schedules, altered roots or pruning.

Use Phase A's identical 32 histories, depths 4/6/8/10, seven paired samples,
64 table warmups, one discarded decision warmup, two memory repetitions.
Same fixed acceptance gates as Phase A (copied below, not inferred post hoc).
No PVS+mirror test unless both independent methods qualify. Budget: measured
and diagnostic 1200s, validation 300s. Max batch 450s; acquire shared lock before
warmup and retain through every child. Record all null-window correctness,
retained bounds versus independent array minimax and broad exact root vectors.

## Fixed acceptance gates

All correctness must pass. Non-tail geometric wall speedup >=1.05; sum median
wall <=0.95*A and CPU <=0.97*A. Broad D8/D10, >=2000 A nodes: geometric >=1.10
and >=70% faster. Each condition including tail: wall <=1.20*A OR added
<=0.25ms; if A>=50ms, wall <=1.20*A AND nodes <=1.25*A. Each retained/peak
<=1.25*A OR added <=64KiB; broad summed retained <=1.15*A. Held-out geometric
>=1.00. Reject if any gate fails. Tail diagnostics do not override gates.

## Reproduction

```sh
.venv/bin/python -m scripts.benchmark_negamax_v3 declare --phase pvs
# Commit/push source-frozen design before timing.
scripts/with_benchmark_lock.sh .venv/bin/python -m scripts.benchmark_negamax_v3 run --phase pvs --start 0 --stop 8
# Repeat independent blocks 8:16, 16:24, 24:32.
scripts/with_benchmark_lock.sh .venv/bin/python -m scripts.benchmark_negamax_v3 audit --phase pvs
.venv/bin/python -m scripts.benchmark_negamax_v3 summarize --phase pvs
```
Use a fresh --directory with copied DESIGN.md for replication. No production
integration before independent acceptance and full backend suite.

Phase A outcome fixed before this declaration: both mirror variants rejected
at audited checkpoint b133403. Baseline is original direct, not mirror.
