# Phase D: exact depth10/12 feasibility

2026-10-10. Original optimized reference60b99b0; strongest accepted terminal-proof51bd80e.
Declaration5e79a7d; measured code identical to accepted production snapshot. Public caps unchanged.

| Position | Depth | Original ms | Accepted ms | Speedup | Nodes | Retained MiB | Peak traced MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| empty | 10 | 218.995 | 193.262 | 1.133x | 128308 | 3.624 | 3.759 |
| empty | 12 | 672.348 | 591.682 | 1.136x | 380468 | 12.354 | 13.252 |
| near-opening | 10 | 176.315 | 155.633 | 1.133x | 99441 | 3.094 | 3.285 |
| near-opening | 12 | 630.324 | 548.005 | 1.150x | 340974 | 11.610 | 13.393 |
| seeded-04 | 10 | 454.313 | 395.805 | 1.148x | 253274 | 7.415 | 7.692 |
| seeded-04 | 12 | 2004.718 | 1765.821 | 1.135x | 1089090 | 42.061 | 53.239 |
| post-hoc-tail | 10 | 746.167 | 655.979 | 1.137x | 423857 | 12.574 | 13.334 |
| post-hoc-tail | 12 | 3319.943 | 2907.495 | 1.142x | 1820265 | 54.523 | 56.486 |
| dense-endgame | 10 | 0.024 | 0.023 | 1.032x | 4 | 0.001 | 0.005 |
| dense-endgame | 12 | 0.022 | 0.021 | 1.048x | 4 | 0.001 | 0.005 |
| heldout-03 | 10 | 0.155 | 0.142 | 1.087x | 65 | 0.004 | 0.009 |
| heldout-03 | 12 | 0.162 | 0.147 | 1.103x | 67 | 0.004 | 0.009 |

Depth12 expensive quiet trees remain seconds-long: tail2.907s/1.820Mnodes,
seeded1.766s/1.089Mnodes. Empty0.592s,near-opening0.548s. Dense endgame
finishes in~0.021ms/4nodes; held-out late~0.147ms/67nodes. These are observed
finite cases, not worst-case bounds. No strength study/inference or cap change.

D12/D10 accepted latency: empty3.06x,near3.52x,seeded4.46x,tail4.43x.
Retained TT growth on those cases3.41/3.75/5.67/4.34x; per-node speed cannot
remove horizon growth. TT objects identical to reference; largest retained54.52MiB,
largest traced peak56.49MiB. Interpreter/native/allocator/RSS excluded.

## Correctness and provenance

All264recorded decisions match full ordered vectors/moves; deterministic tree
counters and complete TT dictionaries identical on all12conditions (payload-audit).
33independent tractable array vectors including draw-prefixD12 match. FiveD10
conditions match inherited data; allfour historical broadD12 original vectors
and counters match exactly, same baseline source digest (historical-depth12-audit).
No old result changed. Full backend2867passed15Torchskips before this study.

## Method and planning limits

Apple M5 MacBook Pro,CPython3.11.17; two shared-lock batches198.8/192.9s,391.7s
plus separate audits, within900s. Seven paired rotated warmed samples,64table
constructor warmups, complete choose_move/normalTTrelease inside wall/processCPU.
Two memory snapshots per condition measure reachable retained and traced heap.
All raw samples, CPU, leaves/hits/probes/cutoffs, peak/current and runtime/owner
records are condition JSONs. RESULTS.md and stability.json provide full detail.
Generic analysis gates are diagnostics, not a new default acceptance decision.

Conservative Mac Mini planning sensitivity only; no host/service inspection or
benchmark. Assuming combined throughput/concurrency penalties of2/4/8times
the largest laptop median gives:

| Assumed multiplier | D10 planning seconds | D12 planning seconds |
| --- | --- | --- |
| 2x | 1.312 | 5.815 |
| 4x | 2.624 | 11.630 |
| 8x | 5.248 | 23.260 |

These multipliers are assumptions, not server measurements/guarantees. Two
largest simultaneous tables imply roughly109MiB retained TT plus interpreter,
allocator, other request data and transient peaks. Do not equate traced heap
with process/container RAM. D12 should remain experimental pending separately
authorized deployment-environment validation and a bounded strength study.

Reproduce DESIGN.md from frozencheckpoint into freshdirectory with copied
design/input sources and manifest fixedsixhistories/depths10/12. Exclusive
evidence writes and missing-condition recovery preserve checkpoints. No live
infrastructure, API limits, frontend, other worktrees or canonicalv2 changes.

Next justified direction: symmetry policy keyed by initial root geometry.
PhaseA opposite-orientation reuse occurred on symmetric empty root, not three
asymmetric diagnostic roots. A separate root-symmetry dispatch hypothesis may
capture savings while avoiding asymmetric reflection overhead; it needs a fresh
predeclared targeted cohort and criteria, not relaxation of rejected PhaseA gates.
