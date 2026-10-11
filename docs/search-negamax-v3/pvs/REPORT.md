# Phase B: reject PVS as the production default

2026-10-10. Baseline optimized direct alpha-beta at
`60b99b0d42145da149f433907585b809060c946d`. Design/sources frozen at59b996f,
scout diagnostics at923fbfd; completed first-half checkpoint7d3ef36.
Production search remains unchanged. Mirror combination was not justified.

| Metric, non-tail unless stated | PVS versus A |
| --- | --- |
| Geometric complete-decision speedup | 1.011x (required1.05) |
| Broad D8/10 speedup | 1.086x (required1.10) |
| Sum median wall | 3.259s / 3.676s =0.886*A |
| Sum median CPU | 0.887*A |
| Held-out geometric speedup | 1.013x |
| Broad summed retained TT bytes | 0.898*A |
| Per-condition regression gate | FAIL immediate-win/D8 |
| Retained/peak memory gates | PASS |

PVS lowers total work on some expensive quiet trees but fails three fixed
gates: geometric, broad geometric, and per-condition regression. Summed time
and memory wins do not override failures. [RESULTS.md](RESULTS.md) lists every
position/depth, raw-node/leaf/hit/entry counters, memory and gate failures;
[analysis.json](analysis.json) has the exact computed acceptance decision.
All rejected sources and full raw condition files are retained.

PVS uses full [-beta,-alpha] on the first child, integer scout window
[-alpha-1,-alpha] afterward, and full re-search when alpha<result<beta.
Alpha=-infinity also searches full. Fail-high closes the window without
unjustified exact conversion; original pre-probe alpha/beta classify TT bounds.
All legal root children receive full windows in historical center order.
Terminal-before-leaf signed horizon values, key depth/mover fields, heuristic,
cache lifetime and finally restoration are unchanged. No iterative schedule,
aspiration, tactical pruning or budget-limited returned result was added.

[Audit](audit.json): all2816 recorded warmup/sample/memory/counted vectors
and deterministic per-variant stats match; 111 independent array-minimax vectors
match both engines; 112 inherited optimized-engine score/move/stat vectors match
A. 89 focused tests passed in7.69s under lock. Retained sizes repeat exactly.
The initial re-search fixture mistake and correction remain in Phase A logs;
its corrected tactical fixture exercises qualifying scout-to-full verification.
Every retained bound on tractable test searches is checked against array truth.

[Scout diagnostics](scout-diagnostics.json) separately count null probes and
full re-searches on six declared histories at D8/D10, matching all primary
vectors/counters. These explain work, not replace complete-decision evidence.
When the best child is already early, scout overhead can add little benefit;
heuristic or tactical ordering mistakes can require repeated work. No new
ordering policy or tuned scout-depth gate was introduced after seeing results.

Environment: Apple M5 MacBook Pro Mac17,2,10cores,32GB, native CPython3.11.17;
platform/load/lock-owner metadata in each condition, hardware.txt in parent.
Seven paired warmed samples, rotating variant order; constructor warmup64;
wall/CPU cover complete choose_move and normal table release. Two separate
traced runs capture peak/current and reachable retained table objects, not RSS.
Shared lock held for each complete measured batch (38.7/57.0/82.4/43.7s,
221.8s total, within1200s), released between blocks. MCTS pilots owned intervening
slots; queued acquisition waited for normal release without removing any lock.
No competing intensive Negamax workload used for acceptance.

[Stability](stability.json): leave-one-repetition-out geometric range1.008–1.015x;
wall sum/A0.887–0.888, zero of seven omissions eligible. Sensitivity is
not a confidence interval. Paired timing variation, thermal/OS activity and
this finite survival-conditioned workload limit generalization; no strength
or Mac Mini latency inference. Acceptance thresholds were never changed.

Reproduction: DESIGN.md commands, frozen source checkpoint and a new
--directory with copied design; exclusive writes refuse frozen evidence.
Run scripts.negamax_v3_stability after complete evidence; separate scout
script/declaration hashes are preserved. Full backend integration gate is
not triggered by this rejection because production source bytes are unchanged.
Next: profile A/direct, then choose bounded per-node experiments from measured
hotspots, preserving complete scores and (for per-node-only changes) counters.
