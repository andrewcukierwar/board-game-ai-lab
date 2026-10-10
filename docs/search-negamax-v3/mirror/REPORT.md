# Phase A: reject full and selective mirror canonicalization

2026-10-10. Baseline: `60b99b0d42145da149f433907585b809060c946d`.
Research design checkpoint `244b0c5`; frozen implementation/manifest `e45cc91`;
partial evidence checkpoint `5372bff`. Production files remain unchanged.

Both candidates preserve exact search but fail the fixed latency gates in
[DESIGN.md](DESIGN.md). Full reflection improves the symmetric empty opening;
most asymmetric histories gain no opposite-orientation reuse and pay overhead.
Selective reflection (remaining depth >=3) limits the cost but fails eligibility.

| Variant | Non-tail geometric speedup | Broad speedup | Sum median wall/A | CPU/A | Broad retained/A | Held-out speedup |
| --- | --- | --- | --- | --- | --- | --- |
| full mirror | 0.884× | 0.879× | 1.124 | 1.124 | 0.956 | 0.869× |
| depth>=3 mirror | 0.966× | 0.971× | 1.010 | 1.010 | 0.959 | 0.935× |

Sum complete-decision medians: A 3.672s, full 4.129s, selective 3.709s across
124 non-tail conditions. Neither memory saving compensates failed timing.
[RESULTS.md](RESULTS.md) lists every condition, leaf/hit/entry/node count,
retained and peak memory and every failed gate. Full has seven material
per-condition regressions; selective none under the OR tolerance, but all
aggregate speed gates fail. Peak/retained per-condition gates pass for both.

## Exact correctness and provenance

[Audit](audit.json): 4,224 complete decision vectors/counters (warmup, seven
samples, two memory runs and counted run for each variant/condition), all exact.
111 independent array-minimax vectors (32 boards × D1/2/3 plus draw prefixes at
D4/6/8/10/12) match all three variants. All 112 inherited v2 score/move/stat
vectors match A exactly. Two repeated retained sizes match in every condition.
Bitboard reflection includes sentinel rows, is an involution, and canonical
packing preserves both boards, mover and unbounded depth. Hints are reflected
on storage/probe only; symmetric boards stay in direct orientation and center
maps to itself. Original-window bound classification and signed entry encoding
are unchanged. Root ties/full windows and exception restoration are preserved.

Initial focused suite: 68 passed. Expanded suite: 88 passed and one PVS-specific
test fixture assertion failed: the empty D4 tree has no qualifying re-search.
This is a test precondition failure, not a score mismatch. The log is retained
in expanded-tests.txt. Replacing that fixture with the existing legal tactical
history [5,4,3,6,2,4] produces 42 null-to-full re-search observations and its
oracle still matches; targeted correction log is preserved. The complete
expanded suite will be rerun under the shared lock before Phase B measurement.
No production-path integration occurred, so the full backend integration gate
is deferred until a candidate actually qualifies.

## Reuse and overhead diagnostics

[Orientation diagnostics](orientation-diagnostics.json) use separately declared,
source-hashed instrumentation and match all primary counters/vectors. Unlike
`reflected_hits`, `cross_orientation_hits` compares the stored writer orientation
with the reader orientation; it measures actual shared symmetry-orbit reuse.
At D10 the empty board has 885 such full-mirror hits (319 selective). Near-opening,
seeded-04 and post-hoc-tail have zero in both variants. Full empty nodes fall
128,308→61,322, median 225.95→123.16ms (1.835×); selective 62,334 nodes and
112.73ms (2.004×). Root move is the historical tied center-first column 2.

Profiled identity cumulative costs at D10: empty full 27.4ms vs selective
11.1ms; seeded-04 full 121.5ms vs selective 42.8ms. These include cProfile
instrumentation and dictionary orientation tracking, so they explain overhead
rather than estimate unprofiled component latency or replace primary timings.
Selective identity still incurs a function call at shallow depths, even when
it skips reflection. The tests do not establish benefits for every possible
symmetry strategy; a future root-symmetry dispatch policy would be a new design.

## Environment, stability and reproduction

Apple M5 MacBook Pro Mac17,2, 10 cores, 32GB; native CPython 3.11.17; runtime
platform/load and owner records in each raw condition file, hardware.txt above.
The four entire measured batches held the shared lock, including warmups:
55.6s, 101.9s, 143.1s, 79.6s (380.2s total), well within 1200s. Lock released
between batches and before the other agent's backend run. No CPU-intensive
concurrent Negamax job ran. Other OS activity remains; results are finite laptop
comparisons, not Mac Mini measurements or strength claims.

Seven paired rotated warmed samples, normal fresh TT disposal included in
latency; memory uses separate retained-table captures/tracemalloc. Retained
bytes include unique reachable table/dictionary/key/value/options objects,
not RSS or allocator overhead. [Stability](stability.json): leave-one-repetition
out geometric range 0.8828–0.8839 full, 0.9653–0.9658 selective; none of seven
omissions qualifies. These are sensitivity diagnostics, not confidence intervals.
No thresholds changed. Raw condition JSONs and all rejected sources are retained.

Reproduce at frozen design/source checkpoint with the commands in DESIGN.md,
a fresh --directory containing copied DESIGN.md, then run
`python -m scripts.negamax_v3_stability --phase mirror` for the preserved primary
directory (or adapt its directory for new evidence). Exclusive writes refuse
existing files. Diagnostic declaration/script hashes are preserved. Never rerun
into this frozen evidence directory. Phase B baseline remains A/direct.
