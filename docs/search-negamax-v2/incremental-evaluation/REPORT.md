# Phase 3B.1: exact incremental Negamax evaluation

October 9, 2026. Laptop repository `andrewcukierwar/board-game-ai-lab`, branch
`search/negamax-v2`; immutable baseline
[`93ee943d8744651064dcf9a3a79108d6ca73af53`](https://github.com/andrewcukierwar/board-game-ai-lab/commit/93ee943d8744651064dcf9a3a79108d6ca73af53).

**Incremental evaluation is accepted:** exact correctness and the declared
latency/memory thresholds pass. Complete backend validation is recorded below.

## Architecture and equivalence

Previously, each nonterminal depth-zero leaf intersected both bitboards with
all 69 four-cell masks and counted their stones. State contained only the two
bitboards, column heights, mover and piece count. The candidate adds a list of
69 window count codes, an X-relative position score and an undo score stack.
Initial array-to-bitboard conversion counts each window once. A precomputed
cell-address-to-window tuple identifies only the windows touched by a drop.
There are at most 13 memberships per playable cell; sentinels have none.

For window counts x and o and unchanged W=(0,1,3,9,81), the contribution is
`[o=0] W[x] - [x=0] W[o]`. This is exactly the original two independent
branches. When both are zero, both terms are W[0]=0; a blocked window has
neither term. A code `x + 5*o` uniquely represents both counts (0..4), so its
lookup value is exact. On an X/O drop the code increases by 1/5. A precomputed
lookup supplies `contribution(new)-contribution(old)` for each touched window.
The maintained score is their X-relative sum. `heuristic()` returns that sum
for X to move and its negation for O, in constant time.

Saved-score undo subtracts 1/5 from only the affected codes, then pops the
previous exact score. No counts are rebuilt or all-window scans performed in
play, undo or leaf evaluation. The bounded-by-search-depth stack reuses a list;
updates allocate no membership lists, ranking tuples or count containers. Small
count codes use Python's existing integer cache; score arithmetic still creates
ordinary Python integers. The original recursive/root `try/finally` blocks
restore all state when downstream evaluation, terminal checks or ordering raise.

No search rule or public setting changes: terminal checks still precede leaves,
win/loss values retain remaining-depth magnitudes, alpha-beta and move ordering
are unchanged, the TT key/layout/bound rules and fresh per-decision lifetime are
unchanged, and every legal root child uses a full window in historical center
order. Exact scores and ties remain depth-specific.

## Independent correctness

The new array evaluator independently enumerates four directions in top-row
array coordinates; it calls no incremental evaluator to derive expected scores.
It compares all 69 geometry masks with the bitboard windows, all count codes,
bitboards, heights, occupied count, mover, cached X score and signed heuristic.
The exhaustive initialization test covers 69 geometries × 81 assignments of
empty/X/O × both movers = **11,178 boards/perspectives**, including empty,
blocked, winning, vertical and both diagonal windows. Floating stones are
intentional in this geometry-only test; move tests enforce gravity and stop at
terminal positions.

The inherited `double-threat-loss` label deserves care: X can win immediately
in column 3 on that fixture; the other six root moves lose. It is a zero-leaf
terminal-heavy fixture, rather than a forced loss of the entire root. Its
history and earlier evidence are preserved exactly.

A seed-31012026 legal path/branch test checks **22,792 play/undo transitions**,
with siblings at varying depths, full rollback and saved-score-stack equality.
Separate seeded legal paths validate all three ablation representations.
The evidence audit additionally checks all 42 legal prefixes of a full draw
and all 42 undos with the independent array oracle. Fixtures cover all seven
columns, both parities/perspectives, tactical wins,
required replies, dense boards, near draws and a full draw. Injected exceptions
from heuristic, terminal evaluation and ordering verify nested and root state
restoration, including caller-board preservation.

The existing independent unpruned array minimax oracle remains active; additional
tractable near-endgame comparisons run at depths 1/2/3/4/6/8/10. Forty-two pinned
baseline comparisons (seven positions × depths 1/2/4/6/8/10) assert the entire
ordered root-score vector, chosen move and all public search counters. The
benchmark separately asserts parity across every fixture/depth/variant and every
measurement mode; it also checks heuristic leaf counts.

## Declared methodology

[DESIGN.md](DESIGN.md) fixes practical acceptance thresholds before timing:
mandatory exact correctness/counters; broad-tree geometric speedup ≥1.15× and
≥80% of broad conditions faster; all non-tail conditions geometric speedup
≥1.05× and sum of median latencies reduced ≥10%; every condition slowdown ≤50%
or added cost ≤0.25 ms; identical retained TT sizes; peak traced growth ≤5% or
≤32 KiB. Broad conditions use deterministic baseline counters (≥2,000 nodes,
≥35% heuristic leaves at depth 6/8/10). This is a finite engineering comparison,
not a statistical claim about a randomly sampled population.

[manifest.json](manifest.json) freezes all eleven base/supplement Phase 3A
histories from the baseline Git object, plus eight seed-20261009 nonterminal
positions (two each at 8/12/16/20 plies, uniform legal moves, terminal rejection
only). Position selection never uses measured speedups. History `[1,4,6,0,6]`
is explicitly a **post hoc diagnostic**, excluded from acceptance aggregates.
Depths are 4/6/8/10 throughout. All three evaluator ablations use the ordinary
unchanged search body. The baseline is loaded directly by `git show` from the
pinned full SHA, not copied from a mutable working file.

Before the final run, 64 table instances per module are constructed and
discarded to equalize CPython shared-key instance-dictionary allocation.
Each condition/variant has one discarded search warm-up and seven timed runs with
rotating variant order. Timings include the entire ordinary `choose_move()`
execution: initial state conversion, all exact root searches, fresh TT and its
release. Game construction, variant selection, wrappers and table sizing are
outside timing. Both wall and process CPU are measured. Leaf counting is a
separate wrapped run; memory is measured separately twice under tracemalloc,
with a diagnostic reference retaining the TT. Instrumented timings are retained
as raw evidence and excluded from latency estimates. CPU covers the process;
these runs are serial but do not isolate the host from unrelated activity.

Isolated microbenchmarks use 20,000 heuristic calls or 2,000
play/heuristic/undo cycles per sample, the same warm-up/seven rotating samples,
and matching checksums. They exclude initial state conversion and search. They
cannot establish end-to-end speedup on their own.

Hardware: MacBook Pro Mac17,2, Apple M5, ten cores (four performance/six
efficiency), 32 GB, native arm64 CPython 3.11.17, macOS 26.6.2. Initial load
averages for the calibrated run were 5.38/4.42/4.06. No other workload was
stopped or changed. This is
laptop evidence, with no production endpoint, concurrency throughput, Mac Mini,
worst-case runtime or process RSS claim. Source/runtime/declaration/fixture
hashes are recorded in the manifest; `system_profiler` emitted a harmless
hw.cpufamily lookup error but returned the recorded hardware metadata.

## Measurement calibration

The entire first run is preserved in [pre-calibration/](pre-calibration/), with
its original manifest, declaration, harness, raw timings, memory, microbenchmarks
and analysis. All latency/correctness/peak-memory criteria passed, but strict
TT-size equality failed: the candidate was 104 bytes smaller on empty depth 4
and 16 bytes smaller on empty depth 6. All three ablations had exactly the same
differences; every other row, including the cost tail, had identical TT size.

This is a measurement artifact in the SearchTable object's attribute dictionary,
not different search entries. CPython reduces the per-instance split-dictionary
allocation across early constructors. Three candidate variants share a table
class and therefore warmed it faster than the separately loaded baseline class.
[calibration.json](calibration.json) records a fresh-process constructor-only
check with the same progression for both
classes: 296, 280, 248, 184, then 120 bytes, remaining at 120 after enough
constructors. The final harness warms both classes equally before measurements
and records this dictionary size in every memory run. The full 80-condition
experiment is repeated with unchanged candidate, fixtures and acceptance
thresholds; first-run observations are not pooled into final medians. The first
analysis's rejected status is preserved rather than retroactively edited.

## Measurements and acceptance

**Accept compact counts with saved-score undo.** All declared performance,
memory and exact-parity criteria pass in the calibrated run. The production
agent uses that evaluator; public interfaces and search semantics are unchanged.

The 28 broad-tree conditions all improve; their geometric speedup is **1.83×**.
Across all 76 non-tail conditions the geometric speedup is **1.47×**
and the sum of median latencies falls **43.9%**.
Equal-condition geometric means include every regression. The cost tail is
excluded from these aggregates. All 80 conditions × four variants completed
seven timed samples, one search warm-up, a leaf-count run and two memory runs.
That is 2,240 timed decisions, 320 warm-ups and 960 separate diagnostic decisions.

[results.jsonl](results.jsonl) contains every score, move, counter, wall/CPU
sample and memory repetition; [performance.csv](performance.csv) exports all
320 variant rows. [analysis.json](analysis.json) records threshold checks and
selection, [microbenchmark.json](microbenchmark.json) keeps isolated samples,
and [audit.json](audit.json) validates coverage, hashes and exact comparisons.

Depth summaries across 19 non-tail positions (sum of per-position medians,
not an endpoint percentile):

| Depth | Wall geometric speedup | CPU geometric speedup | Summed wall ms before → after | Summed CPU ms before → after |
| --- | ---: | ---: | ---: | ---: |
| 4 | 1.59× | 1.59× | 26.109 → 12.498 | 26.106 → 12.491 |
| 6 | 1.56× | 1.56× | 152.347 → 75.641 | 151.987 → 75.382 |
| 8 | 1.45× | 1.45× | 701.108 → 368.966 | 699.919 → 368.738 |
| 10 | 1.30× | 1.31× | 2676.733 → 1536.716 | 2668.010 → 1531.229 |

Per-position medians in ms, before → after / speedup. Every cell is a
fixed-depth, identical-tree comparison:

| Position | D4 | D6 | D8 | D10 |
| --- | ---: | ---: | ---: | ---: |
| empty | 2.364 → 0.853 / 2.77× | 17.560 → 7.302 / 2.40× | 99.870 → 45.365 / 2.20× | 413.645 → 210.096 / 1.97× |
| near-opening | 2.210 → 0.962 / 2.30× | 14.615 → 7.090 / 2.06× | 81.281 → 40.958 / 1.98× | 304.782 → 170.498 / 1.79× |
| midgame | 0.976 → 0.549 / 1.78× | 3.011 → 1.954 / 1.54× | 6.696 → 4.873 / 1.37× | 12.098 → 9.600 / 1.26× |
| midgame-wide | 2.567 → 1.209 / 2.12× | 9.151 → 5.288 / 1.73× | 21.658 → 15.085 / 1.44× | 45.925 → 36.429 / 1.26× |
| late | 0.117 → 0.094 / 1.24× | 0.312 → 0.252 / 1.24× | 0.628 → 0.579 / 1.08× | 1.078 → 1.095 / 0.98× |
| immediate-win | 0.993 → 0.557 / 1.78× | 6.773 → 3.280 / 2.06× | 40.644 → 20.723 / 1.96× | 184.840 → 102.180 / 1.81× |
| forced-reply | 1.221 → 0.560 / 2.18× | 5.113 → 2.583 / 1.98× | 26.313 → 13.854 / 1.90× | 87.941 → 52.060 / 1.69× |
| double-threat-loss | 0.149 → 0.200 / 0.74× | 0.150 → 0.205 / 0.73× | 0.149 → 0.200 / 0.74× | 0.149 → 0.201 / 0.74× |
| dense-endgame | 0.014 → 0.023 / 0.61× | 0.016 → 0.024 / 0.65× | 0.017 → 0.025 / 0.65× | 0.016 → 0.025 / 0.64× |
| quiet-5 | 1.212 → 0.652 / 1.86× | 6.011 → 3.787 / 1.59× | 15.589 → 11.553 / 1.35× | 22.328 → 19.864 / 1.12× |
| supplement-main-003 | 2.573 → 1.145 / 2.25× | 17.528 → 9.033 / 1.94× | 70.585 → 42.172 / 1.67× | 233.060 → 156.775 / 1.49× |
| seeded-00 | 3.478 → 1.412 / 2.46× | 24.968 → 11.407 / 2.19× | 115.249 → 57.420 / 2.01× | 410.893 → 226.557 / 1.81× |
| seeded-01 | 0.335 → 0.296 / 1.13× | 2.142 → 1.313 / 1.63× | 6.015 → 3.442 / 1.75× | 13.700 → 9.379 / 1.46× |
| seeded-02 | 0.487 → 0.470 / 1.03× | 1.653 → 1.043 / 1.58× | 3.756 → 2.549 / 1.47× | 5.067 → 4.642 / 1.09× |
| seeded-03 | 0.384 → 0.313 / 1.23× | 1.233 → 0.910 / 1.35× | 4.642 → 3.303 / 1.41× | 13.590 → 10.794 / 1.26× |
| seeded-04 | 4.224 → 1.605 / 2.63× | 30.317 → 13.502 / 2.25× | 176.518 → 86.602 / 2.04× | 834.722 → 463.355 / 1.80× |
| seeded-05 | 1.626 → 0.778 / 2.09× | 8.677 → 4.404 / 1.97× | 23.884 → 14.809 / 1.61× | 77.100 → 51.158 / 1.51× |
| seeded-06 | 0.906 → 0.531 / 1.70× | 2.598 → 1.683 / 1.54× | 6.755 → 4.452 / 1.52× | 14.534 → 10.548 / 1.38× |
| seeded-07 | 0.272 → 0.287 / 0.95× | 0.520 → 0.578 / 0.90× | 0.859 → 1.001 / 0.86× | 1.266 → 1.461 / 0.87× |
| post-hoc-tail | 4.971 → 1.814 / 2.74× | 40.325 → 16.921 / 2.38× | 277.838 → 128.177 / 2.17× | 1411.609 → 723.438 / 1.95× |

The diagnostic tail improves **1411.6 → 723.4 ms / 1.95×**
(48.8% lower latency). Its 423,857 nodes, 119,958 entries,
47,944 hits, 93,664 cutoffs, complete root vector and move 4 match the
original Phase 3A tail. Its best score remains −4, a heuristic value rather
than a proven forced loss. This result is diagnostic, outside acceptance.

Evaluation-only alternatives (same results/counters):

| Variant | Broad-tree geometric speedup | All non-tail geometric speedup | Summed latency reduction | Eligible |
| --- | ---: | ---: | ---: | --- |
| compact-stack | 1.83× | 1.47× | 43.9% | yes |
| compact-inverse | 1.75× | 1.42× | 41.5% | yes |
| arrays-stack | 1.65× | 1.34× | 38.0% | yes |

[timing-stability.json](timing-stability.json) reports descriptive sample
variation: the median relative median absolute deviation across baseline and
accepted-variant conditions is 0.53%. Seeded-01 depth 6 is noisy (14.5% baseline,
23.6% candidate); its small-tree median should be interpreted cautiously. No
sample is discarded. This condition is outside the broad-tree group. The
complete first run's non-tail geometric speedup was 1.48× versus 1.47× in the
calibrated repeat; those first-run samples are not pooled into final medians.

Both alternatives also pass, but compact saved-score undo has the highest
declared distribution speedup. Inverse-delta undo does additional score
lookup/arithmetic; separate arrays add branch/index work and two containers.
Neither is integrated. No unrelated search optimization is included.

All compact-stack regressions are shown below; no negative row is omitted:

| Position | Depth | Before ms | After ms | Added ms | Slowdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| late | 10 | 1.078 | 1.095 | 0.017 | 1.6% |
| double-threat-loss | 4 | 0.149 | 0.200 | 0.051 | 34.4% |
| double-threat-loss | 6 | 0.150 | 0.205 | 0.055 | 36.7% |
| double-threat-loss | 8 | 0.149 | 0.200 | 0.051 | 34.6% |
| double-threat-loss | 10 | 0.149 | 0.201 | 0.052 | 34.9% |
| dense-endgame | 4 | 0.014 | 0.023 | 0.009 | 63.8% |
| dense-endgame | 6 | 0.016 | 0.024 | 0.009 | 53.7% |
| dense-endgame | 8 | 0.017 | 0.025 | 0.009 | 52.8% |
| dense-endgame | 10 | 0.016 | 0.025 | 0.009 | 57.2% |
| seeded-07 | 4 | 0.272 | 0.287 | 0.015 | 5.4% |
| seeded-07 | 6 | 0.520 | 0.578 | 0.058 | 11.2% |
| seeded-07 | 8 | 0.859 | 1.001 | 0.142 | 16.5% |
| seeded-07 | 10 | 1.266 | 1.461 | 0.195 | 15.4% |

Tiny terminal-heavy searches pay initialization and update costs without
saving many leaf scans. The dense endgame remains only four nodes; relative
percentages are large on its microsecond baseline. Late and seeded-07
regressions remain small in absolute time and pass the declared limit.

## Microbenchmark versus complete search

Isolated heuristic: 126.76× geometric speedup across the 19 non-tail boards; median per-call/cycle cost ranges 2.346–3.678 µs before and 0.023–0.031 µs after.

Play/heuristic/undo cycle: 4.85× geometric speedup across the 19 non-tail boards; median per-call/cycle cost ranges 2.510–4.156 µs before and 0.579–0.925 µs after.

These loops include common Python call/loop/checksum overhead, exclude
initial state conversion and do not visit search nodes. Constant-time score
retrieval is much faster in isolation, but incremental play/undo still runs
at every visited child, including TT hits and terminal children. Therefore
the actual choose_move gains are materially smaller, and terminal-heavy
trees may lose. Acceptance uses complete decisions, never the microbenchmark.

## Memory and counter evidence

TT retained size is identical in all conditions and both memory repetitions.
All measured table attribute dictionaries are 120 bytes after calibration.
Normal production releases the table at the end of each decision. Diagnostic
retention is deliberate and is not a production leak or RSS bound.

Peak traced growth is at most **1,208 bytes** across all 80 conditions.
The additional persistent cell memberships, score and delta lookups occupy **7,652 bytes**
(unique reachable Python objects, allocated at module import and reported
outside per-decision tracemalloc). The 69-code list and search-depth score
stack add roughly a KiB of temporary state; they do not grow with TT entries.

Depth-10 counters are identical before/after; full depth-specific counters
and both peak repetitions are in the raw evidence and CSV:

| Position | Nodes | Leaves | Entries / hits | Cutoffs | TT MiB (both) | Peak MiB before → after | Peak added bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 128,308 | 68,511 | 38,395 / 15,828 | 27,477 | 8.259 | 8.414 → 8.415 | 856 |
| near-opening | 99,441 | 49,039 | 28,999 / 14,340 | 22,095 | 6.746 | 6.868 → 6.869 | 1,112 |
| midgame | 5,389 | 1,543 | 1,881 / 700 | 993 | 0.437 | 0.447 → 0.448 | 1,208 |
| midgame-wide | 19,097 | 5,850 | 6,730 / 2,759 | 5,110 | 1.632 | 1.655 → 1.655 | 868 |
| late | 506 | 78 | 328 / 22 | 166 | 0.070 | 0.074 → 0.075 | 920 |
| immediate-win | 59,934 | 30,391 | 18,565 / 8,028 | 14,178 | 3.998 | 4.084 → 4.084 | 992 |
| forced-reply | 30,407 | 13,904 | 10,297 / 3,522 | 7,563 | 2.200 | 2.259 → 2.260 | 808 |
| double-threat-loss | 81 | 0 | 38 / 0 | 32 | 0.010 | 0.013 → 0.014 | 644 |
| dense-endgame | 4 | 0 | 3 / 0 | 0 | 0.001 | 0.004 → 0.005 | 576 |
| quiet-5 | 10,140 | 2,152 | 4,174 / 1,539 | 3,124 | 0.984 | 1.001 → 1.001 | 880 |
| supplement-main-003 | 87,810 | 36,833 | 28,395 / 16,340 | 22,427 | 6.681 | 6.736 → 6.737 | 920 |
| seeded-00 | 134,238 | 70,300 | 34,557 / 17,305 | 26,855 | 7.972 | 8.096 → 8.097 | 984 |
| seeded-01 | 5,221 | 2,033 | 1,796 / 691 | 1,291 | 0.420 | 0.427 → 0.427 | 616 |
| seeded-02 | 2,290 | 516 | 835 / 367 | 585 | 0.206 | 0.212 → 0.213 | 732 |
| seeded-03 | 5,772 | 1,876 | 2,112 / 1,192 | 1,641 | 0.489 | 0.497 → 0.497 | 772 |
| seeded-04 | 253,274 | 130,691 | 78,484 / 29,767 | 61,262 | 17.338 | 17.435 → 17.436 | 864 |
| seeded-05 | 28,891 | 12,072 | 9,961 / 3,328 | 6,967 | 2.201 | 2.224 → 2.225 | 1,020 |
| seeded-06 | 6,129 | 2,110 | 1,960 / 685 | 1,018 | 0.455 | 0.465 → 0.466 | 1,208 |
| seeded-07 | 730 | 47 | 289 / 94 | 201 | 0.067 | 0.072 → 0.072 | 624 |
| post-hoc-tail | 423,857 | 236,149 | 119,958 / 47,944 | 93,664 | 27.605 | 27.858 → 27.859 | 1,048 |

The independent evidence audit matches all 44 predeclared Phase 3A
score/move/counter/leaf vectors and the original depth-10 tail. It verifies
all 3,200 saved timed/diagnostic decisions and checks unchanged search/terminal/
ordering/TT/public-agent ASTs and constants against the pinned source.

Baseline agent SHA-256: `a07a0d19bf523c6afc1870c1b2deaab811f210f61aa3198db780a1752e436dc2`.
Candidate agent SHA-256: `ac7618cfe9854e355640f17cbe1dd96d0fcc119c5fdeb30667d3d51cfc4ffa09`.
All harness/test/fixture/declaration hashes are in the manifest, with evidence
and auditor hashes in audit.json. Earlier evidence is read from immutable Git
objects and is never rewritten.

## Integration validation

The complete backend suite passes **1,926 tests**, with **15 skips because
PyTorch is unavailable**, in **92.46 seconds**. [backend-tests.txt](backend-tests.txt)
preserves the complete output. This includes the existing MCTS/bitboard,
public concurrency/rate-limit, API, engine, Victor, evidence and Negamax tests,
plus 72 new correctness/benchmark-integrity tests. The isolated incremental
correctness run passed all 66 tests and recorded its 22,792 transitions in
[correctness-tests.txt](correctness-tests.txt); benchmark-integrity tests add six.

The test command uses `PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false
OPENAI_API_KEY=`. Disabling dotenv prevents the laptop research flag from
contaminating Victor's existing default-off test. The suite's global fixture
forbids live HTTPS provider calls. No real providers are contacted.

The evidence audit passes all coverage, source/AST, vector/counter/leaf,
memory and full-draw rollback assertions. Python compilation and final Git
whitespace checks pass. Public signatures, API/frontend files and UI settings
are unchanged, so frontend tests were not run.

## Files and reproduction

Changes are confined to the production Negamax evaluator, two new test files,
`scripts/benchmark_negamax_incremental.py`, evaluation-only ablations in
`scripts/negamax_evaluation_variants.py`,
`scripts/audit_negamax_incremental.py` for evidence/AST verification and CSV
export, and this new evidence directory.
Earlier Phase 1/2/2B/3A and canonical artifacts remain unchanged. No API or
frontend code/presets change, so frontend tests are unnecessary for this change.
No provider calls, services, deployments, merges, secrets, network configuration,
training/model changes or production-checkout operations are involved.

Reproduce in a **new** directory on the laptop checkout with its existing venv:

```sh
mkdir -p /tmp/negamax-incremental-replay
cp docs/search-negamax-v2/incremental-evaluation/DESIGN.md /tmp/negamax-incremental-replay/
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.benchmark_negamax_incremental declare --directory /tmp/negamax-incremental-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.benchmark_negamax_incremental run --directory /tmp/negamax-incremental-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.audit_negamax_incremental --directory /tmp/negamax-incremental-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest tests -q -rs
```

`declare` freezes histories, source hashes, design, hardware and parameters before
measurement. `run` rejects source/runtime drift and refuses to overwrite any
results file. Failed/partial runs are preserved and require a new output
directory; no selective resume or sample deletion is supported. Seven samples
improve stability over the earlier three, but these are descriptive local
medians, not confidence intervals or production percentiles. Results may vary
with scheduling, frequency and Python/runtime changes.

Recommended Phase 3B.2: independently test collision-free compact full-identity
TT keys and entries, retaining mover/depth separation, typed bounds, signed
score width for ±(1,000,000+depth), move hints and a fresh table per decision.
First isolate representation savings and CPU effects. Then separately test
bounded deterministic replacement: an eviction may cause a miss and more work,
never a false hit, early termination or incomplete root scores. Avoid combining
this with mirror keys, new pruning or depth changes.
