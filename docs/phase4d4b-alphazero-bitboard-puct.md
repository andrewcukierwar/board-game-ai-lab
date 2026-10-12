# Phase 4D.4B — Optional bitboard neural PUCT

The optional bitboard backend meets the 1.5× complete-search target with exact
behavioral equivalence. Empty/opening searches improve **1.91–1.93×**, quiet
midgames **1.76–1.77×**, and four paired frozen-network self-play games improve
collection throughput **2.09×**. Python-traced peak allocations decrease in every
comparison. The engine backend remains the default; no merge, deployment,
checkpoint promotion or training campaign is part of this phase.

Acceptance is **PASS after investigation of one retained tail-latency flag**.
The original ten-trial per-checkpoint p95 screen is explicitly **FAIL** for one
of 112 cells. A separate 40-pair repeat finds no persistent regression. The
original flag and observation remain in the primary data; no threshold or
measurement was changed. Details below distinguish observation from inference.

## Implementation and provenance

Development uses branch `phase4d4b-alphazero-bitboard-puct`, based on completed
Phase 4D.4A commit `2fbbdeed452e9d73461f36601296b01e5fb34af6`, in
`experiment-output/phase4d4b-worktree` under the historical checkout. That
checkout remains at `1894bc4d8f035cf738df8e3fc0ce234f1ce2c8be`.

`alphazero_v2/bitboard.py` adapts the validated state geometry and immutable drop
operation from main's `agents/mcts_bitboard.py` at
`8704142702da273e6571f0ca9df3fd4a793be285`. The historical baseline lacks that
module, so the small rule core is isolated here rather than importing main's
classical search. Each column has six bottom-first playable bits and an empty
sentinel; shifts 1/7/6/8 detect fours. Conversion validates the standard board
once at the root. Successors store two piece integers, occupancy, actor and a
cached winner, avoiding deepcopy and repeated board scanning. Terminal drops
are refused and the engine's post-win turn flip is retained.

`V2Inference.predict` dispatches to direct bitboard encoding for these states.
It constructs the unchanged contiguous CPU float32 `(1,1,6,7)` input, physical
columns ascending, top row first, own +1/opponent -1/empty 0. Even negative zero
on empty O-to-move cells matches the historical encoder's bytes. Architecture,
325,896 parameters, output validation, legal-only float64 softmax and inference
checkpoint format remain unchanged.

`PUCTSearch`, `SelfPlayer` and `EvaluationAgent` accept an explicit
`backend="bitboard"`; `backend="engine"` remains the default. Both paths share
the original Node, parent/child dictionaries, score arithmetic, alternating
backup, noise, temperatures, full-child expansion, action selection and RNG
calls. No tactical guards, proofs, batching, transpositions, tree reuse, compact
node rewrite or training-parameter change was introduced.

Example capture compares the full canonical bitboard state with the pre-move
engine board. The historical engine `__dict__` check remains intact. Provenance
now classifies the imported state module as a training-path source; the entire
execution closure remains hashed. This does not authorize any old declaration
or resume boundary against changed sources.

The eight archived checkpoints are the existing seed 42/314159 generations
0/10/15/20. Exact paths and byte hashes are in each run's `provenance.json`, copied
from Phase 4D.4A and independently checked before inference. No checkpoint was
saved or updated. All new outputs use separate fresh directories.

The host/environment matches the principal baseline: Apple M4 Mac mini, macOS
26.6.2 arm64, CPython 3.11.17, PyTorch 2.10.0, NumPy 1.26.4, CPU float32,
deterministic algorithms, one intra-op and one inter-op thread, and all five
BLAS/vector thread settings pinned to one before interpreter startup. The
existing `/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python` was
used without installing packages.

Engineering decisions follow Phase 4D.4A's measured copying/checking costs and
external implementation comparison. Main's latest pinned F1 research changes
rollouts, proofs and exploration; those changes are deliberately outside this
representation experiment. Faster states plus independently batched games remain
the useful computational lessons; external author performance claims are not
used as acceptance evidence.

## Exact validation

The standalone audit passed **10,138 legal successor transitions** from 1,578
checked states, including mirrored histories, both actors, 234 states with full
columns, 80 terminal states, explicit draws, opening/tactical fixtures and both
players' wins. Board cells, actor/piece, ordered legal columns, outcomes,
termination, all successors and tensor bytes match the independent engine.
Regression tests also enumerate every one of the 69 four-cell windows, each
missing-cell alternative and sentinel wraparound cases.

The archived-checkpoint matrix comprises **672 paired validation searches**:
eight checkpoints × seven fixed positions × two budgets × three independent
search/action RNG trials × noise disabled/enabled. It checked **1,045,908 allocated
nodes** and **179,238 neural predictions** per backend. Comparisons include:

- Actual forward input and raw output tensor bytes, and every legal-masked
  policy/value prediction; Python floating values use exact/hex equality.
- Root network/noised priors, Dirichlet samples and private PCG64 state.
- Every allocated node, including unvisited children: full state, actor,
  ordered children, move, parent ownership, terminal/winner, visits, value sum,
  Q and PUCT score.
- Root visits/targets, max depth, selected-leaf depth sum, terminal leaves,
  and exact search/action/noise RNG states.
- Selected actions and distributions at temperatures 1 and 0.

The noisy matrix uses the unchanged epsilon .25/alpha 1 and a controlled noise
seed. Complete self-play additionally uses four independent root-noise streams.
Root expansion remains uncounted: exactly 256 or 512 simulations visit root
edges, and their visits sum to the requested budget.

The 2,240 uninstrumented matrix searches also compare complete trees and search
RNG after each timed pair. Separate profile probes reproduce their uninstrumented
tree/RNG exactly. A further **200 development tactical rows** at 42/g10, 512
simulations pass full-tree/forward/action/RNG differential checks. These are
existing development fixtures, not a new held-out strength evaluation.

Four complete empty-board games per backend use frozen 42/g10 weights and
unchanged V2Config defaults (256 simulations, root noise, temperature 1 before
ply eight then 0). Their lengths are 40/17/36/26 plies, **119 positions** per
backend. Completed trajectories, observations, visits, policy targets,
temperatures, actor-relative outcomes, winners and all three RNG streams match
exactly. Weight hashes before/after are identical.

Final test result: **707 passed, 2 skipped (276.32 seconds)**. Commands and complete test logs are
retained with the evidence. Existing training tests use tiny synthetic temporary
fixtures; no archived-network training or campaign execution occurred.

## Paired complete-search latency

Both backends were warmed for each checkpoint/position/budget. Trial order
alternates engine/bitboard, with identical seed domains. Loading, setup, RNG
creation, explicit GC, hashing, tracing, instrumentation and comparisons are
outside timing. Complete root initialization/expansion, traversal and result
construction are inside timing. Automatic GC remains enabled in both paths.
Action selection is outside search latency and is verified separately.

Latency below is the **median of eight checkpoint medians**, ten trials per
checkpoint, matching Phase 4D.4A's aggregation. Values are paired current-engine
measurements, not a comparison against a stale historical time. CPU/wall,
p95/min/max, simulations/s, inference calls and terminal-leaf tables are in
[the detailed tables](../research/alphazero_bitboard/results/phase4d4b-20261011/TABLES.md).

| Position | Engine 256 ms | Bitboard 256 ms | Speedup | Engine 512 ms | Bitboard 512 ms | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Empty | 114.19 | 59.26 | 1.927× | 227.98 | 119.20 | 1.913× |
| Early opening | 113.29 | 58.81 | 1.927× | 226.84 | 118.21 | 1.919× |
| Quiet midgame | 100.72 | 57.22 | 1.760× | 199.04 | 112.18 | 1.774× |
| Tactical midgame | 100.71 | 52.94 | 1.902× | 202.09 | 107.32 | 1.883× |
| Late game | 31.78 | 13.69 | 2.322× | 48.88 | 18.76 | 2.606× |
| Immediate win | 18.03 | 5.34 | 3.378× | 31.23 | 7.88 | 3.962× |
| Mandatory block | 85.78 | 42.41 | 2.023× | 179.36 | 89.19 | 2.011× |

All neural-call counts, terminal leaves and visit/depth diagnostics are unchanged.
Terminal-heavy speedups are larger because cached outcomes replace expensive
scans even when no neural call occurs. Those speeds should not be extrapolated
to broad opening searches.

### Retained p95 flag and investigation

Aggregate p95 improves in all 14 cells (bitboard/engine ratios .305–.549).
These p95 values describe the fixed 80-observation checkpoint mixture, not an
independent population-tail estimate. Per-checkpoint p95 from ten observations
is an unstable screen; one cell, 42/g10 tactical midgame at 256, flags +36.9%.

Its bitboard trial 3 takes **249.52 ms wall / 100.41 ms CPU**, versus 50.88–54.32
ms wall for its other trials. The paired engine trial is also slower:
**130.53 ms wall / 118.41 ms CPU**, versus 96.46–102.89 ms elsewhere. All trials
perform exactly 225 neural calls and reach 32 terminal leaves, with identical
searched trees. The 149.11 ms wall/CPU gap suggests a transient system disturbance;
the data do not prove the scheduling or frequency cause.

The separate **40 interleaved pairs**, including the original first ten RNG
domains, reproduce identical searches and give engine/bitboard medians
**97.42/50.72 ms**, **1.921×** speedup, and p95 **98.49/52.04 ms** (ratio .528).
The slowest bitboard search, 52.76 ms, is faster than the fastest engine search,
95.97 ms. This supports no persistent meaningful backend regression.

The original observation is never removed, overwritten or pooled away.
`paired/summary.json` intentionally retains **`p95_pass: false`**. Criterion 4
is accepted **after the required investigation**, with that qualification;
this is not a claim that the initial strict p95 screen passed or a guaranteed
latency tail under arbitrary system load.

## Self-play throughput and memory

| Backend | Four games, wall s | Games/hour | Positions/s |
|---|---:|---:|---:|
| Engine | 11.762 | 1,224 | 10.12 |
| Bitboard | 5.617 | 2,564 | 21.19 |

The **2.094×** throughput gain measures complete collection, including action
selection, engine move application, capture and final outcome verification.
It is a small frozen-checkpoint sample, not a claim that all future generations
will improve by that ratio. No optimization/training is included.

Separate allocation probes cover all 112 checkpoint/position/budget cells.
Bitboard peak/engine peak ratios range **.293–.403**, with median **.302**:
about **69.8% lower** median paired Python-traced peaks. None increases.
Tracemalloc omits most native tensor storage.

Fresh isolated processes, each loading 42/g10 and warming quiet 512, report:

| Backend | Traced peak bytes | Process lifetime peak RSS MiB |
|---|---:|---:|
| Engine | 2,980,121 | 271.31 |
| Bitboard | 950,324 | 267.36 |

RSS comes from macOS `resource.getrusage`, includes imports/model loading/warm-up
and tracing, and is a process high-water mark. One process per backend cannot
establish a general RSS reduction. Main-run RSS high-water marks additionally
include earlier trials; those records are retained but are not search-only
incremental memory. Initial current-RSS probes using `ps` were sandbox-refused;
the failed attempts are retained and the supported high-water probe was rerun
in fresh directories.

## Remaining bottlenecks

Exclusive profiles across the same 112 cells per backend show:

| Category | Current engine % | Bitboard % |
|---|---:|---:|
| Neural forward, including contracts | 36.40 | 68.71 |
| Terminal/winner checks | 26.00 | 0.44 |
| State copying | 16.00 | 0.00 |
| Inference wrapper/softmax/scalar extraction | 8.12 | 14.21 |
| Encoding/tensor construction | 4.26 | 2.61 |
| PUCT selection | 3.13 | 5.64 |
| State transitions | 1.51 | 3.59 |

The copy category disappears in the bitboard path; root conversion is accounted
in residual work and remains inside search timing. Exclusive categories plus
other are additive; inclusive times overlap. Instrumentation overhead prevents
interpreting these shares as pure convolution kernel costs. Separate cProfile
records corroborate the shift away from deepcopy/winner scans. Neural inference
now dominates, while compact nodes alone would address a much smaller latency
fraction, though they could still reduce memory/cycles.

## Preservation, review and acceptance

The official directory remains
`/Volumes/Mini SSD/code/board-game-ai-lab-m4/experiment-output/phase4d3c-alphazero-v2-official-m4-20261008-110950`.
Its declaration SHA-256 remains
`c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716`.
All **150 file hashes** match Phase 4D.4A before/after every invocation and the
final verification, including checkpoints, manifests, frozen sources and
sealed bytes. Sealed evaluation rows were never parsed. Historical/Phase 4D.4A/
main-reference worktrees remain clean; the canonical 112-game benchmark,
deployments and Mac mini services were untouched.

The repository's older frozen declaration is also byte-preserved (SHA
`2e39713f6e2791c735406c3bec7def0d336adb6add1cd43d49319e83aac486ab`). Its test
now pins historical bytes/source identity and checks that changed development
sources cannot launch it. It was not re-frozen to accommodate new code.

An independent review found and resolved the slots/example-capture bug,
provenance source-group omission, and optimized-Python assertion bypass. New
regressions cover the capture integration and ownership rejection. The benchmark
refuses `python -O`. Review of core logic, executed source snapshots and measured
evidence found no remaining correctness issue; the reviewer independently
approved the qualified tail-latency conclusion and immutable-declaration test.

| Criterion | Result | Evidence |
|---|---|---|
| 1. Required equivalence checks | PASS | 10,138 transitions; complete archived matrix, neural bytes, noisy searches and self-play |
| 2. Visits/predictions/actions/RNG unchanged | PASS | 672 paired validations, all timed pairs, 200 tactical rows and complete games |
| 3. ≥1.5× on all six primary cells | PASS | 1.760–1.927× |
| 4. No meaningful latency regression; investigate >5% p95 | PASS after investigation | Raw screen FAIL retained; 40-pair follow-up p95 ratio .528 |
| 5. Peak traced allocations do not increase | PASS | All 112 ratios ≤.403 |
| 6. Relevant existing/new tests | PASS | 707 passed, 2 optional retained-v1-checkpoint skips; 44 new tests |
| 7. Official 150-file inventory unchanged | PASS | Each preservation record and final verification |
| 8. Training/evaluation evidence unchanged | PASS | Full inventory equality; frozen declaration bytes retained; no archive writes |

The complete experiment consumed **921.63 seconds (15.36 minutes)** across the
main run, follow-up, RSS probes and failed attempts, excluding ordinary tests.
No model training or multi-hour evaluation was launched. A short initial tool
attempt failed when the differential comparator assumed both states were
bitboards during an engine-only instrumentation check; it was corrected and
covered by a regression. Failed attempts are recorded separately and are not
aggregated into designated results.

Machine-readable evidence, all trial records, profiles, run registry,
preservation inventories, checkpoint/source hashes and compressed executed
source snapshots are in
`research/alphazero_bitboard/results/phase4d4b-20261011/`. The first short failed
probe retains hashes/partial latency; designated measurements and subsequent
attempts retain full snapshots. Main-run core state/search/encoding/capture
sources match the delivered implementation. Post-measurement edits add source
classification and benchmark controls/RSS reporting, without changing search
operations. Reproduction commands and backend usage are in
[the tooling README](../research/alphazero_bitboard/README.md).

## Recommended next phase

Test **batched neural inference across independent self-play games** with the
current architecture, fixed checkpoints and unchanged per-game search/RNG
semantics. The measured 68.7% neural-forward share and Phase 4D.4A's promising
batch-eight CPU probe justify a bounded collection experiment; require at least
1.3× complete-game throughput and exact per-game outputs where operation order
is retained. Batch-dependent floating-point behavior must be investigated
explicitly rather than hidden by tolerances.

Compact nodes are a secondary memory improvement. Root tactical safety and
proof propagation address the unchanged tactical weaknesses and should be
separate fixed-checkpoint strength experiments. This phase demonstrates faster
computation, not stronger neural predictions or changed playing strength. None
of those follow-up features was implemented here.
