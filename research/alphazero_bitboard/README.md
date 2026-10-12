# Phase 4D.4B: optional bitboard neural PUCT

The historical engine backend remains the default. Select the representation
explicitly with `PUCTSearch(inference, simulations, backend="bitboard")`,
`EvaluationAgent(inference, simulations, backend="bitboard")`, or
`SelfPlayer(inference, config, ..., backend="bitboard")`. No campaign configuration,
network weights, search parameters, training targets or production factory is changed.
A caller always supplies a standard engine board; search converts it once.

The state module adapts the geometry/drop core of main's `mcts_bitboard.py`
at `8704142702da273e6571f0ca9df3fd4a793be285`, without its rollout/search changes.
The copied rule core is isolated here because the historical baseline has no
shared bitboard module. Its immutable internal constructor is trusted; public
engine conversion validates shape, symbols, actor/piece agreement, gravity and
ambiguous simultaneous winners. This backend supports legal Connect 4 states,
not artificial gravity holes or two-winner boards. Diagnostic `state.board` is
read-only and is never built by search or inference.

Run from the dedicated Phase 4D.4B worktree, using a **fresh** output path:

```sh
PYTHON=/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python
CAMPAIGN='/Volumes/Mini SSD/code/board-game-ai-lab-m4/experiment-output/phase4d3c-alphazero-v2-official-m4-20261008-110950'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"$PYTHON" -m research.alphazero_bitboard.benchmark \
  --campaign "$CAMPAIGN" --output experiment-output/new-paired \
  --repeats 10 --validation-trials 3 --games 4 --seconds 1500
"$PYTHON" -m research.alphazero_bitboard.benchmark \
  --campaign "$CAMPAIGN" --output experiment-output/new-rss-engine \
  --mode rss --backend engine --seconds 60
"$PYTHON" -m research.alphazero_bitboard.benchmark \
  --campaign "$CAMPAIGN" --output experiment-output/new-rss-bitboard \
  --mode rss --backend bitboard --seconds 60

"$PYTHON" -m pytest -q tests/test_alphazero*.py \
  tests/test_connect4_neural_mcts.py tests/test_neural_mcts_import_isolation.py \
  tests/test_connect4_mcts.py tests/test_connect4_mcts_api.py
```

Use normal Python, without `-O`: assertions enforce exact comparisons.
The harness never creates a runner/optimizer or parses sealed evaluations.
Before any inference it checks the 150-file Phase 4D.4A inventory, declaration
hash and all eight checkpoint hashes. A finally block checks all official file
bytes again, including after failure. Sealed artifacts are hashed only.
Execution source bytes and hashes are retained per run.

The seven fixed positions are read verbatim from the Phase 4D.4A profile.
The paired workload covers both archived seeds and generations 0/10/15/20,
256/512 simulations and ten independent trial domains per cell. Trials alternate
engine/bitboard order. Complete searches are warmed for each backend and cell.
RNG construction, explicit GC, comparisons, loading, tracing and profiling are
outside latency; root conversion/expansion and result construction are inside.
Automatic GC remains enabled identically. CPU and wall clocks bracket the same
search. The search does not select an action; action selection is checked
separately at both temperatures.

Separate validation trials record actual model input/output bytes and every
legal-softmax prediction, then compare all allocated nodes (including unvisited
children), parents, moves, states, terminal status, priors, PUCT scores, visits,
Q/value sums and depth counters. Search/action/noise RNG states are compared
exactly. Three trials run with and without self-play noise for every checkpoint,
position and budget. Complete self-play uses unchanged V2Config defaults,
compares completed examples/outcomes/trajectories and all streams, and verifies
model-weight hashes before/after. Development tactical checks use 200 existing
rows with 42/g10 at 512 simulations; they are differential regressions, not a
new strength tournament.

Memory probes cover every checkpoint/position/budget independently of latency.
Tracemalloc reports Python current/peak bytes and excludes most tensor/native
storage. RSS reports process resident memory: main-process lifetime high-water
marks include all earlier work; fresh-process RSS probes offer a more controlled
comparison but still include imports, checkpoint loading and warm-up. Neither
quantity is interchangeable with traced allocations.

Aggregate latency follows Phase 4D.4A: median of eight checkpoint medians.
Machine-readable trials retain wall/CPU times, min/max, visits, depths, calls,
terminal leaves and simulations/second. P95 over 80 observations per backend
and position/budget describes the fixed checkpoint mixture. Per-checkpoint p95
from ten trials is unstable and is screened for >5% regression, not treated as
an accurate population-tail estimate. Exclusive nested profile categories add
to elapsed time; inclusive categories overlap. cProfile corroborates hotspots.

Results and acceptance decisions are in `docs/phase4d4b-alphazero-bitboard-puct.md`
and `results/phase4d4b-20261011/`. An initial short failed instrumentation
comparison remains recorded separately; it never altered official evidence.

The primary ten-trial matrix flagged one per-checkpoint p95 cell. The preserved
40-pair follow-up does not replace or remove that original observation:

```sh
"$PYTHON" -m research.alphazero_bitboard.benchmark \
  --campaign "$CAMPAIGN" --output experiment-output/new-tail-followup \
  --mode focus --seed 42 --generation 10 --position tactical_midgame \
  --simulations 256 --repeats 40 --seconds 90
```

The raw primary `summary.json` retains `p95_pass: false`. The phase report and
`acceptance.json` explain the investigated acceptance decision. Source snapshots
preserve the exact tool versions for designated measurements and subsequent
attempts. The initial short failed instrumentation probe retains hashes and
partial latency, and is excluded from the designated results.
