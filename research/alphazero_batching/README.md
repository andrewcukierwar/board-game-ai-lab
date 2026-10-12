# Phase 4D.4C: independent-game CPU neural batching

Opt in explicitly; the historical collector and engine default are unchanged:

```python
from games.connect4.alphazero_v2.batching import collect_batched, CollectionStats
stats = CollectionStats()
games = collect_batched(inference, config, generation=21, indices=range(16),
                        batch_size=8, stats=stats)
```

This is a computational experiment with frozen weights, not a campaign runner.
`collect_serial_reference` invokes the unchanged Phase 4D.4B bitboard search
with the same NEW per-game seed protocol. It does not reproduce historical
campaign RNG streams. `V2Config.batch_size` remains the training minibatch field;
self-play concurrency uses the separate explicit `batch_size` argument.

`game_seed` hashes ASCII
`connect4-az-independent-games-v1:{base_seed}:{generation}:{index}:{domain}`
with SHA256 and interprets all 32 bytes big-endian. Domains are `search`,
`action`, `noise`. Search/action use distinct Python `random.Random`; noise
uses the existing V2RootNoise (which additionally hashes its historical domain
and draws from PCG64). Streams restart by game identity, never by pool slot.
Output games sort by ID; input ID order controls dispatch. Reordering can alter
batch composition/rounding with real networks, but has no scheduling RNG draws.

Each game generator owns an engine state for capture and an immutable bitboard,
a history, pending examples and three streams. A nested search generator yields
its root and then nonterminal selected leaves. The driver batches one outstanding
request from each active game, sends each prediction to its owner, and refills
finished slots immediately. Terminal traversals back up exact values without
requests. A tree never has two traversals in flight; there is no virtual loss,
transposition, tree sharing/reuse, tactical guard or altered simulation budget.
Games can reach different plies without a global ply lockstep. No padding is
used: batch occupancy shrinks at the end of a finite requested collection.

`predict_batch` accepts lists/tuples of structurally consistent nonterminal
BitboardState objects, preserving row/column order, actor encoding and signed
empty-cell zeros. It checks bit types/ranges, disjoint pieces, occupancy, turn
counts, gravity and winner consistency. As in the existing trusted immutable
state constructor, this is not an exhaustive historical reachability solver;
callers supply positions from legal histories. It rejects malformed states
instead of repairing them. All scheduler states originate from legal empty-board
play, are cross-checked against the engine at every move, and are replayed at
finalization. Empty inference batches are no-ops; singleton delegates unchanged
`predict`. Larger batches use one float32 CPU `(N,1,6,7)` tensor and forward,
then the existing legal-only float64 softmax independently per row.

Failure returns no dataset. `on_game` can already have observed complete games
before a later failure, as with the historical collector; it never observes
pending examples. `check` runs before batches and every simulation, bounding
cooperative cancellation to one inference/traversal/capture/finalization segment.
There is no thread, queue or asynchronous cancellation of a PyTorch forward.
Do not reuse game-owned RNGs in observers. A bounded active pool plus the requested
completed dataset bounds memory; completed records necessarily grow with the
requested collection, while outstanding inference requests do not.

Read [DESIGN.md](DESIGN.md) for thresholds declared before measurements and
[the report](../../docs/phase4d4c-alphazero-batched-inference.md) for results.
No thresholds may be relaxed after seeing results.

Run from this research worktree, using fresh disjoint output paths:

```sh
PYTHON=/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python
CAMPAIGN='/Volumes/Mini SSD/code/board-game-ai-lab-m4/experiment-output/phase4d3c-alphazero-v2-official-m4-20261008-110950'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

"$PYTHON" -m research.alphazero_batching.benchmark --campaign "$CAMPAIGN" \
  --output experiment-output/new-numerical --mode numerical --seconds 600
"$PYTHON" -m research.alphazero_batching.benchmark --campaign "$CAMPAIGN" \
  --output experiment-output/new-collection --mode collection --games 16 --repeats 2 --seconds 600
"$PYTHON" -m research.alphazero_batching.benchmark --campaign "$CAMPAIGN" \
  --output experiment-output/new-fixed --mode fixed --seconds 90
"$PYTHON" -m research.alphazero_batching.benchmark --campaign "$CAMPAIGN" \
  --output experiment-output/new-profile --mode profile --games 8 --seconds 120
"$PYTHON" -m research.alphazero_batching.benchmark --campaign "$CAMPAIGN" \
  --output experiment-output/new-confirmation --mode confirmation --games 16 --seconds 120
# Run memory in a FRESH process for each condition: serial, 1, 2, 4, 8.
"$PYTHON" -m research.alphazero_batching.benchmark --campaign "$CAMPAIGN" \
  --output experiment-output/new-memory-8 --mode memory --condition 8 --games 16 --seconds 120
"$PYTHON" -m research.alphazero_batching.summarize \
  --numerical experiment-output/new-numerical --collection experiment-output/new-collection \
  --output experiment-output/new-summary

"$PYTHON" -m pytest -q tests/test_alphazero*.py \
  tests/test_connect4_neural_mcts.py tests/test_neural_mcts_import_isolation.py \
  tests/test_connect4_mcts.py tests/test_connect4_mcts_api.py
```

Each benchmark verifies the Phase 4D.4A 150-file campaign inventory, declaration
and all eight checkpoint hashes before any inference; it verifies the inventory
again in `finally`, including failures. Sealed artifacts are hashed as bytes
only. Source snapshots/hashes and runtime/hardware settings accompany each run.
The unchanged baseline search/selfplay/bitboard/config/data sources remain pinned
to df8a586. The tools refuse optimized Python and require vector threads pinned
before imports; no accelerator, quantization, training, strength tournament,
production integration or remote deployment is performed.

Primary complete collection includes capture/finalization, statistics and a
lightweight decision recorder; neural inclusive time includes encoding,
validation, forward and softmax. A separate observer-free confirmation uses the
original V2Inference without counting wrappers. Profiling and tracemalloc probes
are separate from primary timing. Fixed inference has identical inputs/counts;
fixed search has identical positions/budgets, with neural leaf counts recorded.
Per-game observer durations include waiting and must not be summed. Completion
and active residence times are latency measurements. Scheduler overhead is direct
driver residual: collection wall minus measured inference minus generator-advance
work. The latter contains tree traversal, backup, capture and callbacks. Serial
cancellation checks are per move; batched additionally checks every simulation.
These small diagnostics/check overheads are retained, not subtracted from rates.
RSS is fresh-process lifetime high-water including import/load/warmup/tracing;
Python tracemalloc excludes most native tensor memory. Avoid comparing traced
probe wall times to untraced primary rates.

External references are pinned in `external_sources.json`. Eugeneyp batches roots
and leaves across independent trees and refills completed game slots. Suharshkumar
uses lockstep independent trees and excludes terminal leaves from inference.
Kuotunyu explicitly permits one leaf per tree per tick and refills completed
games. AlphaZero.jl separates simulation tasks from an inference server and
optionally fills batches. These are engineering references; their hyperparameters,
root-visit conventions, global RNG usage, hardware and performance claims are
not transplanted or used as evidence of our measured speedup. No external code
was copied into the implementation.


Large machine-readable JSON records in the committed results are gzip-compressed
UTF-8 JSON (`*.json.gz`); use `gzip.open(path, "rt")` followed by `json.load`.
Small summaries, provenance, registry and acceptance files remain plain JSON.
The complete raw uncompressed outputs remain in the ignored experiment folders.
To re-run aggregation on committed evidence, copy/decompress the numerical and
collection folders into fresh output folders first. Executed-source archives
preserve each invocation; the numerical predecessor differs from the final
collector only in optional diagnostic clocks, with no change to game/search/RNG
operations. Later harness changes add profiling/confirmation modes and stricter
postprocessing; primary source hashes match the delivered inference/collector.
