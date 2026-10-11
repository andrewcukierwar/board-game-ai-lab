# Phase 4D.4A read-only baseline tools

These tools read the completed official campaign and verify its declaration,
checkpoint hashes and frozen execution-source hashes. They never create a
GenerationRunner, train, invoke a campaign command, or resume a boundary.
`archive` reads stored replay tensors with `torch.load(weights_only=True)` only.

Run from this dedicated worktree with the existing M4 environment:

```sh
PYTHON=/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python
CAMPAIGN='/Volumes/Mini SSD/code/board-game-ai-lab-m4/experiment-output/phase4d3c-alphazero-v2-official-m4-20261008-110950'
REFERENCE='/Volumes/Mini SSD/code/board-game-ai-lab-m4/experiment-output/phase4d4a-main-reference'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"$PYTHON" -m research.alphazero_baseline.audit --campaign "$CAMPAIGN" \
  --output experiment-output/new-profile --mode profile --seconds 300 --repeats 5
"$PYTHON" -m research.alphazero_baseline.audit --campaign "$CAMPAIGN" \
  --output experiment-output/new-archive --mode archive --seconds 180
"$PYTHON" -m research.alphazero_baseline.audit --campaign "$CAMPAIGN" \
  --output experiment-output/new-state --mode state --main-reference "$REFERENCE" --seconds 90
"$PYTHON" -m research.alphazero_baseline.audit --campaign "$CAMPAIGN" \
  --output experiment-output/new-ladder --mode evaluate --main-reference "$REFERENCE" \
  --seconds 900 --opening-rows 2 --ladder-only
```

`evaluate` without `--ladder-only` assesses all 200 tactical and 300 solved
development rows for each of the eight checkpoints before playing games.
For a positions-focused run, set a small wall ceiling and retain the complete
position records if the ladder stops. Partial ladder records remain in JSONL.
`--opening-rows 2` selects the empty board and the first development prefix;
each is played with the tested agent as X and O. The explicit opening list
is saved before inference. It is a smoke test, not a strength tournament.

Every output directory must be new and disjoint from the campaign, including
after symlink resolution. Before/after SHA-256 inventories cover all campaign
files, including sealed bytes solely for preservation checks. Sealed rows
are never parsed or evaluated. The directory also holds the exact audit
script used, runtime settings, parameters, checkpoint identities and status.

Principal latency measurements exclude instrumentation, explicit pre-trial GC,
warm-up, checkpoint loading, file hashing and action selection. Timed search
includes root expansion, all simulations and constructing the search result.
Separate instrumented trials must reproduce every node's statistics and RNG
state. Exclusive categories add to wall time with `other`; inclusive categories
overlap. cProfile and tracemalloc run in separate probes. CPU peak RSS is a
process lifetime high-water mark; tracemalloc omits most native tensor storage.

The main reference worktree is imported under a separate module namespace.
Only its previously absent Negamax TT helper gets a canonical alias because
the upstream module uses an absolute import. Historical neural search, MCTS,
board and inference modules are untouched. Opponent source hashes and the
reference commit accompany every ladder run.

To rebuild the compact results and tables:

```sh
"$PYTHON" -m research.alphazero_baseline.summarize \
  --profile experiment-output/phase4d4a-20261011-profile-official-runtime \
  --positions experiment-output/phase4d4a-20261011-evaluation \
  --ladder experiment-output/phase4d4a-20261011-ladder \
  --archive experiment-output/phase4d4a-20261011-archive-final \
  --state experiment-output/phase4d4a-20261011-state \
  --output experiment-output/new-summary
```

Family bootstrap intervals preserve X/O pairs but are unreliable with two
opening families. Degenerate intervals are not evidence of certainty. No
Elo or significance estimate is made from this ladder. Development data
have already been used for selection and are not fresh held-out evaluation.

Tests cover full-tree/RNG equivalence with and without root noise, nested
exclusive timing, restoration on exceptions, output isolation through
symlinks, pair-preserving aggregation, and archive/replay schema extraction.
The existing v2 tests remain the algorithmic regression suite.
