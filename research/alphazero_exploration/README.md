# Phase 4D.4D: frozen-checkpoint self-play exploration and diversity

Does a different action-temperature schedule give more diverse self-play data
without more tactical mistakes? Frozen checkpoints only: nothing is trained,
resumed, promoted or deployed. Read [DESIGN.md](DESIGN.md) (committed before
any collection) for the schedules, identities, endpoints and decision rules,
and [the report](../../docs/phase4d4d-alphazero-exploration-diversity.md) for
results.

## What changed in the collector

`collect_batched` accepts an optional `temperature_schedule`:

```python
from games.connect4.alphazero_v2.batching import TemperatureSchedule, collect_batched
schedule = TemperatureSchedule('extended', ((0, 1.0), (16, 0.0)))
games = collect_batched(inference, config, generation=21, indices=range(64),
                        batch_size=8, temperature_schedule=schedule)
```

Segments are `(first zero-based ply, temperature)` steps starting at ply 0.
`None` (the default) keeps `config.exploratory_plies` and is byte-for-byte the
Phase 4D.4C behaviour. `V2Config`, the serial collector, the campaign runner,
PUCT, root noise and the visit targets are untouched; no production path
passes a schedule. A schedule changes only which visited action is executed.
`select_action` draws once per move at every temperature, so matched game
identities consume identical RNG streams until their trajectories differ.

## Tools

| Module | Purpose |
|---|---|
| `schedules.py` | The four declared schedules, checkpoints, cohort seeds and stages |
| `collect.py` | Bounded pool-eight collection: `regression`, `pilot`, `main`, `supplement` |
| `analysis.py` | Verified loading, positions/mirrors, tactics and attribution, jackknife |
| `replay.py` | Dataset replay profile, archived reference, counterfactual replay sampling |
| `analyze.py` | Applies the prospective analysis; writes results, decision, acceptance, tables |

`collect.py` writes one JSON line per COMPLETE game as it finishes (moves,
winner, per-ply visits, executed action, temperature, execution distribution,
root prior and value, record and RNG hashes). A deadline stop marks the cell
incomplete in `cells.json`; `analysis.load_collection` never loads such a
cell, and `analyze.matched_blocks` drops any checkpoint x cohort block that is
not complete for all four schedules with identical identities.

## Reproduce

Run from this worktree with fresh, disjoint output paths:

```sh
PYTHON=/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python
CAMPAIGN='/Volumes/Mini SSD/code/board-game-ai-lab-m4/experiment-output/phase4d3c-alphazero-v2-official-m4-20261008-110950'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

"$PYTHON" -m research.alphazero_exploration.collect --campaign "$CAMPAIGN" \
  --output experiment-output/new-regression --mode regression --seconds 240
"$PYTHON" -m research.alphazero_exploration.collect --campaign "$CAMPAIGN" \
  --output experiment-output/new-pilot --mode pilot --seconds 240
"$PYTHON" -m research.alphazero_exploration.collect --campaign "$CAMPAIGN" \
  --output experiment-output/new-main --mode main --seconds 1500
"$PYTHON" -m research.alphazero_exploration.analyze --campaign "$CAMPAIGN" \
  --regression experiment-output/new-regression --pilot experiment-output/new-pilot \
  --main experiment-output/new-main --output experiment-output/new-analysis

"$PYTHON" -m pytest -q tests/test_alphazero*.py \
  tests/test_connect4_neural_mcts.py tests/test_neural_mcts_import_isolation.py \
  tests/test_connect4_mcts.py tests/test_connect4_mcts_api.py
```

Identical runtime, checkpoint, schedule, base seed, identities, order and pool
size reproduce every record and RNG hash exactly. Each invocation verifies the
150-file official inventory, declaration and eight checkpoint hashes before
inference and hashes the inventory again in `finally`. The analysis reads the
archived generation-20 game log and replay storage read-only and hashes the
inventory before and after.

To re-run the analysis on the committed evidence, point `--regression`,
`--pilot` and `--main` at the folders under `results/`; game files there are
gzip-compressed (`*.jsonl.gz`) and are read transparently.

## Reading the results

- Games are the sampling units. Intervals are delete-one-game jackknife
  intervals, paired by game identity for differences. They are descriptive.
- The duplicate fraction grows with the number of games, so compare it only
  between equal-sized datasets. The pairwise collision probability does not.
- More positions from longer games are not more diversity. Decisions use only
  the stage-conditioned, normalized endpoints.
- The pilot is a cost estimate. Its games are never analyzed.
- Replay figures describe sampling exposure. They do not show better gradient
  updates, generalization or playing strength; that needs a training ablation.
