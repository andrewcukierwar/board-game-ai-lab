# Experimental training infrastructure, Phase 4D.4F

This package supports **validated forks**, then exact experimental continuation.
It never resumes the official campaign. `validate_readiness.py` imports both
archived seeds read-only and does **zero** collection and **zero** updates.
`results/readiness.json` records the independent arm state identities. Bulky
fork checkpoints live in fresh ignored output directories, not in this package.

Run from the Phase 4D.4F worktree with the existing M4 environment:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
AZ_PY=/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python
```

Create a pair without training (substitute the actual source boundary and its
SHA from `results/readiness.json` / the archived `summary.json` `files.resume`):

```sh
"$AZ_PY" -B -m games.connect4.alphazero_v2.experimental_cli fork-pair \
  --parent "$PARENT" --parent-sha256 "$PARENT_SHA" \
  --historical-checkout '/Volumes/Mini SSD/code/board-game-ai-lab-m4' \
  --output experiment-output/new-approved-experiment/seed-42 \
  --lineage phase4d4g-seed-42 --games 64 --updates 64 --new-generations 2
```

This command validates the source with the original strict historical loader
in an isolated process importing the pinned historical checkout. It does not
change that checkout or call `run_generation` on its runner. It verifies the
manifest file hash, architecture, source-content identity, every runtime field,
weights, optimizer state, trainer counters, full replay and historical schedules.
Conversion then verifies ordered AdamW parameter groups, moment shapes and
finiteness, moment steps, and exact reconstructed replay identity again.
Neither the historical loader nor its source contracts were edited.

Both arms have equal `initial_state_sha256` before collection, including model,
optimizer, counters, replay, sampling/reflection state, global/Torch RNGs,
training configuration and parent provenance. Arm name and prospective current
schedule are excluded from that equality identity. Their full checkpoint hashes
and integrity hashes differ because the declared arm/schedule differs.

**Only after separate approval for Phase 4D.4G**, run one generation:

```sh
"$AZ_PY" -B -m games.connect4.alphazero_v2.experimental_cli run-one \
  --boundary "$BOUNDARY" --sha256 "$BOUNDARY_SHA" \
  --output experiment-output/new-approved-experiment/seed-42/control/gen-21
```

The output directory must be new. The command completes exactly one generation,
including persistence. It writes a complete `.pt` boundary and `resources.json`.
Use the printed `boundary_sha256` for the next invocation. The boundary already
contains the model, so every completed generation is also a model checkpoint.
A separately exported inference artifact is unnecessary for recovery.

To recover, load the last atomically published boundary and its trusted hash.
After any failure, discard the live runner; do not save its attempted state.
Remove or choose a different empty scratch attempt directory before retrying.
If a process dies after publication but before printing its hash, compute the
file SHA, inspect/reload the file, and record that recovered hash in your
experiment manifest. Never trust a hidden `.tmp` file as a completed generation.
For a directory-fsync failure after link publication, the visible file is fully
validated, but crash durability is uncertain. Verify it before selecting it;
otherwise rerun from the prior boundary into a fresh directory.

```sh
"$AZ_PY" -B -m games.connect4.alphazero_v2.experimental_cli inspect \
  --boundary "$BOUNDARY" --sha256 "$BOUNDARY_SHA"
```

Python callers can use `check()` for cancellation before batches/simulations,
updates and checkpoint validation, and `progress(event)` for accounting. A
callback must not mutate runner state or consume its RNG streams. No mid-search
state is saved. `evaluate_boundary(path, evaluator, expected_sha256=...)` gives
an independent frozen model and counters to an RNG-neutral evaluation hook.
Evaluation outcomes do not choose a winner, alter training, or promote a model.

Run focused checks (the wrapper executes all 56 cases in an isolated interpreter,
so native Torch inter-op configuration cannot leak into historical tests):

```sh
"$AZ_PY" -m pytest -q -s tests/test_alphazero_experimental.py
```

See [the boundary schema](boundary-schema.md), [independent review](independent-review.md)
and [the phase report](../../docs/phase4d4f-alphazero-training-infrastructure.md).
