# Phase 4D.4E research tools

Frozen inference only; no training or training-runner changes. Read DESIGN.md
before interpreting results. Its commit `89c8822` was pushed before main collection;
analysis was committed `c549021` before collection, with subsequent reviewer-led
robustness fixes tracked. Results use every decision at zero-based ply >= 8.

Use the existing M4 Python and set environment BEFORE interpreter startup:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
C4_PY=/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python
C4_CAMPAIGN='/Volumes/Mini SSD/code/board-game-ai-lab-m4/experiment-output/phase4d3c-alphazero-v2-official-m4-20261008-110950'
```

From this worktree, reproduce analysis from committed exact proofs (no new
solver execution or self-play, fresh output directory required):

```sh
"$C4_PY" -m research.alphazero_exploration_confirmation.analyze \
  --campaign "$C4_CAMPAIGN" --collection research/alphazero_exploration_confirmation/results/main \
  --proofs research/alphazero_exploration_confirmation/results/exact/proofs.json \
  --output /tmp/phase4d4e-reproduction
```

The source loads gzip game files without changing bytes, replays legal engine
moves, checks temperatures/action distributions/visit budget/actor targets,
rebuilds each normalized target record and its hash, and checks matching prefixes.
JSON values are exact W/D/L proof results, never approximate network estimates.
Native queries record history, nodes, wall time, status, and abort reason;
`decisions.jsonl.gz` has every eligible decision; `unresolved.jsonl.gz` explicitly
lists unknowns. Scalar values also have exact deductions from immediate wins,
terminal outcomes and completed successor proofs. The unchanged native source
is pinned to main-reference `8704142` and independent of AlphaZero PUCT.

Fresh collection, if separately wanted, must use a fresh output directory and
stay within the declared combined execution budget:

```sh
"$C4_PY" -m research.alphazero_exploration_confirmation.collect \
  --campaign "$C4_CAMPAIGN" --output /tmp/phase4d4e-fresh-main \
  --mode main --games 64 --seconds 650
"$C4_PY" -m research.alphazero_exploration_confirmation.analyze \
  --campaign "$C4_CAMPAIGN" --collection /tmp/phase4d4e-fresh-main \
  --output /tmp/phase4d4e-fresh-exact --ceiling 600
```

Collection identities/conditions are fixed in schedules.py and DESIGN.md.
Recollection establishes deterministic behavior under this runtime, not
independent replication: new identities are needed for independent research.
Native wall aborts can affect coverage; proof-based reanalysis reproduces the
committed findings without attempting timing-dependent queries again.

Focused and relevant inference/analysis tests (no optimization steps):

```sh
"$C4_PY" -m pytest -q tests/test_alphazero_exploration_confirmation.py \
  tests/test_alphazero_exploration.py tests/test_alphazero_v2_batching.py \
  tests/test_alphazero_v2_bitboard.py tests/test_alphazero_v2_exploration.py \
  tests/test_alphazero_baseline.py
```

Feasibility used old 4D positions only. Regression uses the existing4D driver
with a fresh output directory, retains all 64 complete validation games and is
excluded from confirmatory statistics. Both previous tool trees and campaign
artifacts remain immutable. Per-invocation preservation records cover all 150
campaign files; main provenance records all 8 inference checkpoints and source
hashes. Final manifest covers delivered new artifacts. Delivered statistics are in
`results/analysis-final/`; the prior analysis and failed summary remain preserved. See the phase report
and independent `results/REVIEW.md` for findings and limits.
