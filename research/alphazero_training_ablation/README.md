# Phase 4D.4G managed exploration/training ablation

Read [DESIGN.md](DESIGN.md) before executing. Preparation, detached execution and
final analysis are separate stages. The manager refuses duplicate launch/retry.

All commands run in the phase4d4g-worktree, with the existing Python and
OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
VECLIB_MAXIMUM_THREADS and NUMEXPR_NUM_THREADS set to1. Use `-B`.

Preparation: `python -m research.alphazero_training_ablation.freeze_fixtures`,
then bounded `prepare` (external300s ceiling). Commit/push the final prospective
design, validated sources, frozen fixtures and readiness. Only then:

```
/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python -B -m research.alphazero_training_ablation.manage launch
```

The launcher installs thread controls for the detached supervisor/children. It
returns after a startup handshake. No active Codex session is required.

Progress: `python -B -m research.alphazero_training_ablation.manage status`, or
read `experiment-output/phase4d4g-primary/execution/status.json`. Durable logs are
in that directory and `experiment-output/phase4d4g-primary/supervisor.log`.
Do not infer completed training from a partial progress sidecar. Final status is
COMPLETE_PENDING_FINAL_ANALYSIS, STOPPED_INCOMPLETE or INTEGRITY_FAILURE.

Recovery: preserve failed task directory/logs and status; do not rerun launcher.
Read `last_boundaries` and any failed task's `published_boundary`. Inspect SHA and
strict load using existing `experimental_cli inspect` with the five thread env
settings. If death occurred between boundary publication and sidecar, locate only
the visible generation .pt, compute its file SHA and strictly inspect it. Hidden
.tmp archives are never completion evidence. Reconcile attempted actual games/
updates before proposing any new execution; no automatic repeats permitted.
Changing frozen sources/runtime/fixtures requires a documented fresh fork.

After completion, in a separate session, run evidence-only `analyze`, audit all
boundaries/frozen identities and150 official files, collect compact evidence from
local output into tracked results, re-review independently, finalize the phase
report and commit/push. Analyze refuses a partial run as a full experiment.
No checkpoint is promoted, deployed or merged into main.
