# Phase 4D.4H-B — ResNet cost and fixed-data learning benchmark

The report is `docs/phase4d4h-b-resnet-benchmark.md`. Read [DESIGN.md](DESIGN.md)
first: it is the protocol that was frozen before anything was measured.

All commands run from the worktree root with the campaign interpreter, `-B`,
and the five thread variables set to 1:

```
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python
```

| Step | Command | Writes |
|---|---|---|
| Dataset freeze and leakage audit | `$PY -B -m research.alphazero_resnet_benchmark.dataset` | `results/dataset-freeze.json`, `results/dataset-audit.json` |
| Architectural validation | `$PY -B -m research.alphazero_resnet_benchmark.validate` | `results/validation.json` |
| Cost-only pilot | `$PY -B -m research.alphazero_resnet_benchmark.pilot` | `results/pilot.json` |
| Pipeline smoke test | `$PY -B -m research.alphazero_resnet_benchmark.smoke` | `results/smoke.json` |
| Design and source freeze | `$PY -B -m research.alphazero_resnet_benchmark.plan` | `results/design.json`, `results/freeze.json` |
| Measured run (detached) | `$PY -B -m research.alphazero_resnet_benchmark.manage launch` | `experiment-output/phase4d4h-b/run/` |
| Progress | `$PY -B -m research.alphazero_resnet_benchmark.manage status` | — |
| Analysis | `$PY -B -m research.alphazero_resnet_benchmark.analyze` | `results/benchmarks.json`, `results/learning.json`, `results/comparisons.json`, `results/endpoint-detail.json` |
| Report tables and figures | `$PY -B -m research.alphazero_resnet_benchmark.report` | `results/tables.md`, `results/*.svg`, `results/resources.json` |

`manage launch` requires the frozen design to be committed and pushed, refuses a
second launch, and returns after a handshake. The supervisor runs inference,
training, collection and the two learning seeds in order under one absolute
deadline, with no retry. Final status is `COMPLETE_PENDING_ANALYSIS`,
`STOPPED_INCOMPLETE` or `INTEGRITY_FAILURE`; `analyze` refuses anything but the
first.

| Module | Role |
|---|---|
| `common.py` | Paths, durable JSON, official-file preservation, the source freeze |
| `dataset.py` | Strict read of the generation 13-20 replay, family split, leakage audit |
| `validate.py` | Independent reference forward pass, checkpoint contracts, fresh-process resume, batch norm, optimizer, CNN compatibility, isolation |
| `bench.py`, `stages.py` | Inference, collection and training cost measurements |
| `metrics.py` | Held-out, development-suite, batch-norm and calibration evaluation |
| `learn.py` | Fixed-data training of the three architectures for one seed |
| `manage.py` | Detached supervisor |
| `analyze.py` | Cost summaries, contrasts, bootstrap intervals, verdicts |
| `report.py` | Tables and figures from the analysis output (presentation only) |
| `pilot.py`, `smoke.py`, `plan.py` | Feasibility pilot, pipeline smoke test, design freeze |

Binary outputs (final and 3,000-update model states, per-position scores, full
per-update series) stay in the ignored `experiment-output/phase4d4h-b/run/` tree;
their SHA-256s are in the tracked results. Tests: `tests/test_alphazero_resnet_benchmark.py`.

Nothing here writes to the official campaign, the historical checkout or the
Phase 4D.4G worktree, and nothing trains by self-play.
