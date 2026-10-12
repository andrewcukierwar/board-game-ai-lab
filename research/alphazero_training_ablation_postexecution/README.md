# Phase 4D.4G evidence-only final analysis

These supplementary tools were added after execution in a sibling package so
the original 44-file source freeze stays intact. They do not train, search,
infer with a network, solve new positions, change fixtures or select a model.
`audit` and `overlap` only strictly load existing boundaries and inspect data.

Run from the existing Phase4D.4G worktree with the original M4 Python:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
AZ_ANALYSIS_PY=/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python
```

The unchanged original README's `analyze` and `publish_evidence` commands were
used for the first completed results package. Reproduce frozen statistics into
a fresh file, preserving the committed first analysis:

```sh
"$AZ_ANALYSIS_PY" -B -m research.alphazero_training_ablation.analyze \
  --output /private/tmp/phase4d4g-analysis-reproduction.json
```

Compare parsed JSON to `research/alphazero_training_ablation/results/analysis.json`.
The frozen publisher refuses an existing destination. Verify the retained package
against `results/evidence-manifest.json`; do not delete/overwrite it to republish.
On a separate fresh copy with local checkpoints/output restored, the original
publisher can create its first `results/execution-evidence/`.

Supplementary modules:

- `python -B -m research.alphazero_training_ablation_postexecution.audit`: strict
  twelve-boundary lineage/evidence/hash/source/runtime/targets/accounting audit;
  official inventory and prior tracked-file/HEAD preservation. JSON-normalizes
  int-key/tuple history containers only; Git reporting is not computational state.
- `...summarize`: recorded replay exposure/diversity/loss/resource summaries and
  per-checkpoint tactical/deep/parent contrasts, same frozen bootstrap units.
- `...overlap`: descriptive reflected-board matches against available retained
  replay; no new held-out split, endpoint or model query.
- `...report`: generates the final Markdown tables from the audited JSON.

These commands reproduce current derived files; use a copy for a fresh audit
if preserving the first postexecution timings is required. Timings of new audits
are nondeterministic and are separate from the primary execution budget.

The independent reviewer used separate board logic and multiplicity-weighted
bootstrap implementations. Reproduce its JSON in a copy, in this order:

```sh
"$AZ_ANALYSIS_PY" -B -m research.alphazero_training_ablation_postexecution.independent_review
"$AZ_ANALYSIS_PY" -B -m research.alphazero_training_ablation_postexecution.independent_diagnostics
"$AZ_ANALYSIS_PY" -B -m research.alphazero_training_ablation_postexecution.independent_games
```

Later review annotations/calibration crosschecks are preserved in the review
JSON/Markdown; the scripts reproduce their core raw-record calculations.
Never use optimized Python (`-O`) for reviewer scripts that use assertions.

Relevant checks:

```sh
"$AZ_ANALYSIS_PY" -B -m pytest -q \
  tests/test_alphazero_ablation.py tests/test_alphazero_ablation_results.py
```

Primary binaries remain ignored under `experiment-output/phase4d4g-primary/`.
Exact paths/file hashes/model/optimizer/RNG/state hashes are in
`results/integrity-audit.json`. Recovery needs the unchanged historical/runtime
contracts and the documented strict `experimental_cli inspect`; there is no
permission or need to launch another training/evaluation job here.
