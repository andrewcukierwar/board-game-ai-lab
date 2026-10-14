# Tactical root guard confirmation (Phase 4D.4J-B)

See [DESIGN.md](DESIGN.md) and the [report](../../docs/phase4d4j-b-alphazero-tactical-guard-confirmation.md).
The 4D.4J-A guard is imported unchanged; production app and historical sources are untouched.

Environment: `/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python` with OMP/MKL/OPENBLAS/
VECLIB/NUMEXPR thread variables set to 1. Modules, in order: `openings` (torch-free), `calibrate`,
`validate`, `prepare`, independent prelaunch review, `prepare --freeze`, `manage` (one attempt),
`analyze`, `independent_verify` (results reviewer). `selftest_analysis` is a synthetic, model-free
pre-freeze dry run of the analysis. Raw logs/attempt files live in ignored `experiment-output/phase4d4j-b/`;
match records and diagnostics are copied to `results/tasks/`.
