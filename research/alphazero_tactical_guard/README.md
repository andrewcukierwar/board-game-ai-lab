# Experimental tactical root guard

See [DESIGN.md](DESIGN.md) and the [report](../../docs/phase4d4j-a-alphazero-tactical-guard.md).
The production app and historical AlphaZero sources are unchanged.

All commands use the existing board-game-alphazero-m4 Python environment with
OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, VECLIB_MAXIMUM_THREADS,
NUMEXPR_NUM_THREADS set to1 before importing torch. Modules: `pilot`, `validate`,
`prepare`, `prepare --freeze`, `manage`, `analyze`, `independent_verify`.
Frozen measured run has one attempt only; analysis and independent saved-record
verification can be reproduced without neural evaluation after completion.
Machine-readable identities/evidence live in results/. Checkpoints stay in their
original ignored H-C output paths, accessed strictly read-only.
