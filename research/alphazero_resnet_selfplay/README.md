# Phase 4D.4H-C — controlled CNN vs ResNet 3x64 self-play experiment

Report: [docs/phase4d4h-c-resnet-selfplay.md](../../docs/phase4d4h-c-resnet-selfplay.md).
Protocol, frozen before training: [DESIGN.md](DESIGN.md).

| File | Role | Frozen |
|---|---|---|
| `DESIGN.md` | Prospective protocol and decision rules | Yes |
| `common.py` | Paths, durable JSON, preservation inventories, source freeze | Yes |
| `calibrate.py` | Bounded cost calibration with trained weights; opponent probe | Yes |
| `validate.py` | Model-factory and boundary compatibility, fresh-process resume | Yes |
| `prepare.py` | Budgets, feasibility projection, generation-zero boundaries, freeze | Yes |
| `manage.py` | Detached supervisor: `launch`, `status`, `cancel` | Yes |
| `worker.py` | One bounded task per fresh interpreter: `train`, `positions`, `matches`, `dynamics` | Yes |
| `evaluate.py` | Raw policy, raw PUCT and fixed-opponent matches on development fixtures | Yes |
| `measures.py` | Read-only training-dynamics measures | Yes |
| `analyze.py` | Evidence-only analysis and decision rules | Yes |
| `fixtures.json` | Phase 4D.4G's development fixtures, byte-identical | Yes |
| `results/` | Calibration, smoke, validation, design, freeze, then run evidence and analysis | Design inputs yes |

Checkpoints are local and ignored by git, under
`experiment-output/phase4d4h-c/` in this worktree. Their hashes are in
`results/`.

## Commands

All from this worktree, with the five thread variables set to 1 and the M4
environment's Python (`/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python -B`).

```
python -m research.alphazero_resnet_selfplay.manage status     # ledger summary
python -m research.alphazero_resnet_selfplay.manage cancel     # cooperative stop; nothing is retried
python -m research.alphazero_resnet_selfplay.analyze           # evidence-only; queries no model
python -m pytest tests/test_alphazero_resnet_selfplay.py
```

`manage launch` permits exactly one launch per design and refuses to run from
sources, a runtime or a design other than the frozen ones.

## Recovery

Nothing is retried automatically and no generation beyond the declaration may
be run. Published boundaries are complete generations; inspect one with

```
python -m games.connect4.alphazero_v2.experimental_cli inspect --boundary <path> --sha256 <sha>
```

A generation that did not finish leaves no boundary. Any continuation needs a
new prospective declaration.
