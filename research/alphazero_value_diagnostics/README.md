# AlphaZero frozen value diagnostics

Prospective protocol: [DESIGN.md](DESIGN.md). Independent review:
[independent-prelaunch-review.md](independent-prelaunch-review.md).
No training/collection entrypoint exists. Strict frozen-boundary access is read-only.

Use the established M4 Python environment and all five thread controls at1,
plus PYTHONDONTWRITEBYTECODE=1. Entry points:

- `python -m research.alphazero_value_diagnostics.validate`: focused correctness checks.
- `python -m research.alphazero_value_diagnostics.pilot`: prefreeze cost-only probe.
- `python -m research.alphazero_value_diagnostics.prepare`: prefreeze identities/inventories.
- `python -m research.alphazero_value_diagnostics.manage`: one clean pushed prospective run; no retry.
- `python -m research.alphazero_value_diagnostics.analyze`: full completed evidence only.

Never re-run prepare, pilot or measured tasks after launch. Inspect results/status.json
and task logs after an interruption. Retain all partial evidence. No retries or
expanded evaluation are authorized. Each task stores identities, exact visits,
raw action/match evidence and timing. Checkpoints remain in the original H-C
ignored output tree; the new package records their exact paths and hashes.
The detached manager cleans up only its own process groups.
