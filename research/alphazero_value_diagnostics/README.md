# AlphaZero frozen value diagnostics

Prospective protocol: [DESIGN.md](DESIGN.md). Independent review:
[independent-clean-prelaunch-review.md](independent-clean-prelaunch-review.md).
Incident isolation: [integrity-resolution.md](integrity-resolution.md). The
original reviewer is retired; its review/code are retained as historical artifacts.
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

The approved launch entrypoint is now `python -m research.alphazero_value_diagnostics.report_lifecycle`,
which wraps the unchanged manager plus fixed analysis and fresh independent
reproduction under one absolute total budget. Supplemental source hashes are
in results/integrity-resolution.json. Live lifecycle progress is under the
ignored experiment-output/phase4d4i-a-lifecycle/ tree so the original manager
can enforce its clean pushed-HEAD gate. At completion those records are copied
to tracked results/lifecycle.json; primary progress remains results/status.json.
No lifecycle or task retry is allowed. Final scientific report/publication follows
only after completed evidence and independent reproduction are reviewed.

## Completed evidence

All48 primary tasks and the fixed clean independent reproduction completed.
The final report is docs/phase4d4i-a-alphazero-value-diagnostics.md. The source
protocol/manifest and raw primary/lifecycle records remain unchanged. Final
reporting adds report-supplement.json, final-integrity-audit.json and resource-final.json;
final-status.json records reporting/review completion separately. Independent
results review is independent-clean-results-review.md. evidence-manifest.json
records all published package artifacts and the report by SHA256.

The results show tactical value interference alongside useful match guidance;
zero/half outputs do not establish a universal strength improvement. The only
recommended follow-up is a controlled fixed-data CNN value-head decay probe;
it was not implemented or launched.

report_results.py and report.py aggregate completed evidence only and query no
model. To regenerate the final narrative after existing evidence is available:
`python -m research.alphazero_value_diagnostics.report_results`, followed by
`python -m research.alphazero_value_diagnostics.report`. Do not rerun measured
work, the pilot, preparation or lifecycle. matches-progress.json files are retained
historical progress snapshots; only complete matches.json files define final arenas.
