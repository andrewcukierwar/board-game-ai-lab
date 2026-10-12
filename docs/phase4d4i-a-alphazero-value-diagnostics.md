# Phase 4D.4I-A — AlphaZero value-head and PUCT causal diagnostics

Status: prospective design prepared; no measured results yet.

This phase isolates learned priors, learned nonterminal values and their
interaction using four strictly loaded frozen generation-6 CNN/ResNet checkpoints.
No model is trained. See the [prospective design](../research/alphazero_value_diagnostics/DESIGN.md),
[focused checks](../research/alphazero_value_diagnostics/results/validation.json),
[cost pilot](../research/alphazero_value_diagnostics/results/pilot.json) and
[independent prelaunch review](../research/alphazero_value_diagnostics/independent-prelaunch-review.md).

Five fixed search variants share unchanged PUCT256: learned policy/value,
learned policy/zero value, uniform policy/learned value, uniform policy/zero
value and learned policy/half value. Planned work:15,072 searches on unchanged
nonsealed tactical200/solved300/deep128 and1,600 color-paired negamax3 games
on40 development opening families. G0 baseline networks are negative controls.
Value diagnostics use full g6/g5 replay, g5 predictions on new g6 games and
exact fixture labels. Realized self-play outcomes and optimal WDL are separate
targets. All statistical methods and identities are frozen prospectively.

A reviewer source search accidentally printed partial sealed-fixture text during
orientation. This violates the requested no-sealed-access boundary and is
retained as an unresolved protocol limitation. No design, evaluation or
scientific inference uses that content. Frozen bytes remain unchanged.

Final analysis, independent headline reproduction and preservation/resource
checks will be added after the detached run completes. No merge, deployment,
checkpoint promotion or further training is authorized by this phase.
