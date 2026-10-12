# Phase 4D.4I-A — AlphaZero value-head and PUCT causal diagnostics

Status: prospective design frozen as c993d87; integrity isolation and fresh
independent prelaunch review prepared; no measured results yet.

This phase isolates learned priors, learned nonterminal values and their
interaction using four strictly loaded frozen generation-6 CNN/ResNet checkpoints.
No model is trained. See the [prospective design](../research/alphazero_value_diagnostics/DESIGN.md),
[focused checks](../research/alphazero_value_diagnostics/results/validation.json),
[cost pilot](../research/alphazero_value_diagnostics/results/pilot.json) and
[fresh independent prelaunch review](../research/alphazero_value_diagnostics/independent-clean-prelaunch-review.md).

Five fixed search variants share unchanged PUCT256: learned policy/value,
learned policy/zero value, uniform policy/learned value, uniform policy/zero
value and learned policy/half value. Planned work:15,072 searches on unchanged
nonsealed tactical200/solved300/deep128 and1,600 color-paired negamax3 games
on40 development opening families. G0 baseline networks are negative controls.
Value diagnostics use full g6/g5 replay, g5 predictions on new g6 games and
exact fixture labels. Realized self-play outcomes and optimal WDL are separate
targets. All statistical methods and identities are frozen prospectively.

The original reviewer accidentally accessed sealed fixture text. The
[metadata-only scope audit and isolation resolution](../research/alphazero_value_diagnostics/integrity-resolution.md)
retain the breach:149 visible source-line references in a truncated response,
three sealed files within the traversal scope, no protected content copied into
research artifacts. The exposed reviewer is retired from substantive work. A
fresh reviewer with no inherited history independently reconstructed the
development-only design and added new arithmetic/model reproduction scripts.
The original frozen protocol is unchanged. This isolates the study; it does
not erase the earlier access or its automatic transcript.

Final analysis, independent headline reproduction and preservation/resource
checks will be added after the detached run completes. No merge, deployment,
checkpoint promotion or further training is authorized by this phase.
