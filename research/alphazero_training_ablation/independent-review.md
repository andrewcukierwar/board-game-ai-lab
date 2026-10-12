# Independent prelaunch critical review — Phase 4D.4G

Review performed by a separate agent against the requested bounded experiment,
the Phase 4D.4F proposed protocol and boundary/import contracts, the preceding
development evaluation packages, and all new ablation Python sources and
prospective design. This is independent review on the same host, not external
laboratory replication. The reviewer performed no collection, training or launch.

## Findings and resolutions

1. **Position clustering must include shared openings.** Existing package family
   IDs identify a position and its reflection, rather than all positions sharing
   an opening. The frozen union components additionally link reflection-equivalent
   prefixes through ply eight and short-prefix ancestry. Independent recomputation
   finds 102 solved clusters, 71 tactical clusters and 28 additional deep clusters.
   Mirrors are never resampled separately. Unrecorded generator trajectory
   identities cannot be reconstructed; the protocol explicitly retains this
   limitation. The 128-row deep table has only 28 clusters and may provide poor
   noninferiority precision.
2. **Supervisor termination must clean up its owned task.** Workers create
   separate process sessions. Default supervisor SIGTERM would bypass Python
   cleanup and risk leaving a worker running. Supervisor SIGTERM/SIGINT now raise
   through the owned-child cleanup path; startup timeout termination waits and
   escalates. Cleanup signals only the process group created for that task.
3. **Atomic JSON replacement needs directory durability.** Status/task JSON
   initially fsynced the temporary file but omitted directory fsync after rename.
   The shared writer now fsyncs both contents and the parent directory.
4. **Calibration and one-ply evidence must reach final summaries.** Parent
   calibration, checkpoint calibration changes, avoidable-response errors and
   missed wins are now retained in analysis, alongside absolute rates/counts and
   paired changes. These diagnostics do not replace the declared primary gates.
5. **Interrupted progress counters are not exact final counts.** The existing
   runner checks cancellation before notifying its progress callback. A completed
   operation can therefore precede an interrupted notification. Failed-attempt
   records now identify progress as a lower bound and retain the per-attempt
   64-game/64-update upper bounds. They are not confused with published completed
   generations. No automatic retry is allowed.

All raised implementation findings are resolved in the reviewed sources.

## Independent checks and methodology assessment

The reviewer ran `tests/test_alphazero_ablation.py` with the existing M4 virtual
environment and all five thread variables set to one: **5 passed in 1.42 seconds**.
These tests exercise family ancestry/reflection, complete oracle coverage and
match budgets, paired bootstrap/margin direction, execution order/caps and the
owned-child watchdog. A separate disposable child ignored SIGTERM; the watchdog
escalated to SIGKILL and returned in **3.009 seconds under a four-second ceiling**.
Neither check queried a model or trained one.

A separate read-only fixture audit checked original source-file hashes,
recomputed every cluster mapping, replayed nonterminal legal histories, validated
complete legal-action WDL vectors and optimal sets, compared tactical labels to
the independent scans, and checked reflected action-value mappings. All passed.
The established solved300 and tactical200 original file hashes are respectively
`41ffd4dfcdd85b939c8afaea692566339e612224f16d2a62797c2960192f37b9`
and `106cc315c7e8d9b44a260df1a3407578ce1197411a6e614be3a7c38a5b9e6d9d`.
The additional 4E rows are selected model-blind from existing successful full-vector
queries; scalar or selected-action proofs cannot qualify. There is no new solving.

The implementation reuses strict historical fork import, persistent optimizer,
mixed-schedule replay, independent game RNGs, the CNN factory, pool-eight
collection and atomic boundaries without modifying their source contracts.
Every training task begins with a fresh strict load and verifies published full
model/optimizer/replay/counter/RNG state by strict reload. Evaluation uses frozen
independent models, domain-separated streams, matched arm-independent identities,
raw bitboard PUCT512 and existing color-correct paired arena logic. Thirty-two
base opening rows, each with both colors, produce 64 games per arm rather than
silently doubling the budget with mirror rows.

The bootstrap resamples entire frozen clusters with 10,000 seeded draws and
recomputes row-weighted means. Learning pools equal-weight seed contrasts before
shared-position resampling; strength pools independent within-seed opening
resamples. Deep mistakes include every exact achievable-value decrease, including
win-to-draw. One-sided noninferiority uses the .05/.95 quantiles, independently
of two-sided .025/.975 learning intervals. PASS/FAIL/INCONCLUSIVE rules and
practical thresholds are prospective. Failure of a learning improvement target
does not itself demonstrate harm. Two seeds and reused development fixtures
support conditional pilot conclusions, not broad unseen-position or seed-level
generalization. No automatic selection or promotion is implemented.

Readiness workload arithmetic matches 512 training games, 512 updates, 6,280
position-search roots and 256 strength games, with maximum 42-ply histories,
worst observed probe timings, twofold contingency and 600 seconds for I/O. This
is a feasibility projection, not a completion guarantee. Per-generation and
overall process deadlines enforce stopping if actual costs exceed expectations;
30 seconds are reserved for final preservation/accounting. Completed boundaries,
failed logs and partial evidence remain separate. Recovery never reuses poisoned
memory or silently retries a failed generation.

## Disposition

**PASS for prospective methodology and bounded orchestration implementation,
with no unresolved material source-review finding.** At the time of this review,
real-parent readiness, its measured feasibility decision, the final source/runtime
freeze and pushed prospective design commit remain required launch gates. They
must pass before primary collection. This review establishes no neural-learning,
tactical-reliability or playing-strength result. Final evidence and interpretation
require a separate post-execution review and session, as requested by the user.

## Final readiness audit

The reviewer independently strictly reloaded all four real generation-20 forks
against their declared file SHA-256 values. Model, ordered optimizer, replay,
counters, sampling/reflection/global RNG state, enforced execution source and
runtime matched readiness evidence and matched within each seed. Schedules and
the 64-game/256-simulation/pool-eight/64-update/batch128/two-generation limits
were checked explicitly. The matched initial-state hashes are:

- Seed 42: `2dd15035fd4e10654a80a78cf5dec16425d4ad6e23f21323adcc219ca8bee5e4`.
- Seed 314159: `5aa9916bc689710aa8ed4e6a5d268cb94d913cb8375e5d14542280587be29c77`.

The final 44-file supplemental source map and runtime enforce successfully;
the frozen readiness file hash agrees. Both readiness inventories and the current
inventory equal the prior 150-file official inventory. Independent recomputation
of workload arithmetic gives **9,081.0101 seconds projected**, below 10,800:
858.0951 collection, 24.7478 optimization and 3,357.6622 evaluation seconds,
then the declared factor two plus 600 seconds for I/O. Successful readiness took
**132.6558 seconds**, with **zero primary games and zero primary updates**.
The read-only four-fork audit took 37.1230 seconds and performed no optimizer work.

An initial reviewer comparison of the entire reporting dictionary failed solely
because the Git untracked-file count changed when readiness/freeze documents were
created. The enforced execution-content identity was unchanged. The successful
audit compares computational state and `source.execution`, with Git reporting
kept separately, as required by the existing 4F contract. No loader or validation
was relaxed. The initial preparation ImportError is retained in
`results/readiness-console.txt`; it occurred during imports before output/fork
creation. Its corrected `game_record` import and a new entrypoint import test
were reviewed. The final six ablation tests independently passed in 2.01 seconds.

The added compact game-history export uses the existing validated record
function after strict boundary reload, preserving moves, visits and temperatures.
The postexecution evidence exporter copies reasonable-sized JSON/log evidence
with a SHA manifest, leaves binaries ignored and retains local partial-evaluation
files separately. It performs no checkpoint query, training or solving. Both
exporters are included in the final frozen execution-source map.

**Final prelaunch disposition: PASS.** Measured readiness, strict matched forks,
source/runtime freeze and official preservation satisfy the reviewed launch
gates. A clean, pushed prospective design commit must still be recorded by the
supervisor before primary execution. This remains an execution-readiness finding;
all experimental outcomes await the separately requested final-analysis session.
