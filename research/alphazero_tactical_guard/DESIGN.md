# Phase 4D.4J-A prospective design

Original generation-6 CNN seeds42 and314159 from Phase4D.4H-C, read-only,
strict full checkpoint/config/model/source/runtime/inference validation.
No value-regularized checkpoint, training, replay update, optimizer update,
self-play collection, deployment, promotion or merge.

## Intervention and deterministic protocol

Run historical `EvaluationAgent` unchanged:256 bitboard PUCT simulations,
c=1.41, no root noise, exact terminal values, learned priors and learned values.
Root-only POST-selection classification uses the actual immutable root and
original physical-column visits. Enumerate legal successors and their opponent
replies. Immediate wins override all other choices. Otherwise override an unsafe
original only if at least one safe action exists. Choose greatest original visits
then LOWEST physical column for any override/immediate-win restriction. An
already-safe original is returned EXACTLY, including its historical seeded tie.
No safe action: return original, record NO_SAFE_MOVE, make no theoretical-loss
claim. Terminal roots rejected. Zero-visit chosen actions remain zero in evidence.
No neural inference, RNG draws, prior renormalization, visit rewriting or tree
modification occurs in tactical checks. Diagnostics are versioned in guard.py
and evaluate.py; all legal actions, wins, safe/unsafe sets, opponent witnesses,
original/final selections, visits, reasons and nanosecond-clock elapsed seconds
are retained. Full classification is recorded even when an immediate win exists.

This is conventional tactical filtering, already used by Tactical MCTS and F1
neural MCTS. Those agents filter before search; this experiment postprocesses
unchanged PUCT so learned search guidance and root visits remain available.
The existing immutable bitboard geometry/drop/winner methods are reused rather
than duplicating rules or adding expensive board/deepcopy conversions to guard.
The matrix reference in independent_verify.py is for independent validation only.

## Frozen endpoints and memberships

Nonsealed tactical200/solved300/deep128, inherited H-C/G SHA256
80f6851838b18172d6f818ebd306ce0cd36e4796cac1e3a71df29694feb3ca2c.
All legal-action oracle vectors retained. Theoretical classes: solved100/100/100
W/D/L; deep70/58/0. Tactical100 unique wins/100 unique safe responses.
No incomplete row receives invented oracle labels. Reused development performance
does not establish sealed/generalization performance. The prior I-A sealed-access
incident remains historical provenance; this phase never reads sealed content.
Only exact named development paths are parsed. Protected files may be hashed
without interpreting/printing their contents.

Both fixture variants repeat the SAME raw search with SAME per-row evaluation
stream4404901 and H-C namespace, asserting exact original-action/visit/value/
prior/logit identity.628 rows x2 variants x2 seeds =2512 fixture searches,
1256 paired records. All320 primary matches use reference Negamax3 and all40
established base opening families x both colors x two seeds x two variants.
Opening SHA2563990bc230084bd5c9293c15a44eae05f35749d2a3a18e66566461dae98f82f72.
Historical H-C match namespace/seed4404902; role-specific independent streams.
Pairs start on identical boards/streams; diverging play consumes later draws
differently. No secondary evaluation or ResNet: prospectively omitted to preserve
primary/review budget. Two sequential seed tasks; within task full position suites,
control matches then guarded matches. No outcome-dependent membership/order.

Tactical correctness is a targeted RULE CHECK, not independent strength evidence.
Report wins/safe responses, errors fixed/introduced, missed wins, activation,
changes, no safe alternatives, zero-visit selections. Exact endpoints include
optimal/WDL-preserving choices, class-specific mistakes/W→D/W→L/D→L, changes
and selected-action exact differences. Lost rows' automatically preserving choices
are not evidence of strong gameplay. Any exact-WDL regression blocks positive
publication and requires independent case investigation; no retrospective repair.
Principal outcome is actual match score, guarded minus control, win1/draw.5/loss0.
Retain all W/D/L, colors, families, both seeds, pooled equal seed weights, changes
and activations. Activation/outcome correlations are descriptive, not causal.

## Statistics and verdicts

10000 paired percentile cluster bootstrap replicates, PCG64 seed4410903.
Whole opening families keep both colors and BOTH fixed seeds together; pooled
means first average fixed seed row differences, then resample common families.
Fixture ancestry/reflection clusters retain all rows and row weighting; whole
clusters are sampled with replacement. Two trained networks are FIXED; intervals
do not estimate retraining variability or unseen-development generalization.
No multiplicity correction; match outcome is primary. No best-checkpoint selection.

Match PASS: pooled score increase>=5 percentage points, BOTH seeds nonnegative,
and pooled95% interval lower>0. FAIL: pooled decrease>=5pp and upper<0.
Otherwise INCONCLUSIVE. Correctness: any reproducible rule violation FAIL.
Deeper safety: no confirmed selected-action oracle-WDL regression. A passing local
suite alone cannot justify deployment or stronger-gameplay language.

Latency PASS requires guarded-call p95<=1ms, worst<=5ms, mean relative overhead
<=5%. Separately report original search, classification, whole guard, total call,
absolute/relative timing mean/median/p95/worst, candidate/reply counts, occupancy,
activation in games. Guard clock includes validations/classification/selection up
to final diagnostic assembly; total call includes assembly. CPU/runtime deterministic
one-thread Apple M4; measured workers sequential; unrelated research compute absent
at preflight (normal GUI/OS processes remain). No extra neural inference for guard.

## Resource and lifecycle

Cost-only pilot12 fixture roots and4 games, outcomes discarded; timings only.
Primary projected331.33s, conservative contingency962.66s incl300s overhead.
Successful initial validation14.18s; directed draw/sign enhancements validated
again before freeze (no measured retry). Independent prelaunch reviewer and
separate matrix verifier inspect all frozen implementation/design.
Target20–30 minutes is an approximate upper planning target, not padding; measured
work is expected to finish sooner. HARD total45 minutes from2026-10-13
22:53:39 America/New_York (2026-10-14 02:53:39UTC), includes preparation, validation,
analysis, review and final publication. The manager stops compute180s before that
absolute deadline to reserve final review/publication; owned child groups have
TERM then KILL cleanup. Deadline/CANCEL checked between searches, games and plies,
supervisor polls every.25s. Durable logs/progress; exclusive attempt/task paths;
no retries, extra experiments or automatic continuation.

Strict gate: clean requested branch, remote branch equals HEAD, prelaunch PASS,
all frozen source/runtime/checkpoint identities. Full checkpoint/model/learner
state preservation verified in each task; all historical preservation inventories
nonempty, including150 official files and actual canonical paths. Raw evidence
copied only after both tasks complete; analysis/verifier then run, no partial-run
positive interpretation. Prospective design/review/freeze committed and pushed
BEFORE measured work. Results require independent same-host saved-record review,
final preservation/regression checks, report and final branch push. Never merge.

To inspect: experiment-output/phase4d4j-a/status.json, seed logs and tasks/{seed}/
progress.json. To cancel: create experiment-output/phase4d4j-a/CANCEL. Starting a
second attempt is forbidden; failure evidence remains intact.
