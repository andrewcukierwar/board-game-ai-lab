# Phase 4D.4E — Confirmatory exploration safety and exact values

**INCONCLUSIVE on safety; diversity is independently confirmed.** Across 256
matched identities (512 complete games), original/extended have 390/358 confirmed
value-changing actions at plies>=8, or 1.523/1.398 per game. These are lower bounds,
not fully measured error rates. The confirmed-only contrast is -0.125 with 95%
interval [-0.417, +0.167]. Exact adjudication is 89.4% overall and 66.2% at plies 8–15;
conservative missingness-aware 95% interval [-3.160, +2.548] cannot exclude the
predeclared +0.05 margin. Extended sampling sometimes downgrades perfect-play
value, while the overall game-level effect remains unknown.

A **small controlled training ablation is worthwhile as a guarded development
experiment**, after the runner prerequisites below. This is a risk-bearing
learning experiment, not permission to adopt 16 plies as a safe default. There
is no neural learning experiment or playing-strength improvement in this phase.

## Isolation and prospective commitment

The dedicated `phase4d4e-alphazero-exploration-confirmation` worktree starts at
`08656faa1efb9d76fb9ae546f894164011e0b453`; the historical checkout remains at
`1894bc4d8f035cf738df8e3fc0ce234f1ce2c8be`. No source under `games/` changes.
The official declaration remains SHA-256
`c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716`.
All new artifacts live in `research/alphazero_exploration_confirmation/`.

[DESIGN.md](../research/alphazero_exploration_confirmation/DESIGN.md) was committed
and pushed in `89c8822` before any fresh confirmatory game; design SHA-256
`e90aecd6f0deed6cec2c2034d2be12c9a1a3dd51f18c8bdc1472f05e37b32713`.
Analysis implementation was committed in `c549021` before main collection.
Reviewer-led robustness fixes handle incomplete matching, enforce declared
runtime/sample ceilings, and report exact scalar-value coverage. They do not
change the prospective margin or endpoint. Feasibility used only prior 4D games.

Primary comparison: original 1.0 at plies 0–7 versus extended 1.0 at plies 0–15,
then greedy 0.0; 128 games/checkpoint/schedule, two 64-game independently seeded
cohorts each. Frozen generation 20 checkpoints seeds 42/314159, unchanged pool 8,
256 PUCT simulations, root noise, weights and normalized raw-visit targets.
Fresh independent-game generation identity 22; base seeds 4404511/4404512 and
4404521/4404522; matched indices 0–63 across schedules. Existing M4 virtualenv,
CPU float32, deterministic algorithms, one intra/inter-op and vector thread.

## Method and prospective interpretation

The primary outcome is the count per complete game of executed actions at
plies>=8 that downgrade exact achievable W/D/L for their actor. Every eligible
action is considered. Forced win->draw, win->loss, and draw->loss count once;
already-lost actions and alternative forced wins do not. Realized self-play
outcomes are separate. A terminal winning successor has value -1 for its new
mover; draw 0. No exact proof/tactical behavior is inserted into neural PUCT.

The broader Victor solver/native facilities were inspected: they distinguish
exact endgame outcomes from established bounds and heuristic playing policy.
Those outputs are not substituted for exact W/D/L here. The native oracle is a
byte-identical copy of the project's validated C source
at pinned reference 8704142. It computes complete exact legal-action vectors;
200,000 nodes/0.3 seconds per root first pass, 2,000,000/0.5 seconds second pass,
600 second launch ceiling. Descending ply and canonical-board order, reflection,
shared TT and proven successor values avoid repeated work. Unknowns have explicit
reasons; no partial vector is accepted. Scalar immediate-win and terminal proofs
are exact, and an exactly lost state proves every legal action neutral.

Per-game error lower bound is confirmed errors; upper adds every unresolved
action. Paired effect lower subtracts original upper from extended lower; upper
subtracts original lower from extended upper. Checkpoint-stratified paired
95% normal/jackknife intervals expand these bounds for RNG sampling uncertainty.
Confirmed-only contrasts are descriptive and coverage-dependent. No denominator
is restricted to easy positions. The declared +0.05 errors/game margin is a
permissive practical screen for a small learning experiment, not universal safety.
It is roughly 116% of the previous incomplete post hoc 0.043 baseline. With 256 pairs,
the prior paired half-width 0.044 already warned of limited noninferiority power.

MATERIAL HARM requires pooled conservative lower bound>0.05; NO MATERIAL HARM
WITHIN THE DECLARED MARGIN requires pooled and both checkpoint upper bounds<0.05,
99% exact adjudication in every checkpoint/schedule/stage, complete matching and
correctness. Otherwise INCONCLUSIVE. These thresholds remain frozen.

## Experimental correctness and the retained diagnostic deviation

All 512 complete primary games were replayed legally and their winners and
actor-relative targets checked. All 15,964 searched moves have 256 root visits;
all targets are normalized raw visits and execution distributions equal the
requested temperature policy. Every game and cell record hash reconstructs.
The 8 checkpoint file hashes and every loaded model's before/after weights match.
The historical checkout is clean; production/game code and prior research are
unchanged. All 150 official files remain byte-identical, verified before/after
collection/analysis and independently by the reviewer. No training, resume,
merge, deployment or service operation occurred.

The existing 4D regression driver collected 64 validation games in a fresh directory:
8 checks pass, including original/default equivalence with retained 4C and
repeat records/RNGs. These games are excluded from primary evidence.
All 256 matched identities have identical before-ply-8 states, actions, visits,
temperatures, priors, distributions, depths and terminal-leaf diagnostics.

**Literal prospective full-record equality did not pass.** Three before-ply-8 neural
root-value diagnostics differ by at most 5.960464477539063e-8, for final game ID 63:
314159/cohort 0/ply 3, 42/cohort 1/ply 3, 314159/cohort 1/ply 1. The analyzer initially
stopped on exact equality, preserving the proofs and failure record. Independent
review confirmed zero other stored decision-field mismatches. Batch-tail float32
rounding is a supported explanation, not a proven hardware cause; the drift is
below the previously declared 4C value tolerance 2e-6, and root values are uncounted
diagnostics. The user's identical-decision criterion passes; the design's stricter
literal full-record criterion remains marked false. Reanalysis reports both,
and that literal failure prevents a NO MATERIAL HARM label. No primary endpoint,
margin, RNG identity or sample was changed; the safety result already fails on
coverage/precision. No game or solver query was repeated to address this failure.

Focused contracts and the relevant inference/analysis suite: **245 passed**,
including 38 new exact-confirmation tests; no optimizer steps ran. Independent
native validation checked 8 late old histories and their mirrors against both
Python BitboardSolver and exhaustive minimax, with no discrepancies. A second
reference check on 12 new, non-one-reply confirmed errors made 24 bounded Python
state queries: 11 resolved and matched, 13 explicitly unknown, 0 discrepancies.
The independent evidence-only script rebuilt the complete scalar cache from
native proofs/immediate wins/successors and reproduced all classifications,
attribution, sampling comparisons, coverage, intervals and headline diversity.
See [REVIEW.md](../research/alphazero_exploration_confirmation/results/REVIEW.md).

## Exact-value comparison

Counts cover every eligible decision and keep every completed-game denominator.
Each arm has 256 games. An unresolved action contributes 0..1 additional error in
the conservative bounds. Confirmed-count contrast is neither a lower bound on
the true difference nor evidence of equivalence.

| Arm | Win→draw | Win→loss | Draw→loss | Confirmed total | Confirmed/game | Unresolved actions | Full-rate identification bounds/game |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 141 | 130 | 119 | 390 | 1.523 | 678 | [1.523, 4.172] |
| extended | 115 | 118 | 125 | 358 | 1.398 | 587 | [1.398, 3.691] |

| Checkpoint | Original confirmed/game | Extended confirmed/game | Confirmed paired difference [95%] | Identification difference bounds | Conservative paired 95% interval |
|---|---:|---:|---|---|---|
| 42/g20 | 1.344 | 1.102 | -0.242 [-0.654, +0.170] | [-2.047, +1.328] | [-2.555, +1.812] |
| 314159/g20 | 1.703 | 1.695 | -0.008 [-0.422, +0.406] | [-3.500, +3.008] | [-4.084, +3.594] |
| Pooled, equal checkpoint weights | 1.523 | 1.398 | -0.125 [-0.417, +0.167] | [-2.773, +2.168] | [-3.160, +2.548] |

The achieved confirmed-only half-width is 0.292 errors/game, almost 6 times the
0.05 margin. If paired variance were similar under complete adjudication, this sample
would still have a half-width much too large for the 0.05 margin. The primary uncertainty bound
is much wider because unknowns concentrate in the changed exploration window.
The intervals are approximate paired game-level normal/jackknife intervals;
positions are never treated as independent. They describe RNG variability under
these 2 checkpoints and do not establish safety across networks or generations.

The much higher absolute counts than 4D's post hoc 0.043/0.059 are expected from
examining all eligible actions: many deeper result downgrades are invisible to
one-reply flags. These estimates have different scopes and cannot be pooled.
They expose a substantial existing deep-error burden in both frozen networks.
The observed lower extended count does not imply safer play: missingness,
shorter trajectories and changing downstream states prevent that inference.

Confirmed errors by stage (rates use all 256 games/arm):

| Stage | Original confirmed count/game | Extended confirmed count/game |
|---|---:|---:|
| 8-15 | 167 / 0.652 | 186 / 0.727 |
| 16-23 | 176 / 0.688 | 136 / 0.531 |
| 24+ | 47 / 0.184 | 36 / 0.141 |

These stage contrasts are descriptive and coverage-dependent; plies 0–7 were
not exactly analyzed. Full per-ply counts are in results.json/coverage.

## Solver coverage and unresolved evidence

The runtime-only feasibility pilot used 24 old 4D positions at plies 8/12/16/24/32/40:
native 17/24 exact at 200,000 nodes, Python 18/24 at 20,000 nodes; 0.906 seconds, excluded
from all confirmatory statistics. Those full native vectors are harder than
scalar state values, so the main analysis also reuses exact successor proofs.

Main analysis examined 7,705 unique canonical eligible states. It made 7,124 first
pass queries (6,001 exact/1,123 node aborts) and 1,123 second pass queries
(718 exact/405 node aborts). Total 8,247 queries, 63.492 seconds; no wall timeout,
no total-budget stop and no extra retry. Every successful vector establishes
all legal action values; aborted vectors contribute none. The scalar cache has
30,648 proven states, including successors; 7,351/7,705 eligible unique states
have exact cached scalar values. Individual action adjudication still requires
both value perspectives (or a valid exact neutral deduction).

| Checkpoint/arm | Eligible decisions | Exactly adjudicated | Unresolved | Action coverage | Coverage 8–15 /16–23 /24+ |
|---|---:|---:|---:|---:|---|
| 42/original | 2914 | 2683 | 231 | 92.1% | 74.7% / 100.0% / 100.0% |
| 42/extended | 2550 | 2349 | 201 | 92.1% | 77.4% / 100.0% / 100.0% |
| 314159/original | 3398 | 2951 | 447 | 86.8% | 54.5% / 99.9% / 100.0% |
| 314159/extended | 3022 | 2636 | 386 | 87.2% | 59.7% / 100.0% / 100.0% |
| All | 11884 | 10619 | 1265 | 89.4% | 66.2% / 99.97% / 100% |

Only 1 unknown occurs after ply 15; 1,264 are at plies 8–15. All 1,265 are missing
pre-state/successor values after declared node limits, explicitly listed in
`unresolved.jsonl.gz`; none is counted safe or as a confirmed error. Exact
pre-state values are available for 10,626 eligible observations and exact
executed-action values for 10,867; the action classification is resolved for
10,619. Counts differ because either perspective alone can remain insufficient.
Coverage by each ply, checkpoint, schedule, stage and max/non-max/tactical category
is stored; sampled non-max coverage is 83.8% at 42 and 62.1% at 314159. Mandatory-block
coverage is 98.9–99.7%, immediate-win 100%, and other states 85.3–90.7%.
This unevenness makes a solved-only rate particularly misleading. The predeclared
99% coverage requirement is not met where exploration differs most.

## Same-search sampling regret

Extended has 1,850 eligible decisions at plies 8–15. Sampling chooses a non-max
visit move 508 times (27.5%). Exact comparisons use every tied max-visit action:

| Executed value versus all maximum-visit alternatives | All 1850 decisions | Among 508 non-max selections |
|---|---:|---:|
| Worse than every alternative | 69 | 69 |
| Better than every alternative | 23 | 23 |
| Equal to every alternative | 1249 | 286 |
| Mixed exact tie values | 0 | 0 |
| Unresolved comparison | 509 | 130 |

Four eligible searches have multiple max-visit columns; their ties were kept
explicitly. Over 1,341 resolved comparisons, mean uniform-max-visit regret is
0.0522 W/D/L units (not a game outcome or trajectory effect). Among non-max
selections, 69/508 (13.6%) are confirmed worse and 23/508 (4.5%) confirmed better;
130/508 (25.6%) remain unknown. These denominators retain unresolved cases.

Of 358 confirmed extended game-theoretic errors, 57 are strictly
sampling-attributable: all max-visit alternatives preserve the optimal state
value; 299 occur where every max-visit alternative also downgrades it; 1 has mixed
tie attribution and 1 unresolved attribution. Original has 390 search-attributable
confirmed errors at plies>=8. The 69 counterfactual worse actions include further
worsening when the greedy preference would already err, explaining why 69 exceeds
57. Immediate regret is a real exploration cost, and does not reconstruct a
complete alternative greedy game or establish the total learning effect.

## Independent diversity confirmation and secondary behavior

| Checkpoint/arm | Duplicates>=8 | Mirror duplicates>=8 | Collision probability>=8 | Unique trajectories/128 | Mean length | X wins/O wins/draws |
|---|---:|---:|---:|---:|---:|---|
| 42/original | 34.0% | 35.4% | 0.0210 | 104 | 30.70 | 68/24/36 |
| 42/extended | 10.7% | 12.4% | 0.0058 | 125 | 27.86 | 74/35/19 |
| 314159/original | 37.4% | 39.9% | 0.0413 | 109 | 34.55 | 66/21/41 |
| 314159/extended | 12.5% | 15.1% | 0.0195 | 128 | 31.61 | 72/27/29 |

Paired duplicate reductions: seed 42 -23.2 points [95%: -31.0, -15.4], seed 314159
-24.9 points [-32.1, -17.6]. Mirror duplicates and collision probabilities also
fall in both checkpoints, and every checkpoint/cohort duplicate fraction falls.
This meets the prospective diversity replication rule. The pooled stratified
paired duplicate reduction is -24.0 points [-29.4, -18.7]. Prefix 12/16/20/24 counts,
length/outcome distributions, policy entropy and unique tactical states are all
retained in results.json. Complete trajectories rise 104→125 and 109→128.

| Checkpoint/arm | Unique 12/16/20/24 prefixes | One-reply errors | Missed immediate wins | Missed mandatory blocks | Target entropy>=8 (nats) | Unique tactical states |
|---|---|---:|---:|---:|---:|---:|
| 42/original | 76/74/69/66 | 63 | 20 | 28 | 0.683 | 747 |
| 42/extended | 102/92/83/72 | 78 | 25 | 41 | 0.734 | 952 |
| 314159/original | 70/77/73/77 | 36 | 7 | 16 | 0.741 | 805 |
| 314159/extended | 90/106/97/92 | 55 | 12 | 26 | 0.744 | 920 |

Pooled one-reply errors rise 99→133, 0.387→0.520/game. First-player outcomes are
X-win/draw/O-win; second-player W/D/L reverse the wins, giving original pooled
134/77/45 (first) and 45/77/134 (second), extended 146/48/62 and 62/48/146.
Draw share falls 30.1%→18.8%; paired intervals by checkpoint are provided.
The networks play both sides, so this is an outcome-label shift, not a head-to-head
network win rate. Shorter games are reported descriptively; no inherent quality
judgment follows. More diverse positions are evidence about collected data only.

## Decision and learning evidence

The fixed three-way safety decision is **INCONCLUSIVE**, for both solver coverage
and attainable uncertainty. MATERIAL HARM is not supported; NO MATERIAL HARM
WITHIN THE DECLARED MARGIN is not established. The observed confirmed-only
-0.125 contrast must not be read as improvement or safety. The literal root-value
record-equality deviation is disclosed separately and cannot rescue the decision.

There is **no neural learning evidence**. The practical next step is a bounded
original-versus-extended training pilot with an original control and explicit
stop/acceptance guards, after implementing the experimental runner. Reasons to
try it: large independent diversity replication and no observed confirmed-count
increase across the full game. Reasons for caution: a large deep-error burden in
both networks, 57 strict sampling-caused downgrades, 69 worse versus 23 better
counterfactuals, severe early coverage gaps and a shifted value-label distribution.
The pilot asks whether improved learning offsets this risk; it is not certified
safe by this study. Keep the original default pending actual learning evidence.
Do not automatically schedule another frozen-checkpoint confirmation experiment.

## Runtime, evidence and reproduction

Main 512 games took 469.649 seconds; regression 64 games 60.146; feasibility 0.906.
Native main solving 63.492 seconds is inside the retained failed-summary invocation,
whose filesystem-derived duration is 68.753 seconds (a monotonic total was not
saved before its exception). Conservatively account 90 seconds for that attempt.
Successful proof-only analysis takes 7.023 seconds. Independent reference solving
adds 1.537 seconds; prior independent solver validation 0.00956 seconds. Final proof-only reproduction takes 7.011 seconds and exactly preserves every
initial result key; it adds the declared pooled secondary intervals. Review
timings and aggregate accounting are in the run registry. Conservatively accounted aggregate experimental execution is 656.6 seconds
(10.94 minutes), including the retained failure and both independent
recomputations, below the 30-minute maximum. Including ordinary test runtimes
still stays below 12 minutes. No cost-filling extra work was launched.
Ordinary tests and documentation are reported separately. All stopped/failed work
and its proofs remain on disk; no observation was discarded.

The delivered evidence includes prospectively frozen DESIGN, exact feasibility,
all 512 fresh complete records, all 64 regression records, source/checkpoint hashes,
8,247 native query records and canonical scalar proofs, all 11,884 decision rows,
1,265 explicit unknown rows, full secondary profiles/intervals, preservation
inventories, independent validation/recomputation, tests and final artifact hashes.
[README.md](../research/alphazero_exploration_confirmation/README.md) gives fresh-
directory commands for proof-based reproduction; it does not recollect games.
Reanalysis reproduces result numbers from committed bytes. New bounded solving
can have timing-dependent coverage, so stored-proof reproduction is the exact
reproducibility claim. Independent review is another agent on this host, not an
external laboratory replication.

## Next training phase — design only

A useful first bound is two new generations in each arm, on both development
checkpoints, targeting roughly 64 complete games per generation with matched
search/optimization caps declared prospectively. Commit its own learning and
tactical decision rules before launching; end at a saved generation boundary.
A short ablation must warm-start paired arms from identical development boundary
states, including model, AdamW state and replay. Inference-only checkpoints are
insufficient: import the corresponding development generation boundary read-only. Use pool-eight collection,
separate new development identities and a small fixed number of generations,
with matched collection-compute and optimization-update budgets. Equal game
counts alone do not match computation when lengths differ. Prospectively choose
updates independent of outcomes and record simulations/positions/time separately.
Evaluate network-only learning and full-PUCT strength on reproducible development
opening families, paired colors, and treat tactical accuracy as a separate guard.
No official sealed evaluation should be repurposed for development.

The prerequisite engineering task is an explicitly experimental generation
runner plus per-generation execution-schedule storage and validation. Currently
`generation.rebuild_game` at line 105 and replay reconstruction in
`load_resume_boundary` validate every stored game's temperatures against one
global `config.exploratory_plies`. Changing 8 to 16 cannot correctly reconstruct a
mixed historical/new replay window. Store each generation's immutable schedule,
use it to rebuild that generation's games, validate schedule provenance on load,
retain source/runtime/RNG/config checks, and support a read-only warm-start import
into a new declared experiment rather than resuming the official campaign.
Wire only this new runner to `collect_batched`; keep optimizer/replay unchanged
at initialization and snapshot experiment RNGs, schedule map and active collector
protocol at complete generation boundaries. Boundary-resume equality and mixed-
schedule replay rejection tests are prerequisites. None of this is implemented
in this phase; no neural learning evidence is produced here.
