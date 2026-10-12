# Phase 4D.4G prospective protocol

Status: prospective, authorized bounded pilot; final results are a separate session.
Starting revision: 9325f0a19e6b0106ea1dcd265316b57323f8eb4f.
Historical strict reader: 1894bc4d8f035cf738df8e3fc0ce234f1ce2c8be.
The launch ledger records the final pushed prospective commit SHA before any
primary collection. No outcome-dependent changes, checkpoint promotion or expansion.

## Four matched forks

Seeds 42 and 314159, control and treatment, archived generation-20 boundaries.
`results/readiness.json` freezes trusted archived summary hashes and full strict
fork/reload state identities. Parent models, ordered AdamW state, replay, counters,
sampling/reflection/process RNG states and lineage identities match within seed.
Both arms are new experimental forks, never official continuations.

Only prospective action temperature differs: control tau1 at zero-based plies
0–7, tau0 afterward; treatment tau1 at 0–15, tau0 afterward. Existing 4F schedule
objects, factory, importer, pool-eight collection and generation barriers are
reused unchanged. Each arm runs generations 21 and 22, exactly 64 complete games,
256 simulations per move and 64 AdamW updates per generation, minibatch128.
The original three-layer CNN, LR3e-4, AdamW betas/eps, weight decay1e-4, clipping5,
noise epsilon.25/alpha1, PUCT formula, reflection.5, losses, eight-generation
replay expiration and per-game RNG protocol remain unchanged. Config is fully
recorded in each boundary. Totals are 512 primary games and 512 primary updates.
Different positions, lengths, inference batch tails and times are measured.

Order: 42/control21, 42/treatment21, 314159/treatment21, 314159/control21,
42/treatment22, 42/control22, 314159/control22, 314159/treatment22.
After each boundary: SHA validation, strict full reload and equality of model,
optimizer, replay, counters, sampling, augmentation and global RNG hashes;
then all development position evaluations. No optimizer updates during collection.
All completed game histories/visits/temperatures are stored in the boundary.

## Frozen evaluation

`fixtures.json` embeds solved300, tactical200, a full-vector 4E subset, 32 existing
development opening base rows, all identities/seeds and source hashes. Sealed
inputs are never evaluation data. This is reused development evidence, not a
claim of unseen generalization. No model participates in selection.

solved300 has complete exact legal-action WDL vectors (actor-relative), exact
state values and optimal sets. Labels are inherited from the established exact
solver with existing independent checks; legal coverage/max/set consistency and
mirror properties are revalidated. The additional 4E table is the first 64
canonical board families, sorted by family key, from previously committed exact
full-vector queries at ply>=16, with no immediate win and nonconstant action
values. Both orientations are frozen (128 rows). Selected-action/scalar proofs
never qualify. No new solving or adaptive coverage repair during execution.

Clusters are union components linking original mirror-position families and
shared reflection-equivalent move prefixes through ply8. Shorter tactical
prefixes link descendants by ancestry. All positions from such opening families
remain together; mirrored rows are never independent. Frozen generation histories
do not identify original generator trajectories beyond their prefixes, so this
clustering cannot recover unrecorded provenance and remains a limitation.

Parents are evaluated once per seed and explicitly reused for both identical
arms. Every generation21/22 checkpoint gets the same positions and seeds.
Raw neural policy and raw neural PUCT512, no noise, tau0 seeded ties, no guards
or proof propagation. Existing validated bitboard search backend and independent
arena procedure are used; neither mathematical search nor training changes.
One search seed per position, fixed 4404701, identities independent of checkpoint,
arm and training streams. NN argmax ties use the existing independent NN helper.
Total position search roots: 10 checkpoints x628 rows=6,280; each exactly512.
No repeated evaluation seeds or automatic broad tournament.

At generation22 each arm plays its own frozen parent on the same 32 nonempty
development opening families (first32 base rows in existing bank order), both
colors. 64 complete games per arm, 256 total; no mirror copies added. Evaluation
seed4404702, namespace includes training seed but omits arm/generation. Agent color
is explicitly recorded; opening turn is respected. Max42 total plies/game.
No root noise/tactical guard, raw PUCT512 for both players. Matches occur after
all training; evaluation never changes trajectory or continuation decisions
except integrity/deadline failure.

## Endpoints and exact decisions

Primary generation22 treatment-minus-control neural contrasts, equal seed weight:
optimal-set CE=-log(sum legal policy on exact-optimal actions), raw argmax
optimal-set accuracy, value MSE versus exact WDL. Also optimal-set probability,
all checkpoint/parent changes, training losses/gradient norms and fixed ten-bin
policy confidence versus optimal-set accuracy/value predictions versus exact
mean (calibration diagnostics only). Values always belong to actor to move.

Family bootstrap10,000 replicates, seed4404703, resamples entire clusters and
recomputes row-weighted means, retaining unequal cluster sizes. Position contrasts
are paired. For pooled learning, average two seeds' per-position contrasts before
shared family resampling. Two-sided95% percentile intervals use .025/.975;
one-sided95% bounds use .05/.95. All point results/intervals/per-seed comparisons
remain published. Conditional inference covers these fixed networks/fixtures,
not the distribution of arbitrary training seeds. No multiplicity correction;
requiring all gates reduces but does not remove exploratory interpretation.

Neural-learning PASS requires mean CE<=-.02 nats, each seed CE<=0, mean accuracy
improvement>=.02, each seed accuracy>=0, each seed MSE increase<=.02, plus pooled
CE upper two-sided95% bound<0 and accuracy lower bound>0. Preliminary observed
gain with either improvement CI overlapping zero is INCONCLUSIVE unless a FAIL
condition applies. FAIL means a seed CE CI wholly positive, accuracy CI wholly
negative, a seed point MSE regression>.02, pooled CE lower95% bound>-.02 or
pooled accuracy upper95% bound<.02 (practical improvement unsupported at this
precision). All other cases INCONCLUSIVE. FAIL of an improvement target is not
proof of harm; distinction explicitly reported. Threshold equality is inclusive.

Tactical raw point gate: each seed, generation21/22, method and category,
treatment immediate-win/unique-safe accuracy may drop at most .02 against its
parent AND matched control. Both arms' absolute results and parent changes are
published; FAIL if either treatment drop exceeds .02. Mandatory-block accuracy
is the unique-safe response category: this suite does not separately sample
multiple-safe alternatives. Avoidable one-ply mistakes are actions outside the
nonempty safe set, and missed wins are separately counted. Raw measurements and
point decision are distinct; family CIs remain visible.

Deep mistakes are action_value < exact_state_value, including win->draw and
win->loss/draw->loss, not merely moves that ultimately lose. On solved300 and
the separately frozen4E table, for each seed, generation and raw/PUCT method:
PASS if one-sided95% upper bound on treatment-control mistake-rate difference
<=+.05; FAIL if lower bound>+.05; otherwise INCONCLUSIVE. Every included row
has complete legal-action coverage. Coverage defects or insufficient precision
yield INCONCLUSIVE, not safe/erroneous unresolved labels. Tactical overall FAIL
if any raw/deep gate fails, INCONCLUSIVE if none fails but a deep gate is
inconclusive, PASS only if every gate passes. No margin changes after collection.

Strength: score=(wins+.5draws)/64, wins/draws/losses and color-specific scores.
Within seed pair treatment/control opening-color rows. Each opening's two colors
stay together. Strength NI PASS if each seed paired-family one-sided95% lower
bound>=-.05; FAIL if any seed upper bound<-.05; otherwise INCONCLUSIVE. Separate
supporting positive signal: treatment score>.5 both seeds and equal-seed pooled
two-sided95% lower bound>=.5. Pool independent family resampling within seeds
using bootstrap seeds4404703+training_seed. Positive support alone never promotes
a model or passes the learning/tactical gates.

Execution is separate: PASS only complete four arms/two generations,512 games,
512 updates, all evaluations and identity/preservation checks; FAIL for integrity
defect; incomplete/deadline execution INCONCLUSIVE. No composite score/champion.
Final report recommends a single subsequent phase based on evidence; no further
experiment, ResNet, replay change or tactical proof search is bundled here.

## Budgets, readiness and independent management

Readiness hard ceiling300 seconds, zero primary collection/updates. Strict parent
fork/reload, each seed three groups of eight simultaneous256 search roots,
12 raw512 probes including empty/early/mid states, four disposable CNN AdamW
updates per seed at128. Scratch/probe identities990–992/4404799 are outside
primary data. Parents/arms receive zero updates. Bounds use worst observed
seconds/move, maximum42 plies for512 training and256 match games,6,280 position
roots,512 update cost, factor2 runtime contingency plus600 seconds I/O. Launch
only if total projected<10,800 seconds. No workload reduction to fit.

Primary wall ceiling10,800 seconds including parent/new position and match
evaluations, strict reloads and resource logging. Each generation child has
1,200-second attempt ceiling including load/train/verify. Cooperative callbacks
and external supervisor-owned process-group watchdog; termination begins before
the hard deadline and escalates to SIGKILL only for that owned child. Sequential
tasks; no unrelated processes/services touched. Detached Python supervisor has
durable fsynced status/task logs, process identity, design commit and deadlines.

No automatic retry, no repeated failed attempts, no partial generation in dataset.
Attempt logs/progress remain, including complete-game/update counts before an
abort. Failed progress counts are lower bounds (cancellation can precede the
postoperation callback); upper bounds are64 games/64 updates for that attempt.
No retry ensures even unknown partial work cannot exceed the overall caps.
Atomic complete boundary is the only training completion marker. After
failure use only a freshly strict-loaded published boundary; preserve uncertain
postpublication boundaries for inspection, never poisoned memory or hidden tmp
archives. The manager refuses a second launch. Recovery that recollects data
requires auditing remaining actual budgets and a separately recorded decision;
this run does not authorize automatic recovery/expansion.

Strict original package source/runtime checks remain mandatory and unchanged.
Supplemental freeze additionally enforces every new orchestration/evaluation/
analysis Python source plus this design and fixtures at each task. Any execution
change requires a new documented fork, not weakened validation. Source freeze
map and runtime are in results/freeze.json; design commit is recorded by launch.
All150 official files equal the prior4F inventory before readiness, before launch
and at supervisor completion. Declaration remains c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716.

## Diagnostics, dilution and publication

Each complete boundary retains full histories, games, targets, model/optimizer,
per-game RNG identities, schedule metadata, trainer/replay/reflection RNG and
resource records. Sidecars add state digests, win/draw/loss, duplicates including
mirror/postply8, opening concentration, entropy, one-reply errors, retained replay
generations and exact immediate-proof diagnostics. Each update's loss/gradient
metrics, sampling counts by source generation and historical/experimental
exposure are retained. Approx65–69k parent examples dwarf two64-game additions;
quantify surviving replay share and actual sampled fraction rather than assuming
equal dilution. A null learning signal cannot establish diversity's irrelevance.

Large binary boundaries remain ignored under experiment-output/phase4d4g-primary;
tracked readiness/protocol/source/fixture evidence and later compact results are
published on the dedicated branch. No main merge, production/API/deployment/
canonical benchmark change. Final statistics, independent critical re-review,
report completion and results commit/push occur after the managed process ends,
in a separate user session. The current report must say results pending.
