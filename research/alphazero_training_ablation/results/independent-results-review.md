# Independent postexecution review — Phase 4D.4G

Reviewer: a separate reviewer agent that did not author the frozen implementation,
prospective design, or main results analysis. This review uses only frozen fixtures,
completed raw JSON records and execution sidecars. It launched no training,
model evaluation, solver query or tournament and modified no frozen source,
configuration, fixture or checkpoint.

The independently reconstructed outcomes are **neural learning FAIL**, **tactical
reliability FAIL**, and **playing strength INCONCLUSIVE**. These labels apply the
prospective rules; the neural improvement failure does not establish that extended
exploration generally harms learning. Execution integrity is separately audited by
the main agent, including strict binary checkpoint identities and preservation.

Machine-readable independent calculations: [independent-recomputed.json](independent-recomputed.json).

## Evidence and independent calculation

The reviewer reconstructed all 6,280 completed position records directly from
stored policy probabilities, values, chosen actions and frozen labels. All IDs,
row counts, completed markers, legal actions, policy normalization, argmax actions,
and exactly 512 visits per search matched. CE was recomputed as negative log of
optimal-set probability, accuracy as membership in that set, MSE against
actor-relative exact state WDL, and deeper mistakes as selected action value below
state value. No unavailable action labels were interpreted as correct. Every
solved300 and deep128 row has complete legal-action coverage, consistent maximum
and optimal set; mirror action vectors and state labels agree. This checks the
frozen labels' internal consistency, not an independent rerun of exact solving.

A separate plain Python Connect 4 board implementation independently validated
all 200 immediate-win/safe-response fixture labels and classifications. It
reconstructed every strength game's terminal result and verified all 256 complete
games, 32 openings per arm, both colors per opening, and 512 visits per decision.
The opening turn is respected by the frozen arena: side swapping assigns agent
ownership while play uses the existing board's current player.

Clustering was independently reproduced as connected components of a graph joining
mirror families or reflection-equivalent prefix ancestry through ply 8. Frozen
memberships matched exactly: solved300 has 102 clusters, tactical200 has 71,
and deep128 has 28. Cluster sizes differ and some related opening descendants
remain in large components. Category-filtered tactical summaries retain each
category's contributing rows under the original cluster labels.

The independent bootstrap used a multinomial multiplicity matrix applied to
cluster sums and row counts, rather than importing the frozen bootstrap routine.
It retained unequal cluster sizes and paired contrasts, 10,000 replicates and RNG
4404703. Shared solved positions were paired by ID across seeds; pooled learning
averaged each position's two seed contrasts before the common family bootstrap.
One-sided upper bounds were used for deep error-rate increases and lower bounds
for score noninferiority. Threshold inclusivity, strict positivity requirements
for improvement intervals and the full FAIL precedence were checked against DESIGN.md.

## Headline independent results

Generation 22 treatment minus control, equal seed weight:

| Metric | Difference | Two-sided 95% family interval |
|---|---:|---:|
| Optimal-set CE | +0.0014204 nats | [-0.0066562, +0.0107742] |
| Raw optimal-action accuracy | -0.8333 percentage points | [-2.8718, +1.0246] pp |
| Value MSE | +0.0135128 | [-0.0125841, +0.0421848] |
| Optimal-action probability | -0.0009898 | [-0.0032614, +0.0010226] |

The pooled CE lower interval bound exceeds -0.02 nats and pooled accuracy upper
bound is below +2pp. Seed 314159's point value-MSE regression is +0.041322, exceeding
+0.02. Each independently causes failure under the frozen neural rules.
Seed 42 CE is +0.0019781, accuracy -1.6667pp and MSE -0.0142963; seed 314159 CE is
+0.0008627, accuracy 0pp and MSE +0.0413220. CE/accuracy intervals include zero;
these are failed improvement targets, not decisive evidence of policy harm.

The tactical point-margin failures are:

| Seed / generation / method / category | Comparison | Difference |
|---|---|---:|
| 42 / 22 / policy / unique safe | Treatment minus control | -3pp |
| 314159 / 21 / policy / unique safe | Treatment minus parent | -6pp |
| 314159 / 21 / PUCT / immediate win | Treatment minus parent | -3pp |
| 314159 / 22 / policy / unique safe | Treatment minus control | -3pp |

The 2pp point gate is prospective. Its failure should be distinguished from a
confidence-bound claim of generalized tactical harm. The machine-readable review
retains all per-seed, generation, method, category and deep margin calculations.
The unique-safe category is the suite's mandatory-block measurement; it does not
provide a separate multiple-safe alternative stratum.

On deep128, seed 42 raw-policy mistake-rate noninferiority is INCONCLUSIVE at
generations 21 and 22: differences are +3.125pp and +1.5625pp, with one-sided
upper bounds +5.8333pp and +5.5556pp, respectively. Both exceed the +5pp
noninferiority bound. Their lower bounds do not exceed the margin, so neither
is a deep-margin FAIL. All other declared deep gates PASS. The overall tactical
FAIL is driven by the four point-margin failures listed above.

Strength scores against each arm's own identical seed-specific parent:

| Seed | Control W/D/L, score | Treatment W/D/L, score | T−C | One-sided 95% lower | NI |
|---|---|---|---:|---:|---|
| 42 | 30/7/27, 52.34375% | 31/7/26, 53.90625% | +1.5625pp | -7.03125pp | INCONCLUSIVE |
| 314159 | 28/5/31, 47.65625% | 31/6/27, 53.125% | +5.46875pp | -3.90625pp | PASS |

Each opening's two colors remain together in score resampling. The aggregate
strength result is INCONCLUSIVE because seed 42 does not demonstrate the -5pp
noninferiority margin; its upper bound also does not establish regression. Small
observed score advantages do not establish superiority, and these matches are
separate parent comparisons, not direct treatment-versus-control games.

## Replay, budgets and scientific interpretation

Eight completed generation sidecars agree with 512 stored complete game histories,
512 update metric records, exactly 64 games and 64 updates per boundary, 256 visits
per self-play decision, and the expected eight/sixteen-ply schedules. Sampling
source-generation totals are exactly 8,192 appearances per generation. New replay
counts equal the sum of retained generations 21/22 and expiration is 13 at generation 21,
14 at generation 22. All stored loss and gradient scalars are finite; combined losses
agree with policy plus value components. New-data appearance fractions are only
2.42–3.94% at generation 21 and 6.21–7.80% at generation 22. Historical replay dominates.
This makes replay dilution and a small update budget plausible explanations for
weak learning, but the experiment does not causally separate either mechanism.

The manager records 22 completed tasks: 8 training, 10 position evaluations and 4
strength tasks. The measured wall time is 2,016.158602s; child wall times sum to
1,986.274245s, leaving 29.884357s for supervisor/process/I/O overhead. Both the
10,800s total ceiling and 1,200s generation ceilings were respected. Actual search
work is 4,197,632 self-play simulations, 3,215,360 position-evaluation simulations,
and 3,724,288 strength simulations, totaling 11,137,280. Training generated 16,397
positions; strength searched 7,274 decisions. No extra workload is warranted by
unused time budget.

Collection timing is recorded directly. There is no isolated optimizer-only timer:
first before_update to before_checkpoint is an approximately 3.40–3.66s stage that
includes training and summary/RNG bookkeeping; training-and-persistence timings
must not be described as pure optimization time. Per-process peak RSS is not
concurrent total machine memory. Collection batch counters measure inference
throughput; no full evaluation neural-query throughput counter was recorded.

## Critical findings and limits

No material arithmetic, grouping, missing-label or gate-direction defect was found
in the frozen analysis. Two source-level limitations require care in publication:

1. `analyze.py` derives its execution label from manager status and official
   preservation alone. It is not a replacement for the separate full task,
   checkpoint-lineage/source/runtime and historical-preservation audit.
2. Pooled learning uses positional zip in the frozen implementation. Here all
   completed files preserve the exact frozen row order, and ID-based independent
   pairing confirms the same result. Future tooling should verify pooled IDs before
   relying on order; no source change is necessary for these records.

All evaluations reuse development fixtures and opening banks, including the 4E
subset. Neither calibration nor loss differences demonstrate unseen generalization.
Training policy loss against visit distributions differs from exact optimal-set CE;
their numerical subtraction is not a meaningful policy generalization gap. Value
MSE gaps across sampled replay outcomes and exact development WDL are diagnostics
with different data/label distributions, not a held-out test. Two seeds provide
very limited between-run inference. Bootstrap uncertainty is conditional on these
fixed networks and fixture families; it does not infer arbitrary future training
seeds. Shared unrecorded historical generator trajectories cannot be recovered
beyond the frozen prefix ancestry rule. No multiplicity correction was specified.
Full per-seed endpoints and failed gates must remain visible to avoid selective
reporting, and no treatment adoption or checkpoint promotion follows from diversity.

Recommendation: retain eight exploratory plies as the working schedule and proceed
with the already planned controlled Phase 4D.4H-B CNN/ResNet inference-cost,
throughput and fixed-data learning comparison using read-only historical replay
and a leakage-aware held-out split. The current evidence does not justify adopting
sixteen plies or automatically adding exploration training.

## Auxiliary integrity-audit review

The reviewer examined the new postexecution `audit.py`, which is outside the
frozen execution-source closure, and the final [integrity-audit.json](integrity-audit.json).
It reports PASS for 12 strict generation20/21/22 boundaries, the full lineage/state,
targets/labels/RNG/configuration/replay checks, 512 games, 512 updates and 256
matches, and all 150 official file hashes. It also documents evaluation identities
as reconstructed from frozen manager transitions, rather than independently
signed per-row attestations, and atomic-save evidence as frozen source plus
successful strict completion, rather than an independent syscall trace.

Two auxiliary validation failures and one auxiliary metadata error were resolved without changing experiment
sources, evidence, training or decisions. First, JSON sidecars stringify integer
histogram keys and convert tuples to arrays; canonical JSON normalization makes
comparison against binary history structurally correct. Second, an unrelated
active ResNet H-B worktree acquired untracked review/report files during the
read-only audit. The corrected protection gate compares tracked bytes and HEAD;
full before/current inventories and untracked-status drift remain reported. This
is a reporting change from external work, not experimental recovery or retry.
The historical and ResNet protected tracked evidence and HEAD identities passed.
Third, calling `runner.boundary_payload()` during reporting recomputes current
Git reporting metadata, so its reconstructed integrity digest is not the original
archived integrity field. The audit now records the archived `integrity` directly
using `torch.load(..., weights_only=True)` after strict SHA/loader validation.
Original-versus-reconstructed digests are preserved in
[archive-integrity-reporting-correction.json](archive-integrity-reporting-correction.json).
This corrects reported archive metadata; checkpoint bytes and strict integrity
validation remain unchanged. The reviewer inspected the frozen payload/loader
implementation and found all three corrections appropriate, with no scientific
alteration.

Frozen main-analysis crosscheck PASS: 1,316 numeric scalars matched the independent
calculations within 5.56e-16 absolute difference. This includes every learning
parent/checkpoint/contrast interval, all tactical labels/contrasts and category
absolute metrics, strength scores/color metrics/noninferiority and supporting
signal, and all three scientific decisions. The independent treatment parent-score
support is 53.515625%, pooled interval [47.265625%, 59.765625%]; its lower bound
fails the 50% supporting-signal criterion.

Final report disposition **PASS** after minor wording/readability corrections.
All learning/tactical/deep/strength/diversity/replay/resource tables, decisions,
uncertainty interpretation, evidence links, historical preservation statements
and the single next-phase recommendation were reviewed against the raw or audited
machine records. All direct report links resolve. Two minor wording findings
were resolved: intervals concern fixed checkpoints from two seeds, and reductions
in parent-relative value MSE are described as point estimates when intervals
include zero. There is no unresolved material scientific or arithmetic finding.

Calibration was separately recomputed with vectorized bin assignments for all
10 checkpoints; bin counts, policy ECE and value bin error agree. The reviewer
inspected the observed-replay-overlap implementation and independently reconstructed
all 12 experimental-generation overlap tables from exported generation21/22 game
histories using its separate board implementation. Identities and producing
generations agree. Historical overlap counts were checked through the strict
six-boundary source/data audit rather than duplicated binary loads. The report
correctly treats this as descriptive retained-replay overlap, not an unseen
split or a prospective generalization endpoint. The additional behavior/resource
summaries preserve the frozen bootstrap/endpoints and make no causal claim.
