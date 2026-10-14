# Independent completed-results review — Phase 4D.4J-A

Disposition: **PASS for complete execution, arithmetic, preservation and the
qualified report.** The scientific playing-strength verdict remains
**INCONCLUSIVE** under the frozen practical-improvement rule. This review does
not authorize integration or imply that the guard met that rule.

Reviewer: `/root/independent_review`, October 13, 2026 local time. I did not
write the guard, evaluation runner or primary analyzer. I independently
authored the matrix reference, saved-record checker and separate bootstrap
implementation before freeze. The completed review uses only saved evidence,
with zero neural queries, training updates, self-play or new search experiments.
It is same-host verification, not external replication or a new exact solver.

## Execution and independent arithmetic

Both seed tasks are complete: 2,512 fixture searches, 1,256 paired fixture
records and 320 matches. Each seed retains every 200/300/128 fixture in both
variants and all 40 opening families with both colors against Negamax-3.
There are no abandoned games, missing tasks or measured retries. The launch
record names prospective commit `ffb8009b014ab6c84b9eb00ccab986fcb6bc4b14`;
the manager required the clean branch's remote ref to equal that commit.

The frozen independent verifier reconstructs all fixture positions and full
match histories with immutable top-first matrices and directional four scans.
It checks 6,287 agent decisions and all 56 changed actions, including complete
original visit vectors, maximum-visit original selections, legal masks,
original/final actions, wins, safe/unsafe sets, every opponent win witness,
zero-visit flags, reasons, candidate/reply counts, actor turns and terminal
results. Opponent selections agree with their saved score maxima. Every match
has the required opening prefix/color/family and the independently replayed
winner/result/score. Every paired fixture has identical original action,
visits, root logits, priors and value before the guard.

The separate verifier reproduces all main counters, means, match W/D/L,
color/family scores, paired confidence intervals, pooled contrasts, latency
and verdicts: **279 numerical comparisons PASS**. Its independently implemented
frozen stream 4410903 reproduces the analyzer; reviewer stream 20261013 also
gives the same headline match intervals and verdict. Clusters retain both
colors and both fixed networks inside each shared opening family. The code
imports neither production rules nor primary analysis arithmetic. I additionally
recalculated the raw tactical category counts and match W/D/L/scores during
the final review; all agree with the report. A final saved-row supplement
independently reproduces all four per-seed tactical category intervals, and
confirms the latest editorial report's five distinct score-changing families,
six changed solved positions that remain lost, and single zero-visit tactical
winning selection. The report's separate future hypotheses are clearly untested;
no follow-up or public integration is represented as complete.

## Scientific findings

| Seed | Control wins / safe responses | Guarded wins / safe responses | Baseline mistakes fixed |
|---|---:|---:|---:|
| 42 | 97/100; 90/100 | 100/100; 100/100 | 13 |
| 314159 | 99/100; 87/100 | 100/100; 100/100 | 14 |

All 27 targeted mistakes are fixed, with no new targeted mistake. These
directly targeted one-ply tests are correctness evidence, not independent
evidence of stronger general gameplay. Inactive and all-unsafe choices retain
the original action. The prelaunch independent reference and measured-record
checks find no rule violation or illegal action.

On solved300, each seed makes seven changes and fixes four WDL mistakes.
Seed 42 improves accuracy 229/300 to 233/300; seed 314159, 233/300 to 237/300.
Of the 14 changes, eight improve exact WDL and six retain it. Seed 42 has
three -1 to 0 changes and one -1 to +1; seed 314159 has four -1 to 0 changes.
All remaining legal-action labels and changed selections are checked directly
against the complete frozen oracle vectors. There is no WDL regression.
Deep128 has no changed action: 98/128 and 96/128 correct for seeds 42 and
314159 respectively in both variants. The correct conclusion is **no observed
WDL regression on these development boards**, with a small solved300 gain,
not proven universal strategic safety. Lost-row denominators remain explicit;
automatically WDL-preserving choices on lost boards are not strong-play evidence.
The review checks full-vector consistency and saved labels, without independently
re-solving their entire game trees.

| Seed | Control W/D/L; score | Guarded W/D/L; score | Change, percentage points [95%] |
|---|---|---|---:|
| 42 | 44/4/32; 57.500% | 45/6/29; 60.000% | +2.500 [0.000, +5.625] |
| 314159 | 35/4/41; 46.250% | 36/5/39; 48.125% | +1.875 [0.000, +5.000] |

Equal-weight pooled change: **+2.1875pp [ +0.625, +4.375 ]**. This is a positive,
small conditional effect at the declared development Negamax-3 endpoint. It
does not meet the prospectively required +5pp practical target, so the declared
playing-strength verdict **INCONCLUSIVE** is correct even though the pooled
interval excludes zero. Both seeds improve in point estimate; their separate
intervals include zero. Neither a favorable seed nor a checkpoint is selected
retrospectively. These fixed-network/family intervals do not estimate retraining
variation, unseen-opening performance or strength against other opponents.

The guard changes eight of 994 guarded match decisions on seed 42 and seven
of 938 on seed 314159. Fourteen changes avoid a winning reply and one fixes a
missed immediate win. All are independently classified. IMMEDIATE_WIN often
activates without changing an already-winning move. The 68 NO_SAFE_MOVE
match decisions are recorded without inventing an escape or theoretical-loss
claim. Descriptive activation/outcome association is not substituted for the
paired intervention contrast.

## Latency, preservation and resources

Across 3,188 guarded fixture/match calls, mean guard time is **0.050229ms**,
median 0.054292ms, p95 0.072235ms and worst 0.102667ms. Mean per-call guard/search
ratio is 0.240151%; this differs from dividing pooled mean guard by pooled mean
search because search times vary. Classification mean is 0.043795ms. Guard
and total-call clocks remain separate; the former ends before final diagnostic
object assembly. The observed p95, worst and mean-relative measurements pass
all prospectively declared budgets. These are host/workload measurements,
not a guarantee for other devices or app concurrency.

I directly rehashed all **61 frozen source/design/review/input identities**,
both original checkpoint files, all **150 official campaign files** against
the original official inventory and launch inventory, and all **1,473 inherited
canonical files** against launch hashes. Every hash matches. The saved
preservation audit has 22 nonempty actual historical scopes and no changed
file, covering G/H/H-B/H-C/I-A/I-B outputs, sources, reports and evidence.
The prior sealed-access incident remains disclosed; no sealed content was
read for this review. Canonical/protected preservation uses only byte hashing.

Each seed's checkpoint/configuration identity and complete learner before/after
state agree. Worker CPU/wall ratios are approximately 0.9985 and 0.9986, consistent
with one-thread execution without a competing heavy workload. Frozen regression
evidence reports 113 passing MCTS/bitboard tests, excluding two self-play-generating
tests. No historical/app/API/deployment source or model checkpoint was edited;
the public agent remains unchanged. The corrected prelaunch launcher finding
remains in the prelaunch review rather than being erased.

Measured tasks plus analysis/verification finish at task-origin charge
1,149.65 seconds (19.16 minutes), including all prelaunch elapsed time. Seed
tasks consume 145.72 and 139.69 seconds in the supervisor; analysis 1.04 seconds,
independent full verification 14.19 seconds. The original hard deadline remains
03:38:39 UTC (October 13, 23:38:39 local). Final review, integrity audit and
publication must remain charged to that same origin in additive final accounting.
This review does not claim publication has already occurred.

## Recommendation

The evidence establishes mechanical correctness, reliable targeted tactics,
no observed development-suite WDL regression and low overhead. It shows a small
positive pooled endpoint effect but **does not establish the frozen practical
playing-strength improvement target or broad stronger gameplay**. Keep the
public baseline unchanged. A separate prospectively sized evaluation with more
opening families and additional fixed opponents is a reasonable next research
step; no integration or follow-up experiment is implemented here. No unresolved
scientific or preservation finding blocks publication of this qualified result.

## Publication supplement

The additive publication audit correctly hashes each inherited file as a Git
blob against the exact base, reporting 1,487 matching inherited files. I
inspected this hash-only code and independently compared all 160 control
histories/results with the named H-C baseline: all 1,280 field comparisons
match. The 61 frozen identities still match. The audit's 1,536.77-second
prepublication snapshot is 25.61 minutes, below the unchanged 45-minute limit;
the report should describe that snapshot as approximately 26 minutes. These
checks made no model query, search or bootstrap rerun and changed no frozen
source. Evidence is `results/publication-review.json`. Final push timing remains
the lead's durable publication record.
