# Phase 4D.4J-A — AlphaZero tactical root guard

**Playing-strength verdict: INCONCLUSIVE.** The guard fixed all 27 targeted tactical errors and produced a small positive match-score effect: **+2.1875 percentage points**, paired 95% interval **[+0.625, +4.375]**. Both seeds improved, but the gain missed the prospectively required +5-point practical target. This supports reliable one-ply safeguards and a small conditional endpoint gain; it does not establish the requested meaningful playing-strength improvement or justify public integration.

| Question | Verdict |
|---|---|
| Mechanical correctness | PASS: independent rules and original PUCT preservation |
| One-ply tactical reliability | PASS |
| Deeper exact-value safety | PASS_NO_OBSERVED_WDL_REGRESSION |
| Actual playing strength | INCONCLUSIVE |
| Latency budget | PASS |

## Problem, motivation and established methods

CNN remains preferred: H-C found no equal-compute ResNet advantage. I-A showed useful strategic information in learned policy and value: zero values raised CNN tactics but lost 20pp match score; uniform-policy/zero-value PUCT solved tactical200 yet scored 11.875%. I-B reduced held-out MSE without establishing preserved strength. This phase uses ORIGINAL H-C generation 6 CNNs, not I-B treatment.

Immediate-win detection and one-ply loss avoidance are conventional tactical safeguards. Existing Tactical MCTS (`MCTSAgent`) takes wins/filters unsafe moves before UCT; F1 neural MCTS filters root expansion and renormalizes allowed priors. Existing `tactical_value.py` also contains exact engine successor/reply checks. This experiment reuses immutable bitboard drop/winner rules and applies selection AFTER unchanged PUCT. It does not introduce a novel AI algorithm or replace learned search guidance.

## Algorithm and implementation

Run 256 original PUCT simulations, then inspect the ACTUAL immutable root and unmodified seven-column visits. Immediate winning actions take priority, selecting most visits then lowest physical column. With no win, preserve an already-safe original exactly; otherwise choose the most-visited safe action with the same deterministic restricted tie rule. If all legal moves allow a winning opponent reply, return original and record NO_SAFE_MOVE, without a theoretical-loss claim. A terminal draw successor is safe. Terminal roots are rejected. No recursion, model call, RNG draw, policy/value modification, tree change, visit rewrite, proof propagation or inference interface change.

Versioned contracts: `connect4-tactical-root-guard-v1` and `connect4-guarded-decision-v1`. Every guarded decision stores original/final actions, legal/winning/safe/unsafe sets, winning reply witnesses, complete original visits, selected/zero visits, change/reason/no-safe flags, occupancy, candidate/reply counts and timing. Zero visits are not replaced with synthetic evidence.

## Correctness and preservation of original search

Prelaunch [validation](../research/alphazero_tactical_guard/results/validation.json): {'directed_terminal_draw_rejection': 1, 'directed_terminal_rejection': 14, 'puct_model_rng_preservation': 6, 'reference_decisions': 4872, 'terminal_rejection': 79}. Coverage: {'all_unsafe': 13, 'both_players_0': 185, 'both_players_1': 185, 'full_column': 69, 'immediate_win': 136, 'multiple_safe': 292, 'multiple_wins': 20, 'zero_visit_override': 1020}. Random attempt cap 160; 81 nonterminal legal samples, plus synthetic physical-column wins, horizontal/vertical/both diagonals, both colors, mirrors, multiple wins/safe moves, all-unsafe, full columns, terminal wins/draws, last-cell draw, zero visits and deterministic ties.

A separately authored matrix rules engine scans directional cells rather than bit operations. Both checkpoint preservation tests compare disabled/enabled wrapper searches against canonical PUCT: exact actions before guard, visits, neural logits/priors/values, evaluation RNG states, board bytes, predictions, full learner/RNG/optimizer/replay state and weights. Model checks use 6 positions across two seeds. All measured fixture pairs additionally assert the same raw search identities. No original source under games/ or live app path changed.

## Checkpoint and execution provenance

| Seed | Original generation 6 boundary SHA256 |
|---|---|
| 42 | `ba2bde2ec71b3ac7486046d1cc26ab591bb59d6b764193ece08c6d6a6f24f58b` |
| 314159 | `2646d7cdee7d7b069412688bf95fd9794bea30402b4952120c904dbcb399401d` |

CNN identity `connect4-conv64-128-128-policy7-scalar-tanh-v2`,325,896 parameters. Strict original loader validates full architecture/configuration/model state, original schedule, generation, original source/runtime/inference contract and checkpoint hash. Checkpoints are read-only; no optimizer/replay modification. Runtime and complete execution-source identities: [freeze.json](../research/alphazero_tactical_guard/results/freeze.json). Per-seed before/after states and weight hashes: [tasks](../research/alphazero_tactical_guard/results/tasks/).

## Prospective design and execution

Prospective pushed commit `ffb8009b014ab6c84b9eb00ccab986fcb6bc4b14`; base `37ec500ce70bd2bbe63ac6578cd7bf576004def6`; dedicated `phase4d4j-a-tactical-root-guard` branch/worktree. Protocol [DESIGN.md](../research/alphazero_tactical_guard/DESIGN.md), [design.json](../research/alphazero_tactical_guard/results/design.json), independent prelaunch [review](../research/alphazero_tactical_guard/independent-prelaunch-review.md). All freezes preceded measured work.

Both original fixed CNN seeds; PUCT256/c1.41/no root noise; unchanged values/priors/terminal handling. Fixtures have identical board and search RNG identities; matches have same opening/color/role streams and reference Negamax3 configuration. Diverging trajectories may consume later random numbers differently. All 40 development base opening families, both colors. No secondary/ResNet evaluation, chosen before freeze. No sealed input was accessed; the prior I-A accidental sealed-text incident remains disclosed history.

Execution: 2512 fixture searches, 1256 paired records, 320 matches, two completed seed tasks, zero retries/training/self-play/abandoned games. Durable logs and exclusive attempt identity remain in ignored experiment-output/phase4d4j-a/; raw decisions/results are tracked under results/tasks/.

## One-ply tactical results

Perfect targeted tactical performance is principally a correctness check, not independent evidence of stronger gameplay.

| Seed | Variant | Immediate wins /100 | Unique safe /100 | Avoidable errors /200 | Missed wins /200 |
|---|---|---:|---:|---:|---:|
| 314159 | control | 99 | 87 | 13 | 1 |
| 314159 | guarded | 100 | 100 | 0 | 0 |
| 42 | control | 97 | 90 | 10 | 3 |
| 42 | guarded | 100 | 100 | 0 | 0 |

Seed 314159: 14 changes, 14 baseline mistakes fixed, 0 new mistakes, 186 already-correct decisions, 14 incorrect despite safe alternative, 0 no-safe rows. Reasons: {'AVOID_IMMEDIATE_LOSS': 13, 'IMMEDIATE_WIN': 100, 'UNCHANGED': 87}. Category contrasts pp [95%]: win +1.000 [+0.000, +3.409]; safe +13.000 [+4.630, +23.529].

Seed 42: 13 changes, 13 baseline mistakes fixed, 0 new mistakes, 187 already-correct decisions, 13 incorrect despite safe alternative, 0 no-safe rows. Reasons: {'AVOID_IMMEDIATE_LOSS': 10, 'IMMEDIATE_WIN': 100, 'UNCHANGED': 90}. Category contrasts pp [95%]: win +3.000 [+0.000, +7.609]; safe +10.000 [+3.922, +17.594].

Tactical activation rates (IMMEDIATE_WIN or AVOID_IMMEDIATE_LOSS) were 55.0% for seed 42 and 56.5% for seed 314159; changed-decision rates were 6.5% and 7.0%. None of the tactical200 positions lacked a safe action. Across seeds, 373 baseline choices were already correct and all 27 incorrect choices had a safe alternative.

One seed-42 tactical decision selected a winning action with **zero original PUCT visits**. Its original visit vector is retained exactly; no synthetic visits were assigned. All other measured fixture selections and all match selections had positive visits. Full original/final vectors and every tactical row are in the saved task records.

## Exact solved300 and deep128 decisions

Existing COMPLETE legal-action oracle vectors are used without invented labels. Solved300 has 100 each win/draw/loss; deep128 has 70 wins, 58 draws, 0 losses. All actions in theoretically lost rows preserve WDL automatically; that100% stratum says nothing about good gameplay. Optimal-action and WDL-preserving accuracy coincide for these WDL vectors.

| Seed | Suite | Variant | Optimal/WDL-preserving | Avoidable mistakes | W→D | W→L | D→L |
|---|---|---|---:|---:|---:|---:|---:|
| 314159 | solved | control | 77.667% | 67 | 4 | 20 | 43 |
| 314159 | solved | guarded | 79.000% | 63 | 4 | 20 | 39 |
| 314159 | deep | control | 75.000% | 32 | 9 | 11 | 12 |
| 314159 | deep | guarded | 75.000% | 32 | 9 | 11 | 12 |
| 42 | solved | control | 76.333% | 71 | 7 | 22 | 42 |
| 42 | solved | guarded | 77.667% | 67 | 7 | 21 | 39 |
| 42 | deep | control | 76.562% | 30 | 7 | 10 | 13 |
| 42 | deep | guarded | 76.562% | 30 | 7 | 10 | 13 |

Seed 314159 solved: accuracy difference +1.333 [+0.000, +3.226]pp; 7 changed decisions, reasons {'AVOID_IMMEDIATE_LOSS': 7, 'UNCHANGED': 293}; selected exact-value differences {'0': 296, '1': 4}; regressions [].

Seed 314159 deep: accuracy difference +0.000 [+0.000, +0.000]pp; 0 changed decisions, reasons {'UNCHANGED': 128}; selected exact-value differences {'0': 128}; regressions [].

Seed 42 solved: accuracy difference +1.333 [+0.000, +3.285]pp; 7 changed decisions, reasons {'AVOID_IMMEDIATE_LOSS': 7, 'UNCHANGED': 293}; selected exact-value differences {'0': 296, '1': 3, '2': 1}; regressions [].

Seed 42 deep: accuracy difference +0.000 [+0.000, +0.000]pp; 0 changed decisions, reasons {'UNCHANGED': 128}; selected exact-value differences {'0': 128}; regressions [].

All class denominators, change IDs/reasons and original/guarded exact labels: [analysis.json](../research/alphazero_tactical_guard/results/analysis.json) and raw suite rows. No observed WDL regression is a finding for these development boards; it does not establish universal strategic safety. Solved300 gained four optimal decisions per seed. Deep128 had no changed action and no improvement. Six changed solved decisions remained theoretically lost, so their local safe responses are not deeper improvements.

## Actual playing strength — principal outcome

| Seed | Variant | W/D/L | Score | X score | O score |
|---|---|---|---:|---:|---:|
| 314159 | control | 35/4/41 | 46.250% | 47.500% | 45.000% |
| 314159 | guarded | 36/5/39 | 48.125% | 50.000% | 46.250% |
| 42 | control | 44/4/32 | 57.500% | 56.250% | 58.750% |
| 42 | guarded | 45/6/29 | 60.000% | 57.500% | 62.500% |

Seed 314159 guarded−control: +1.875 [+0.000, +5.000] percentage points.

Seed 42 guarded−control: +2.500 [+0.000, +5.625] percentage points.

Equal-weight pooled guarded−control: **+2.188 [+0.625, +4.375]pp**. Frozen rule: PASS requires>=+5pp, both seeds nonnegative, pooled95% lower>0; FAIL requires<=−5pp and upper<0; otherwise INCONCLUSIVE. Result: **INCONCLUSIVE**. The pooled interval excludes zero, supporting a small positive conditional effect against this opponent/opening set, but its entire interval remains below the +5pp practical target. Each seed’s interval includes zero. The two seed effects arose in five distinct opening families; those few families, rather than 320 independent trials, determine the evidence.

10,000 prospective paired cluster percentile resamples, PCG64 seed4410903: whole 40 opening families retain both colors and both fixed seeds together. Fixtures retain ancestry/reflection clusters and row weighting. Intervals condition on these networks/development families and omit retraining-seed variability; no multiplicity correction. No favorable seed/checkpoint chosen. All per-family/color summaries and per-seed/pooled intervals are machine readable.

## Guard activation and action changes

| Seed | Guarded decisions | Win-rule activations | Avoid-loss overrides | No safe move | Changed decisions | Changed rate |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 994 | 45 | 8 | 29 | 8 | 0.805% |
| 314159 | 938 | 36 | 6 | 39 | 7 | 0.746% |

Across both seeds, 15 of 1,932 guarded match decisions changed: 14 avoided an immediate losing reply and one recovered a missed immediate win. No selected match action had zero visits. Win-rule plus safe-response activation rates were 5.332% (42) and 4.478% (314159); activation includes already-winning moves returned unchanged. All changed match decisions were independently checked against reconstructed actual boards.

Seven seed-42 games and six seed-314159 games contained a changed decision. By guarded outcome, these were respectively 1 win/2 draws/4 losses and 2 wins/1 draw/3 losses. The counts are descriptive and do not attribute individual outcomes to the guard.

IMMEDIATE_WIN can activate without changing an already-winning choice. NO_SAFE_MOVE is recorded, not an escape or theoretical proof. All changed selections were independently reclassified; changes fix local omissions/unsafe choices, not necessarily all deeper mistakes. Descriptive game-activation/outcome counts do not identify whether an activation caused a win; paired intervention matches provide the causal contrast.

## Performance and latency

| Measure | Mean | Median | p95 | Worst |
|---|---:|---:|---:|---:|
| search_seconds, ms | 42.59487 | 51.31031 | 57.98767 | 136.10121 |
| classification_seconds, ms | 0.04379 | 0.04788 | 0.06525 | 0.09596 |
| guard_seconds, ms | 0.05023 | 0.05429 | 0.07224 | 0.10267 |
| total_seconds, ms | 42.64676 | 51.36202 | 58.06101 | 136.17642 |

Mean guard/search overhead 0.240%, median 0.121%, p95 1.169%, worst 3.006%. Practical budget p95<=1ms,worst<=5ms,mean relative<=5%: PASS. Timing pools guarded fixture/game decisions (not independent samples). Whole guard clock ends before final diagnostic object assembly; total-call clock includes assembly. Root bitboard adaptation occurs identically before timed search in both variants.

Candidates per decision: mean 5.467, median 6, p95/worst 7. Opponent replies inspected: mean 31.540, median 35, p95/worst 49. Full win/reply scans used no neural inference.

| Occupied cells | Decisions | Mean guard ms | Median ms | p95 ms | Worst ms |
|---|---:|---:|---:|---:|---:|
| 0–6 | 182 | 0.06914 | 0.06923 | 0.07341 | 0.08175 |
| 7–13 | 713 | 0.06773 | 0.06954 | 0.07372 | 0.08008 |
| 14–20 | 1062 | 0.05700 | 0.05533 | 0.07195 | 0.10267 |
| 21–27 | 682 | 0.04019 | 0.04019 | 0.06795 | 0.07838 |
| 28–34 | 360 | 0.02456 | 0.02258 | 0.04013 | 0.06592 |
| 35–41 | 189 | 0.01312 | 0.01279 | 0.02187 | 0.02958 |

One-thread deterministic CPU on Apple M4, sequential workers; exact runtime in freeze. No competing neural/training workload was found at preflight; ordinary GUI/OS load remains, so wall-clock outliers are observed system-dependent measurements. Per-task CPU/wall/RSS are in identity.json. No guard neural inference.

## Independent review and historical preservation

Independent reviewer authored the matrix rules/record checker and prelaunch review without writing the guard or analyzer. The frozen independent verifier checked 6,287 agent decisions and all 56 changed actions (27 tactical, 14 solved, 15 matches), replayed all 320 games, and reproduced 279 numerical comparisons. Its separate bootstrap stream also returned +2.1875pp [+0.625,+4.375] for the pooled match contrast. It checked every fixture pair, complete visits, labels and original selections, colors, results, openings and latency. This is same-host verification, not external replication or a fresh full-game oracle solve. See [independent-verification.json](../research/alphazero_tactical_guard/results/independent-verification.json) and [results review](../research/alphazero_tactical_guard/independent-results-review.md).

The prelaunch and results reviews found no unresolved blocker. A launcher omission that would have run only the verifier self-check was found and corrected before freeze; its evidence remains in the prelaunch review. Relevant existing regression tests passed **113/113**, with two self-play-related checks excluded to avoid new self-play generation.

All 150 official campaign files remain byte-identical; required G/H/I checkpoints/reports/data/results and actual canonical source/fixture/app paths are protected by nonempty hash-only inventories. [preservation.json](../research/alphazero_tactical_guard/results/preservation.json) records every scope. The final hash-only Git-blob audit additionally verifies all **1,487 inherited tracked files**, including configuration paths, against the exact base commit. All **160 control match histories/results** reproduce the historical H-C baseline exactly (1,280 field comparisons). No historical file, original checkpoint, deployment configuration, API, public agent or live app was changed. Only the new package/report are committed; no oversized model is committed.

## Resource accounting and recommendation

Cost-only pilot: 12.54s:12 roots/4 games, only times retained; primary projection 331.33s. Prelaunch validation 14.08s final successful pass, independent review and all preparation charged from actual task origin. Measured phases: [{'name': '42', 'seconds': 145.721581697464}, {'name': '314159', 'seconds': 139.69165897369385}, {'name': 'analyze', 'seconds': 1.044724941253662}, {'name': 'independent_verify', 'seconds': 14.193912029266357}]. Execution finish charge was 1,149.65 seconds (19.16 minutes from task origin). The completed results review and final integrity audit brought the charge to approximately 26 minutes, below the 45-minute hard maximum. [final-audit.json](../research/alphazero_tactical_guard/results/final-audit.json) records the precise prepublication charge; the durable local publication record adds the actual final push completion time. Actual measured seed work plus analysis and verification took 300.65 seconds; the shorter evaluation was not padded with extra experiments. Absolute total limit2700s includes ALL work, no artificial padding to20–30min. No measured retry/additional experiment.

**Tactics improved; stronger gameplay remains inconclusive.** Keep the public baseline unchanged. The next highest-value step is a separate, prospectively powered confirmation of this root guard across more opening families and a stronger fixed opponent, preserving both seeds. If a new search improvement is pursued, compare a higher PUCT simulation budget with this guarded 256-simulation baseline under a separate compute-controlled protocol; this experiment did not test that hypothesis. No follow-up or integration is implemented.
