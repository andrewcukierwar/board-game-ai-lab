# Phase 4D.2e — Value-target audit

Completed October 5, 2026. **Analysis only: no training, research-model optimizer steps, model inference, new checkpoint, gameplay campaign or learning/search changes.** Freshly fetched `origin/main` and local `HEAD` both matched the expected `e12684c673e6c353bad55aa4ca0ce5fb7f126b07`; the initial working tree was clean. No commit, push, merge, API/UI work, deployment or paid request occurred.

**Recommendation: one isolated engine-proven value-anchoring experiment.** Of 10,608 collected examples, 1,352 have an exact bounded tactical value; 212 stored outcome targets disagree with it: **15.68% of proven examples**, or **2.00% of all collected examples**. These are valid actor-relative labels for the games that were actually played. They are contradictory supervision if the intended value is the optimal game-theoretic outcome. The audit found no historical reconstruction or label-contract failure.

## Evidence and integrity

Read the Phase 4D.1, 4D.2b, 4D.2c and 4D.2d reports, trainer, self-play driver and neural agent. Used the completed 4D.2b **replacement1** run; its failed pre-training predecessor supplies no collected examples. The symmetry and symmetry-plus-noise runs are the completed retained runs. Each has 200 completed games, 1,970 recorded updates and 63,040 optimizer sample appearances.

Both existing driver and supplemental manifests were hash-verified before analyzing each run. The noisy preflight manifest and its 13 entries were also verified, including the supplemental manifest’s bound external handoff/preflight hashes. Checkpoints were hashed as bytes, without loading or evaluating them. The original manifests were not replaced.

| Run | Driver / supplemental entries verified | Candidate SHA-256 |
| --- | --- | --- |
| 4D.2b | 15 / 37 | `d16cb516775146084bc2438c0166d241c1224efc69b2aa7e1b40596de77d1521` |
| 4D.2c | 17 / 40 | `a70910085f1d8be5886d60ce36ee80877bc2a4de6104ac6311462477c5b2cde7` |
| 4D.2d | 16 / 33 | `346435b1c22fccefc69fae103cd9cc96d343a266a9d1d0c49976290869b28ed7` |

Before/after SHA-256 inventories cover **347 pre-existing files** under `experiment-output/`, `models/`, `research/` and `docs/`, including retained fixtures, event logs, checkpoints and prior reports: **347 unchanged, zero missing or modified**. New audit files live in a separate directory. Existing tracked implementation modules are unchanged; the API isolation test only adds the audit module to its forbidden-import list.

The read-only [auditor](../games/connect4/neural_value_target_audit.py) and [correctness tests](../tests/test_neural_value_target_audit.py) establish the method. The [complete JSON](../experiment-output/phase4d2e-value-target-audit-20261005/analysis.json) contains all **10,608 reconstructed rows**, original run/source identities and exploration configurations, proofs and successor witnesses, summaries by actor/cohort/stage, actor-within-cohort summaries, separate score families and exact sampling summaries. Each row retains physical and canonical pre-move boards, actor, game, one-based ply, zero-based collection index, selected action, all visits/policy, winner, stored outcome and contemporaneous prediction. No frozen diagnostic labels supplied proofs.

Replay starts from a new engine game for every completed episode and applies only recorded legal actions. Before every move it checks board and actor, action support, 32-visit accounting, temperature-1 policy against visits, guard/noise metadata where recorded, and the pending example’s exact SHA-256 fingerprint. It verifies the terminal winner, every finalized example/label/encoding/policy contract, per-actor counts and report history. Updates must follow completed games; all **5,910 minibatch index lists** independently reproduce the seeded sampling stream against the collection size available at that update. There are no partial games. Reflections preserve actor/outcome classes and do not create extra stored examples. Integrity failures raise errors and never repair inputs.

A development check initially rejected the stored `partial_game: null` marker because it expected the key to be absent. The auditor was corrected to accept the documented null marker; historical evidence was untouched. This was a reader-contract mistake, not an evidence failure. All subsequent replay/integrity checks passed.

## Exact tactical proof contract

For **+1**, enumerate legal engine successors and find at least one immediate win for the pre-move actor. For **−1**, first exclude every immediate actor win; then require **every legal action** to produce a nonterminal successor in which the opponent has an immediate winning reply. All legal-action/reply witnesses are retained. This is sufficient under this engine: drops are deterministic, there is no pass, `make_move` switches actor/piece even after winning, and winner identity is checked against the mover captured before the move. A terminal draw successor defeats the loss proof. No terminal input, empty-action vacuity, fabricated opponent turn or diagnostic helper label is accepted.

An ordinary forced block with a safe response receives **unknown**, even if that response is unique. Unknown does not mean draw. These are conservative partial proofs, not a full Connect4 solver; unclassified positions may have exact values discoverable by deeper reasoning. “One-ply loss” here means unavoidable loss on the opponent’s next reply, two actions from the current state. Tests cover both actors and reflections, immediate-win precedence, forced losses, ordinary blocks, a last-move draw escape, terminal rejection and immutability.

## Contradiction prevalence

All W/D/L counts below are **actor-relative eventual targets**, not winner IDs. Contradiction rate uses proven positions as its denominator; prevalence uses all collected positions. Fractions are rounded, counts exact.

### Proven +1

| Run | All examples | Proven count / prevalence | Stored W/D/L | Contradictions / proven |
| --- | --- | --- | --- | --- |
| 4D.2b | 3673 | 345 / 9.39% | 302/0/43 | 43/345 (12.46%) |
| 4D.2c | 3613 | 384 / 10.63% | 326/8/50 | 58/384 (15.10%) |
| 4D.2d | 3322 | 335 / 10.08% | 268/0/67 | 67/335 (20.00%) |
| Pooled | 10608 | 1064 / 10.03% | 896/8/160 | 168/1064 (15.79%) |

### Proven −1

| Run | All examples | Proven count / prevalence | Stored W/D/L | Contradictions / proven |
| --- | --- | --- | --- | --- |
| 4D.2b | 3673 | 95 / 2.59% | 7/0/88 | 7/95 (7.37%) |
| 4D.2c | 3613 | 119 / 3.29% | 16/3/100 | 19/119 (15.97%) |
| 4D.2d | 3322 | 74 / 2.23% | 18/0/56 | 18/74 (24.32%) |
| Pooled | 10608 | 288 / 2.71% | 41/3/244 | 44/288 (15.28%) |

Combined proven-value contradiction rates are **50/440 = 11.36%**, **77/503 = 15.31%**, and **85/409 = 20.78%** for 4D.2b/2c/2d. As fractions of all collected examples they are **1.36% / 2.13% / 2.56%**. Immediate-win contradiction rates are **12.46% / 15.10% / 20.00%**; loss-proof contradiction rates are **7.37% / 15.97% / 24.32%**. Symmetry/noise did not reduce contradictions in these observed collections. One evolving run per method and correlated positions within games do not establish a causal or statistically independent method effect.

### Actor breakdown

| Run / actor | All examples | +1 count; W/D/L | +1 contradictions | −1 count; W/D/L | −1 contradictions |
| --- | --- | --- | --- | --- | --- |
| 4D.2b / X | 1897 | 238; 205/0/33 | 33/238 (13.87%) | 15; 1/0/14 | 1/15 (6.67%) |
| 4D.2b / O | 1776 | 107; 97/0/10 | 10/107 (9.35%) | 80; 6/0/74 | 6/80 (7.50%) |
| 4D.2c / X | 1873 | 292; 247/8/37 | 45/292 (15.41%) | 27; 6/0/21 | 6/27 (22.22%) |
| 4D.2c / O | 1740 | 92; 79/0/13 | 13/92 (14.13%) | 92; 10/3/79 | 13/92 (14.13%) |
| 4D.2d / X | 1715 | 191; 155/0/36 | 36/191 (18.85%) | 33; 8/0/25 | 8/33 (24.24%) |
| 4D.2d / O | 1607 | 144; 113/0/31 | 31/144 (21.53%) | 41; 10/0/31 | 10/41 (24.39%) |
| Pooled / X | 5485 | 721; 607/8/106 | 114/721 (15.81%) | 75; 15/0/60 | 15/75 (20.00%) |
| Pooled / O | 5123 | 343; 289/0/54 | 54/343 (15.74%) | 213; 26/3/184 | 29/213 (13.62%) |

### Game cohorts

Smoke = games 1–2; the other cohorts separate initial learning and subsequent 50-game blocks. `visited / took` counts searches that visited any winning action and samples that immediately won. All actor-by-cohort combinations and their target distributions/rates are retained in JSON, including empty proven subsets.

| Run / games | Examples; W/D/L | +1 n; contradictions | −1 n; contradictions | Winning visits | Visited / took win |
| --- | --- | --- | --- | --- | --- |
| 4D.2b / 1-2 | 34; 17/0/17 | 2; 0/2 (0.00%) | 0; 0/0 (—) | 52 | 2 / 2 |
| 4D.2b / 3-20 | 348; 182/0/166 | 33; 0/33 (0.00%) | 8; 0/8 (0.00%) | 483 | 24 / 18 |
| 4D.2b / 21-50 | 691; 355/0/336 | 50; 6/50 (12.00%) | 11; 1/11 (9.09%) | 1009 | 42 / 30 |
| 4D.2b / 51-100 | 899; 462/0/437 | 84; 15/84 (17.86%) | 25; 3/25 (12.00%) | 1505 | 72 / 50 |
| 4D.2b / 101-150 | 853; 443/0/410 | 81; 6/81 (7.41%) | 20; 1/20 (5.00%) | 1528 | 67 / 50 |
| 4D.2b / 151-200 | 848; 438/0/410 | 95; 16/95 (16.84%) | 31; 2/31 (6.45%) | 1741 | 75 / 50 |
| 4D.2c / 1-2 | 34; 17/0/17 | 2; 0/2 (0.00%) | 0; 0/0 (—) | 52 | 2 / 2 |
| 4D.2c / 3-20 | 222; 119/0/103 | 20; 1/20 (5.00%) | 5; 0/5 (0.00%) | 476 | 20 / 18 |
| 4D.2c / 21-50 | 598; 308/0/290 | 54; 5/54 (9.26%) | 20; 0/20 (0.00%) | 884 | 39 / 30 |
| 4D.2c / 51-100 | 938; 463/42/433 | 107; 19/107 (17.76%) | 38; 6/38 (15.79%) | 1494 | 73 / 49 |
| 4D.2c / 101-150 | 893; 464/0/429 | 114; 28/114 (24.56%) | 33; 12/33 (36.36%) | 1494 | 65 / 50 |
| 4D.2c / 151-200 | 928; 481/0/447 | 87; 5/87 (5.75%) | 23; 1/23 (4.35%) | 1648 | 67 / 50 |
| 4D.2d / 1-2 | 31; 16/0/15 | 2; 0/2 (0.00%) | 0; 0/0 (—) | 57 | 2 / 2 |
| 4D.2d / 3-20 | 332; 172/0/160 | 29; 8/29 (27.59%) | 4; 0/4 (0.00%) | 669 | 28 / 18 |
| 4D.2d / 21-50 | 481; 249/0/232 | 39; 3/39 (7.69%) | 5; 0/5 (0.00%) | 933 | 37 / 30 |
| 4D.2d / 51-100 | 847; 437/0/410 | 96; 21/96 (21.88%) | 19; 4/19 (21.05%) | 1756 | 77 / 50 |
| 4D.2d / 101-150 | 790; 408/0/382 | 83; 21/83 (25.30%) | 24; 8/24 (33.33%) | 1698 | 72 / 50 |
| 4D.2d / 151-200 | 841; 433/0/408 | 86; 14/86 (16.28%) | 22; 6/22 (27.27%) | 1725 | 73 / 50 |

### Position age

Early = pre-move occupied cells 0–13 (plies 1–14); middle = 14–27 (plies 15–28); late = 28–41 (plies 29–42). Fixed cutoffs were used across runs, rather than fractions of each game’s eventual length.

| Run / stage | Examples | +1 prevalence; contradictions | −1 prevalence; contradictions |
| --- | --- | --- | --- |
| 4D.2b / early | 2586 | 115 (4.45%); 18/115 (15.65%) | 39 (1.51%); 4/39 (10.26%) |
| 4D.2b / middle | 1005 | 202 (20.10%); 21/202 (10.40%) | 47 (4.68%); 2/47 (4.26%) |
| 4D.2b / late | 82 | 28 (34.15%); 4/28 (14.29%) | 9 (10.98%); 1/9 (11.11%) |
| 4D.2c / early | 2631 | 152 (5.78%); 19/152 (12.50%) | 51 (1.94%); 5/51 (9.80%) |
| 4D.2c / middle | 886 | 206 (23.25%); 32/206 (15.53%) | 57 (6.43%); 9/57 (15.79%) |
| 4D.2c / late | 96 | 26 (27.08%); 7/26 (26.92%) | 11 (11.46%); 5/11 (45.45%) |
| 4D.2d / early | 2536 | 132 (5.21%); 21/132 (15.91%) | 25 (0.99%); 2/25 (8.00%) |
| 4D.2d / middle | 762 | 187 (24.54%); 42/187 (22.46%) | 46 (6.04%); 16/46 (34.78%) |
| 4D.2d / late | 24 | 16 (66.67%); 4/16 (25.00%) | 3 (12.50%); 0/3 (0.00%) |

## Immediate wins: search discovery versus sampled action

Every run used raw 32-simulation PUCT 1.41, temperature 1 and guard OFF. 4D.2b had no symmetry/noise; 4D.2c had 0.5 horizontal augmentation and no noise; 4D.2d retained 0.5 augmentation and used root Dirichlet ε=.25, α=.30 on legal actions. Winning visits below are summed over all immediate-winning actions, not the maximum or modal action. Sampling is the actual recorded temperature-1 action.

| Run | Win states | Any winning action visited | Total winning visits / actions | Zero-visit winning actions | Immediate win sampled | Missed win sample |
| --- | --- | --- | --- | --- | --- | --- |
| 4D.2b | 345 | 282 (81.74%) | 6318 / 428 | 133 | 200 (57.97%) | 145 |
| 4D.2c | 384 | 266 (69.27%) | 6048 / 481 | 200 | 199 (51.82%) | 185 |
| 4D.2d | 335 | 289 (86.27%) | 6838 / 390 | 96 | 200 (59.70%) | 135 |

At the state level, no win was visited in **63 / 118 / 46** searches. Noise broadened winning-action discovery relative to symmetry only, but sampled win-taking rose only **199/384 → 200/335**. The extra action exploration did not align final-outcome labels with exact state values.

| Run | Any win visited? | Immediate win taken? | Eventual actor W/D/L = stored +1/0/−1 |
| --- | --- | --- | --- |
| 4D.2b | no | no | 44/0/19 |
| 4D.2b | yes | no | 58/0/24 |
| 4D.2b | yes | yes | 200/0/0 |
| 4D.2c | no | no | 85/6/27 |
| 4D.2c | yes | no | 42/2/23 |
| 4D.2c | yes | yes | 199/0/0 |
| 4D.2d | no | no | 22/0/24 |
| 4D.2d | yes | no | 46/0/43 |
| 4D.2d | yes | yes | 200/0/0 |

Taking an immediate win always terminates with a +1 label, as verified in all **599** such examples. Missing it can still end in a later win, loss or draw. In the noisy run, **43** contradictory +1 states had visited a winning action but sampled another action; **24** had no visited winning action. Thus unvisited wins are only part of the mechanism. Conservative −1 contradictions likewise arise when an opponent fails to exploit a winning reply under actual play.

Concrete retained examples: baseline game 22, ply 31, X could win in column 5 with 19 visits, but sampled column 0 and eventually lost (target −1). Symmetry game 6, ply 8, O could win in column 3 with 25 visits, but sampled column 4 and lost. Noise game 4, ply 12, O could win in column 4 with 28 visits, but sampled column 0 and lost. All columns are zero-based; full boards, visits and winners are in JSON.

## Draw supervision and exact optimizer appearances

Stored examples and sampled appearances are different denominators. Every appearance below comes from an actual recorded minibatch index; reflection retains its W/D/L class. No sampling proportion was inferred from collection proportions.

| Run | Stored W/D/L | Collected draw fraction | Optimizer W/D/L | Appearance draw fraction | Updates containing draw |
| --- | --- | --- | --- | --- | --- |
| 4D.2b | 1897/0/1776 | 0.00% | 32524/0/30516 | 0.00% | 0/1970 |
| 4D.2c | 1852/42/1719 | 1.16% | 32761/719/29560 | 1.14% | 543/1970 |
| 4D.2d | 1715/0/1607 | 0.00% | 32597/0/30443 | 0.00% | 0/1970 |
| Pooled | 5464/42/5102 | 0.40% | 97882/719/90519 | 0.38% | 543/5910 |

All 42 stored draw examples come from **4D.2c game 73** (21 X, 21 O), cohort 51–100. All 42 appear at least once in recorded updates. The 719 total appearances average 17.12 per stored draw example, but are repeated exposures to one correlated game. Symmetry updates **1–700** (after games 3–72) had no available draws. Even after game 73, only 719/40,640 = **1.77%** of sample appearances are draws, and **543/1,270** updates contain at least one. Across the full run, **1,427/1,970** updates contain none. Baseline/noise value losses never saw the draw class as the target.

By **update cohort** (the game after which the optimizer ran):

| Run / after-game cohort | Updates; appearances | Optimizer W/D/L | Updates containing draw |
| --- | --- | --- | --- |
| 4D.2b / 3-20 | 180; 5760 | 2972/0/2788 | 0 |
| 4D.2b / 21-50 | 300; 9600 | 4983/0/4617 | 0 |
| 4D.2b / 51-100 | 500; 16000 | 8280/0/7720 | 0 |
| 4D.2b / 101-150 | 500; 16000 | 8332/0/7668 | 0 |
| 4D.2b / 151-200 | 490; 15680 | 7957/0/7723 | 0 |
| 4D.2c / 3-20 | 180; 5760 | 3019/0/2741 | 0 |
| 4D.2c / 21-50 | 300; 9600 | 5048/0/4552 | 0 |
| 4D.2c / 51-100 | 500; 16000 | 8340/233/7427 | 160 |
| 4D.2c / 101-150 | 500; 16000 | 8191/278/7531 | 209 |
| 4D.2c / 151-200 | 490; 15680 | 8163/208/7309 | 174 |
| 4D.2d / 3-20 | 180; 5760 | 2977/0/2783 | 0 |
| 4D.2d / 21-50 | 300; 9600 | 4969/0/4631 | 0 |
| 4D.2d / 51-100 | 500; 16000 | 8222/0/7778 | 0 |
| 4D.2d / 101-150 | 500; 16000 | 8316/0/7684 | 0 |
| 4D.2d / 151-200 | 490; 15680 | 8113/0/7567 | 0 |

By **source-game cohort** (where sampled examples originated), the JSON records exact W/D/L counts separately. Every draw appearance originates in 51–100: 719 for symmetry, zero elsewhere. Collection W/D/L per source cohort is in the earlier cohort table. Updates in later cohorts continue drawing the single earlier draw game. The final game in each run is collected but has no subsequent update; its examples receive zero optimizer appearances.

The draw game itself contains **8 proven wins and 3 proven losses**, all labeled draw by its actual outcome. Those 11 labels are included in contradiction counts. Observed draws are not proofs of optimal draws. This audit proves no exact draw values and makes no inference that the draw class is impossible. Anchoring these proven states in a future experiment would reduce, rather than solve, the already sparse draw supervision.

The value loss saw contradictory proven-state labels on **581 / 949 / 1,384** recorded sample appearances: **0.92% / 1.51% / 2.20%** of each run’s 63,040 appearances, pooled **2,914/189,120 = 1.54%**. This measures actual optimizer exposure, beyond collection prevalence.

## A. Predictions against eventual self-play outcomes

Use the retained **contemporaneous pre-search** W/D/L probabilities only. No checkpoint inference was rerun. These scores describe prediction of the actual subsequent self-play result under evolving search/action behavior. They are neither held-out evaluation nor game-theoretic calibration. Brier is the unnormalized three-class squared-error sum, NLL uses natural logs, scalar prediction is P(win)−P(loss), and scalar errors use the actor-relative target.

| Run | N | Brier | NLL | Scalar MAE | Scalar MSE |
| --- | --- | --- | --- | --- | --- |
| 4D.2b | 3673 | 0.6058 | 1.2713 | 0.8567 | 1.2082 |
| 4D.2c | 3613 | 0.5896 | 1.2084 | 0.8633 | 1.1322 |
| 4D.2d | 3322 | 0.7182 | 1.2770 | 1.0270 | 1.4326 |
| Pooled | 10608 | 0.6355 | 1.2517 | 0.9123 | 1.2526 |

## B. Predictions against exact tactical values

The same contemporaneous probabilities are scored only where an engine proof exists, using the proven value. This evaluates exact tactical-value prediction; it is not on-policy outcome calibration. The underlying position populations differ across methods, and repeated positions/game siblings are correlated. Lower scores on these collections do not establish a causal method improvement or general optimal-value calibration.

| Run / proof class | N | Brier | NLL | Scalar MAE | Scalar MSE |
| --- | --- | --- | --- | --- | --- |
| 4D.2b / all proven | 440 | 0.4679 | 1.0285 | 0.6267 | 0.9339 |
| 4D.2b / +1 | 345 | 0.5117 | 1.1075 | 0.6648 | 1.0211 |
| 4D.2b / −1 | 95 | 0.3085 | 0.7419 | 0.4885 | 0.6171 |
| 4D.2c / all proven | 503 | 0.3507 | 0.6995 | 0.5128 | 0.6859 |
| 4D.2c / +1 | 384 | 0.3872 | 0.7753 | 0.5610 | 0.7568 |
| 4D.2c / −1 | 119 | 0.2329 | 0.4548 | 0.3572 | 0.4573 |
| 4D.2d / all proven | 409 | 0.5350 | 1.0337 | 0.7447 | 1.0679 |
| 4D.2d / +1 | 335 | 0.5687 | 1.1185 | 0.7795 | 1.1349 |
| 4D.2d / −1 | 74 | 0.3824 | 0.6500 | 0.5871 | 0.7648 |
| Pooled / all proven | 1352 | 0.4446 | 0.9077 | 0.6200 | 0.8822 |
| Pooled / +1 | 1064 | 0.4847 | 0.9911 | 0.6634 | 0.9615 |
| Pooled / −1 | 288 | 0.2963 | 0.5996 | 0.4596 | 0.5890 |

On the **contradictory subset only**, keep both notions of target separate:

| Run | N | A: outcome Brier / NLL | B: exact Brier / NLL |
| --- | --- | --- | --- |
| 4D.2b | 50 | 1.3772 / 4.4485 | 0.3610 / 0.6384 |
| 4D.2c | 77 | 1.6788 / 6.1242 | 0.1009 / 0.1669 |
| 4D.2d | 85 | 1.6116 / 4.4541 | 0.1642 / 0.2935 |

Predictions on contradictory states often already favor the proven value: scoring them against subsequent mistakes makes them look much worse. The stored target then pushes in the opposite direction. All retained probabilities are finite, normalized within the original float32 tolerance and strictly positive for the target classes here, so no NLL clipping or infinite-score substitution was needed. The auditor explicitly represents zero-probability NLL as null plus an infinity flag for strict JSON, rather than silently clipping. Score families and class/cohort/actor breakdowns remain separate in JSON.

## Loss mathematics: functioning objective, mismatched semantics

The existing three-class cross-entropy is mathematically functioning as intended. With raw value logits u, p=softmax(u), and stored class y, its unbatched logit gradient is **p−onehot(y)**; a mean minibatch divides it by batch size. At a proven win subsequently labeled loss, gradient descent lowers the win logit and raises the loss logit. If the exact win were used instead, the gradient would be p−onehot(win). Their difference is **(1,0,−1)** in W/D/L order, independently of confidence (and divided by 32 within these batches). A draw-versus-proven-value contradiction similarly changes the two relevant class components.

For a discarded synthetic prediction p≈(.976,.0066,.0179), the loss-target gradient is approximately (.976,.0066,−.982), while the win-target gradient is (−.024,.0066,.0179). The exact numbers are illustrative, not research-model optimizer updates. A correctness test differentiates only discarded logits through the existing `training_loss`, with no model or optimizer, and confirms these gradient identities. There is no softmax-as-logits bug, label reversal or broken cross-entropy exposed here. The objective estimates eventual behavioral outcomes when supplied behavioral outcome labels; it does not automatically estimate optimal outcomes when the behavior can miss proven wins.

## One next controlled hypothesis

**Engine-proven value anchoring will improve learned tactical values when behavioral mistakes make eventual-outcome supervision contradict exact state values.** Contradictions affect 12–20% of immediate-win examples, 7–24% of conservatively proven losses, both actors and multiple trained cohorts. The noisy run improved winning-action discovery while having the highest contradiction rate and 1,384 contradictory optimizer appearances. This is material enough to test target semantics before changing the value head or directly optimizing a scalar.

Recommended future A/B contract: fresh matched initial weights/seeds and equal budgets; control retains eventual outcomes; treatment substitutes **+1 or −1 only when this conservative engine proof succeeds**, otherwise retaining the eventual actor-relative outcome. Use no diagnostic fixture labels or tactical action guards. Keep architecture, three-class value loss and head weights, policy targets, PUCT, 32 simulations, temperature 1, optimizer, augmentation and root noise fixed to the chosen 4D.2d parent configuration (augmentation .5, ε=.25, α=.30). Both runs start fresh from matched weights and use the same exploration configuration.

Predeclare independent exact-proof +1/−1 value scores and scalar errors as the primary evaluation, alongside actual-outcome prediction scores reported separately; inspect both actors and mirrors. Retain the existing matched playing-strength protocol as a secondary outcome in that separately authorized experiment. Label contradictions becoming zero on the anchored subset is tautological and must not be treated as evidence of better predictions or strength. Ordinary blocks remain unanchored. This hypothesis does not solve deeper unknown-state targets or missing exact draw supervision. No second head/loss experiment is recommended concurrently, and no anchoring implementation or training occurred here.

## Verification and handoff

The new tests cover both actors/reflections, immediate wins, conservative losses with all reply witnesses, safe ordinary blocks, draw escapes, exact completed win/loss/draw replay, corruption rejection, no input mutation/output overwrite, exact sampling counts, score separation, strict JSON and discarded-logit loss gradients. Existing neural/backend/API-isolation regressions were run with live provider requests blocked by the existing test fixture. Those existing regressions include discarded synthetic learning tests; no research-model weights or optimizer were loaded or updated.

| Gate | Result |
| --- | --- |
| neural-tests.log | 293 passed in 59.97s |
| backend-tests.log | 387 passed, 12 skipped in 7.44s |
| api-isolation-tests.log | 2 passed in 0.42s |
| Focused audit tests | 34 passed in 0.89s |
| Whitespace | git diff --check passed |
| Pre-existing evidence | 347/347 SHA-256 unchanged |

Reproduce the analysis into a **new** destination (the command refuses existing outputs and historical input-directory overlap):

```sh
/tmp/board-game-phase4c1-venv/bin/python -m games.connect4.neural_value_target_audit \
  --output experiment-output/<new-audit-directory>/analysis.json
```

Evidence includes the complete analysis JSON, before/after preservation inventories, verified original-manifest/checkpoint references, test logs and a supplemental audit-only manifest. No old log, checkpoint, fixture, manifest or report was edited. Work stops at this audit and recommendation; no candidate or subsequent experiment was produced.
