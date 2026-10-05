# Phase 4D.2f — Controlled engine-proven value anchoring

Completed October 5, 2026. **Exactly one fresh seed-42 treatment completed: 200 games, 3,583 collected plies, 1,970 updates and 132.988 training seconds.** The predeclared exact-value primary score improved overall, driven by proven −1 positions; proven +1 prediction regressed, particularly for O. Forced-block search improved, but missed wins, some mirror measures and NN Only playing strength regressed. This is partial support for the hypothesis, not a uniformly better candidate.

## Preflight, authoritative proof and optimizer boundary

Fetched `origin/main`; both it and local HEAD matched expected `295525671bd7cfe986c960634bfbaf32f101a473`. Started with a clean tree and created local branch `phase4d2f-neural-value-anchoring`. Read the 4D.2d/2e handoffs and all requested implementation files. The saved control was verified before launch and at handoff at SHA-256 `346435b1c22fccefc69fae103cd9cc96d343a266a9d1d0c49976290869b28ed7`. It was used only for inference, never retrained or used as initialization.

The proof implementation was extracted verbatim into [`tactical_value.py`](../games/connect4/tactical_value.py), imported by audit and anchoring. +1 requires an immediate winning move. −1 requires no current immediate win and a nonterminal successor with at least one immediate opponent winning reply for every legal current action. A terminal draw successor or any safe successor defeats −1; ordinary forced blocks remain unknown. No deeper solver was added. Before training, the entire rebuilt 4D.2e analysis matched retained bytes exactly: 10,608 positions, 1,352 proven positions, 212 contradictions and all successor witnesses. The focused regression also reconstructs and checks every retained board.

[`ValueTargetAnchorer`](../games/connect4/train_mcts_nn.py) inverts immutable canonical observations using the explicit actor, checking cells, dimensions, gravity, physical piece counts, turn and nonterminal status. Completed replay supplies legal-history provenance. Minibatches follow base index selection → deterministic anchoring → unchanged probability-.5 symmetry decisions. Only a temporary optimizer outcome changes, using a validated replacement when needed. The original observations, actor, metadata and policy targets retain their contracts; stored completed outcomes remain actual actor-relative eventual results. Aligned proofs count as anchored but do not numerically change a target. Unknowns retain behavioral outcomes.

Every appearance records `behavioral_outcome`, `proven_value` (or null), `training_outcome`, `anchored` and `contradiction`. No RNG is consumed. Tests prove anchoring commutes semantically with reflection; post-run replay verifies proof-value consistency on all 3,583 stored states and reflections. Proofs are never injected into action selection, noise, leaf evaluation, PUCT or backup.

## Gates before the sole launch

| Gate | Result |
| --- | --- |
| Established neural regressions plus initial anchoring tests | 339 passed, 63.13 s |
| Final focused anchoring suite, including scripted driver integration | 47 passed, 9.88 s |
| Six established DQN suites | 466 passed, 5.44 s |
| Backend `.venv` suite | 387 passed, 29 optional skips, 7.23 s |
| Both API import-isolation gates in `.venv` | 2 passed, 0.69 s |
| Serialization / fresh seed-42 initialization | Passed; zero training calls or optimizer updates |
| Complete historical audit regeneration | Byte-identical before training |
| `git diff --check` | Passed before launch and at handoff |

Neural/DQN checks used existing `/tmp/board-game-phase4c1-venv`; backend/isolation used `.venv`. Optional neural/DQN dependencies were covered in their configured environment. No correctly configured pre-launch gate failed. A supplemental integration development assertion initially assumed the standard scripted draw history contained provable positions; it contains none. Replacing that unsuitable fixture with retained 4D.2c game 73 (11 proofs) passed without implementation changes. The failed log and explanation remain in preflight evidence. Tests cover both actors, +1/−1 proofs, ordinary blocks, draw escape, reconstruction, immutable behavioral collection, value-only replacement, exact disabled tensors, aligned/contradictory labels, reflection, RNG isolation, strict JSON/events and all retained witnesses.

## Frozen primary evaluation set

Before training or either learned checkpoint’s evaluation on this set, froze **128 rows / 64 base situations**: 32 rows per actor/value cell, 64 proven +1 and 64 proven −1, with explicit mirrors. Seed 420404; predeclared target 16 bases per actor/value, cap 200,000 random prefixes, largest balanced fallback if needed. All quotas were achieved without fallback. A standalone drop-and-four-cell-window scanner independently validates proofs and complete reply witnesses against actual engine successors; every history legally replays. All existing frozen/blind suites, original probes and their board-plus-actor reflections were excluded.

File: [`exact-value-blind.json`](../experiment-output/phase4d2f-neural-value-anchoring-preflight-20261005/exact-value-blind.json). SHA-256 **`53a8bf0d9c99d16e165a08505cd651545a73f18dd4db195b51cd599dca3b89b0`**. The frozen file supplied no training examples, tuning, stopping or model-selection decisions; the trainer never reads it.

**Incidental overlap limitation:** treatment game 129, ply 8 independently reached `exact_23_mirror`, a proven +1 O state. That source example appeared in 11 optimizer batches; augmentation can expose its mirror. Thus one direct frozen row, and its reflected equivalent, were encountered during normal self-play. The control collection had no direct/reflected overlap. We retained all 128 frozen primary rows and bytes. A separately labeled post hoc sensitivity excludes the entire pair, leaving 126 rows (62 +1, 64 −1), for both checkpoints. It preserves the conclusion: aggregate Brier 0.7393→0.5656; +1 0.5701→0.6569 worsens; −1 0.9031→0.4772 improves. This collision limits a strict fully unseen interpretation; no replacement set or training run was created.

## Primary: exact-proof value prediction

A = saved 4D.2d root-noise checkpoint; B = fresh anchored checkpoint. Brier is the sum of three class squared errors, NLL uses natural logs without clipping, scalar prediction is P(win)−P(loss). Opposite sign means scalar × exact value < 0. Saturation means maximum W/D/L probability ≥ .99. Mirror discrepancy is mean WDL L1 per explicit pair. All target probabilities were positive; no infinite NLL handling was needed.

| Proof | Model | N | Brier | NLL | MAE | MSE | Class accuracy | Opposite sign | Saturated | Mirror WDL L1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| +1 | A | 64 | 0.5523 | 1.0047 | 0.7612 | 1.1047 | 62.50% | 37.50% | 25.00% | 0.5794 |
| +1 | B | 64 | 0.6376 | 1.3296 | 0.8350 | 1.2753 | 57.81% | 42.19% | 17.19% | 0.4327 |
| −1 | A | 64 | 0.9031 | 1.8121 | 1.0712 | 1.8063 | 46.88% | 53.12% | 28.12% | 0.3965 |
| −1 | B | 64 | 0.4772 | 0.9653 | 0.6880 | 0.9544 | 67.19% | 32.81% | 35.94% | 0.4953 |

| All 128 | Brier | NLL | MAE | MSE | Accuracy | Saturated | Mirror WDL L1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.7277 | 1.4084 | 0.9162 | 1.4555 | 54.69% | 26.56% | 0.4880 |
| B | 0.5574 | 1.1474 | 0.7615 | 1.1148 | 62.50% | 26.56% | 0.4640 |

| Proof / actor | N | Brier A → B | NLL A → B | Accuracy A → B |
| --- | --- | --- | --- | --- |
| 1 / X | 32 | 0.7417 → 0.4683 | 1.4818 → 0.9051 | 53.12% → 71.88% |
| 1 / O | 32 | 0.3630 → 0.8070 | 0.5276 → 1.7541 | 71.88% → 43.75% |
| -1 / X | 32 | 0.8790 → 0.4753 | 1.6599 → 0.9053 | 43.75% → 68.75% |
| -1 / O | 32 | 0.9273 → 0.4790 | 1.9643 → 1.0252 | 50.00% → 65.62% |

Overall Brier falls about 23.4%, but +1 and −1 do not both improve. Both actors improve on −1; X improves on +1 while O worsens sharply. +1 correct classifications fall 40→37/64; −1 rise 30→43/64. Correct-class saturation is +1 20.31%→9.38%, −1 15.63%→26.56%; total max-class saturation remains 34/128 for both checkpoints. +1 mirror value discrepancy improves, while −1 discrepancy worsens. These narrow correlated tactical measurements do not establish general calibration.

## Sole treatment and target accounting

The sole launch used a new Git-ignored directory and the following command. There was no retry, sweep, fresh control, extension or additional research training run.

```sh
PYTHONHASHSEED=42 /tmp/board-game-phase4c1-venv/bin/python -m games.connect4.neural_self_play --profile scaled --horizontal-symmetry-probability 0.5 --root-noise-epsilon 0.25 --root-dirichlet-alpha 0.30 --value-anchoring --evaluation-baseline experiment-output/phase4d2d-neural-root-noise-seed42-20261005/candidate.pt --output experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005
```

| Fixed setting | Value |
| --- | --- |
| Initialization | Fresh canonical Connect4Net, seed 42; CPU float32; 326,026 parameters |
| Runtime | Native ARM64; one intra-op/inter-op thread; deterministic algorithms |
| Bounds | 200 completed games / 8,400 plies / 2,000 updates / 900 training seconds |
| Self-play search | 32 simulations; PUCT 1.41; temperature 1; guard OFF |
| Root noise | epsilon .25; alpha .30; existing private domain-separated seed 42 |
| Symmetry | Probability .5; existing independently seeded stream |
| Optimizer / loss | Persistent Adam .001; batch 32; existing equal policy/value losses |
| Schedule | Two collection-only games; ten updates after games 3–199; none after game 200 |
| Only learning-method change | Temporary optimizer value outcomes on conservatively proven states |

Initial weight-byte SHA-256 `559620773060542d4569b25eb92f2e60c862f44a45299dca00c666cef9d4033e` exactly matches the parent. Configuration comparison has only the `value_anchoring` addition. Smoke histories/31 plies precede optimization. All 200 games completed without partial/quarantined episodes or invariant failures. Game 200 is collected and receives no later optimization. Loss snapshots are fitting diagnostics on changing batches, not model-selection criteria.

| Measurement | A: root noise | B: anchoring |
| --- | --- | --- |
| Completed games | 200 | 200 |
| Stored completed examples | 3322 | 3583 |
| Optimizer updates | 1970 | 1970 |
| Training seconds | 85.6644 | 132.9880 |
| X / O wins / draws | 108 / 92 / 0 | 113 / 87 / 0 |

| Denominator | Total | Proven | Aligned | Contradictory | Targets actually changed |
| --- | --- | --- | --- | --- | --- |
| Stored behavioral collection | 3583 | 418 | 371 | 47 | 0 |
| Sampled optimizer appearances | 63040 | 6653 | 5775 | 878 | 878 |

| Exact value | Stored proven / aligned / contradictory | Sampled proven / aligned | Sampled targets changed |
| --- | --- | --- | --- |
| 1 | 337 / 296 / 41 | 5744 / 4912 | 832 |
| -1 | 81 / 75 / 6 | 909 / 863 | 46 |

**878/63,040 = 1.39% of optimizer appearances actually changed target:** 832 behavioral losses became +1 and 46 behavioral wins became −1. All 6,653 exact-proof appearances are anchored; 5,775 aligned appearances are numerically unchanged. The 418 stored proofs remain labeled by behavior (47 contradict proofs); stored target changes are exactly zero. Different self-play populations mean the lower collection contradiction prevalence is not itself a controlled held-out result. Making optimizer contradictions zero on proven appearances is tautological and is not evidence of prediction quality.

| Labels / exposure | Behavioral W/D/L | Optimizer W/D/L | Draw before / after |
| --- | --- | --- | --- |
| Stored collection | 1848/0/1735 | Not rewritten | 0 / 0 |
| Sampled appearances | 32693/0/30347 | 33479/0/29561 | 0 / 0 |

| Grouping | Cell | Appearances | Proven | Changed | Behavioral W/D/L | Training W/D/L |
| --- | --- | --- | --- | --- | --- | --- |
| Actor | 0 | 32624 | 3950 | 609 | 19276/0/13348 | 19867/0/12757 |
| Actor | 1 | 30416 | 2703 | 269 | 13417/0/16999 | 13612/0/16804 |
| Stage | early | 47250 | 2413 | 458 | 24116/0/23134 | 24574/0/22676 |
| Stage | late | 921 | 458 | 0 | 553/0/368 | 553/0/368 |
| Stage | middle | 14869 | 3782 | 420 | 8024/0/6845 | 8352/0/6517 |

| Source games | Stored | Stored proven | Stored contradictions | Sampled | Sampled proven | Changed | Behavioral → training W/D/L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-2 | 31 | 2 | 0 | 2725 | 173 | 0 | 1398/0/1327 → 1398/0/1327 |
| 3-20 | 292 | 36 | 9 | 16544 | 2017 | 466 | 8630/0/7914 → 9096/0/7448 |
| 21-50 | 571 | 52 | 6 | 18223 | 1586 | 197 | 9374/0/8849 → 9571/0/8652 |
| 51-100 | 859 | 100 | 3 | 15757 | 1774 | 48 | 8207/0/7550 → 8255/0/7502 |
| 101-150 | 897 | 108 | 15 | 7503 | 832 | 108 | 3913/0/3590 → 3971/0/3532 |
| 151-200 | 933 | 120 | 14 | 2288 | 271 | 59 | 1171/0/1117 → 1188/0/1100 |

Early/middle/late are pre-move occupied-cell cutoffs 0–13 / 14–27 / 28–41, matching 4D.2e. [`analysis.json`](../experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/analysis.json) retains exact per-game, actor, source cohort, source-cohort×actor, stage and update-cohort counts, with separate behavioral/training distributions and draw counts. Repeated optimizer appearances are not independent new positions.

## Behavioral-outcome prediction, kept separate

Contemporaneous pre-search probabilities are scored against each run’s actual eventual behavioral outcomes. Final checkpoints are additionally scored on their own retained collections, explicitly **in-sample**. Runs encounter different evolving positions, so own-run differences are descriptive rather than a same-position causal comparison. Neither measurement is general calibration or an optimal-value test.

| Measurement | Model | N | Brier | NLL | MAE | MSE | Outcome-class accuracy | Mean P(draw) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contemporaneous | A | 3322 | 0.7182 | 1.2770 | 1.0270 | 1.4326 | 48.83% | 0.0045 |
| contemporaneous | B | 3583 | 0.6261 | 1.1044 | 0.9274 | 1.2487 | 55.23% | 0.0042 |
| final_checkpoint | A | 3322 | 0.2239 | 0.3396 | 0.4568 | 0.4478 | 81.52% | 0.0003 |
| final_checkpoint | B | 3583 | 0.2512 | 0.4010 | 0.4981 | 0.5025 | 80.97% | 0.0001 |

Contemporaneous behavioral prediction improves, while final own-collection fitting worsens (Brier .2239→.2512; NLL .3396→.4010). No completed self-play draw occurred in either run, so collected and sampled draw coverage are both zero, with zero updates containing a draw target. No prediction chooses draw. Mean final P(draw) on own collections is approximately 2.51e−4→8.15e−5; on the primary exact set 6.43e−6→1.27e−5; on pooled prior fixtures 1.83e−5→4.36e−5. Draw suppression persists despite small fixture increases. There is no draw-outcome prediction score to estimate, and no exact-draw supervision was introduced.

## Noise-free policy and search evaluation

All evaluation search uses root noise OFF, raw 32-simulation PUCT 1.41 and guard OFF. Diagnostic samples use temperature 1; modal is the lowest-index maximum-visit action. NN Only is the lowest-index legal-policy maximum. Existing suites remain frozen, including both earlier blind sets. Nothing here guides moves or training.

| Suite | Tactical rows | A: NN / modal / sampled correct | B: NN / modal / sampled correct |
| --- | --- | --- | --- |
| original_probes | 8 | 6/8/7 | 6/7/7 |
| diagnostic | 48 | 19/35/25 | 24/37/35 |
| validation | 96 | 31/62/49 | 34/66/56 |
| replication | 96 | 26/67/59 | 29/68/60 |
| pooled_prior | 240 | 76/164/133 | 87/171/151 |
| phase4d2c_blind | 96 | 30/60/51 | 38/75/57 |
| phase4d2d_blind | 96 | 29/61/55 | 40/73/62 |

| Suite | Win modal A → B | Block modal A → B | Winning actions with zero visits | Max visit share A → B | Visit entropy A → B |
| --- | --- | --- | --- | --- | --- |
| original_probes | 4 → 3 / 4 | 4 → 4 / 4 | 0 → 1 / 4 | 0.6362 → 0.7500 | 1.0321 → 0.6586 |
| diagnostic | 20 → 20 / 24 | 15 → 17 / 24 | 2 → 3 / 24 | 0.6318 → 0.7198 | 1.0132 → 0.8200 |
| validation | 37 → 38 / 48 | 25 → 28 / 48 | 9 → 8 / 48 | 0.6891 → 0.6768 | 0.8365 → 0.8481 |
| replication | 41 → 38 / 48 | 26 → 30 / 48 | 4 → 6 / 48 | 0.6751 → 0.7119 | 0.8781 → 0.8084 |
| pooled_prior | 98 → 96 / 120 | 66 → 75 / 120 | 15 → 17 / 120 | 0.6701 → 0.7004 | 0.8944 → 0.8263 |
| phase4d2c_blind | 39 → 41 / 48 | 21 → 34 / 48 | 7 → 6 / 48 | 0.6663 → 0.7194 | 0.8986 → 0.7794 |
| phase4d2d_blind | 42 → 41 / 48 | 19 → 32 / 48 | 6 → 3 / 48 | 0.6963 → 0.7217 | 0.8356 → 0.8114 |

Pooled modal tactics improve 164→171/240, NN 76→87, sampled 133→151. Both earlier blind suites improve substantially in modal accuracy (60→75 and 61→73/96), driven mainly by blocks: 21→34 and 19→32/48. Pooled blocks improve 66→75/120. However pooled immediate-win modal correctness declines 98→96/120 and zero-visit wins rise 15→17/120. Original probes lose one immediate win. Better blocking is therefore accompanied by a missed-win regression, rather than universal tactical improvement. Root action concentration generally rises; lower entropy alone is not evidence of stronger play.

| Suite | Mirror pairs | NN policy L1 | Visit-policy L1 | WDL L1 | NN action agreement | Modal action agreement |
| --- | --- | --- | --- | --- | --- | --- |
| pooled_prior | 126 | 0.5334 → 0.5667 | 0.5739 → 0.5367 | 0.5318 → 0.4142 | 72 → 66 | 85 → 85 |
| phase4d2c_blind | 48 | 0.5671 → 0.5319 | 0.5352 → 0.5026 | 0.4690 → 0.3288 | 23 → 32 | 32 → 34 |
| phase4d2d_blind | 48 | 0.5059 → 0.6347 | 0.4557 → 0.5247 | 0.4290 → 0.5462 | 28 → 23 | 38 → 33 |

Mirror consistency is mixed: pooled NN policy L1 worsens while visit/value L1 improve; 4D.2d blind NN policy, visit-policy and WDL discrepancies worsen and modal mirror agreement falls 38→33/48. Policy targets and losses were unchanged, but shared learned features and changed future self-play can still produce policy/search regressions.

## Matched 144-game playing-strength campaign

The unchanged combined protocol completed all 144 games in 8.341 seconds: each checkpoint, NN Only/raw MCTS, Random/Negamax depth 1/depth 2, six matched openings and both sides. Raw MCTS uses temperature zero, 32 simulations, PUCT 1.41, guard OFF and root noise OFF. Each table cell below has 12 games (six per side). All 72 control histories/results exactly reproduce the parent’s saved final-model campaign. No opponent budget was extended.

| Mode | Opponent | A W/D/L | B W/D/L |
| --- | --- | --- | --- |
| nn_only | random | 11/0/1 | 10/0/2 |
| nn_only | negamax1 | 3/0/9 | 1/0/11 |
| nn_only | negamax2 | 0/0/12 | 0/0/12 |
| raw_mcts | random | 11/0/1 | 12/0/0 |
| raw_mcts | negamax1 | 5/0/7 | 6/0/6 |
| raw_mcts | negamax2 | 1/0/11 | 1/0/11 |

Raw MCTS has a small depth-1 gain (5→6/12 wins), while depth-2 remains 1/12. NN Only depth-1 strength regresses 3→1/12; depth-2 remains 0/12. Across both modes, Negamax wins fall 9→8/48. Against Random, NN wins fall 11→10/12 and raw MCTS rises 11→12/12. Overall wins fall 31→30/72 per checkpoint. These small correlated campaigns do not establish a general strength gain. Exact side-specific results remain in the opponent JSON and analysis.

## Interpretation and next priority

1. **Did exact-proof value prediction improve?** Overall yes on the predeclared set: Brier .7277→.5574, NLL 1.4084→1.1474, scalar MSE 1.4555→1.1148; the overlap-pair sensitivity agrees.
2. **Did both signs improve?** No. Proven −1 improves for both actors. Proven +1 worsens overall, driven by O, despite X improvement.
3. **How many optimizer targets actually changed?** 878 of 63,040 appearances (1.39%), comprising 832 +1 changes and 46 −1 changes; 5,775 other proven appearances were already aligned.
4. **Behavioral-outcome prediction?** Contemporaneous own-run prediction improves; final own-collection in-sample prediction worsens. Neither measures general optimal-value calibration.
5. **Tactical MCTS?** Modal and sampled accuracy improve on pooled tactics and both earlier blind suites, but original-probe modal accuracy regresses.
6. **Forced blocks?** Improve: pooled 66→75/120, blind 21→34 and 19→32/48.
7. **Negamax playing strength?** Mixed: raw MCTS improves slightly at depth 1, depth 2 does not improve, and NN Only regresses; combined Negamax wins decline.
8. **Draw suppression?** Persists. Zero draw games/labels/appearances and very low draw probabilities; anchoring supplies no draw coverage.
9. **Policy/search regressions?** Yes: proven-win value scores, pooled immediate-win modal correctness and zero visits, selected mirror consistency, original probes and NN Only opponent strength.
10. **Next priority?** Draw/value coverage, including both actors and both proven signs, before a scalar-head change. The WDL loss is already mathematically correct; this run shows selective value improvement but leaves draw supervision absent and actor/sign coverage uneven. Further data/coverage work should be separately controlled; a temperature schedule or scalar-head experiment is not implemented here.

Only one training seed was used. Reflected rows are correlated, stored sibling positions and repeated appearances are correlated, and self-play trajectories diverge after changed targets. The balanced exact set is narrow and has an incidental encountered pair; the post hoc sensitivity is secondary, not a replacement primary. Do not treat these results as general calibration, reliable Elo or proof that anchoring is universally beneficial.

## Artifacts, preservation and handoff

Canonical inference checkpoint: [`candidate.pt`](../experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/candidate.pt), SHA-256 **`78276912bdef5f1b1b53ac6b2ec3ede2eed20a8ebb5796a9bdcbb57e4cb28c51`**. It contains only the existing contract and model state dict. `weights_only=True`, CPU reload verifies all float32 weights; the driver verifies exact tensor, logit and prediction equality. One candidate checkpoint was saved; discarded serialization preflight used a temporary file that was removed. Optimizer/resume state is not packaged.

Verified all 3,583 root Dirichlet draws/mixed priors and final PCG64 state, 1,970 base sampling draws and final state, 63,040 unchanged symmetry flags/final state, and 63,040 anchoring records. Reflection flags exactly match the parent (31,390 reflected / 31,650 retained). Python-global, NumPy-global and PyTorch RNG states are unchanged during training. Stored outcomes all agree with legally replayed actual winners. All 331 preflight-inventoried artifact files remain byte-identical; additionally all 347 entries in the retained 4D.2e evidence inventory remain unchanged, including prior reports, fixtures and research evidence.

The original driver [`manifest.json`](../experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/manifest.json) is preserved and verified. Supplementary manifests bind preflight, analysis and this report. Source patch and untracked source copies match tested launch source. A post-run read-only analysis initially stopped on `statistics.mean` receiving NumPy `bool_`; Python scalar conversion fixed the analysis without changing training/checkpoints. Both scripts and logs are retained. No model or fixture was tuned to the results.

Main evidence: [`analysis.json`](../experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/analysis.json), [`overlap-sensitivity.json`](../experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/overlap-sensitivity.json), [`events.jsonl`](../experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/events.jsonl), [`opponent-evaluation.json`](../experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/opponent-evaluation.json), [`verification.json`](../experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/verification.json), plus preflight audit regression, protocol, test logs, inventories and reproduction scripts. The preflight analysis script only reads checkpoints/events and refuses to overwrite its outputs.

No architecture/loss/weighting/class-balancing change, synthetic draw labels, search/temperature/guard change, DQN change, API/UI integration, hosting/deployment work or paid request occurred. No commit, push or merge was made. Work stops at this treatment experiment and handoff; the subsequent recommendation is not implemented.
