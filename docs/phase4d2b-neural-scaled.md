# Phase 4D.2b — Scaled neural self-play baseline

Completed October 4, 2026. **The authorized replacement succeeded: 200 completed games, 3,673 applied/collected plies and 1,970 optimizer updates in 84.821 training seconds.** Training stopped at the game cap. The bounded opponent evaluation completed all 144 games in 7.667 seconds. There was one real training experiment, following one earlier CLI attempt that failed before training.

Additional experience improved some policy actions and known-win value scores, but **did not establish improved playing strength**. Raw MCTS tactical accuracy fell relative to the initial model on both the original probes and the larger frozen suites. Values remained asymmetric and frequently saturated. Mathematical correctness, successful optimization, tactical behavior and measured playing strength are separate conclusions below.

## Failure history, replacement authorization and reconciliation

The original branch was created from `5ab044b4e836dc374509cfe639b5e7a60dfe01f5`. Its initial verification gate passed, but its CLI launch failed while serializing the initial nested diagnostic snapshot: a NumPy comparison sum produced `numpy.int64`, which `json.dumps(..., allow_nan=False)` rejected. The failure happened **before opening `events.jsonl` or entering `run_training`: zero games, zero plies, zero optimizer updates**. It was a measurement serialization defect, not a mathematical or numerical training failure.

The repair converts each saturation comparison to a native boolean before summation. Full nested snapshot and synthetic evaluation JSON serialization were added to verification. No automatic retry occurred under the original restriction. The user then reviewed the failed handoff and explicitly authorized **one fresh replacement launch**, superseding the no-retry restriction for that launch only. No third attempt occurred.

For the replacement, fetched `origin/main`, now **`b892f14b060a101184c8cc88b12fef206d150e4e`**, which already contained the repaired implementation. Local `phase4d2b-neural-scaled` matched that commit with a clean working tree; no merge or source rewrite was needed. No unrelated local changes existed to reconcile. The commit was already present upstream when this authorization began; no commit, push, merge or deployment was performed during replacement work.

The [failed evidence directory](../experiment-output/phase4d2b-neural-scaled-seed42-20261004/) remains unchanged, including its original launch source patch, untracked source/test archive, traceback, failure report, integrity manifest and separately identified repaired-source archive. The exact prior handoff also survives as `handoff-before-replacement.md` in the new output directory. Original and replacement configurations, bounds, checkpoint/target contracts, measurement protocols and initial weight digests compare exactly equal.

## Preserved contracts and exact settings

Reviewed the [Phase 4D.1 foundation](phase4d1-neural-mcts-foundation.md), [Phase 4D.2 pilot](phase4d2-neural-training.md), shared neural contract, search, trainer and driver. No mathematical ambiguity required a change. Architecture, canonical encoding, current-player W/D/L perspective, alternating backup, existing PUCT and logits-based loss/visit targets remain unchanged.

| Setting | Value |
| --- | --- |
| Initialization | Fresh canonical Connect4Net; 326,026 CPU float32 parameters; no checkpoint initialization |
| Seeds | Python/NumPy/PyTorch 42; independent search and minibatch Random(42) |
| Search | 32 simulations; PUCT 1.41; temperature 1 throughout self-play |
| Disabled | Tactical guard, root noise, symmetry augmentation, temperature scheduling |
| Optimizer | One persistent Adam, learning rate 0.001; batch size 32 |
| Loss | Original equally weighted policy cross-entropy and actor-relative W/D/L cross-entropy |
| Sampling | Uniform without replacement within each batch from all completed examples; no eviction |
| Execution | CPU float32; one intra-op and inter-op thread; deterministic algorithms |
| Global limits | 200 completed games; 8,400 applied plies; 2,000 updates; 900 training seconds |
| Update schedule | Zero updates through the two-game smoke gate; up to ten after each subsequent completed game, checking all limits first |
| Diagnostics | Initial, after games 50/100/150, final; separate seeded diagnostic searches; no training feedback |
| Opponent evaluation | Initial/final, NN Only/raw MCTS, Random/Negamax depth 1/depth 2; combined 180-second deadline |

`Bounds()` preserves the original pilot ceilings; explicit `--profile scaled` selects the scaled bounds. The 2,000-update ceiling permits the complete schedule. The final game's update block is skipped at the game limit: games 3–199 yield **197 × 10 = 1,970 updates**. The run did not stop near game 103.

The sole replacement command was:

```sh
PYTHONHASHSEED=42 /tmp/board-game-phase4c1-venv/bin/python \
  -m games.connect4.neural_self_play --profile scaled \
  --output experiment-output/phase4d2b-neural-scaled-replacement1-seed42-20261004
```

This is a record of a completed launch, not authorization to repeat it. The directory was new and Git-ignored. No architecture, learning method, seed, setting or budget changed to compensate for the earlier failure.

## Serialization preflight and verification gate

One final serialization-only preflight initialized a discarded seed-42 model and serialized the complete fourteen-position initial diagnostic payload and nested snapshot containing those fourteen plus **252 frozen research positions**. Training entry points were blocked. It verified JSON round-trips with `allow_nan=False`, unchanged model tensors and unchanged process RNG states; no real self-play collection or optimizer update occurred. The native-integer repair was confirmed present. The preflight record and full verification logs are retained in the new evidence directory.

| Replacement gate | Result |
| --- | --- |
| Neural correctness + pilot/scaled driver tests | **215 passed**, 17.68 s |
| All six existing DQN regression modules | **466 passed**, 5.71 s |
| Backend `pytest -q tests` | **354 passed, 9 skipped**, 6.67 s |
| Explicit DQN + neural API import isolation | **2 passed**, 0.69 s |
| Serialization-only preflight | Passed; 14 initial + 266 nested rows; zero updates |
| `git diff --check` | Passed before launch and at handoff |

Neural tests cover `test_connect4_neural_mcts.py`, `test_neural_self_play.py`, and `test_neural_scaled.py`. DQN regressions cover the existing agent, experiment, diagnostics, symmetry, validation and replication modules. Backend skips are optional PyTorch-dependent modules, verified separately in the optional environment. The isolation tests ran in backend `.venv`; no production dependency was added. Existing HTTPS prohibition/mocks remained active in tests.

Focused synthetic coverage proves the 200-game/1,970-update schedule, unchanged pilot limits, smoke gate, update/ply/time/interruption first-limit stopping, partial-game quarantine, numerical abort, immutable diagnostics/model/RNG behavior, complete nested serialization, frozen fixture integrity, balanced matched evaluation settings, complete-history W/L/D accounting, illegal-action abort and deadline quarantine. No verification failure preceded the replacement launch.

## Actual collection, schedule and outcomes

| Measurement | Actual |
| --- | ---: |
| Started / completed games | 200 / 200 |
| Applied / collected plies | 3,673 / 3,673 |
| Optimizer updates | 1,970 |
| Training elapsed time | 84.821 s |
| End-to-end driver time, excluding imports | 100.074 s |
| Stop status / reason | `bounded_stop` / `max_games` |
| Partial games / numerical or mathematical failures | None / none |
| Smoke gate | Passed at two games, 34 labeled plies, zero updates |
| Game plies: mean / median / range | 18.365 / 18 / 7–41 |

The training timer includes collection, replay validation, event writes, optimization and the three intermediate measurement snapshots. Limits are cooperative: an in-flight operation may finish beyond a deadline, but new training operations cannot start afterward. No deadline was approached here. End-to-end time additionally includes initialization/source identity, initial/final diagnostics, checkpoint integrity and opponent evaluation.

The smoke games were the same O wins as the pilot, with 18 and 16 plies and empty Adam state. Every completed game passed pre-move observation/player replay, immutable capture, legal finite normalized policy, exact 32-visit accounting, disabled-guard and actor-relative actual-winner checks. Optimization occurred only between completed games. Snapshots ran after games **50/100/150 at 480/980/1,480 updates**; final was **200 games/1,970 updates**.

An independent audit found exact equality with the pilot's **first twenty move histories and first 170 loss/gradient/entropy records**. The scaled run additionally trained after game 20, whereas the pilot stopped there. At game 200, the final 28 examples were validated and collected but never used for optimization. The final update used the first 3,645 examples, after game 199.

| Self-play games | X wins | O wins | Draws |
| --- | ---: | ---: | ---: |
| 1–50 | 35 | 15 | 0 |
| 51–100 | 25 | 25 | 0 |
| 101–150 | 33 | 17 | 0 |
| 151–200 | 28 | 22 | 0 |
| Total | **121** | **79** | **0** |

| Captured actor | Win targets +1 | Draw targets 0 | Loss targets −1 | Total |
| --- | ---: | ---: | ---: | ---: |
| X | 1,124 | 0 | 773 | 1,897 |
| O | 773 | 0 | 1,003 | 1,776 |
| Both | **1,897** | **0** | **1,776** | **3,673** |

Winner IDs and W/D/L class IDs remain distinct. Zero draw coverage is observed experience, not a labeling failure; synthetic draw tests still pass. Self-play outcomes establish neither optimal outcomes nor independent opponent strength.

## Optimization and policy concentration

All 1,970 updates had finite gradients and finite post-update parameters. Gradient L2 norms ranged **0.474–9.887**. No clipping, added regularizer, head reweighting or optimizer replacement occurred. Losses are pre-update minibatch means, not held-out generalization losses.

| Measurement | Policy loss | Value loss | Combined | Target entropy, nats |
| --- | ---: | ---: | ---: | ---: |
| First update | 1.9454 | 1.0523 | 2.9977 | 1.9049 |
| First ten mean | 1.9384 | 0.8158 | 2.7542 | 1.8734 |
| Game 50, preceding ten mean | 1.1874 | 0.2823 | 1.4698 | 0.9555 |
| Game 100, preceding ten mean | 1.1624 | 0.2978 | 1.4602 | 0.9400 |
| Game 150, preceding ten mean | 1.2014 | 0.2689 | 1.4702 | 0.9583 |
| Final preceding ten mean, after game 199 | 1.1411 | 0.3443 | 1.4854 | 0.9096 |
| Last update | 1.3101 | 0.3422 | 1.6524 | 1.0805 |

Lower fitting losses demonstrate functioning optimization. Changing minibatches and targets prevent a controlled generalization interpretation. The final value fitting loss exceeds the twenty-game pilot's final-ten mean of 0.0961 as the collection becomes larger and more varied; that alone is not a regression in generalization.

Completed-game target entropy averaged **1.8740** in smoke games, **1.0903** in games 3–20, **0.8614** in games 21–50, **0.9057** in games 51–100, **0.9549** in games 101–150 and **0.9274** in games 151–200. Targets became more concentrated early, without a monotonic subsequent trend.

On the original fourteen probes, mean NN legal-policy entropy fell **1.9237 → 1.3285** and maximum legal probability rose **0.1492 → 0.4873**. Raw MCTS visit entropy fell **1.5450 → 0.8510**, and maximum visit probability rose **0.3839 → 0.6942**. The final modal MCTS action was column 3 on **11/14** probes. Across the larger suites, NN entropy fell to **1.3044/1.0493/1.1088**, and MCTS visit entropy to **0.8761/0.8191/0.8029** (original/validation/replication). Concentration is not tactical correctness.

## Value predictions, known-win scores and saturation

Full W/D/L distributions, scalar/root values, draw probabilities and concentration are preserved per position at every snapshot. On the original fourteen probes:

| Snapshot | Mean W / D / L | Mean maximum WDL probability | WDL max ≥ .99 | Known-win Brier / NLL |
| --- | --- | ---: | ---: | --- |
| Initial | .332272 / .301443 / .366285 | .3663 | 0/14 | .6708 / 1.1017 |
| Game 50 | .575860 / .000424 / .423716 | .8273 | 2/14 | .9025 / 1.5229 |
| Game 100 | .635282 / .000524 / .364194 | .7252 | 0/14 | .3959 / .5785 |
| Game 150 | .678306 / .000151 / .321543 | .7570 | 1/14 | .3445 / .4699 |
| Final | .763893 / .000782 / .235326 | .8106 | 1/14 | **.2103 / .2927** |

The known-win metrics concern **four immediate-win positions**, whose optimal mover-relative label is provably win. Brier is the unnormalized three-class squared-error sum; NLL is −log P(win), using natural logs. The twenty-game pilot scored **1.3552 / 2.4106** on these same four positions, with 3/14 saturated predictions. Additional experience improved this narrow score substantially; saturation was not monotonically increasing on the fourteen probes.

The larger suites provide broader, still previously inspected tactical coverage:

| Frozen suite | Known wins | Brier initial → final | NLL initial → final | Final mean max WDL | Final saturation | Final mean P(draw) |
| --- | ---: | --- | --- | ---: | --- | ---: |
| Original, 60 positions | 24 | .6712 → .3069 | 1.1023 → .4544 | .8520 | 20/60 | 1.846e−4 |
| Validation, 96 positions | 48 | .6708 → .2778 | 1.1018 → .8419 | .9453 | 58/96 | 4.366e−6 |
| Replication, 96 positions | 48 | .6711 → .3992 | 1.1022 → .9289 | .9290 | 52/96 | 3.202e−6 |

Across all 120 known wins, pooled Brier improved **.6710 → .3321**, NLL **1.1021 → .7992**. Nevertheless, **25/120** final known-win positions assigned greater probability to loss than win. The worst (`holdout_06_win_vertical_1` in the validation suite) predicted W/D/L approximately **.000001632 / 1.41e−21 / .9999983**, despite having an immediate win. Its NN/MCTS action happened to find the winning column 5: correct action and correct value are independent properties.

Across all 252 frozen positions, saturated counts were **0 → 132 → 99 → 91 → 130** at initial/50/100/150/final. Strong concentration persists rather than uniformly increasing. Draw suppression is consistent with zero training draw targets; it does not establish that draws are impossible. The final original X-win position predicts P(win)=.9447, but its mirror predicts .9995; the O-win position predicts .9128, while its mirror predicts only .3597.

These proper-score improvements on proven wins are **not evidence of general W/D/L calibration**. Forced-block positions do not have proven optimal outcome labels, and no blind outcome holdout or reliability curve was used. Online pre-move predictions scored against eventual self-play winners gave cohort Brier means **.6364/.3230/.7452/.6072/.5763/.6351** for games 1–2/3–20/21–50/51–100/101–150/151–200; changing behavior and positions prevent a controlled calibration comparison. The correct conclusion is narrower known-win improvement alongside persistent saturation, draw undercoverage and mirror inconsistency.

## NN Only, raw MCTS and tactical behavior

The original fixed probes are unchanged. NN Only uses the lowest-index legal policy maximum. Diagnostic MCTS uses 32 simulations, PUCT 1.41 and **temperature 1**, reporting both a sampled action and the lowest-index maximum-visit modal action. Engine tactical annotations are computed afterward and never filter moves; guards remain disabled.

Each cell below is **NN greedy / MCTS modal / MCTS temperature-1 sampled** correct tactics:

| Snapshot | Original probes, 8 tactics | Original frozen, 48 tactics | Validation, 96 tactics | Replication, 96 tactics |
| --- | --- | --- | --- | --- |
| Initial | 2 / 6 / 5 | 6 / 35 / 31 | 16 / 79 / 62 | 15 / 73 / 61 |
| Game 50 | 1 / 4 / 2 | 9 / 25 / 19 | 18 / 39 / 35 | 16 / 50 / 39 |
| Game 100 | 2 / 4 / 1 | 11 / 22 / 19 | 22 / 40 / 36 | 21 / 47 / 38 |
| Game 150 | 2 / 5 / 4 | 14 / 27 / 19 | 22 / 42 / 36 | 20 / 53 / 47 |
| Final | **3 / 3 / 3** | **10 / 28 / 24** | **18 / 47 / 39** | **21 / 48 / 42** |

Across the 240 labeled frozen tactical positions, NN accuracy improved **37/240 → 49/240**; MCTS modal accuracy fell **187/240 → 123/240**, sampled accuracy **154/240 → 105/240**. Search still outperformed the learned direct policy on those suites, but learned search was worse than initial search. Relative to the twenty-game pilot's fourteen probes, NN improved **1/8 → 3/8**, modal MCTS declined **5/8 → 3/8**, and sampled MCTS rose **2/8 → 3/8**. None establishes competitive strength.

All actions are zero-based physical columns; arrows are initial → final:

| Position | Required | NN greedy | MCTS modal | MCTS sampled |
| --- | --- | --- | --- | --- |
| Opening X | — | 6 → 3 | 1 → 3 | 1 → 3 |
| Opening O | — | 6 → 3 | 1 → 3 | 1 → 3 |
| Middle X | — | 6 → 2 | 1 → 3 | 1 → 2 |
| Middle O | — | 6 → 1 | 1 → 3 | 1 → 2 |
| Win X | 0 | 6 → 1 | 0 → 3 | 0 → 3 |
| Win O | 1 | 6 → 1 | 1 → 1 | 1 → 1 |
| Block X | 1 | 6 → 1 | 1 → 1 | 1 → 2 |
| Block O | 0 | 6 → 3 | 1 → 3 | 1 → 0 |
| Full column 0 | Legal only | 6 → 3 | 3 → 3 | 5 → 3 |
| Win X mirror | 6 | 6 → 3 | 6 → 3 | 6 → 3 |
| Win O mirror | 5 | 6 → 5 | 5 → 5 | 5 → 5 |
| Block X mirror | 5 | 6 → 2 | 1 → 3 | 1 → 3 |
| Block O mirror | 6 | 6 → 3 | 6 → 3 | 3 → 2 |
| Full column 6 | Legal only | 4 → 0 | 1 → 3 | 4 → 3 |

The final X-win search never visited winning column 0: root visits were **[0,1,1,26,4,0,0]**. Its mirror devoted **all 32 visits to column 3**, also missing the immediate win. The unchanged finite-budget PUCT contract allows a low-prior winning action to remain unvisited. This is a concrete learned-prior/value/search failure at the declared budget, not a reason to improvise a PUCT change. The final block-X modal action is correct, while temperature-1 sampling chooses column 2; exploration and evaluation must remain distinct.

## Mirrors and action concentration

Across the 126 explicit mirror pairs in the frozen suites, NN action reflection agreement rose **0/126 → 18/126**, MCTS modal agreement fell **74/126 → 43/126**, and sampled agreement fell **62/126 → 38/126**. The near-uniform initial network's small left/right policy differences already drove an asymmetric argmax; action matching alone is an incomplete equivariance measure.

| Frozen suite | Mean NN mirror-policy L1, initial → final | Mean MCTS visit-policy L1, initial → final | Mean WDL mirror L1, initial → final |
| --- | --- | --- | --- |
| Original | .0243 → 1.1862 | .2271 → .9625 | .00116 → .4840 |
| Validation | .0264 → 1.2967 | .2487 → 1.2422 | .00168 → .8634 |
| Replication | .0259 → 1.3358 | .2318 → 1.1628 | .00164 → .7872 |

L1 compares a mirror's distribution with the original policy reversed, or the original WDL distribution unchanged; its range is 0–2. Learned distributional asymmetry increased markedly. On the original five mirror pairs, final modal agreement is 4/5, but much of that reflects repeated center-column choices; the mirrored X-win pair is consistently wrong. Legal masking continued to give exactly zero probability to full columns.

## Bounded playing-strength evaluation

All **144/144** games completed in **7.667 seconds**, within one shared 180-second budget. No partial/deadline-limited games occurred. The protocol evaluated both models, both configurations, all three opponents and both sides; no extra opponent games or budget extension were added.

The six predeclared legal prefixes were `[3,2]`, `[3,4]`, `[0,3,2]`, `[6,3,4]`, `[2,4,3,2]`, `[4,2,3,4]`. Each model played each prefix as both X and O. Opponent seeds were `420202 + 2*opening_index + side`, and MCTS seeds `420203 + 2*opening_index + side`, matched across models. Random uses the existing agent with isolated per-game RNG; Negamax depth 1/2 uses the existing agent, fresh cache per game and existing tie order. Full histories, winners, move timing and W/L/D by opponent/model/configuration/side are preserved and replay-validated.

**Playing-strength MCTS used temperature zero**, with seeded maximum-visit ties; this is distinct from exploratory self-play and sampled diagnostics. There was no tactical guard, root noise, augmentation or learning. NN Only used legal greedy actions.

Each cell is **W/L/D**, six completed games per side:

| Configuration | Opponent | Initial X | Initial O | Final X | Final O |
| --- | --- | --- | --- | --- | --- |
| NN Only | Random | 4/2/0 | 3/3/0 | 3/3/0 | 2/4/0 |
| NN Only | Negamax depth 1 | 4/2/0 | 1/5/0 | 1/5/0 | 0/6/0 |
| NN Only | Negamax depth 2 | 0/6/0 | 0/6/0 | 0/6/0 | 0/6/0 |
| Raw NN + MCTS | Random | 6/0/0 | 5/1/0 | 6/0/0 | 5/1/0 |
| Raw NN + MCTS | Negamax depth 1 | 1/5/0 | 0/6/0 | 1/5/0 | 2/4/0 |
| Raw NN + MCTS | Negamax depth 2 | 1/5/0 | 0/6/0 | 0/6/0 | 0/6/0 |

Across these opponents, NN Only went **12/24/0 → 6/30/0**; raw MCTS **13/23/0 → 14/22/0**. Raw MCTS retained 11/12 against Random, gained two wins against depth 1 and lost its sole depth-2 win. These small paired-opening results do not establish a reliable improvement or competitive strength. The pilot had no matching opponent evaluation, so no pilot-versus-scaled opponent-strength claim is possible.

## Resources and whole-search timing

Native macOS ARM64, Python 3.11.17, torch 2.10.0, NumPy 1.26.4; existing optional environment reused. Memory is process `ru_maxrss` converted from macOS bytes to MiB, not current RSS or system usage.

| Measurement | Actual |
| --- | ---: |
| Self-play whole searches | 3,673, each 32 simulations |
| Mean / median whole search | 12.338 / 13.291 ms |
| p95, nearest rank | 14.040 ms |
| Minimum / maximum | .762 / 61.664 ms |
| Total self-play whole-search time | 45.317 s |
| Evaluation NN Only model turns: mean / p95 | .274 / .296 ms |
| Evaluation raw MCTS model turns: mean / p95 | 12.038 / 13.819 ms |
| Fresh initialization | 4.491 ms |
| Checkpoint writing / safe loading | 4.666 / 2.712 ms |
| Initial / final maximum process peak memory | 268.063 / 307.203 MiB |

Self-play search timers include root initialization/copying, all traversals, leaf evaluation, expansion, backup and sampling; they exclude the separately measured pre-search prediction, capture validation, training and event writing. Evaluation move timers measure each complete model action call. Per-ply/per-update memory and timings remain in `events.jsonl`; trees are released after immutable capture. These measurements do not establish serving concurrency, long-lived stability or performance on other hardware.

## Checkpoint, source provenance and preservation

[Replacement evidence](../experiment-output/phase4d2b-neural-scaled-replacement1-seed42-20261004/) contains configuration/runtime, clean source identity at `b892f14`, an empty hashed source patch, initial/final RNG states, complete events/report, five nested snapshots, original initial/final diagnostic files, complete opponent evaluation, launch log, serialization preflight, verification logs, analysis, prior handoff and preservation records. External audit files supplement the driver's original manifest without replacing it. Full RNG information aids reproduction; there is no exact interrupted training-resume artifact.

The initial weight-byte digest matches the pilot and failed launch: `559620773060542d4569b25eb92f2e60c862f44a45299dca00c666cef9d4033e`. The run initialized once from scratch, not from that pilot checkpoint. The source and configuration were not changed between verification and launch; only this handoff was updated afterward.

The new [candidate.pt](../experiment-output/phase4d2b-neural-scaled-replacement1-seed42-20261004/candidate.pt) is **1,309,427 bytes**, SHA-256:

```text
d16cb516775146084bc2438c0166d241c1224efc69b2aa7e1b40596de77d1521
```

It contains exactly `contract` and `model_state_dict`, with format version 1, unchanged architecture, canonical `connect4-current-player-1x6x7-v1`, physical columns 0–6, `[win,draw,loss]`, current-player perspective and logits outputs. All 16 state tensors are finite CPU float32. Optimizer, replay, RNG and experiment metadata are external, not embedded in the inference envelope.

The driver safely reloaded with `map_location="cpu", weights_only=True`: **every weight tensor, logit and prediction matched exactly** on all fourteen fixed positions. Independent post-run safe loading verified the minimal schema, dtype/device/finiteness and exact saved prediction equality on the **fourteen original and all 252 supplemental positions**. All fifteen files in the original driver manifest were independently hash-verified. The supplemental audit manifest additionally covers the launch, preflight, preservation, verification and final handoff evidence.

All **195 pre-existing artifact/evidence files** remain unchanged: the original **181 historical files** plus **14 failed-launch files**, including their manifest. The ten historical neural checkpoints and all DQN experiments/checkpoints remain untouched. The Phase 4D.2 candidate still hashes to `ddca8f8ec49bd3cf74a6ddb05c6c7c0b4a2068ffe3169ac227dc2ad40435fe54`. No checkpoint was promoted or integrated into production; no DQN, API/UI, dependency, hosting or provider change occurred.

## Interpretation and recommended next learning-method change

- **Mathematical correctness:** unchanged contracts and synthetic tests passed; no invariant failure occurred during this run.
- **Successful optimization:** the persistent optimizer performed 1,970 finite updates and lowered fitting losses; this does not demonstrate tactical or strategic improvement.
- **Tactical behavior:** direct NN actions improved modestly, known-win value proper scores improved, but raw MCTS tactical accuracy regressed and learned mirror distributions became strongly asymmetric.
- **Measured playing strength:** this small balanced evaluation found no consistent gain; NN Only declined, raw MCTS results were mixed, and depth 2 remained dominant.

More experience alone did not remedy the pilot's central weaknesses under this learning method. The next separately authorized learning-method change should be **horizontal symmetry augmentation of completed canonical examples with policy columns reversed**, preserving W/D/L labels and the current contracts. This directly tests the measured distributional mirror asymmetry. Keep architecture, search budget, noise and temperatures fixed so its contribution can be isolated; do not train on these diagnostic labels or adaptively select against these previously inspected suites.

Draw coverage, wrong saturated value predictions and finite-budget search failures still require later investigation; symmetry alone is not promised to solve them. A future strength/calibration acceptance protocol needs more independent games and outcome coverage. No further training, inference campaign, method change, promotion, commit, push, merge or deployment is authorized by this completed baseline. Work stops here for review.
