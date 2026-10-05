# Phase 4D.2d — Controlled root exploration noise

Completed October 5, 2026. **Exactly one fresh seed-42 experiment completed: 200 games, 3,322 applied/collected plies and 1,970 optimizer updates in 85.664 training seconds.** Root noise broadened self-play exploration and reduced unvisited immediate wins. It did not establish an overall tactical improvement over symmetry-only: direct NN tactics regressed; modal MCTS was mixed, including a substantial decline on the existing blind set. The small matched opponent campaign showed modest Negamax gains, including one depth-2 win. Value and mirror results remained mixed. The candidate is a research artifact.

## Preflight and preservation

Fetched `origin/main`, matching expected **`05f96b948e70b0a708556e7b58c43e86c0656b8f`** and the clean local HEAD. Created local branch `phase4d2d-neural-root-noise` from that reference. Read the foundation, scaled and symmetry handoffs, neural agent, self-play driver, trainer, shared contract and evaluation implementation.

The supplied symmetry-only checkpoint `experiment-output/phase4d2c-neural-symmetry-seed42-20261005/candidate.pt` matched SHA-256 **`a70910085f1d8be5886d60ce36ee80877bc2a4de6104ac6311462477c5b2cde7`** before launch and afterward. It was loaded only into a separate read-only inference object. The experimental model started from the same fresh canonical seed-42 initialization, never a checkpoint.

All **274 prior artifact/evidence files** under `experiment-output/` and `models/` retain their inventoried preflight SHA-256, including earlier failed-run evidence, neural and DQN checkpoints, manifests and logs. No commit, push, merge, DQN change, public API/UI integration, dependency installation, paid request, hosting or deployment occurred.

## Root-noise contract and RNG ownership

[`RootDirichletNoise` and `MCTSNNAgent`](../games/connect4/agents/mcts_nn_agent.py) expose `root_noise_epsilon=0.0`, `root_dirichlet_alpha=0.30`, and `root_noise_seed=42`. Epsilon must be finite in [0,1]; alpha must be finite and positive. With epsilon zero, the normal priors are returned verbatim, no noise draw or extra prior normalization occurs, and existing child priors/search remain numerically identical.

Root expansion first follows the existing legal masking/normalization and optional permitted-action filtering. This experiment keeps tactical filtering OFF. One Dirichlet sample is then drawn over **only permitted legal root children, in ascending physical-column order**. For the experimental epsilon .25 and alpha .30:

```text
P_noisy = normalize(0.75 * P + 0.25 * Dirichlet(alpha=.30))
```

Illegal/excluded actions retain exact zero. Only root-child `prior_p` changes. Network outputs, below-root expansion, PUCT formula, simulation counting, terminal evaluation, alternating value backups, action temperature and visit-policy targets retain their prior contracts. There is no exploration floor or additional tactical rule. `SearchResult` explicitly records epsilon/alpha, pre-noise and used root priors, the seven-column noise sample and draw index. Immutable training targets remain the existing temperature-adjusted root visits; external event provenance records the search configuration that generated them.

The private `numpy.random.Generator(PCG64(...))` uses the integer derived from SHA-256 of **`connect4-self-play-root-dirichlet-v1:42`**. Seed digest:

```text
70f0245938e9326b15476afffbefeafa5c27f23d1d210dfecf0d7405c3d4f0d1
```

Config records domain, source seed, full derived integer as a decimal string, digest, NumPy version, bit generator and action order. Initial/final PCG64 states and every sample/mixed prior permit exact replay. Noise consumes none of the search tie/action RNG, base minibatch sampling RNG, symmetry RNG, Python-global RNG, NumPy-global RNG or PyTorch RNG. Search naturally takes different paths and therefore can consume different search draws; stream ownership remains independent.

The unchanged horizontal augmentation uses probability **0.5**, its original private `random.Random` stream, domain `connect4-completed-horizontal-symmetry-v1` and digest **`c1a46d134746ddc6d95e0e14aad0aee19081b5badafa7da77482de841b9c61f4`**. Its implementation, completed-example reflection, W/D/L labels and sampling semantics were not changed. Noise is the only additional learning-method change.

## Correctness gates before the sole launch

| Gate | Result |
| --- | --- |
| Neural foundation, driver, scaled, symmetry and root-noise tests | 259 passed, 56.74 s |
| Six existing DQN suites | 466 passed, 5.27 s |
| Backend `.venv` suite | 354 passed, 11 optional modules skipped, 6.63 s |
| Both API import-isolation tests | 2 passed, 0.40 s |
| Serialization-only preflight | 458 original/frozen/blind rows; zero training calls/updates; weights and RNGs unchanged |
| `git diff --check` | Passed before launch and at handoff |

Neural/DQN gates used existing `/tmp/board-game-phase4c1-venv`; backend/isolation used `.venv`. Optional skips were covered in the neural/DQN environment. No correctly configured pre-training gate failed. An initial development assertion compared below-root priors directly to `predict_legal` and omitted the existing second `legal_policy` normalization; two assertions failed on roundoff. Correcting that oracle required no search implementation change. The failed development result is recorded.

[Focused tests](../tests/test_neural_root_noise.py) use fake logits and independently seeded Dirichlet samples to establish exact disabled-path priors/whole-tree statistics, legal-only dimensionality including one-action and filtered roots, exact 75/25 mixing and normalization, finite nonnegative/zero-illegal support, fixed-seed/state replay, domain/stream ownership, no noise below root, no caller/network mutation, exact 32-visit accounting, unchanged temperatures and capture targets, reflection compatibility, and nested JSON provenance. Full synthetic diagnostic/144-game evaluation tests assert noise remains disabled. Existing foundation tests cover PUCT, terminal/backup and value semantics; no learned-strength correctness threshold was added.

## Frozen additional model-blind set

Before training and before querying either learned model on the new set, froze **96 positions** with seed **420403** using the existing model-blind procedure: 48 unique bases plus explicit mirrors; 12 bases per actor/category; 48 immediate wins and 48 unique forced blocks. Excluded the three prior suites, original fourteen probes, Phase 4D.2c blind set, and all their board-plus-actor reflections. Standalone four-cell drop/scanner labels were cross-checked against actual engine successors and tactical helpers; every legal history was replayed. Forced blocks have unique immediate-reply safety, not proven ultimate W/D/L labels.

File: [`blind-positions.json`](../experiment-output/phase4d2d-neural-root-noise-preflight-20261005/blind-positions.json). SHA-256 **`53775d757c268fc76cb51e429679693b26fa2e11bcf11918be84ba0db11986cc`**. The seed/exclusion parameters extend the existing freezer while its default procedure/output remains unchanged. No fixture was regenerated, tuned, trained on or used to select a checkpoint. Reflection pairs are correlated; 96 rows represent 48 base situations.

The Phase 4D.2c blind set was reused unchanged at SHA-256 `ba31268418872fb2b5d89879d48fa644cd8cb576db0b6fd2d80d1af41f3727d5`. Frozen research suites retain their previously recorded hashes.

## Single fresh experiment and accounting

The sole real training launch was:

```sh
PYTHONHASHSEED=42 /tmp/board-game-phase4c1-venv/bin/python \
  -m games.connect4.neural_self_play --profile scaled \
  --horizontal-symmetry-probability 0.5 \
  --root-noise-epsilon 0.25 --root-dirichlet-alpha 0.30 \
  --evaluation-baseline experiment-output/phase4d2c-neural-symmetry-seed42-20261005/candidate.pt \
  --output experiment-output/phase4d2d-neural-root-noise-seed42-20261005
```

This records a completed launch. The output directory was new and Git-ignored. There was no retry, sweep, extension or additional training run.

| Setting | Value |
| --- | --- |
| Initialization | Fresh canonical 326,026-parameter Connect4Net; CPU float32; seed 42 |
| Runtime | One intra-op/inter-op thread; deterministic algorithms; native ARM64 |
| Search | 32 simulations; PUCT 1.41; temperature 1 throughout self-play; guard OFF |
| Root noise | ε .25; α .30; private domain-separated seed 42 |
| Symmetry | Probability .5; unchanged minibatch augmentation and RNG |
| Optimizer | Persistent Adam .001; batch 32; unchanged equally weighted policy/value loss |
| Schedule | Two collection-only games; ten updates after games 3–199; none after final cap |
| Limits | 200 games / 8,400 plies / 2,000 updates / 900 training seconds; first reached stops |

| Measurement | Symmetry only | Symmetry + noise |
| --- | --- | --- |
| Games | 200 | 200 |
| Applied/collected plies | 3613 | 3322 |
| Updates | 1970 | 1970 |
| Training seconds | 85.30576683301479 | 85.6643747498747 |
| X wins | 133 | 108 |
| O wins | 66 | 92 |
| Draws | 1 | 0 |
| Smoke plies; zero updates | 34 | 31 |

Both runs stopped at `max_games`. The noise run had no partial episodes or invariant failures. Every ply passed pre-move replay, legality, immutable capture, actor-relative winner labeling and exact 32-visit checks. Its two smoke histories differ from symmetry-only because root noise operates from the first search, before any optimizer update. Smoke still supplied 31 examples; game 3 raised collection above batch size, preserving the 1,970-update schedule. Snapshots occurred after games 50/100/150 at 480/980/1,480 updates. Game 200 was collected but not optimized.

The initial weight-byte digest **`559620773060542d4569b25eb92f2e60c862f44a45299dca00c666cef9d4033e`** exactly matches Phase 4D.2c's seed-42 fresh initialization. All updates had finite gradients/parameters. Augmentation transformed **31,390** and retained **31,650** of **63,040** sample appearances, exactly the prior stream's flags/counts; appearances are repeated sampling, not new data. All 1,970 base-index draws and all 63,040 reflection flags replay exactly against the current collection sizes. Self-play produced no draws; draw targets were therefore absent in this run.

| Actor | Win +1 | Draw 0 | Loss −1 | Total |
| --- | --- | --- | --- | --- |
| X | 903 | 0 | 812 | 1715 |
| O | 812 | 0 | 795 | 1607 |

| Snapshot games / updates | Policy loss | Value loss | Combined | Target entropy |
| --- | --- | --- | --- | --- |
| 50 / 480 | 1.6606 | 0.3929 | 2.0536 | 1.3154 |
| 100 / 980 | 1.6382 | 0.2817 | 1.9199 | 1.2897 |
| 150 / 1480 | 1.6336 | 0.3415 | 1.9751 | 1.3042 |
| 200 / 1970 | 1.5773 | 0.3756 | 1.9529 | 1.2181 |

These are fitting losses on changing sampled batches/targets, not held-out generalization scores.

## Exploration during self-play

Baseline exploration was reconstructed from saved Phase 4D.2c events, original raw predictions/legal-prior normalization and legally replayed histories; no historical model was trained or noisy baseline campaign run. Each row compares whole-run, search-weighted observations. Networks and visited positions evolve differently, so these are observed process differences rather than a same-position causal isolation. Entropies use natural logs; root shares use visits/32.

| Measurement | Symmetry only | Symmetry + noise |
| --- | --- | --- |
| Searches | 3613 | 3322 |
| Distinct visited actions, mean | 4.7210 | 5.8167 |
| Distinct visited actions, median | 5 | 7.0 |
| Every legal action visited | 47.33% | 60.75% |
| Visit entropy | 0.9792 | 1.2559 |
| Maximum-visit share | 0.6257 | 0.5295 |
| Neural legal-prior entropy before noise | 1.3525 | 1.6705 |
| Used/noisy-root-prior entropy | 1.3525 | 1.7175 |
| Zero-visit legal fraction, mean per search | 28.91% | 15.09% |
| Zero-visit legal fraction, pooled action opportunities | 29.37% | 15.09% |
| Known-win searches / winning actions | 384 / 481 | 335 / 390 |
| Winning actions with zero visits | 200 | 96 |
| Winning-action zero-visit fraction | 41.58% | 24.62% |
| Mean visits per immediate-winning action | 12.5738 | 17.5333 |
| Known-win searches with no winning action visited | 118 | 46 |
| Known-win modal tactical accuracy | 57.03% | 75.22% |

Multiple immediate-winning actions can occur in a self-play position. Per-action zero rates and mean visits count each winning action; the no-winning-action count is per search. Modal correctness uses the lowest-index maximum-visit action, independently of the sampled temperature-1 move.

| Game cohort | Visited actions, mean A → B | Every legal visited A → B | Visit entropy A → B | Zero legal fraction A → B |
| --- | --- | --- | --- | --- |
| smoke | 7.0000 → 6.7419 | 100.00% → 93.55% | 1.8740 → 1.7689 | 0.00% → 3.69% |
| 3-50 | 4.9073 → 5.8216 | 55.61% → 62.12% | 1.0444 → 1.2895 | 23.60% → 13.90% |
| 51-100 | 4.6887 → 5.7769 | 45.10% → 58.68% | 0.9773 → 1.2410 | 29.63% → 15.79% |
| 101-150 | 4.5308 → 5.7835 | 43.34% → 59.49% | 0.9390 → 1.2479 | 32.41% → 16.30% |
| 151-200 | 4.6886 → 5.8490 | 44.18% → 61.47% | 0.9292 → 1.2268 | 30.55% → 14.80% |

Breadth increased across trained cohorts, not in the near-uniform untrained smoke cohort: adding a spiky Dirichlet sample reduced its prior entropy. Whole-run prior entropy increased by .0470 within the noisy run, while the learned pre-noise priors also became broader than baseline. Higher entropy by itself is not evidence of improved strategy.

## Noise-free frozen and blind tactical evaluation

All diagnostics use the unchanged raw 32-simulation PUCT 1.41 search, guard OFF and **root noise OFF**. NN Only is the lowest-index maximum legal policy. Modal MCTS is lowest-index maximum root visits. Sampled diagnostics use temperature 1; playing-strength searches below use temperature zero. Tactical labels annotate after search and never filter actions. A means the saved Phase 4D.2c symmetry model; B means this run.

| Suite | Tactical rows | A: NN / modal / sampled | B: NN / modal / sampled |
| --- | --- | --- | --- |
| Original 14 probes | 8 | 4 / 6 / 5 | 6 / 8 / 7 |
| Original frozen 60 | 48 | 21 / 35 / 29 | 19 / 35 / 25 |
| Validation 96 | 96 | 32 / 59 / 53 | 31 / 62 / 49 |
| Replication 96 | 96 | 31 / 73 / 56 | 26 / 67 / 59 |
| Pooled prior 252 | 240 | 84 / 167 / 138 | 76 / 164 / 133 |
| 4D.2c blind 96 | 96 | 44 / 71 / 59 | 30 / 60 / 51 |
| New 4D.2d blind 96 | 96 | 31 / 58 / 53 | 29 / 61 / 55 |

Prior pooled modal accuracy was **167→164 / 240**, sampled **138→133**, direct NN **84→76**. Existing blind modal was **71→60 / 96**, sampled **59→51**, direct **44→30**. New blind modal improved **58→61 / 96**, sampled **53→55**, while direct declined **31→29**. Root noise therefore did not produce a consistent tactical gain beyond symmetry-only.

| Suite | Known wins | Zero-visit winning actions A → B | Zero fraction A → B | Mean winning visits A → B | Win modal correct A → B |
| --- | --- | --- | --- | --- | --- |
| Original 14 probes | 4 | 0 → 0 | 0.00% → 0.00% | 31.2500 → 30.2500 | 4 → 4 |
| Original frozen 60 | 24 | 5 → 2 | 20.83% → 8.33% | 22.0000 → 23.5000 | 19 → 20 |
| Validation 96 | 48 | 11 → 9 | 22.92% → 18.75% | 20.9792 → 22.0000 | 35 → 37 |
| Replication 96 | 48 | 6 → 4 | 12.50% → 8.33% | 21.3958 → 23.4583 | 38 → 41 |
| Pooled prior 252 | 120 | 22 → 15 | 18.33% → 12.50% | 21.3500 → 22.8833 | 92 → 98 |
| 4D.2c blind 96 | 48 | 8 → 7 | 16.67% → 14.58% | 22.4792 → 22.3542 | 37 → 39 |
| New 4D.2d blind 96 | 48 | 13 → 6 | 27.08% → 12.50% | 19.2708 → 23.8958 | 33 → 42 |

Every held-out known-win row has one immediate-winning action. On pooled prior tactics, immediate-win modal correctness rose **92→98 / 120**, but forced-block correctness fell **75→66 / 120**. On the old blind set, wins rose **37→39 / 48**, blocks fell **34→21 / 48**; new blind wins **33→42**, blocks **25→19**. Fewer missed wins did not guarantee better blocking decisions.

| Noise-free suite | Visited mean / median A → B | Every legal visited A → B | Visit entropy A → B | Max visit share A → B | Neural prior entropy A → B | Zero legal fraction A → B |
| --- | --- | --- | --- | --- | --- | --- |
| Pooled prior 252 | 4.3373 / 4 → 4.4444 / 4 | 36.11% → 33.73% | 0.8261 → 0.8944 | 0.7018 → 0.6701 | 1.2697 → 1.4322 | 33.72% → 32.05% |
| 4D.2c blind 96 | 3.9271 / 4 → 4.5104 / 4 | 35.42% → 39.58% | 0.7625 → 0.8986 | 0.7093 → 0.6663 | 1.1569 → 1.3976 | 39.29% → 30.75% |
| New 4D.2d blind 96 | 4.3229 / 4 → 4.1875 / 4 | 37.50% → 35.42% | 0.8286 → 0.8356 | 0.6947 → 0.6963 | 1.2391 → 1.3467 | 34.44% → 37.11% |

Used root-prior entropy equals neural-prior entropy in every evaluation because noise is disabled. Noise-free breadth did not improve uniformly: new-blind mean visited actions fell 4.323→4.188, despite fewer missed wins. Full suite-level and pooled legal-action denominators remain in `analysis.json`.

## Mirror policy, value and visits

| Suite / pairs | NN legal-policy L1 A → B | WDL L1 A → B | Visit-policy L1 A → B | NN / modal action agreement A → B |
| --- | --- | --- | --- | --- |
| Original frozen 60 / 30 | 0.4136 → 0.5051 | 0.4451 → 0.3885 | 0.4604 → 0.6479 | 23 / 23 → 19 / 17 |
| Validation 96 / 48 | 0.4659 → 0.5658 | 0.4401 → 0.5916 | 0.4453 → 0.5365 | 31 / 36 → 27 / 35 |
| Replication 96 / 48 | 0.5286 → 0.5188 | 0.4909 → 0.5615 | 0.5677 → 0.5651 | 29 / 30 → 26 / 33 |
| Pooled prior 252 / 126 | 0.4773 → 0.5334 | 0.4607 → 0.5318 | 0.4955 → 0.5739 | 83 / 89 → 72 / 85 |
| 4D.2c blind 96 / 48 | 0.4808 → 0.5671 | 0.4905 → 0.4690 | 0.5052 → 0.5352 | 33 / 33 → 23 / 32 |
| New 4D.2d blind 96 / 48 | 0.5391 → 0.5059 | 0.3792 → 0.4290 | 0.4453 → 0.4557 | 29 / 35 → 28 / 38 |

L1 compares the original policy reversed against its physical mirror; WDL is unchanged under reflection. Lower is better, range 0–2. Symmetry gains largely survive relative to the earlier nonaugmented learned model, but attenuate: pooled NN-policy L1 .4773→.5334, WDL .4607→.5318, visits .4955→.5739. The older nonaugmented reference was 1.2853/.7441/1.1453 (from Phase 4D.2c), so this run retains substantial improvement over it without adding another comparison campaign. Old-blind value L1 and new-blind NN-policy L1 improve, while other dimensions regress. Exact equivariance is not imposed.

## Known-win proper scores, value saturation and draw probability

Only immediate wins have proven mover-relative W/D/L labels. Brier is the unnormalized three-class squared-error sum; NLL is −log P(win), in nats. Forced-block rows are not outcome-calibration labels.

| Suite / known wins | Brier A → B | NLL A → B | Known wins favoring loss A → B |
| --- | --- | --- | --- |
| Original 14 probes / 4 | 0.3138 → 0.0088 | 0.4313 → 0.0601 | 1 → 0 |
| Original frozen 60 / 24 | 0.6847 → 0.7809 | 1.3855 → 1.4504 | 10 → 12 |
| Validation 96 / 48 | 0.9523 → 0.5705 | 2.1002 → 1.2515 | 26 → 16 |
| Replication 96 / 48 | 0.5719 → 0.6112 | 1.0693 → 1.2360 | 18 → 19 |
| Pooled prior 252 / 120 | 0.7467 → 0.6288 | 1.5449 → 1.2851 | 54 → 47 |
| 4D.2c blind 96 / 48 | 0.4433 → 0.6181 | 0.8965 → 1.3688 | 12 → 18 |
| New 4D.2d blind 96 / 48 | 0.5566 → 0.4725 | 1.0829 → 0.8397 | 16 → 13 |

| Suite | Mean W / D / L, A | Mean W / D / L, B | Mean max WDL A → B | Saturated ≥.99 A → B |
| --- | --- | --- | --- | --- |
| Pooled prior 252 | 0.477614 / 0.000927 / 0.521459 | 0.545556 / 0.000018 / 0.454425 | 0.8673 → 0.8573 | 67 → 59 |
| 4D.2c blind 96 | 0.564556 / 0.000265 / 0.435179 | 0.521987 / 0.000005 / 0.478008 | 0.8684 → 0.8937 | 28 → 21 |
| New 4D.2d blind 96 | 0.520976 / 0.000006 / 0.479018 | 0.581662 / 0.000003 / 0.418335 | 0.8859 → 0.8891 | 32 → 21 |

Pooled known-win Brier/NLL improved **.7467/1.5449→.6288/1.2851**; new blind **.5566/1.0829→.4725/.8397**. Old blind worsened **.4433/.8965→.6181/1.3688**. Pooled proven wins favoring loss fell 54→47 / 120, still substantial. Saturated counts declined, but the old blind mean maximum WDL rose .8684→.8937: fewer ≥.99 predictions does not ensure better calibration.

Pooled mean P(draw) fell **9.265e−4→1.835e−5**; old blind **2.646e−4→4.947e−6**; new blind **6.425e−6→2.710e−6**. The experimental collection had zero draw labels versus 42 in symmetry-only. Noise did not fix draw coverage or value reliability. These narrow proper scores do not establish general W/D/L calibration.

## Matched 144-game playing-strength evaluation

All 144 games completed in **9.539 seconds** within the original shared 180-second budget. `initial` aliases the read-only symmetry baseline and `final` the new candidate in the evaluation JSON; `snapshot-initial` still denotes the fresh untrained network. Reused six prefixes `[3,2]`, `[3,4]`, `[0,3,2]`, `[6,3,4]`, `[2,4,3,2]`, `[4,2,3,4]`; both sides; NN Only/raw MCTS; Random and existing Negamax depth 1/2; matched opponent seeds `420202 + 2*opening_index + side` and search seeds `420203 + 2*opening_index + side`. Fresh Negamax cache/tie behavior is unchanged.

Raw playing-strength MCTS uses **temperature zero**, guard/noise OFF, seeded maximum-visit ties and 32 simulations. NN Only uses legal greedy actions. Every history is legal/replay-validated. All **72 baseline histories/results exactly reproduce** Phase 4D.2c final-model evaluation. Each cell is W/L/D, six games per side.

| Mode | Opponent | A X | A O | B X | B O |
| --- | --- | --- | --- | --- | --- |
| nn_only | random | 6/0/0 | 5/1/0 | 6/0/0 | 5/1/0 |
| nn_only | negamax1 | 1/5/0 | 0/6/0 | 2/4/0 | 1/5/0 |
| nn_only | negamax2 | 0/6/0 | 0/6/0 | 0/6/0 | 0/6/0 |
| raw_mcts | random | 6/0/0 | 6/0/0 | 6/0/0 | 5/1/0 |
| raw_mcts | negamax1 | 3/3/0 | 1/4/1 | 3/3/0 | 2/4/0 |
| raw_mcts | negamax2 | 0/5/1 | 0/6/0 | 1/5/0 | 0/6/0 |

NN Only totals **12/24/0→14/22/0**: Random unchanged 11/1/0; depth 1 **1/11/0→3/9/0**; depth 2 unchanged 0/12/0. Raw MCTS totals **16/18/2→17/19/0**: Random **12/0/0→11/1/0**, depth 1 **4/7/1→5/7/0**, depth 2 **0/11/1→1/11/0**. Negamax gains are modest: one additional raw depth-1 win replaces a draw, and the depth-2 draw becomes the first win in this learned comparison. Noisy learning improved these small opponent results while aggregate tactical accuracy regressed. Twelve games per opponent/mode and one seed do not establish reliable superiority.

## Timing and memory

| Measurement | Actual |
| --- | --- |
| 3,322 self-play searches, mean / median / p95 | 13.236 / 13.350 / 16.396 ms |
| Whole-search min / max / total | 1.598 / 124.657 ms / 43.970 s |
| Training / end-to-end | 85.664 / 102.774 s |
| Initialization / checkpoint writing / safe loading | 4.413 / 4.070 / 2.206 ms |
| Initialization / maximum driver peak memory | 269.734 / 304.781 MiB |

| Evaluation model / mode | Mean / p95 whole action ms |
| --- | --- |
| initial/nn_only | 0.275 / 0.293 |
| initial/raw_mcts | 11.670 / 13.925 |
| final/nn_only | 0.272 / 0.287 |
| final/raw_mcts | 11.871 / 13.821 |

Native macOS ARM64, Python 3.11.17, torch 2.10.0, NumPy 1.26.4. Whole-search timers include root copying/initialization, noise, traversals, leaf inference, expansion, backup and action sampling; they exclude separate pre-search predictions, tactical annotations, capture validation and optimizer work. Training time includes replay, logging, annotations, updates and intermediate snapshots; driver end-to-end adds initial/final diagnostics, checkpoint checks and opponent games, excluding imports. Bounds are cooperative; none approached a deadline. `ru_maxrss` is process peak, not current/system memory. These are local research timings, not serving/concurrency claims.

## Canonical artifact, replay and evidence integrity

One canonical [`candidate.pt`](../experiment-output/phase4d2d-neural-root-noise-seed42-20261005/candidate.pt), **1,309,427 bytes**, SHA-256:

```text
346435b1c22fccefc69fae103cd9cc96d343a266a9d1d0c49976290869b28ed7
```

The unchanged minimal safe contract contains exactly `contract` and `model_state_dict`: format 1, canonical encoding/architecture, physical columns, current-player `[win,draw,loss]` perspective and logits. All 16 tensors are finite CPU float32. Safe reload uses `map_location="cpu", weights_only=True`. The driver proved every tensor/logit/prediction exactly equal to its in-memory trained model on fourteen original probes. Independent safe-loading checks matched saved raw-policy/WDL predictions on **916 model-position checks** (458 positions per model: 266 original/frozen plus both 96-row blind sets). All 96 saved baseline blind NN predictions, visit counts, root values and actions reproduced exactly.

All **3,322 Dirichlet draws and mixed priors replay exactly**, including illegal zeros, draw indices and final PCG64 state. All **63,040 augmentation flags**, final symmetry state, all 1,970 base minibatch-index lists and final sampling state replay exactly. Python-global, NumPy-global and PyTorch states remain unchanged between before-collection and final recording. Augmentation stream flags match the symmetry-only run because its update/sample-appearance schedule is identical; collection, sampled-example identities and search histories differ.

The driver's original **16-file manifest** was independently hash-verified without replacement. Supplemental audit manifests cover later analyses, both frozen sets, all preflight/log evidence and this handoff. All 274 prior files remain byte-identical. Evidence is in [`experiment-output/phase4d2d-neural-root-noise-seed42-20261005/`](../experiment-output/phase4d2d-neural-root-noise-seed42-20261005/) and its [preflight directory](../experiment-output/phase4d2d-neural-root-noise-preflight-20261005/): source identity/patch and untracked test archive, config, initial/final RNG states, complete events/report, five snapshots, diagnostics, frozen protocol, blind evaluations, opponent histories/timings, checkpoint verification, analyses, gates and preservation inventory.

Standalone post-run audit development needed three corrections: sampling replay must use recorded ascending-column order rather than the engine's center-first legal order; JSON lists and in-memory tuples must be compared in the same representation; an added NumPy comparison count needed conversion to native bool before JSON serialization. Failed audit scripts/logs are retained. These changed only supplemental analysis scripts; experiment source/checkpoint/events, frozen fixtures and original manifest were untouched. There was one training launch and one 144-game opponent campaign; audit inference reproduced the same frozen rows. Exact interrupted training resumability remains unsupported; noise and augmentation streams are replayable.

## Explicit interpretation and next priority

1. **Did root exploration broaden meaningfully? Yes during self-play.** Mean visited actions increased 4.721→5.817; median 5→7; every-legal coverage 47.33%→60.75%; pooled zero-visit fraction 29.37%→15.09%. This persists across trained cohorts. Noise-free held-out breadth is mixed; entropy alone does not demonstrate strategy.

2. **Did unvisited immediate wins decrease? Yes.** Self-play per-winning-action zeros fell 200/481→96/390; known-win searches with no visited win 118/384→46/335. On identical noise-free suites: pooled 22→15/120, old blind 8→7/48, new blind 13→6/48. More immediate wins were found, while low-prior wins still remain unvisited.

3. **Did MCTS tactical accuracy improve beyond symmetry-only? No consistent improvement.** Prior modal 167→164/240, old blind 71→60/96, new blind 58→61/96. Improved winning-action discovery coexists with worse forced-block accuracy. Sampled results are also mixed.

4. **Did direct NN policy improve or regress? It regressed on these tactical sets.** Prior 84→76/240, old blind 44→30/96, new blind 31→29/96. NN Only depth-1 opponent wins nevertheless increased; the two measurements assess different situations.

5. **Did Negamax playing strength improve? Small observed gains.** NN Only depth-1 wins 1→3/12; raw depth-1 4→5/12; raw depth-2 0→1/12 with a draw disappearing. Neither mode gained a direct-policy depth-2 win. One seed/small correlated opening schedule does not establish reliable strength improvement.

6. **What happened to value calibration/saturation? Mixed proper scores, persistent wrong values and worse draw suppression.** Prior/new-blind known-win scores improve, old blind worsens; 47/120 prior proven wins still favor loss. Saturation counts fall, while old-blind mean maximum probability rises. Mean draw probabilities shrink with no draw labels. General calibration remains unestablished.

7. **Did symmetry gains survive noise? Largely, with partial regression.** Pooled policy/value/visit mirror L1 worsened relative to symmetry-only but remains substantially below the earlier nonaugmented learned baseline. Blind dimensions vary; augmentation still provides approximate, not exact, consistency.

8. **Next priority: a separately controlled value-learning hypothesis.** Broader exploration reduced missed wins without fixing blocking decisions, wrong value predictions or draw undercoverage. Investigate value learning and outcome coverage before spending a new experiment on temperature scheduling, more simulations or simply more of the same data. The present outcome targets/backup contracts passed correctness checks; this recommendation concerns learned behavior, not an established mathematical bug. No subsequent learning change was implemented.

Work stops after this single experiment and verified handoff. No checkpoint promotion, further training, commit, push, merge or deployment occurred.

