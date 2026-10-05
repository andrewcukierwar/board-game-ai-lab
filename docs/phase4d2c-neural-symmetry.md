# Phase 4D.2c — Controlled neural symmetry augmentation

Completed October 5, 2026. **One fresh seed-42 experiment completed: 200 games, 3,613 applied/collected plies and 1,970 optimizer updates.** Horizontal augmentation improved mirror consistency, direct-policy tactical accuracy and raw MCTS tactics relative to the 200-game baseline. Search recovered part of the regression from the original untrained network, but did not exceed it overall. Known-win value proper scores worsened. The small opponent evaluation showed limited gains, without a depth-2 win.

## Preflight and scope

Fetched `origin/main`, which matched the expected **`0201231776799a4c3b8cfce002678f2e5f995da6`**. The initial tree was clean. Created local branch `phase4d2c-neural-symmetry` from that reference. Read the [foundation](phase4d1-neural-mcts-foundation.md), [pilot](phase4d2-neural-training.md), [scaled baseline](phase4d2b-neural-scaled.md), trainer, driver, shared neural contract and evaluation/search implementation.

The scaled baseline `experiment-output/phase4d2b-neural-scaled-replacement1-seed42-20261004/candidate.pt` was verified before launch and again afterward against SHA-256 **`d16cb516775146084bc2438c0166d241c1224efc69b2aa7e1b40596de77d1521`**. It was loaded into a separate read-only inference object for evaluation; training started from a fresh network. No historical checkpoint initialized training.

All **233 pre-existing artifact/evidence files** under `experiment-output/` and `models/` retain their preflight SHA-256, including the failed 4D.2b launch evidence, ten historical neural checkpoints, DQN artifacts and successful baseline. Their full before-hash inventory and final verification are preserved. No commit, push, merge, production dependency change, DQN change, API/UI integration, provider request, hosting or deployment occurred.

## Implementation and RNG ownership

[`reflect_completed_example`](../games/connect4/train_mcts_nn.py) accepts only completed immutable `TrainingExample` instances. It reverses each of the six physical board rows and the seven-action policy tuple, then reconstructs using `dataclasses.replace`, invoking the existing validated constructor. Acting player, actor-relative outcome, temperature, encoding, tactical-guard and policy-target metadata are preserved exactly. The input is never changed; two reflections recover exact dataclass equality. No encoding, network, PUCT, loss or outcome-label mathematics changed.

[`SymmetryAugmenter`](../games/connect4/neural_self_play.py) defaults to probability **0.0**. Experimental probability is **0.5**. Each selected completed example receives an independent Bernoulli decision while constructing the existing 32-example minibatch. Stored examples remain original; no collection duplication, extra examples, batch expansion or additional optimizer updates occur. With augmentation disabled it returns the original objects and consumes no augmentation RNG draws.

Its private `random.Random` uses the integer derived from SHA-256 of `connect4-completed-horizontal-symmetry-v1:42`. Seed digest: **`c1a46d134746ddc6d95e0e14aad0aee19081b5badafa7da77482de841b9c61f4`**. It neither uses nor advances search, base sampling, Python-global, NumPy or PyTorch RNGs. Full initial/final augmentation states accompany the original RNG records. Every update records base indices and all 32 reflection flags. The final audit replayed the entire augmentation and sampling streams exactly against the original completed collection.

## Correctness gates before launch

| Gate | Result |
| --- | --- |
| Neural foundation, driver, scaled and symmetry suites | 234 passed, 43.24 s |
| Six existing DQN suites | 466 passed, 5.46 s |
| Backend `pytest -q tests` | 354 passed, 10 optional PyTorch modules skipped, 6.81 s |
| Both explicit API isolation tests | 2 passed, 0.75 s |
| Complete nested JSON serialization preflight | Passed: 266 original/frozen + 96 blind rows; zero training calls/updates |
| `git diff --check` | Passed before launch and at handoff |

Neural/DQN gates used existing `/tmp/board-game-phase4c1-venv`; backend/isolation gates used `.venv`. No dependencies were installed. Optional skips were covered in the neural/DQN environment. [New tests](../tests/test_neural_symmetry.py) cover full completed X-win, O-win and draw histories with both actors and full columns; reflection/involution, policy mapping, immutable input ownership and metadata; deterministic isolated RNGs; invalid probabilities/pending examples; and exact disabled-path loss, gradient metrics, weight tensors and Adam state across four discarded synthetic updates. A 200-game scripted synthetic regression verifies unchanged collection, episodes, base indices, 1,970-update schedule and nested event/report serialization. Existing tests retain stop limits, smoke gate, quarantine, numerical checks and evaluation isolation.

The serialization-only preflight blocked both training entry points and proved unchanged model tensors/RNG state. No correctness gate failed before launch. The earlier scaled launch serialization failure was preserved and was not repeated.

## Frozen model-blind tactical set

[`neural_symmetry_diagnostics.py`](../games/connect4/neural_symmetry_diagnostics.py) generated a new **96-row** set before training and before evaluating either learned checkpoint on it. Seed `420402` generated legal nonterminal prefixes without model access. There are 48 unique base boards plus their explicit reflections: 12 bases per actor/category, yielding 48 immediate-win and 48 unique forced-block rows. Board-plus-actor duplicates and all boards/reflections of the three prior suites were excluded. Reflection pairs are correlated; 96 rows are not 96 independent tactical situations.

Labels were derived by an independent four-cell horizontal/vertical/diagonal window scanner on dropped-piece boards, then cross-checked against actual engine successor wins and the existing tactical helpers. Every position is legally replayed. Forced-block actions have unique immediate-reply safety, not proven ultimate W/D/L outcomes. The set was not altered after any model evaluation, and its labels never entered training or selected the checkpoint.

Frozen file: [`preflight/blind-positions.json`](../experiment-output/phase4d2c-neural-symmetry-seed42-20261005/preflight/blind-positions.json). SHA-256:

```text
ba31268418872fb2b5d89879d48fa644cd8cb576db0b6fd2d80d1af41f3727d5
```

## Single experiment and accounting

The sole real training launch was:

```sh
PYTHONHASHSEED=42 /tmp/board-game-phase4c1-venv/bin/python \
  -m games.connect4.neural_self_play --profile scaled \
  --horizontal-symmetry-probability 0.5 \
  --evaluation-baseline experiment-output/phase4d2b-neural-scaled-replacement1-seed42-20261004/candidate.pt \
  --output experiment-output/phase4d2c-neural-symmetry-seed42-20261005
```

This records the completed launch, not permission to repeat it. The directory was new, Git-ignored, and never reused. There were no retries, budget extensions or additional training runs.

| Setting | Value |
| --- | --- |
| Initialization | Fresh canonical 326,026-parameter Connect4Net; CPU float32 |
| Seeds | Python/NumPy/PyTorch 42; private search and base sampling Random(42); domain-separated augmentation |
| Search | 32 simulations/move, PUCT 1.41, temperature 1 throughout self-play |
| Guard / root noise | OFF / OFF |
| Optimizer | Persistent Adam, lr .001, batch 32, original equally weighted policy/value losses |
| Schedule | Two collection-only games; ten updates after games 3–199; none after final game cap |
| Ceilings | 200 completed games, 8,400 plies, 2,000 updates, 900 training seconds |
| Runtime | One intra-op/inter-op thread; deterministic algorithms; existing native ARM64 environment |
| Only intended learning-method change | horizontal_symmetry_probability = 0.5 |

| Measurement | Baseline | Augmented |
| --- | --- | --- |
| Completed games | 200 | 200 |
| Applied / collected plies | 3,673 / 3,673 | 3,613 / 3,613 |
| Optimizer updates | 1,970 | 1,970 |
| Training seconds | 84.821 | 85.306 |
| Self-play X wins / O wins / draws | 121 / 79 / 0 | 133 / 66 / 1 |
| Smoke gate | 2 games, 34 examples, 0 updates | 2 games, 34 examples, 0 updates |
| Status / stop reason | bounded_stop / max_games | bounded_stop / max_games |
| Partial episodes / invariant failures | None / none | None / none |

The initial weight-byte digest **`559620773060542d4569b25eb92f2e60c862f44a45299dca00c666cef9d4033e`** matches the original fresh seed-42 initialization. The two smoke histories remain unchanged (18- and 16-ply O wins) because augmentation first operates after game 3. Snapshots occurred at games 50/100/150 with 480/980/1,480 updates. The final game is collected but not optimized after the game limit. All 3,613 plies passed the existing replay, immutable pre-move, legality, 32-visit and actor-relative winner gates.

| Actor | Win labels | Draw labels | Loss labels | Total |
| --- | --- | --- | --- | --- |
| X | 1217 | 21 | 635 | 1873 |
| O | 635 | 21 | 1084 | 1740 |
| Both | 1852 | 42 | 1719 | 3613 |

Augmentation transformed **31,390** samples and left **31,650** untransformed: **49.7938%** of **63,040 = 1,970 × 32** training sample appearances. These are repeated sampling appearances, not new stored examples. The single draw supplied 42 draw labels; data still contain very limited draw coverage. Self-play outcomes do not establish strength.

All updates had finite gradients and parameters. Last-ten minibatch means:

| After games / updates | Policy loss | Value loss | Combined | Target entropy (nats) |
| --- | --- | --- | --- | --- |
| 50 / 480 | 1.3722 | 0.3555 | 1.7277 | 1.0644 |
| 100 / 980 | 1.2480 | 0.4263 | 1.6743 | 0.9980 |
| 150 / 1480 | 1.3339 | 0.4475 | 1.7814 | 1.0597 |
| 200 / 1970 | 1.2662 | 0.3631 | 1.6293 | 1.0060 |

The final loss block is after game 199. Changing sampled batches and generated targets prevent interpreting these fitting losses as held-out generalization losses.

## Frozen diagnostic comparison

Original diagnostic histories, labels, frozen hashes and RNG protocol were reused. All diagnostics use raw 32-simulation neural MCTS, PUCT 1.41, guard/noise OFF and **temperature 1**. NN Only is the lowest-index maximum legal policy; modal MCTS is the lowest-index maximum root visits; sampled MCTS is exploratory sampling. Tactical labels annotate after inference/search and never filter actions. “Untrained” uses the identical original seed-42 fresh weights; “baseline” is the supplied scaled checkpoint. Original baseline diagnostic predictions match exactly on CPU reload.

Each cell is **NN greedy / MCTS modal / MCTS sampled** correct actions:

| Suite | Tactical rows | Untrained | Baseline | Augmented |
| --- | --- | --- | --- | --- |
| Original 14 probes | 8 | 2 / 6 / 5 | 3 / 3 / 3 | 4 / 6 / 5 |
| Original frozen 60 | 48 | 6 / 35 / 31 | 10 / 28 / 24 | 21 / 35 / 29 |
| Validation 96 | 96 | 16 / 79 / 62 | 18 / 47 / 39 | 32 / 59 / 53 |
| Replication 96 | 96 | 15 / 73 / 61 | 21 / 48 / 42 | 31 / 73 / 56 |
| Pooled prior frozen | 240 | 37 / 187 / 154 | 49 / 123 / 105 | 84 / 167 / 138 |
| New blind 96 | 96 | 5 / 74 / 63 | 22 / 54 / 42 | 44 / 71 / 59 |

On the prior 240 tactics, modal search recovered **123 → 167**, still below **187** untrained; sampled **105 → 138**, below **154**. On the blind set, modal **54 → 71**, below **74**; sampled **42 → 59**, below **63**. Direct policy improved on every suite, including blind **22 → 44 / 96**. This supports tactical generalization relative to the learned baseline for this seed and these fixtures, without establishing superiority to untrained search. The fourteen original probes alone returned to the untrained modal/sample counts.

## Mirror consistency

L1 compares reversed original physical-column policy against its mirror, and unchanged original W/D/L against its mirror; range 0–2. Legal NN policies are used for the main table. Lower is better. Pair names are qualified by suite when pooling to avoid cross-suite name collisions.

| Suite / pairs | NN policy L1 baseline → augmented | W/D/L L1 baseline → augmented | MCTS visit-policy L1 baseline → augmented |
| --- | --- | --- | --- |
| Original / 30 | 1.1862 → 0.4136 | 0.4840 → 0.4451 | 0.9625 → 0.4604 |
| Validation / 48 | 1.2967 → 0.4659 | 0.8634 → 0.4401 | 1.2422 → 0.4453 |
| Replication / 48 | 1.3358 → 0.5286 | 0.7872 → 0.4909 | 1.1628 → 0.5677 |
| Pooled / 126 | 1.2853 → 0.4773 | 0.7441 → 0.4607 | 1.1453 → 0.4955 |
| Blind / 48 | 1.3038 → 0.4808 | 0.5366 → 0.4905 | 1.1302 → 0.5052 |

Across the 126 prior pairs, NN reflected-action agreement rose **18 → 83**, MCTS modal **43 → 89**, and sampled **38 → 68**. Pooled raw unmasked NN-policy L1 fell **1.2992 → .4856**; mean absolute scalar-value mirror difference fell **.7440 → .4591**. The new blind set also improved NN action agreement **7 → 33 / 48**, modal **21 → 33 / 48**.

Consistency improved relative to the learned baseline but remains much worse than near-uniform untrained distributions: prior pooled legal-policy/WDL/MCTS L1 **.0257/.00155/.2371**, versus augmented **.4773/.4607/.4955**. Symmetry augmentation encourages approximate consistency; it does not impose exact equivariance, and correct reflected actions do not imply correct tactics or values.

## Known-win scores, distributions and saturation

Brier is the unnormalized three-class squared-error sum; NLL is −log P(win) in nats. Only immediate-win rows have proven optimal mover-relative W/D/L labels. Blocks are not outcome calibration labels. These scores are narrower than general W/D/L calibration.

| Suite / known wins | Untrained Brier / NLL | Baseline Brier / NLL | Augmented Brier / NLL |
| --- | --- | --- | --- |
| Original probes / 4 | 0.6708 / 1.1017 | 0.2103 / 0.2927 | 0.3138 / 0.4313 |
| Original frozen / 24 | 0.6712 / 1.1023 | 0.3069 / 0.4544 | 0.6847 / 1.3855 |
| Validation / 48 | 0.6708 / 1.1018 | 0.2778 / 0.8419 | 0.9523 / 2.1002 |
| Replication / 48 | 0.6711 / 1.1022 | 0.3992 / 0.9289 | 0.5719 / 1.0693 |
| Pooled / 120 | 0.6710 / 1.1021 | 0.3321 / 0.7992 | 0.7467 / 1.5449 |
| Blind / 48 | 0.6711 / 1.1021 | 0.2984 / 0.7194 | 0.4433 / 0.8965 |

Pooled known-win Brier/NLL worsened from **.3321/.7992 → .7467/1.5449**, also worse than untrained **.6710/1.1021**. Blind scores worsened **.2984/.7194 → .4433/.8965**, though still better than untrained **.6711/1.1021**. Among 120 prior known wins, predictions favoring loss over win increased **25 → 54**. Improved action accuracy and mirror agreement therefore did not ensure correct value predictions.

| Suite / model | Mean W / D / L | Mean max WDL | Max WDL ≥ .99 |
| --- | --- | --- | --- |
| Prior pooled / untrained | 0.332202 / 0.301713 / 0.366085 | 0.3661 | 0 / 252 |
| Prior pooled / baseline | 0.679481 / 0.000047 / 0.320472 | 0.9169 | 130 / 252 |
| Prior pooled / augmented | 0.477614 / 0.000927 / 0.521459 | 0.8673 | 67 / 252 |
| Blind / untrained | 0.332291 / 0.301703 / 0.366006 | 0.3660 | 0 / 96 |
| Blind / baseline | 0.738777 / 0.000006 / 0.261217 | 0.9019 | 42 / 96 |
| Blind / augmented | 0.564556 / 0.000265 / 0.435179 | 0.8684 | 28 / 96 |

Prior-suite saturated counts fell **130 → 67 / 252** (original **20 → 12**, validation **58 → 31**, replication **52 → 24**). Blind saturation fell **42 → 28 / 96**. Fewer saturated predictions coexist with worse proper scores; this is not a calibration improvement. Mean prior draw probability rose **4.683e−5 → 9.265e−4**; blind **5.595e−6 → 2.646e−4**. The draw class remains strongly suppressed despite one collected draw. All per-position distributions, ranges and entropies are retained in snapshots and blind-evaluation JSON.

## Action concentration and unvisited wins

| Suite / model | NN entropy | NN mean max | Greedy modal-column share | Visit entropy | Visit mean max | MCTS modal-column share |
| --- | --- | --- | --- | --- | --- | --- |
| Prior pooled / untrained | 1.8661 | 0.1591 | 0.6905 | 1.2715 | 0.5466 | 0.3294 |
| Prior pooled / baseline | 1.1327 | 0.5701 | 0.2421 | 0.8265 | 0.6896 | 0.2262 |
| Prior pooled / augmented | 1.2697 | 0.5248 | 0.1944 | 0.8261 | 0.7018 | 0.2063 |
| Blind / untrained | 1.8754 | 0.1571 | 0.7917 | 1.2610 | 0.5544 | 0.2604 |
| Blind / baseline | 1.1576 | 0.5585 | 0.2917 | 0.9094 | 0.6510 | 0.2083 |
| Blind / augmented | 1.1569 | 0.5812 | 0.2188 | 0.7625 | 0.7093 | 0.2083 |

Prior NN distributions became somewhat broader, but root visits remain concentrated: mean maximum visit probability **.6896 → .7018**, and entropy **.8265 → .8261**. Blind visits became more concentrated (**.6510 → .7093**, entropy **.9094 → .7625**). Greedy/modal column frequency is a separate measure from within-position probability concentration. The fourteen-probe augmented NN and search are especially concentrated (mean maxima **.6699/.7969**).

Across 120 prior known wins, **22** augmented searches never visited any immediate-winning action, compared with **34** baseline and **0** untrained. For `diagnostic:win_diagonal_0_1`, win column 0 has legal prior **.005578** and root visits **[0,24,0,0,0,8,0]**; modal action 1 and sampled action 5 both miss the win. The unchanged finite-budget PUCT can still neglect low-prior wins. This observed failure supports exploration as a subsequent hypothesis; no PUCT repair or exploration change was made here.

## Playing strength: matched 144-game evaluation

All **144/144** games completed in **9.343 seconds** within the original single shared 180-second budget. The model pair is now **scaled baseline versus augmented candidate**. In the evaluation JSON, `initial` aliases the scaled baseline and `final` the augmented model; diagnostic `snapshot-initial` still means the fresh untrained network. The explicit model mapping and baseline path/hash are in `config.json`. No additional untrained opponent campaign was run.

Reused six prefixes `[3,2]`, `[3,4]`, `[0,3,2]`, `[6,3,4]`, `[2,4,3,2]`, `[4,2,3,4]`; both sides; NN Only and raw MCTS; Random and existing Negamax depth 1/2. Opponent seeds `420202 + 2*opening_index + side` and search seeds `420203 + 2*opening_index + side` match across models. Fresh Negamax cache/tie behavior is unchanged. **Playing-strength MCTS is temperature zero**, using seeded maximum-visit ties; it is separate from temperature-1 sampled tactical diagnostics. Histories are legal and replay-validated. All 72 baseline histories/results exactly reproduce the previous scaled final-model evaluation.

Each cell is **W/L/D**, six games per side:

| Mode | Opponent | Baseline X | Baseline O | Augmented X | Augmented O |
| --- | --- | --- | --- | --- | --- |
| nn_only | random | 3/3/0 | 2/4/0 | 6/0/0 | 5/1/0 |
| nn_only | negamax1 | 1/5/0 | 0/6/0 | 1/5/0 | 0/6/0 |
| nn_only | negamax2 | 0/6/0 | 0/6/0 | 0/6/0 | 0/6/0 |
| raw_mcts | random | 6/0/0 | 5/1/0 | 6/0/0 | 6/0/0 |
| raw_mcts | negamax1 | 1/5/0 | 2/4/0 | 3/3/0 | 1/4/1 |
| raw_mcts | negamax2 | 0/6/0 | 0/6/0 | 0/5/1 | 0/6/0 |

NN Only totals improved **6/30/0 → 12/24/0**, entirely from Random (**5 → 11 wins / 12**); depth 1 remains **1/11/0**, depth 2 **0/12/0**. Raw MCTS totals improved **14/22/0 → 16/18/2**: Random **11/1/0 → 12/0/0**, depth 1 **3/9/0 → 4/7/1**, depth 2 **0/12/0 → 0/11/1**. Negamax gains are small and occur in raw search; direct policy has no Negamax improvement. There is still no depth-2 win. Twelve games per opponent/configuration and one training seed cannot establish reliable playing-strength superiority.

## Timing and process memory

Existing runtime: native macOS ARM64, Python 3.11.17, torch 2.10.0, NumPy 1.26.4. `ru_maxrss` is macOS process peak bytes converted to MiB, not current RSS or system usage. Whole-search timing includes root initialization, copies, traversals, inference, expansion, backup and sampling.

| Measurement | Actual |
| --- | --- |
| 3,613 self-play searches, mean / median / p95 | 12.553 / 13.359 / 14.090 ms |
| Whole-search min / max / total | 0.940 / 59.992 ms / 45.353 s |
| Training / driver end-to-end | 85.306 / 102.207 s |
| Initialization / checkpoint write / safe load | 5.492 / 5.633 / 2.270 ms |
| Initialization / maximum training-driver peak memory | 270.297 / 304.219 MiB |

| Evaluation model / mode | Mean / p95 action ms |
| --- | --- |
| Baseline / nn_only | 0.274 / 0.289 |
| Baseline / raw_mcts | 11.386 / 13.756 |
| Augmented / nn_only | 0.277 / 0.297 |
| Augmented / raw_mcts | 11.544 / 13.818 |

Training timing includes collection, replay/event writes, optimization and intermediate snapshots; end-to-end additionally includes initial/final diagnostics, checkpoint checks and opponent games, excluding imports. Bounds are cooperative; no new operation begins after a limit, while an in-flight operation may finish beyond it. The run was well inside time/ply/update ceilings. Trees are released after capture. Per-ply/per-update times and peaks are saved; these are local research timings, not serving/concurrency performance.

## Canonical artifact and provenance

Evidence: [`experiment-output/phase4d2c-neural-symmetry-seed42-20261005/`](../experiment-output/phase4d2c-neural-symmetry-seed42-20261005/). One minimal canonical [`candidate.pt`](../experiment-output/phase4d2c-neural-symmetry-seed42-20261005/candidate.pt), **1,309,427 bytes**, SHA-256:

```text
a70910085f1d8be5886d60ce36ee80877bc2a4de6104ac6311462477c5b2cde7
```

The checkpoint contains exactly `contract` and `model_state_dict`, with unchanged format/architecture/encoding, physical actions, current-player W/D/L perspective and logits outputs. All 16 state tensors are finite CPU float32. Safe CPU reload uses `weights_only=True`, with no unsafe fallback. The driver verified exact weight/logit/prediction equality on fourteen probes. Independent post-run verification proved exact saved raw-policy and W/D/L prediction equality on **all 266 original/frozen positions** for both the candidate and baseline; checkpoint tensors also match exactly. Blind inference used these safely loaded models.

Complete config/runtime, code identity at the expected origin commit plus source patch/untracked source archive, fresh weight digest, private RNG provenance and states, all events and reflection flags, five nested snapshots, frozen blind fixture/protocol, three-model blind evaluation, 144 opponent histories/timings, checkpoint verification, analyses, preflight scripts/logs and preservation records are retained. The original driver manifest was independently hash-verified; a supplementary audit manifest covers subsequent evidence and this final handoff without replacing it. Exact interrupted resumability is not implemented. The checkpoint was not promoted or integrated.

## Explicit answers and next priority

1. **Did mirror consistency improve? Yes relative to the trained baseline.** Prior pooled legal NN-policy L1 fell 1.2853 → .4773, W/D/L L1 .7441 → .4607 and visit-policy L1 1.1453 → .4955. Blind improvements agree, but value consistency improves less and exact equivariance is not established.

2. **Did tactical search quality recover? Partly.** Prior modal MCTS 123 → 167 / 240 and blind 54 → 71 / 96 recover toward, but remain below, untrained 187/240 and 74/96. Sampled search also remains below untrained. Recovery on the original fourteen probes alone is not full recovery across larger sets.

3. **Did the direct neural policy improve? Yes on these tactical sets.** Prior 49 → 84 / 240, blind 22 → 44 / 96. It also improves against Random, but has no Negamax strength gain.

4. **Did playing strength improve against Negamax? A small raw-search gain was observed.** Depth 1 gains one win and one draw, depth 2 one draw and zero wins. NN Only is unchanged. This small single-seed schedule does not establish reliable strength improvement.

5. **Did action concentration or value saturation persist? Yes.** Prior root visit maxima average .7018 and 67/252 value predictions remain ≥ .99. Draw predictions stay tiny; known-win proper scores worsen, and 54/120 proven wins favor loss. Blind visits become more concentrated.

6. **Next priority: a separately controlled root exploration-noise experiment.** Twenty-two known-win searches still leave their winning action unvisited, and root targets remain concentrated under augmentation. Root noise directly tests exploration/data diversity at the current method, while keeping temperature scheduling, architecture and PUCT fixed. Additional data alone was already insufficient in 4D.2b; temperature scheduling is a separate future hypothesis. Wrong values and limited draw coverage still need investigation, and noise is not guaranteed to fix them. **No root noise, temperature change, additional training or other subsequent learning-method change was implemented.**

The symmetry hypothesis is supported for policy/mirror and tactical improvements relative to this learned baseline, with partial search recovery and persistent value/strength limitations. Work stops after this experiment and handoff for review.
