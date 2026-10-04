# Phase 4C.2c — Controlled horizontal symmetry augmentation

Exactly one fresh seed-42 training run was executed after the complete verification gate passed. It stopped at **5000 completed games**, with stop reason `max_games`, in **215.820 training seconds**. No retries, sweeps or extensions were run.

## Direct result

| Measure | B: No augmentation | C: Probability 0.5 |
| --- | --- | --- |
| Tactical accuracy | 15/48 (31.25%) | 26/48 (54.17%) |
| Diagonal accuracy | 2/16 (12.50%) | 7/16 (43.75%) |
| Aligned mirrored legal-Q MAE | 0.576036 | 0.433437 |
| Mirror action matches | 10/30 | 12/30 |
| Random wins / losses / draws | 75 / 25 / 0 | 92 / 8 / 0 |

The observed results support the hypothesis in this single-seed comparison, with substantial residual failures:

1. **Did tactical accuracy improve?** Yes: 31.25% → 54.17%, a gain of 11/48 rows (+22.92 percentage points). Immediate wins improved 7/24 → 12/24 and forced blocks 8/24 → 14/24. Mean tactical margin improved from −0.248615 to +0.009956.
2. **Did mirrored-Q discrepancies decrease?** The mean aligned legal-Q discrepancy fell **24.76%**, from 0.576036 to 0.433437. The worst discrepancy increased from 2.536337 to 3.399748, so improvement was not uniform. Mirror action agreement changed separately, from 10/30 to 12/30; the policy remains far from symmetric.
3. **Did diagonal recognition improve?** Yes: 2/16 → 7/16 (+31.25 percentage points), while still missing nine diagonal tactics.
4. **Did Random performance remain stable or improve?** It improved: 75% → 92% wins, with X wins 40/50 → 49/50 and O wins 35/50 → 43/50. No draws occurred.
5. **Did previously correct behaviors regress?** Yes: four rows regressed (three immediate wins and one block), while 15 improved. Expected-column accuracy fell from 2/3 to 0/3 for column 5 and from 1/4 to 0/4 for column 6. All four regressions are listed below.
6. **Is this sufficient for additional training or integration?** It supports proposing a separately approved, bounded replication across seeds before a longer training campaign. It does **not** justify DQN integration: the model still misses half the immediate wins, 10/24 blocks, and has significant symmetry and calibration failures. No additional run was started.

The augmented model used 1,615 more optimizer updates (+1.70%) because its self-play games were longer; this was not an independently fixed-update ablation. Mean improvement in this one seed does not establish general causality. Greedy diagnostic entropy also fell (2.628308 → 2.197523 bits), with more concentration on columns 2 and 4. These limits temper the favorable aggregate results.

## Preflight and implementation

Fetched `origin/main`; it matched the supplied reference `a18b03b060df52abf103c9d0825e670f8bbe1783`. The initial tree was clean. Created local branch `phase4c2c-dqn-symmetry` from that commit. No commits, pushes, merges, deployments, API/UI integration, dependency changes, opponent changes, or VictorAgent/neural-MCTS modifications were made.

The [reflection helper](../games/connect4/dqn/training.py) reverses the seven physical columns separately within each of six rows in both canonical states, maps action `c` to `6-c`, reverses the successor legal mask, and preserves piece signs, actor, reward and done. It reconstructs the existing validated immutable `Transition`. Reflection twice restores the original exactly.

`DQNConfig.horizontal_symmetry_probability` defaults to `0.0`; the existing CLI automatically exposes `--horizontal-symmetry-probability`. The trainer samples the unchanged replay once, then independently reflects each sampled transition with the configured probability before building optimization tensors. It neither duplicates replay entries nor changes batch size or sampled reward/terminal/actor composition. A private `random.Random` stream is seeded with `connect4-horizontal-symmetry-v1:42`, separate from action selection, replay sampling and global RNGs. Disabled augmentation draws nothing and follows the original numerical optimization path. Counts describe constructed optimization samples, including any batch followed by an invariant failure; this run had no such failure.

Canonical encoding, signed target `r - gamma * max_legal Q_target(next_state)`, terminal target `r`, legal masking, selected-action Smooth L1 loss, Adam, epsilon decay, target synchronization, architecture and inference checkpoint contract are unchanged. No tactical labels enter training. No explicit symmetry loss, Double DQN, prioritized replay or reward shaping was added.

## Verification gate

All checks passed before training. The new tests use fixed legal sequences and synthetic updates, not learning-strength thresholds.

```sh
/tmp/board-game-phase4c1-venv/bin/python -m pytest -q \
  tests/test_connect4_dqn.py tests/test_dqn_experiment.py \
  tests/test_dqn_diagnostics.py tests/test_dqn_symmetry.py
# 257 passed in 4.13 seconds

.venv/bin/python -m pytest -q tests
# 353 passed, 4 optional DQN modules skipped in 6.22 seconds

.venv/bin/python -m pytest -q tests/test_dqn_import_isolation.py
# 1 passed in 0.18 seconds

git diff --check
# Passed
```

The 83 new checks cover state/action/mask reflection, both player perspectives, terminal X/O wins, draws, full columns, immutability, involution, independently replayed mirrored successors for every legal action of all 60 frozen fixtures, contract reconstruction, configuration validation, deterministic per-item decisions and optimization, RNG ownership, exact disabled losses/weights/targets/epsilon/RNG behavior against the legacy optimization path, replay replacement/composition, warmup, episode/update/synchronization accounting, and signed/masked terminal-excluding Bellman targets. Backend provider calls remain mocked and live HTTPS is blocked by the existing test guard.

The frozen 60-position suite and diagnostic/evaluation implementation were not edited. Fixture SHA-256 before and after:

```text
ec99e0ee080370ab8cfc678c00a49ff552466de73ffe0955778ac87ee340543a
```

## One authorized invocation and configuration

This command was executed once; it is an execution record, not authorization to repeat:

```sh
/tmp/board-game-phase4c1-venv/bin/python -m games.connect4.dqn.experiment \
  --output experiment-output/phase4c2c-seed42-20261003 \
  --seed 42 --max-games 5000 --max-plies 110000 --max-updates 110000 --max-seconds 300 \
  --epsilon-start 1.0 --epsilon-min 0.10 --epsilon-decay 0.99997 \
  --threads 1 --interop-threads 1 --evaluation-games 200 --evaluation-seconds 120 \
  --horizontal-symmetry-probability 0.5 \
  --archived-checkpoint experiment-output/phase4c2b-seed42-20261003-2059/candidate.pt
```

The legacy driver field `archived` refers to **B, the 5,000-game baseline**, in this run. The older 200-game artifact is preserved but was not evaluated again. `--evaluation-games 200` allocates 100 initial + 100 final games, with 100 additional baseline games. All three models started evaluation with the same protocol.

| Setting | Value |
| --- | --- |
| Seed / initialization | 42 / fresh online network, target, Adam, replay and RNGs |
| Replay / batch | 10,000 / 64 |
| Gamma / Adam learning rate | 0.99 / 0.001 |
| Epsilon start / floor / decay | 1.0 / 0.10 / 0.99997 per successful update |
| Hard target sync | Every 100 successful updates |
| Limits | 5,000 games; 110,000 plies; 110,000 updates; 300 training seconds |
| Device / threads / determinism | CPU float32 / 1 intra-op, 1 inter-op / deterministic algorithms enabled |
| Runtime | Python 3.11.17; PyTorch 2.10.0; macOS 26.6.2 ARM64 |
| Only intended method change | Horizontal symmetry probability 0.0 → 0.5 |

An independent post-run audit verified exact equality with the baseline configuration (except the new probability), limits and runtime. Policies naturally induce different self-play trajectories, plies, replay contents and update counts; equal game budgets do not mean equal optimizer budgets. The baseline took 118.695 seconds. No old checkpoint was used to initialize training.

| Completed budget / accounting | B: Baseline | C: Augmented |
| --- | --- | --- |
| Completed games | 5000 | 5000 |
| Started games | 5000 | 5000 |
| Collected plies | 94846 | 96461 |
| Optimizer updates | 94783 | 96398 |
| Training seconds | 118.69463391578756 | 215.81999433296733 |
| X wins | 2760 | 2911 |
| O wins | 2233 | 2082 |
| Draws | 7 | 7 |
| Final epsilon | 0.1 | 0.1 |
| Final replay size | 10000 | 10000 |

No partial episode. Successful updates: 96,398; sampled items: 6,169,472 = updates × 64. **Transformed: 3,086,198; untransformed: 3,083,274; observed augmentation frequency: 50.023697%.** First 63 plies warmed replay; subsequent plies received one update. Scheduled target copies: 963, last at update 96300.

Collected terminal/nonterminal: 5,000/91,461; reward counts: `{'0': 91468, '1': 4993}`. Final replay composition: `{'size': 10000, 'terminal': 488, 'nonterminal': 9512, 'rewards': {'0': 9513, '1': 487}, 'actors': {'0': 5133, '1': 4867}}`. These collected distributions may differ from baseline because gameplay differs; augmentation itself preserves each sampled item's reward, done and actor.

All recorded losses were finite; mean/min/max = 0.019656381/0.000883874/0.084118001. Training Q range across all seven outputs: -3.546850 to 4.890163; maximum absolute gradient component 0.157744. Invariant failures/artifact errors: zero. Loss is measured against changing replay/targets, not a fixed validation objective.

## Frozen tactical comparison

A = fresh seed-42 network; B = previous 5,000-game baseline without augmentation; C = this candidate. Accuracy is on the 48 tactical rows. All 60 positions contribute to greedy histograms and 30 mirror comparisons. Tactical margin is `Q(expected) - max Q(other legal action)`. Physical columns are zero-based.

| Model | Overall | Immediate wins | Forced blocks | Mean margin | Margin min / max | Legal Q min / max |
| --- | --- | --- | --- | --- | --- | --- |
| A: Fresh seed 42 | 10/48 (20.83%) | 6/24 (25.00%) | 4/24 (16.67%) | -0.047792 | -0.182887 / 0.062967 | -0.141177 / 0.137868 |
| B: Baseline 5,000 games | 15/48 (31.25%) | 7/24 (29.17%) | 8/24 (33.33%) | -0.248615 | -1.150934 / 0.422447 | -2.648116 / 1.876006 |
| C: Augmented | 26/48 (54.17%) | 12/24 (50.00%) | 14/24 (58.33%) | 0.009956 | -0.865629 / 0.844254 | -4.020732 / 1.585482 |

| Tactical category | A: accuracy; mean margin | B: accuracy; mean margin | C: accuracy; mean margin |
| --- | --- | --- | --- |
| block | 4/24 (16.67%); -0.053466 | 8/24 (33.33%); -0.248196 | 14/24 (58.33%); 0.045028 |
| win | 6/24 (25.00%); -0.042117 | 7/24 (29.17%); -0.249034 | 12/24 (50.00%); -0.025116 |

| Threat direction | A: accuracy; mean margin | B: accuracy; mean margin | C: accuracy; mean margin |
| --- | --- | --- | --- |
| diagonal | 3/16 (18.75%); -0.045608 | 2/16 (12.50%); -0.262150 | 7/16 (43.75%); -0.062453 |
| horizontal | 2/16 (12.50%); -0.061168 | 6/16 (37.50%); -0.261935 | 7/16 (43.75%); -0.166580 |
| vertical | 5/16 (31.25%); -0.036599 | 7/16 (43.75%); -0.221760 | 12/16 (75.00%); 0.258901 |

| Player (0 = X, 1 = O) | A: accuracy; mean margin | B: accuracy; mean margin | C: accuracy; mean margin |
| --- | --- | --- | --- |
| 0 | 5/24 (20.83%); -0.046603 | 6/24 (25.00%); -0.253374 | 10/24 (41.67%); -0.038599 |
| 1 | 5/24 (20.83%); -0.048980 | 9/24 (37.50%); -0.243856 | 16/24 (66.67%); 0.058511 |

| Expected physical action column | A: accuracy; mean margin | B: accuracy; mean margin | C: accuracy; mean margin |
| --- | --- | --- | --- |
| 0 | 1/4 (25.00%); -0.051613 | 1/4 (25.00%); -0.191570 | 3/4 (75.00%); 0.486548 |
| 1 | 0/3 (0.00%); -0.117005 | 0/3 (0.00%); -0.633204 | 1/3 (33.33%); -0.430763 |
| 2 | 3/12 (25.00%); -0.030170 | 5/12 (41.67%); -0.067349 | 9/12 (75.00%); 0.147395 |
| 3 | 0/10 (0.00%); -0.075207 | 2/10 (20.00%); -0.329225 | 5/10 (50.00%); -0.039909 |
| 4 | 5/12 (41.67%); -0.015333 | 4/12 (33.33%); -0.330538 | 8/12 (66.67%); 0.054466 |
| 5 | 0/3 (0.00%); -0.103489 | 2/3 (66.67%); -0.068106 | 0/3 (0.00%); -0.318839 |
| 6 | 1/4 (25.00%); -0.031991 | 1/4 (25.00%); -0.249103 | 0/4 (0.00%); -0.310684 |

## Greedy actions and symmetry

| Model | 60-position histogram, columns 0–6 | Entropy, bits | Mirror action matches | Aligned legal-Q MAE | Maximum aligned Q difference | Tactical pairs both / one / neither correct |
| --- | --- | --- | --- | --- | --- | --- |
| A: Fresh seed 42 | [1, 0, 19, 0, 23, 3, 14] | 1.860051 | 8/30 | 0.049140 | 0.206188 | 3 / 4 / 17 |
| B: Baseline 5,000 games | [5, 2, 14, 12, 10, 11, 6] | 2.628308 | 10/30 | 0.576036 | 2.536337 | 5 / 5 / 14 |
| C: Augmented | [5, 2, 20, 7, 22, 1, 3] | 2.197523 | 12/30 | 0.433437 | 3.399748 | 7 / 12 / 5 |

Mirror action agreement checks whether two greedy argmax choices map by `c ↔ 6-c`. Mirrored-Q agreement compares the actual Q-values after aligning physical columns; its reported MAE is the mean of 30 pair-level legal-action means, weighting each pair equally. These are different properties. A matching argmax does not establish matching values; lowest-column tie breaking can itself break action symmetry. Absolute Q differences and margins also depend on the Q scale.

The 48 tactical rows are **24 correlated original/mirror pairs**, not 48 independent trials. There are 30 X/30 O positions overall, 24 X/24 O tactical rows, 24 wins/24 blocks, 16 tactical rows per direction, and expected-column counts `[4, 3, 12, 10, 12, 3, 4]`. No independent-trial significance test is claimed. This fixed one-ply suite cannot establish general playing strength or longer forcing-sequence competence.

## Fixed Random evaluation

Each model played exactly 100 greedy games, 50 as X and 50 as O. Game index `i` uses a private `random.Random(42+i)`, with seeds 42–141 and the unchanged engine legal-move enumeration. The post-run audit reproduced the old initial and baseline full game transcripts and diagnostic Q-vectors exactly. Different model choices can induce different trajectories despite matching opponent seeds. Evaluation does not learn or consume trainer RNG streams.

| Model | W / L / D (rates) | As X: W / L / D | As O: W / L / D | Greedy histogram | Entropy, bits |
| --- | --- | --- | --- | --- | --- |
| A: Fresh seed 42 | 64 (64%) / 36 (36%) / 0 (0%) | 31 / 19 / 0 | 33 / 17 / 0 | [56, 2, 143, 60, 309, 21, 183] | 2.194406 |
| B: Baseline 5,000 games | 75 (75%) / 25 (25%) / 0 (0%) | 40 / 10 / 0 | 35 / 15 / 0 | [128, 66, 189, 177, 80, 132, 37] | 2.645998 |
| C: Augmented | 92 (92%) / 8 (8%) / 0 (0%) | 49 / 1 / 0 | 43 / 7 / 0 | [13, 78, 107, 183, 216, 66, 29] | 2.426038 |

Side-specific W/L/D rates are A: X 62%/38%/0%, O 66%/34%/0%; B: X 80%/20%/0%, O 70%/30%/0%; C: X 98%/2%/0%, O 86%/14%/0%. Random is a weak opponent; these 100 paired-seed game schedules do not establish robust strength or independence between outcomes across models.

## Calibration and tactical regressions

| Model | Raw Q min / max, all 7 columns | Known winning-action Q: mean / min / max | Negative winning-action Q |
| --- | --- | --- | --- |
| A: Fresh seed 42 | -0.141177 / 0.137868 | 0.013374 / -0.064137 / 0.110806 | 9/24 |
| B: Baseline 5,000 games | -2.648116 / 1.876006 | 0.053413 / -1.823957 / 1.426230 | 10/24 |
| C: Augmented | -4.020732 / 1.585482 | 0.353267 / -0.780573 / 1.585482 | 8/24 |

An immediate winning action has exact terminal target +1. Negative predictions for those actions show calibration/generalization failures even when another action has a lower Q-value. Values outside the terminal reward scale also warrant caution; finite outputs alone do not establish useful calibration. Full raw Q-vectors and margins are retained in `report.json`.

Relative to B, C corrects **15** previously missed tactical rows and regresses on **4** previously correct rows. These are descriptive row counts, not independent trials. All regressions:

| Fixture | Kind / direction | Player | Expected column | Baseline → augmented action |
| --- | --- | --- | --- | --- |
| block_vertical_1_0_mirror | block / vertical | 1 | 6 | 6 → 4 |
| win_diagonal_1_1_mirror | win / diagonal | 1 | 5 | 5 → 1 |
| win_horizontal_0_1_mirror | win / horizontal | 0 | 3 | 3 → 2 |
| win_horizontal_1_0 | win / horizontal | 1 | 5 | 5 → 0 |

The complete improved/regressed position lists are in `audit.json`. Diagnostic labels were not modified or used to tune parameters or select among candidates.

## Artifact integrity and provenance

New output directory: [`experiment-output/phase4c2c-seed42-20261003/`](../experiment-output/phase4c2c-seed42-20261003/), entirely Git-ignored and local-only. It contains exactly one new version-1 inference checkpoint, `candidate.pt`, plus full configuration, events, report, source patch/untracked source snapshot, audit, test logs, console record and preservation manifest. The checkpoint is not a resumable training state.

Candidate SHA-256:

```text
3327863c8d6ed45b70871cd7b336d35950f2c9962a204ebcc67baa86f6c86f08
```

`save_candidate` safely reloaded with `weights_only=True`, verified exact equality of every in-memory trained weight tensor and metadata, and exact equality of all 60 diagnostic Q-vectors/actions. The separate read-only audit repeated safe loading, payload/model tensor equality, predictions, metadata, SHA-256, event accounting, source hashes, fixture identity and historical-artifact checks. The external report records the checkpoint hash; the embedded metadata was captured immediately before checkpoint serialization.

All 51 previously existing experiment/historical checkpoint files match their preflight SHA-256 manifest. Preserved primary baseline:

```text
experiment-output/phase4c2b-seed42-20261003-2059/candidate.pt
29c1b6c3288811b449b4d90744607e572fca17bc631d5cff4334fc2b4efaf56e
```

Preserved original 200-game candidate:

```text
experiment-output/phase4c2-seed42/candidate.pt
1be2836c252575e65ce6c46891dcca1ab5b426e08bffb552cee0b3ec570267a9
```

Source provenance at launch:

```text
Commit: a18b03b060df52abf103c9d0825e670f8bbe1783
Branch: phase4c2c-dqn-symmetry
Source identity SHA-256: dcb48a46c23b86dbc15be031e34b87b63e83c719af9729bbe75616c93ab353ed
Tracked diff SHA-256: d171a010b6d7ecbe970e7b6b95867814db9c1ed48d04a4abf60e47bdb0d4708b
Started UTC: 2026-10-04T01:16:55.426001+00:00
Started local: October 3, 2026, 21:16:55 EDT (America/New_York)
```

The launch tree intentionally contained the augmentation implementation and tests, captured as the tracked patch and untracked source bytes. This handoff was written afterward. No claim that the commit alone identifies the experiment source is made. Timing uses the monotonic training loop budget; evaluation, initialization and artifact writing are outside that limit. Runtime timestamps are recorded verbatim in provenance.

The single seed, fixed correlated suite, weak opponent and naturally different update counts limit attribution. This experiment tests the stated symmetry hypothesis under one configuration; it cannot establish a general causal benefit. Stop here for review. No further training or integration was executed.
