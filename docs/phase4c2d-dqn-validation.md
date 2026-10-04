# Phase 4C.2d — Independent DQN validation

**Completed the authorized evaluation after the corrected verification gate passed.** Horizontal augmentation shows limited tactical generalization: holdout accuracy rises **24/96 → 31/96 (25.00% → 32.29%, +7.29 percentage points)**, with improved mirrored-Q consistency, but blocks, horizontal threats and O-side accuracy regress. Neither model gains an aggregate advantage against Negamax: both score **6–34–0 at depth 1 and 2–38–0 at depth 2**. The augmented model remains unsuitable for experimental public gameplay as a standalone raw DQN policy.

All **220 scheduled games completed**, with no partial games, in **1.777971 seconds** of aggregate driver time, below the 600-second deadline. Both preserved checkpoints and both frozen suites retain their original hashes. No training, model selection, tactical guard, dependency change or integration was performed.

The first attempt stopped before inference because the production API import-isolation test was invoked in the optional DQN environment, which lacks `python-dotenv`. The user reviewed that environment mismatch and explicitly authorized resumption. The failure record and its resolution are retained below; the full stopped handoff is preserved locally as `pre-resume-handoff.md`.

## 1. Preflight and exact evaluation configuration

Fetched `origin/main`; it exactly matched the requested reference:

```text
76ba2b6902fc0ce819d2b24212c8c4885e939f8b
```

At the original preflight, the working tree was clean. Created local branch `phase4c2d-dqn-validation` from that reference. Reviewed the Phase 4C.2b and Phase 4C.2c handoffs. All implementation changes are new evaluation files; existing training, networks, opponents, checkpoint loader, and diagnostics are unchanged.

At resumption, the existing branch was clean at `1489f93aa8991e85c49ca8f9b122333425321977`, which contains the preserved validation work. No branch switch, source edit, holdout reconstruction or refetch was needed. The evaluation source, tests, checkpoints, old suite, new suite, and frozen configuration were preserved. `git diff --check` passed before evaluation and at final handoff.

Preserved files, verified before construction, at resumption, and after evaluation:

| Artifact | SHA-256 |
| --- | --- |
| Baseline `experiment-output/phase4c2b-seed42-20261003-2059/candidate.pt` | `29c1b6c3288811b449b4d90744607e572fca17bc631d5cff4334fc2b4efaf56e` |
| Augmented `experiment-output/phase4c2c-seed42-20261003/candidate.pt` | `3327863c8d6ed45b70871cd7b336d35950f2c9962a204ebcc67baa86f6c86f08` |
| Original 60-position `diagnostic_positions.json` | `ec99e0ee080370ab8cfc678c00a49ff552466de73ffe0955778ac87ee340543a` |

The unchanged [evaluation harness](../games/connect4/dqn/validation.py) uses only the existing versioned `load_checkpoint`, whose `torch.load` call requires `weights_only=True` and CPU loading, with no unsafe fallback. Both trained checkpoints were loaded only for evaluation after the corrected gate passed. The new validation test of the loading path uses a synthetic in-memory payload without writing a checkpoint.

| Setting | Value |
| --- | --- |
| Inference | Existing raw DQN greedy action selection, legal mask, lowest legal column on ties |
| Exploration / learning / safeguards | None / none / none |
| Device / dtype | CPU / float32 |
| Threads | 1 intra-op, 1 inter-op |
| Deterministic algorithms | Enabled for actual evaluation |
| Opening construction seed | 420205 |
| Random opponent seed | `420206 + 2 × opening_index + DQN_side` |
| Negamax | Existing depth 1 and depth 2; fresh agent each game, existing cache retained within each game |
| Negamax target | 40 games per model per depth, 20 as X and 20 as O |
| Random target | 20 games per model, first 10 prefixes × both sides; separately limited to 30 seconds |
| Head-to-head target | 20 games, first 10 prefixes × both assignments; separately limited to 30 seconds |
| Aggregate deadline | 600 seconds, including model loading, tactical inference, and gameplay |
| Schedule | Depth, opening index, DQN side, model; then Random; then head-to-head |
| Completion/accounting | Only terminal games count toward W/L/D; partial transcripts retained separately |
| Deadline enforcement | Cooperative checks before and after each move; a single bounded-depth call may overrun the deadline before returning |

Random uses the existing `RandomAgent`. The harness seeds and restores Python's module RNG around each game. Negamax retains its existing heuristic, move order, cache behavior and tie breaking; only its console output is suppressed. No stronger-opponent implementation was modified.

The 20 predeclared prefixes below are replayed legally from an empty board. Each is nonterminal, and each is evaluated with the DQN assigned to both X and O. X always places the first piece under the engine's rules; assigning DQN to O does not change that rule. At takeover, 12 prefixes have X to move and eight have O to move; because each prefix is used for both model assignments, DQN-to-move and opponent-to-move are balanced. Each adjacent pair is reflected:

```text
 0: [5,0]                 1: [1,6]
 2: [5,6,2]               3: [1,0,4]
 4: [2,1,0,2]             5: [4,5,6,4]
 6: [5,4,3,2,5]           7: [1,2,3,4,1]
 8: [6,0,0,4,4,0]         9: [0,6,6,2,2,6]
10: [6,5]                11: [0,1]
12: [1,6,0]              13: [5,0,6]
14: [3,1,3,3]            15: [3,5,3,3]
16: [5,3,1,2,0]          17: [1,3,5,4,6]
18: [3,0,5,1,4,1]        19: [3,6,1,5,2,5]
```

The driver recorded every opening move, every subsequent move with actor and agent role, terminal winner/result, seed, model/opponent assignment, and runtime in `games.jsonl` and `report.json`. Every loaded state tensor was exactly unchanged afterward, and all preserved file hashes matched.

Executed once, after the passing gate:

```sh
/tmp/board-game-phase4c1-venv/bin/python -m games.connect4.dqn.validation evaluate
```

Execution began at `2026-10-04T01:42:50.533677+00:00` (October 3, 2026, 21:42:50 EDT). Runtime: Python 3.11.17, PyTorch 2.10.0, macOS 26.6.2 ARM64. The measured 1.777971 seconds covers loading, tactical inference, gameplay and final integrity checks; interpreter startup, pre-run validation, final JSON serialization, tests and post-run audit are outside the reported driver interval. No expensive extension was used just because budget remained.

## 2. Frozen independent holdout

New fixture: [validation_positions.json](../games/connect4/dqn/validation_positions.json). Construction and validation source: [validation_holdout.py](../games/connect4/dqn/validation_holdout.py).

```text
SHA-256: 1c79db45a09e00a8e8806448362eb319b5e9c81e431de03c7b5ece8b2d04bb2f
Construction seed: 420204 (original suite: 42002)
Rows: 96
Distinct original position families: 48
Additional reflected rows: 48
Freeze timestamp: 2026-10-04T01:36:26.447139+00:00
Local timestamp: October 3, 2026, 21:36:26 EDT
Configuration SHA-256: 577f4025dfab1496910982377f8761dc6c868584d8be4e18f023d2fc34a678ee
```

Each original is selected from a separately sampled random legal rollout, with no model access. Rollout length is sampled from 6 through 34 plies. Terminal boards are rejected. Acceptance quotas were defined by tactic, direction, player, and reflected column bucket before sampling. Canonical board-plus-player identities reject duplicates and reflected equivalents, including every original diagnostic position. A mirror row is deliberately added exactly once per accepted original; it is not counted as another independent family. Tests confirm 96 distinct literal boards and no overlap with the original suite after reflection canonicalization.

| Dimension | Counts |
| --- | --- |
| Immediate wins / uniquely safe immediate blocks | 48 / 48 |
| X / O to move | 48 / 48 |
| Horizontal / vertical / diagonal | 32 / 32 / 32 |
| Each tactic × player × direction group | 8 rows = 4 original/mirror families |
| Correct physical columns 0–6 | `[14, 14, 14, 12, 14, 14, 14]` |

Wins require exactly one engine-winning move. Blocks require no own immediate win, exactly one opposing immediate threat, and exactly one legal response that prevents an immediate winning reply. Threat orientation must be unambiguous. “Safe” means safe against the next reply, not a solved-game guarantee.

The separate tests enumerate engine successors independently and check direction by counting contiguous pieces through the move's landing cell, rather than repeating the constructor's full-board window scan. Both diagonal slopes occur through paired reflection. This holdout is independent of the diagnostic fixtures and model predictions; it cannot be certified disjoint from every historical self-play training state. Distinct families are not a claim of statistically independent draws after quota selection.

The freeze operation completed before testing or inference. Neither the holdout nor the opening protocol was regenerated during resumption. Local ignored artifacts reside in [`experiment-output/phase4c2d-validation/`](../experiment-output/phase4c2d-validation/). They include the original configuration/freeze/test logs; corrected test logs; `source-provenance.json`; `run-console.txt`; full `report.json` and `games.jsonl`; `audit_results.py` and `audit.json`; and `pre-resume-handoff.md`. These artifacts need separate backup because Git ignores them. The fixture, harness, and tests were already present in the resumption commit.

## 3. Correctness gate

New tests: [test_dqn_validation.py](../tests/test_dqn_validation.py). They cover all 96 independent tactical labels, direction checks, frozen identity/coverage, old-suite exclusion, duplicate and reflected-duplicate rejection, malformed labels, legal nonterminal opening prefixes, balanced assignments, reproducible schedules and games, terminal replay, deadline partials, accurate W/L/D accounting, inference without gradients/optimizer steps, safe-loaded tensor immutability, margins, and comparison lists.

Original failed-gate invocations and results (preserved history):

```sh
/tmp/board-game-phase4c1-venv/bin/python -m pytest -q \
  tests/test_connect4_dqn.py tests/test_dqn_experiment.py \
  tests/test_dqn_diagnostics.py tests/test_dqn_symmetry.py \
  tests/test_dqn_validation.py tests/test_dqn_import_isolation.py
# 368 passed, 1 failed in 4.36s
# Failure: test_api_starts_without_neural_modules
# Cause: DQN-only environment lacks dotenv, imported by api/app.py.

.venv/bin/python -m pytest -q tests
# 353 passed, 5 optional DQN modules skipped in 6.08s
# Includes the passing existing API import-isolation test.
```

The 111 new validation checks passed; the existing 257 DQN checks passed. The extra API import-isolation invocation in the DQN-only environment caused the failing exit code. Existing suites ran their normal synthetic test operations; no training experiment or new trained candidate was created. Production dependencies were not changed. After the user's explicit review and resumption authorization, the corrected gate ran in the intended environments:

```sh
/tmp/board-game-phase4c1-venv/bin/python -m pytest -q \
  tests/test_connect4_dqn.py tests/test_dqn_experiment.py \
  tests/test_dqn_diagnostics.py tests/test_dqn_symmetry.py \
  tests/test_dqn_validation.py
# 368 passed in 4.53s

.venv/bin/python -m pytest -q tests
# 353 passed, 5 optional DQN modules skipped in 6.16s

.venv/bin/python -m pytest -q tests/test_dqn_import_isolation.py
# 1 passed in 0.42s

git diff --check
# Passed
```

The API isolation test still actively raises on imports of torch/torchvision/torchaudio and DQN submodules/agent, checks health endpoint startup, and asserts torch is absent from loaded modules. No assertion was removed, skipped or weakened. Nothing was added to `requirements-dqn.txt`.

The post-run read-only audit replayed **all 220 transcripts**, checked legality, actor attribution, opening schedule/seeds, terminal outcomes, side counts and W/L/D accounting, and reproduced every Random and Negamax opponent action using the existing agents. It verified tactical argmax choices/margins against recorded raw Q-values, comparison lists, artifact integrity and historical diagnostics. Both models' complete original-suite rows, including Q-vectors and selected actions, match the Phase 4C.2c report **exactly**. The audit did not load models or start additional games.

## 4. Baseline versus augmentation tactical results

B = baseline without augmentation; A = augmented checkpoint. All actions are raw greedy DQN inference with legal masking, zero exploration and no learning or tactical safeguards. The original 60-position suite has 48 labeled tactical rows and 12 unlabeled probes; the holdout has 96 labeled rows. Accuracy denominators exclude unlabeled probes.

### Original diagnostic suite

| Group | B accuracy | A accuracy | B mean margin | A mean margin |
| --- | --- | --- | --- | --- |
| Overall | 15/48 (31.25%) | 26/48 (54.17%) | -0.248615 | 0.009956 |
| win | 7/24 (29.17%) | 12/24 (50.00%) | -0.249034 | -0.025116 |
| block | 8/24 (33.33%) | 14/24 (58.33%) | -0.248196 | 0.045028 |
| horizontal | 6/16 (37.50%) | 7/16 (43.75%) | -0.261935 | -0.166580 |
| vertical | 7/16 (43.75%) | 12/16 (75.00%) | -0.221760 | 0.258901 |
| diagonal | 2/16 (12.50%) | 7/16 (43.75%) | -0.262150 | -0.062453 |
| X | 6/24 (25.00%) | 10/24 (41.67%) | -0.253374 | -0.038599 |
| O | 9/24 (37.50%) | 16/24 (66.67%) | -0.243856 | 0.058511 |
| Column 0 | 1/4 (25.00%) | 3/4 (75.00%) | -0.191570 | 0.486548 |
| Column 1 | 0/3 (0.00%) | 1/3 (33.33%) | -0.633204 | -0.430763 |
| Column 2 | 5/12 (41.67%) | 9/12 (75.00%) | -0.067349 | 0.147395 |
| Column 3 | 2/10 (20.00%) | 5/10 (50.00%) | -0.329225 | -0.039909 |
| Column 4 | 4/12 (33.33%) | 8/12 (66.67%) | -0.330538 | 0.054466 |
| Column 5 | 2/3 (66.67%) | 0/3 (0.00%) | -0.068106 | -0.318839 |
| Column 6 | 1/4 (25.00%) | 0/4 (0.00%) | -0.249103 | -0.310684 |

Overall margin range: B **-1.150934 to 0.422447**; A **-0.865629 to 0.844254**.

### Independent holdout

| Group | B accuracy | A accuracy | B mean margin | A mean margin |
| --- | --- | --- | --- | --- |
| Overall | 24/96 (25.00%) | 31/96 (32.29%) | -0.341036 | -0.162705 |
| win | 10/48 (20.83%) | 18/48 (37.50%) | -0.419037 | -0.145580 |
| block | 14/48 (29.17%) | 13/48 (27.08%) | -0.263035 | -0.179829 |
| horizontal | 7/32 (21.88%) | 5/32 (15.62%) | -0.299123 | -0.174735 |
| vertical | 11/32 (34.38%) | 18/32 (56.25%) | -0.316994 | -0.016423 |
| diagonal | 6/32 (18.75%) | 8/32 (25.00%) | -0.406991 | -0.296956 |
| X | 8/48 (16.67%) | 18/48 (37.50%) | -0.398332 | -0.129533 |
| O | 16/48 (33.33%) | 13/48 (27.08%) | -0.283740 | -0.195876 |
| Column 0 | 5/14 (35.71%) | 3/14 (21.43%) | -0.395674 | -0.265359 |
| Column 1 | 2/14 (14.29%) | 4/14 (28.57%) | -0.486292 | -0.011211 |
| Column 2 | 5/14 (35.71%) | 6/14 (42.86%) | -0.126624 | -0.102125 |
| Column 3 | 5/12 (41.67%) | 4/12 (33.33%) | -0.121023 | -0.360692 |
| Column 4 | 3/14 (21.43%) | 10/14 (71.43%) | -0.577752 | 0.109229 |
| Column 5 | 2/14 (14.29%) | 3/14 (21.43%) | -0.260291 | -0.184074 |
| Column 6 | 2/14 (14.29%) | 1/14 (7.14%) | -0.388166 | -0.352985 |

Overall margin range: B **-3.492206 to 1.158755**; A **-1.570494 to 0.728591**.

Margin is `Q(correct) − max Q(other legal action)`. Positive values favor the labeled action. A higher mean margin does not ensure higher accuracy in every group; the holdout block margin improves while its correct count falls. Absolute margins also reflect Q scale.

The original-suite gain is reproduced exactly: +11/48 rows (+22.92 points). On the new holdout, the gain is only +7/96 (+7.29 points). Immediate wins improve 10→18, but uniquely safe blocks fall 14→13. Vertical gains (11→18) account for the entire net direction gain; diagonal gains (6→8) are offset by horizontal regressions (7→5). X improves 8→18 while O falls 16→13. The augmented checkpoint still misses **30/48 immediate wins and 35/48 forced blocks**.

### Greedy actions and mirror consistency

Histograms use physical columns 0–6 and include all rows, including the original suite's unlabeled probes. Mirrored-Q MAE averages each pair's mean absolute discrepancy over aligned legal columns; maximum discrepancy is the worst aligned legal action across all pairs.

| Suite / model | Greedy histogram | Entropy (bits) | Mirror action agreement | Aligned-Q MAE | Max discrepancy |
| --- | --- | --- | --- | --- | --- |
| existing / baseline | `[5, 2, 14, 12, 10, 11, 6]` | 2.628308 | 10/30 | 0.576036 | 2.536337 |
| existing / augmented | `[5, 2, 20, 7, 22, 1, 3]` | 2.197523 | 12/30 | 0.433437 | 3.399748 |
| holdout / baseline | `[19, 11, 18, 17, 10, 11, 10]` | 2.753690 | 7/48 | 0.783288 | 4.370158 |
| holdout / augmented | `[11, 9, 27, 9, 31, 5, 4]` | 2.452837 | 16/48 | 0.525572 | 2.264727 |

On the holdout, mirror agreement increases from 14.58% to 33.33%, and aligned-Q MAE falls **32.90%**. This aspect generalizes, although two-thirds of mirror pairs still disagree on reflected actions. Augmented choices concentrate in columns 2 and 4 (58/96 versus baseline 28/96), despite near-balanced correct-column labels. Lowest-column argmax tie breaking can itself break mirror action agreement; aligned-Q differences measure a separate property.

### Correlated families, improvements and regressions

| Suite / model | Tactical families | Both orientations correct | One correct | Neither correct |
| --- | --- | --- | --- | --- |
| existing / baseline | 24 | 5 | 5 | 14 |
| existing / augmented | 24 | 7 | 12 | 5 |
| holdout / baseline | 48 | 3 | 18 | 27 |
| holdout / augmented | 48 | 7 | 17 | 24 |

| Suite | Previously correct | Retained correct | Improved rows | Regressed rows | Neither correct |
| --- | --- | --- | --- | --- | --- |
| existing | 15 | 11 | 15 | 4 | 18 |
| holdout | 24 | 7 | 24 | 17 | 48 |

The holdout comprises **48 original/mirror families**, not 96 independent observations. At the family level, 18 improve their number of correct orientations, 13 regress, and 17 have no net change. Specifically, changes in correct orientations per family are −2: 1 family; −1: 12; 0: 17; +1: 15; +2: 3. Thus the small net gain hides substantial replacement of previously correct behavior: only seven of 24 baseline-correct rows remain correct. A zero family delta can include an orientation exchange.

These are descriptive paired-family results, not independent-row significance tests. The appendix lists every retained, improved and regressed position exactly; “previously correct” is the union of retained and regressed entries. Full fixture moves, Q-vectors and margins are retained in the frozen fixtures and `report.json`.

## 5. Negamax, Random, and head-to-head results

W/L/D below is from the named model's perspective. Every requested game completed, with zero partial games. Negamax results use all 20 prefixes × both side assignments, identically for both checkpoints.

| Model / opponent | Games | Overall W/L/D | As X W/L/D | As O W/L/D | Win rate | Summed game seconds |
| --- | --- | --- | --- | --- | --- | --- |
| baseline / negamax_1 | 40 | 6 / 34 / 0 | 4 / 16 / 0 | 2 / 18 / 0 | 15% | 0.082816 |
| augmented / negamax_1 | 40 | 6 / 34 / 0 | 4 / 16 / 0 | 2 / 18 / 0 | 15% | 0.089439 |
| baseline / negamax_2 | 40 | 2 / 38 / 0 | 2 / 18 / 0 | 0 / 20 / 0 | 5% | 0.255555 |
| augmented / negamax_2 | 40 | 2 / 38 / 0 | 2 / 18 / 0 | 0 / 20 / 0 | 5% | 0.256787 |
| baseline / random | 20 | 17 / 3 / 0 | 8 / 2 / 0 | 9 / 1 / 0 | 85% | 0.025499 |
| augmented / random | 20 | 12 / 8 / 0 | 6 / 4 / 0 | 6 / 4 / 0 | 60% | 0.027419 |

At depth 1, augmentation converts three paired-opening losses to wins and three wins to losses, with 31 shared losses and three shared wins. At depth 2, outcomes are identical for every opening/side assignment: two shared wins and 38 shared losses. **There is no measured stronger-opponent improvement.** Both models lose all 20 O-side depth-2 games.

The small Random check reverses the earlier advantage: baseline wins 17/20, augmented 12/20, with two loss→win conversions and seven win→loss regressions. This uses varied forced prefixes and new seeds, unlike the earlier empty-board 100-game benchmark. The small, different protocol does not establish a broad Random ranking or invalidate the old result; it shows that the old 92% score is not portable to this schedule.

The optional head-to-head completed 20 games in **0.041755 seconds**: baseline **8–12–0**, augmented **12–8–0**. Baseline as X was 3–7–0 and as O 5–5–0; equivalently augmented as X was 5–5–0 and as O 7–3–0. This small direct advantage is insufficient to overcome the tactical failures and identical Negamax aggregates.

Total: **160 Negamax games + 40 Random games + 20 head-to-head games = 220**. Summed game-loop times exclude model loading, tactical diagnostics, opponent construction and artifact overhead; aggregate driver time is **1.777971 seconds**, not the sum of the rounded table values. The Random and head-to-head stages each stayed below their separate 30-second caps and the overall 600-second deadline. No partial or deadline-censored results were omitted.

## 6. Confidence and limitations

The answer to the generalization question is **partial and inconsistent generalization**. The augmentation-associated reduction in symmetry discrepancies appears on the new suite, and overall tactical accuracy improves modestly. The large original-suite accuracy gain does not carry over in magnitude or across categories, players and playing strength. The new results do not support a broadly stronger, tactically reliable policy.

Correctness confidence is supported by the passing independent-label tests, exact reproduction of historical original-suite predictions, full transcript audit and unchanged loaded tensors/files. Empirical confidence remains limited by one training seed; only 48 correlated holdout families; quota-selected one-reply tasks; just ten original Negamax opening families plus mirrors; and shallow existing heuristic opponents. Reflections and side-swapped games are paired, not independent trials. Random and head-to-head use only five original opening families plus mirrors. No independent-row confidence interval or significance claim is made.

The holdout is new and disjoint from the diagnostic suite up to reflection, but there is no exhaustive proof that no board appeared during training. The two training runs also had different realized update counts and trajectories, so this is not a fixed-update causal ablation. Different column/position distributions help explain why original and holdout scores differ; there is no evidence here of fixture leakage or a specific learning defect. Existing Negamax uses its unchanged heuristic terminal scoring and cache semantics; these are reproducible shallow opponents, not an oracle or a competitive-strength rating.

Runtime is a measurement of this small CPU workload on this machine, not a production latency or throughput guarantee. Cooperative deadlines can overshoot by one bounded-depth move; no overshoot occurred. No network requests or paid providers were used.

## 7. Suitability for experimental public gameplay

**Do not expose the augmented checkpoint as a standalone public DQN opponent yet.** It solves only 31/96 holdout tactics, misses most immediate wins and blocks, and loses 85% and 95% of games against the existing depth-1/depth-2 opponents. The direct head-to-head result and improved Q symmetry do not establish acceptable reliability. Keep it available for offline research and comparison.

These measurements describe learned DQN behavior. A future deterministic win/block guard would create a hybrid policy with different performance, requiring separate guarded and unguarded evaluation. Its successes must not be credited to learned tactical competence. No such safeguard, API/UI integration or hosting change was added.

## 8. Learning recommendation and stopping point

**Bounded multi-seed replication is warranted before a longer training campaign or a methodology decision**, subject to separate authorization. Replication would test whether the small holdout gain, subgroup reversals and absent Negamax gain persist across training seeds. A future comparison should predeclare matched budgets and independent evaluation, and report realized updates and trajectories.

Further learning is needed for reliable gameplay, but simply extending this checkpoint's training is not justified by these measurements. This evaluation does not identify a particular Bellman, replay, architecture or optimizer change as the solution. If appropriately controlled replications retain these failures, reassess the learning method and tactical coverage rather than treating symmetry augmentation alone as sufficient. The now-inspected holdout must not become a tuning or checkpoint-selection target; any future selection process needs a separate final holdout.

Stopped after this authorized evaluation and report. Existing source/tests and both fixtures were preserved. No new training run, fine-tuning, model selection, trained-checkpoint creation or modification, Bellman/hyperparameter change, production dependency change, API/UI integration, hosting change, MCTS work, paid request, commit, push, merge or deployment was performed during resumption. The only tracked change is this completed handoff; full results remain in the ignored artifact directory.

## Appendix — Exact position-level changes

Columns are zero-based. Fixture IDs map to the frozen moves; X/O are the player to move. Retained and regressed entries together enumerate every previously correct position. Improved entries were previously incorrect. Rows wrong under both checkpoints are fully retained in `report.json` and omitted from this appendix.

### existing

| Position | Player | Expected | Baseline → augmented | Status |
| --- | --- | --- | --- | --- |
| `block_diagonal_1_0` | O | 2 | 2 → 2 | retained correct |
| `block_horizontal_1_0` | O | 4 | 4 → 4 | retained correct |
| `block_horizontal_1_0_mirror` | O | 2 | 2 → 2 | retained correct |
| `block_vertical_0_0` | X | 4 | 4 → 4 | retained correct |
| `block_vertical_0_0_mirror` | X | 2 | 2 → 2 | retained correct |
| `block_vertical_1_0` | O | 0 | 0 → 0 | retained correct |
| `block_vertical_1_1_mirror` | O | 4 | 4 → 4 | retained correct |
| `win_horizontal_0_1` | X | 3 | 3 → 3 | retained correct |
| `win_horizontal_1_1_mirror` | O | 2 | 2 → 2 | retained correct |
| `win_vertical_0_1` | X | 2 | 2 → 2 | retained correct |
| `win_vertical_0_1_mirror` | X | 4 | 4 → 4 | retained correct |
| `block_diagonal_0_0` | X | 3 | 2 → 3 | improved |
| `block_diagonal_0_1` | X | 2 | 6 → 2 | improved |
| `block_diagonal_1_1` | O | 4 | 2 → 4 | improved |
| `block_diagonal_1_1_mirror` | O | 2 | 5 → 2 | improved |
| `block_horizontal_1_1_mirror` | O | 3 | 2 → 3 | improved |
| `block_vertical_0_1` | X | 3 | 4 → 3 | improved |
| `block_vertical_1_1` | O | 2 | 6 → 2 | improved |
| `win_diagonal_1_0_mirror` | O | 3 | 6 → 3 | improved |
| `win_diagonal_1_1` | O | 1 | 5 → 1 | improved |
| `win_horizontal_0_0` | X | 4 | 3 → 4 | improved |
| `win_horizontal_1_1` | O | 4 | 5 → 4 | improved |
| `win_vertical_0_0_mirror` | X | 0 | 3 → 0 | improved |
| `win_vertical_1_0` | O | 4 | 6 → 4 | improved |
| `win_vertical_1_0_mirror` | O | 2 | 0 → 2 | improved |
| `win_vertical_1_1_mirror` | O | 0 | 1 → 0 | improved |
| `block_vertical_1_0_mirror` | O | 6 | 6 → 4 | regressed |
| `win_diagonal_1_1_mirror` | O | 5 | 5 → 1 | regressed |
| `win_horizontal_0_1_mirror` | X | 3 | 3 → 2 | regressed |
| `win_horizontal_1_0` | O | 5 | 5 → 0 | regressed |

### holdout

| Position | Player | Expected | Baseline → augmented | Status |
| --- | --- | --- | --- | --- |
| `holdout_07_win_vertical_1` | O | 2 | 2 → 2 | retained correct |
| `holdout_15_win_vertical_0` | X | 5 | 5 → 5 | retained correct |
| `holdout_20_win_vertical_0` | X | 4 | 4 → 4 | retained correct |
| `holdout_26_block_vertical_1_mirror` | O | 4 | 4 → 4 | retained correct |
| `holdout_27_win_vertical_0` | X | 4 | 4 → 4 | retained correct |
| `holdout_28_block_vertical_1_mirror` | O | 1 | 1 → 1 | retained correct |
| `holdout_32_block_diagonal_1` | O | 3 | 3 → 3 | retained correct |
| `holdout_00_block_vertical_0` | X | 0 | 4 → 0 | improved |
| `holdout_07_win_vertical_1_mirror` | O | 4 | 2 → 4 | improved |
| `holdout_09_win_horizontal_1` | O | 4 | 1 → 4 | improved |
| `holdout_12_win_diagonal_1` | O | 2 | 5 → 2 | improved |
| `holdout_12_win_diagonal_1_mirror` | O | 4 | 1 → 4 | improved |
| `holdout_13_win_diagonal_1` | O | 3 | 6 → 3 | improved |
| `holdout_14_block_vertical_0_mirror` | X | 1 | 5 → 1 | improved |
| `holdout_16_block_diagonal_0` | X | 4 | 6 → 4 | improved |
| `holdout_18_win_diagonal_0` | X | 1 | 6 → 1 | improved |
| `holdout_20_win_vertical_0_mirror` | X | 2 | 3 → 2 | improved |
| `holdout_23_win_horizontal_1` | O | 6 | 3 → 6 | improved |
| `holdout_24_win_diagonal_0` | X | 4 | 2 → 4 | improved |
| `holdout_25_block_horizontal_0` | X | 5 | 1 → 5 | improved |
| `holdout_27_win_vertical_0_mirror` | X | 2 | 0 → 2 | improved |
| `holdout_28_block_vertical_1` | O | 5 | 3 → 5 | improved |
| `holdout_30_win_diagonal_0` | X | 3 | 4 → 3 | improved |
| `holdout_31_win_horizontal_1` | O | 1 | 0 → 1 | improved |
| `holdout_33_win_vertical_0_mirror` | X | 0 | 6 → 0 | improved |
| `holdout_35_block_horizontal_0` | X | 0 | 6 → 0 | improved |
| `holdout_40_block_vertical_0` | X | 4 | 1 → 4 | improved |
| `holdout_40_block_vertical_0_mirror` | X | 2 | 1 → 2 | improved |
| `holdout_41_block_vertical_0` | X | 2 | 6 → 2 | improved |
| `holdout_41_block_vertical_0_mirror` | X | 4 | 1 → 4 | improved |
| `holdout_45_win_vertical_1` | O | 3 | 0 → 3 | improved |
| `holdout_00_block_vertical_0_mirror` | X | 6 | 6 → 1 | regressed |
| `holdout_02_block_vertical_1_mirror` | O | 0 | 0 → 2 | regressed |
| `holdout_03_win_horizontal_1` | O | 5 | 5 → 4 | regressed |
| `holdout_05_win_diagonal_0_mirror` | X | 0 | 0 → 1 | regressed |
| `holdout_10_block_horizontal_1_mirror` | O | 1 | 1 → 2 | regressed |
| `holdout_13_win_diagonal_1_mirror` | O | 3 | 3 → 2 | regressed |
| `holdout_17_win_horizontal_0_mirror` | X | 6 | 6 → 0 | regressed |
| `holdout_21_block_horizontal_0_mirror` | X | 2 | 2 → 5 | regressed |
| `holdout_23_win_horizontal_1_mirror` | O | 0 | 0 → 4 | regressed |
| `holdout_26_block_vertical_1` | O | 2 | 2 → 6 | regressed |
| `holdout_29_block_horizontal_1` | O | 2 | 2 → 4 | regressed |
| `holdout_32_block_diagonal_1_mirror` | O | 3 | 3 → 2 | regressed |
| `holdout_34_block_diagonal_1_mirror` | O | 0 | 0 → 2 | regressed |
| `holdout_36_win_horizontal_0_mirror` | X | 0 | 0 → 1 | regressed |
| `holdout_38_block_vertical_1` | O | 3 | 3 → 2 | regressed |
| `holdout_38_block_vertical_1_mirror` | O | 3 | 3 → 0 | regressed |
| `holdout_43_block_diagonal_1_mirror` | O | 2 | 2 → 4 | regressed |
