# Phase 4C.2d — Independent DQN validation (stopped at verification gate)

**No trained-checkpoint evaluation was performed.** The verification gate returned a failure, and the user explicitly required stopping without evaluation if verification failed. The independent holdout and opening protocol were frozen before the gate. No candidate predictions were inspected during construction, acceptance, or testing.

The new evaluation checks passed, but an existing API import-isolation test was additionally invoked in the optional DQN environment, which does not contain `python-dotenv`. Its subprocess failed with `ModuleNotFoundError: No module named 'dotenv'`. The same test passed as part of the complete backend suite in `.venv`. This is an environment/invocation failure, not evidence of checkpoint performance or of a neural-import regression. Nevertheless, the aggregate gate did not pass. It was not bypassed or retried, and no checkpoint evaluation followed.

## 1. Preflight and exact prepared configuration

Fetched `origin/main`; it exactly matched the requested reference:

```text
76ba2b6902fc0ce819d2b24212c8c4885e939f8b
```

The initial working tree was clean. Created local branch `phase4c2d-dqn-validation` from that reference. Reviewed the Phase 4C.2b and Phase 4C.2c handoffs. All implementation changes are new evaluation files; existing training, networks, opponents, checkpoint loader, and diagnostics are unchanged.

Preserved files, verified before construction and again at handoff:

| Artifact | SHA-256 |
| --- | --- |
| Baseline `experiment-output/phase4c2b-seed42-20261003-2059/candidate.pt` | `29c1b6c3288811b449b4d90744607e572fca17bc631d5cff4334fc2b4efaf56e` |
| Augmented `experiment-output/phase4c2c-seed42-20261003/candidate.pt` | `3327863c8d6ed45b70871cd7b336d35950f2c9962a204ebcc67baa86f6c86f08` |
| Original 60-position `diagnostic_positions.json` | `ec99e0ee080370ab8cfc678c00a49ff552466de73ffe0955778ac87ee340543a` |

The prepared [evaluation harness](../games/connect4/dqn/validation.py) uses only the existing versioned `load_checkpoint`, whose `torch.load` call requires `weights_only=True` and CPU loading, with no unsafe fallback. Neither trained checkpoint was loaded in this milestone. Tests of the loading path use a synthetic in-memory payload without writing a checkpoint.

| Prepared setting | Value |
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

The driver would record every opening move, every subsequent move with actor and agent role, terminal winner/result, seed, model/opponent assignment, and runtime. It compares all loaded state tensors and preserved file hashes afterward. No game transcript or result file was produced because evaluation never started.

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

The freeze operation completed before testing or inference. The local ignored artifact directory is `experiment-output/phase4c2d-validation/`, containing `configuration.json`, `freeze.json`, `dqn-tests.txt`, and `backend-tests.txt`. These files need separate preservation from Git; the new fixture and construction source are in the working tree for review.

## 3. Correctness gate

New tests: [test_dqn_validation.py](../tests/test_dqn_validation.py). They cover all 96 independent tactical labels, direction checks, frozen identity/coverage, old-suite exclusion, duplicate and reflected-duplicate rejection, malformed labels, legal nonterminal opening prefixes, balanced assignments, reproducible schedules and games, terminal replay, deadline partials, accurate W/L/D accounting, inference without gradients/optimizer steps, safe-loaded tensor immutability, margins, and comparison lists.

Actual invocations and results:

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

The 111 new validation checks passed; the existing 257 DQN checks passed. The extra API import-isolation invocation in the DQN-only environment caused the failing exit code. Existing suites ran their normal synthetic test operations; no training experiment or new trained candidate was created. Production dependencies were not changed. A future continuation should use each suite's established environment, but that corrected gate and any actual evaluation were not run here.

## 4. Baseline versus augmentation tactical results

**Not measured in this milestone.** No accuracy, Q margin, greedy distribution, mirror agreement/discrepancy, or exact improved/regressed position result can be reported from the new holdout.

The harness is prepared to retain raw Q-vectors, masks, selected actions and margins for both suites, with overall/category/player/direction/column accuracy, histograms and entropy, pair-level aligned legal-Q differences, and exact previously correct/retained/improved/regressed/neither-correct position lists. The old suite contains 48 labeled tactical rows plus 12 unlabeled probes; the new suite contains 96 labeled tactical rows. These denominators will be kept distinct.

Historical Phase 4C.2c results remain in [its handoff](phase4c2c-dqn-symmetry.md). They have not been independently reproduced here and are not holdout evidence.

## 5. Negamax, Random, and optional head-to-head results

| Evaluation | Baseline completed | Augmented completed | W/L/D | Evaluation runtime |
| --- | --- | --- | --- | --- |
| Negamax depth 1 | 0 | 0 | Not measured | Not started |
| Negamax depth 2 | 0 | 0 | Not measured | Not started |
| Random | 0 | 0 | Not measured | Not started |
| Baseline vs augmented | 0 total | 0 total | Not measured | Not started |

Synthetic correctness-test games are not trained-model strength measurements and are excluded from these counts. No evaluation deadline was consumed by an actual checkpoint run.

## 6. Confidence and limitations

There is strong test evidence for the holdout's legal reachability, one-reply labels, coverage, and exclusion of old diagnostic boards. There is **no new evidence about learned performance or generalization**. The gate's environment failure prevents this milestone from answering its core empirical question.

Even a completed run would remain a single-training-seed comparison, with correlated reflected rows and paired openings, a small number of opening families, one-reply tactical labels, and the existing shallow heuristic Negamax implementations. Identical openings/seeds do not force identical trajectories after model choices diverge. The original training runs also had different realized update counts. Such results should be reported descriptively without treating reflected rows as independent trials or attributing all differences causally to augmentation.

## 7. Suitability for experimental public gameplay

**Not validated for public gameplay by this milestone.** The prior handoff already records substantial missed immediate wins/blocks and recommends against integration. This stopped run supplies no evidence to change that conclusion. No API/UI or hosting work was performed.

All prepared measurements use the learned DQN's raw greedy policy. Future deterministic tactical guards could improve a combined agent's behavior, but that would be a different system and would not establish that the DQN learned those tactics. No such guard was added here.

## 8. Multi-seed replication and stopping point

Multi-seed replication remains scientifically warranted to assess whether the earlier single-seed improvement is repeatable, as proposed in Phase 4C.2c. This milestone has not strengthened or refuted that rationale. Complete the independent validation under a passing gate before using it to design or justify further training. No additional training is authorized or started by this report.

Stopped after the verification failure. No actual checkpoint evaluation, new training run, checkpoint modification, diagnostic-fixture modification, hyperparameter or Bellman change, production dependency change, API/UI integration, hosting change, MCTS work, paid request, commit, push, merge, or deployment occurred.
