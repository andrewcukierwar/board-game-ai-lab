# Phase 4C.2 — Bounded offline DQN training

Implemented and ran **exactly one experiment on October 3, 2026 (America/New_York)**. The result is **concerning policy bias, with useful tactical learning still inconclusive**. The candidate remained numerically finite and improved its tiny seeded Random score, but still missed all four explicit tactical probes and preferred column 4 on nine of ten fixed positions. This is not evidence of a strong or production-ready agent.

## Preflight and scope

Read [the Phase 4C.1 foundation handoff](phase4c1-dqn-foundation.md). The working tree was clean. Fetched `origin/main`, which was exactly the supplied reference `7461549af833a41f8c0d5ecabcf7a10e573c2563`, and created local branch `phase4c2-dqn-training` from it.

All work remains uncommitted. No push, merge, deployment, paid provider request, production dependency change, API/UI integration, VictorAgent change, or neural-MCTS training occurred. Historical checkpoints are untouched. The existing optional `/tmp/board-game-phase4c1-venv` environment was used; the backend environment remains separate.

## Driver implementation

- [experiment.py](../games/connect4/dqn/experiment.py) provides an import-inert CLI, validated game/ply/update/time bounds, persistent trainer lifecycle, per-ply numerical checks, logging, source provenance, graceful stop requests, and candidate saving/reloading.
- [diagnostics.py](../games/connect4/dqn/diagnostics.py) provides ten fixed legal positions and bounded greedy evaluation against uniform legal Random.
- [test_dqn_experiment.py](../tests/test_dqn_experiment.py) adds 31 synthetic checks. Test outcomes do not depend on learning strength.
- `.gitignore` explicitly excludes `/experiment-output/`.

Each episode initializes `Connect4`. Both players use the **same** `DQNTrainer.choose_move()` policy with the existing canonical current-player encoding. Every collected ply goes through `collect_transition()` and `trainer.memory.append()`. There is at most one call to `optimize()` per collected ply. The existing trainer owns the single Adam optimizer, replay memory, epsilon progression and target synchronization schedule for the entire experiment. No Bellman-target, reward, mask, loss or checkpoint mathematics were duplicated or changed.

Before collection, the driver checks all seven online Q-values even when epsilon exploration selects the action. The foundation checks target Q-values, selected loss and gradients. The driver checks online/target parameters and buffers at startup and after optimizer calls, plus gradients after updates; nonfinite values produce an invariant-failure report and prevent candidate saving. CPU-only state is enforced. Each successful ply logs actor, action, reward, terminal status, Q extrema, loss, maximum absolute gradient, replay size, epsilon, update count, synchronization event and elapsed time.

Every completed episode records its full move sequence, winner/draw, length, initial state, start/end elapsed time, update and epsilon boundaries, and mean loss. Partial episodes are separate and never counted as completed games or draws. SIGINT/SIGTERM set a cooperative stop flag rather than raising in the middle of an optimizer step. The report retains completed operations and any partial game. The CLI never retries a run and refuses an existing output directory or candidate path.

Limits are checked before action collection and again before optimization. Reaching a ply/time limit after collection skips optimization for that ply. Wall-time and signal bounds are cooperative: an already-running Python/PyTorch operation can finish just after a deadline; no new training operation starts once the deadline is observed. This run ended far below the time limit.

## Exact experiment configuration

The CLI invocation below was executed **once**; it is an execution record, not authorization to run it again:

```sh
/tmp/board-game-phase4c1-venv/bin/python -m games.connect4.dqn.experiment \
  --output experiment-output/phase4c2-seed42 \
  --seed 42 \
  --max-games 200 --max-plies 8400 --max-updates 8400 --max-seconds 600 \
  --epsilon-start 1.0 --epsilon-min 0.10 --epsilon-decay 0.9997 \
  --threads 1 --interop-threads 1 \
  --evaluation-games 24 --evaluation-seconds 120
```

| Setting | Value |
|---|---|
| Seed | 42 |
| Replay capacity / batch size | 10,000 / 64 |
| Gamma / Adam learning rate | 0.99 / 0.001 |
| Epsilon start / floor / per-update decay | 1.0 / 0.10 / 0.9997 |
| Hard target synchronization | Every 100 successful optimizer updates |
| Training limits | 200 games, 8,400 plies, 8,400 updates, 600 seconds; first reached stops |
| Device / dtype | Native CPU / float32; no MPS |
| PyTorch intra-op / inter-op threads | 1 / 1 |
| Deterministic algorithms | Enabled |
| Runtime | Python 3.11.17; PyTorch 2.10.0; macOS 26.6.2 ARM64 |
| Evaluation | 12 before + 12 after, six DQN games per starting side per model |
| Evaluation randomness | Private `random.Random(42 + game_index)`, reset to the same suite per model |
| Evaluation time bounds | At most 60 seconds per model, with remaining combined 120-second allowance enforced |

The Random policy samples uniformly from the engine's legal-move enumeration, matching existing RandomAgent behavior while using an isolated RNG. Evaluation uses greedy inference with no replay insertion, optimizer update, or consumption of the trainer's exploration/replay RNGs. Diverging policies can produce different game trajectories despite shared seed indices.

Source provenance at execution:

```text
HEAD:                  7461549af833a41f8c0d5ecabcf7a10e573c2563
Dirty source identity: 4339a69deddb8a8b73009774000ce2def556e943d10716521c25cb61366509c1
Tracked diff SHA-256:  f747a7d7b133aed647f3b34dbf66899cd1a2752bae6a5c1ccfe21d8790cf2f5e
```

The experiment captured branch/status, the binary-capable `git diff HEAD` patch, hashes and copies of the three untracked implementation/test files. The dirty diff was the output-ignore rule. Thus HEAD alone is **not** the experiment's source identity. `configuration.json` and checkpoint metadata preserve the full manifest. The driver/test file hashes were independently compared with the archived source after the run. This handoff document was written afterward and is not part of that source snapshot.

## Actual completed budget and numerical diagnostics

| Measurement | Actual |
|---|---:|
| Stop reason | `max_games` |
| Started / completed games | 200 / 200 |
| Partial games | 0 |
| Collected training plies | 3,779 |
| Successful optimizer updates | 3,716 |
| Final replay size | 3,779 |
| Self-play X wins / O wins / draws | 113 / 87 / 0 |
| Training elapsed time, monotonic | 4.078934833 seconds |
| Total driver experiment time after trainer setup | 4.158623708 seconds |
| Final epsilon | 0.32792601716902486 |
| Scheduled target synchronizations | 37: updates 100, 200, …, 3,700 |
| Nonfinite invariant failures | 0 |

The first 63 plies warmed replay without updates; every remaining ply received exactly one update. Target initialization also copied the online network before training; it is distinct from the 37 scheduled copies. No training state reset occurred between games. The epsilon floor was not reached.

| Completed games | Updates | Epsilon |
|---:|---:|---:|
| 1 | 0 | 1.000000 |
| 25 | 450 | 0.873698 |
| 50 | 900 | 0.763349 |
| 100 | 1,831 | 0.577306 |
| 150 | 2,677 | 0.447884 |
| 200 | 3,716 | 0.327926 |

All 3,716 scalar losses and all per-ply epsilon values are retained. Mean selected-action Smooth L1 loss was **0.006552968**, median **0.005359881**, minimum **0.000393894**, maximum **0.040097263**. The first/last losses were **0.027449012 / 0.007845887**. The first 100 updates averaged **0.004474907** and the last 100 averaged **0.007335600**. Across successive 500-update blocks, means stayed approximately **0.0060–0.0072**; there was no numerical explosion, but also no monotonic improvement. The replay distribution and target network change throughout training, so these losses are not a fixed validation metric.

Observed pre-action Q-values during training ranged from **−2.239085436 to +1.858297944**, and the largest recorded absolute gradient component was **0.043145135**. Finite raw Q-values are not necessarily calibrated; the transient Q range exceeded the magnitude of the terminal reward. Diagnostics and checkpoint weights were also finite.

## Fixed-position diagnostics

Each position is constructed by replaying checked legal moves from the empty game; tests independently verify the unique immediate-winning or safe-blocking action. All column indices below are **zero-based**. Full-column fixtures fill column 0 legally with alternating pieces. Exact sequences and all seven initial/final Q-values are in the report and in the fixed source fixtures.

| Probe | Player to move | Required tactical action | Initial greedy action | Final greedy action |
|---|---|---:|---:|---:|
| Empty opening | X | — | 4 | 4 |
| Opening after column 3 | O | — | 4 | 4 |
| Middlegame | X | — | 6 | 4 |
| Middlegame | O | — | 4 | 4 |
| Immediate horizontal win | X | 3 | 2 | 4 |
| Immediate horizontal win | O | 3 | 4 | 4 |
| Forced horizontal block | X | 3 | 4 | 4 |
| Forced horizontal block | O | 3 | 4 | 4 |
| Full column 0 | X | — | 4 | 4 |
| Full column 0 | O | — | 5 | 3 |

All Q-vectors changed; greedy actions changed on **3/10** probes. All selected actions were legal. Tactical accuracy remained **0/4 → 0/4**: neither improvement nor binary regression was observed on these four examples. Final Q-values for the immediate winning action were still negative (**−0.184656 for X, −0.036699 for O**), whereas its actor-relative terminal target is +1. That is a concrete failure to generalize obvious wins.

Column 4 preference increased from **7/10 to 9/10** probes, a concerning concentration. This is **not proof of global single-column collapse**: across the final Random evaluation the model selected six different columns, with action counts `{0: 5, 1: 4, 2: 22, 3: 16, 4: 24, 5: 17}`. Self-play selected all seven columns, but epsilon exploration makes that weaker evidence against greedy collapse. The small probe set cannot establish broad policy diversity.

## Exploratory Random evaluation

Exactly **24 evaluation games total** completed, with no partial games and no learning. The initial suite took **0.016953750 seconds**, and the final suite **0.014422250 seconds**, totaling **0.031376000 seconds**. No subsequent evaluation games were run.

| Model | Wins | Losses | Draws | DQN as X | DQN as O |
|---|---:|---:|---:|---|---|
| Initial untrained | 8 | 4 | 0 | 4 wins / 2 losses | 4 wins / 2 losses |
| Final candidate | 11 | 1 | 0 | 6 wins / 0 losses | 5 wins / 1 loss |

The score rose from **66.7% to 91.7%**, which is a favorable exploratory signal. Twelve games per model against a weak opponent, with different induced trajectories and an already favorable untrained baseline, cannot establish strength or attribute the change to general tactical learning. The unchanged failed tactical decisions and concentrated probe actions prevent a positive learning-quality conclusion. Training loss alone was not used as evidence of improvement.

## Candidate checkpoint and artifacts

Output directory, intentionally Git-ignored and local-only:

```text
/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/experiment-output/phase4c2-seed42/
```

Files include:

- `candidate.pt`: one final inference candidate, **245,229 bytes**, using the unchanged Phase 4C.1 versioned contract.
- `configuration.json`: full pre-run configuration, runtime and source identity.
- `events.jsonl`: all 3,779 ply and 200 completed-episode events.
- `report.json`: actual budget, losses, outcomes, full diagnostics/evaluation transcripts, checkpoint location, SHA-256 and reload result.
- `source.patch` and `untracked-source/`: exact dirty implementation/test provenance.

Candidate SHA-256:

```text
1be2836c252575e65ce6c46891dcca1ab5b426e08bffb552cee0b3ec570267a9
```

`save_checkpoint()` stored final model weights plus configuration/source/training/diagnostic/evaluation metadata. `load_checkpoint()` reloaded the candidate with **exact equality of every weight tensor, metadata, and all ten diagnostic Q-vectors/actions**. A separate read-only audit rechecked its SHA-256, loaded predictions, source hashes, episode/ply/update/synchronization counts and evaluation budget.

The SHA-256 and reload confirmation are in the external report, not recursively embedded in the hashed checkpoint. Checkpoint metadata reflects the report immediately before saving (`checkpoint: null`); the external report adds checkpoint integrity details and total driver time afterward. The checkpoint contains **inference weights only**, not optimizer, replay, target-network or RNG resume state. Exact training resumability is not claimed. Ignored artifacts must be preserved separately if they need to survive this workspace.

## Verification

All synthetic tests passed **before** the experiment began. There was no failed synthetic-test gate, restart, sweep or automatic repeat.

```sh
/tmp/board-game-phase4c1-venv/bin/python -m pytest -q \
  tests/test_dqn_experiment.py tests/test_connect4_dqn.py
# 103 passed: 31 new driver/diagnostic tests + all 72 existing DQN tests

.venv/bin/python -m pytest -q tests
# 353 passed, 2 optional DQN modules skipped in the backend-only environment

.venv/bin/python -m pytest -q tests/test_dqn_import_isolation.py
# 1 passed: API health/startup with neural imports actively blocked

git diff --check
# Passed
```

Coverage includes independently triggered game/ply/update/time stops, warmup, partial-game interruption, both actors and draws, persistent optimizer/replay/synchronization, deterministic fixed-fixture updates, nonfinite weights/Q/gradients/losses including post-step corruption, legal tactical fixtures, seeded balanced inference-only evaluation, evaluation timeout accounting, source provenance, inference checkpoint round-trip, output overwrite refusal and suppression of candidates after invariant failure. Backend tests retained their autouse live-HTTPS prohibition; no providers were called.

## Limitations and recommendation

This was a short infrastructure and learning-signal experiment, not a training campaign. There was one seed, one configuration, 200 games, no held-out validation-loss set, and no stronger-opponent benchmark. Exploration still accounted for a substantial fraction of late training. Four tactical probes all concern horizontal threats in column 3, so the set is deliberately small and does not cover vertical/diagonal tactics, mirrors, varied threat columns or long forcing sequences. CPU determinism within this runtime is supported by controlled tests, not a claim of bitwise reproducibility across PyTorch/platform versions. Time bounds are cooperative, and abrupt process termination such as SIGKILL cannot produce a clean final report.

**Recommendation: review the column bias and failed win/block decisions before authorizing a longer run.** A separately approved next experiment should first fix a broader diagnostic suite with mirrored positions and multiple winning/blocking columns, and track greedy action distributions independently of epsilon exploration. Then compare a single predeclared bounded configuration against this archived candidate. Do not choose more training merely because the Random score rose, and do not tune to these ten probes or claim tactical competence from this result.

The current foundation showed no mathematical/numerical invariant failure requiring a correction. The observed policy weaknesses warrant investigation, not unverified changes to the corrected negamax target. Stop here for review; no further training, API integration or production promotion is authorized by this handoff.
