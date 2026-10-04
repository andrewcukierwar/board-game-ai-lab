# Phase 4D.2 — First bounded neural self-play experiment

Completed October 4, 2026. **Exactly one actual experiment was launched. The complete learning pipeline passed; learned playing strength was not established.** The run completed 20 games, collected 382 legal plies, and performed 170 optimizer updates. The two-game collection-only smoke gate passed. Training stopped at the completed-game cap; no restart, sweep, extension or subsequent training occurred.

## Implementation and preflight

Fetched `origin/main`; it matched the expected `09157aefdecc8b8a33909c874a1489007fad81e7`. The starting working tree was clean. Created local branch `phase4d2-neural-training` from that commit. Reviewed the [Phase 4B audit](phase4b/audit.md), [Phase 4D.1 foundation](phase4d1-neural-mcts-foundation.md), search, inference and training implementations. No applicable AGENTS.md was found.

Implementation:

- [Offline driver](../games/connect4/neural_self_play.py): explicit CLI launch, bounded collection, game-history replay validation, two-game gate, sequential updates, fixed diagnostics, RNG/source provenance, JSONL measurements and reload verification. Importing the module launches nothing.
- [Canonical checkpoint writer](../games/connect4/neural_mcts.py): validates using the existing reader's tensor contract, independently clones CPU tensors, copies contract metadata and refuses to overwrite a file.
- [Authoritative trainer](../games/connect4/train_mcts_nn.py): a minimal optional loss-components return and `last_metrics` expose separate head losses, target entropy, gradient norm and finite checks. The existing scalar loss/step interface remains compatible; loss mathematics and persistent Adam are unchanged.
- [Synthetic driver/writer tests](../tests/test_neural_self_play.py), plus the new driver module added to [backend neural import isolation](../tests/test_neural_mcts_import_isolation.py).

One freshly initialized canonical `Connect4Net` was shared by X and O. There is no checkpoint load on its initialization path. All 167 pre-existing files under `models/` and `experiment-output/` were compared by SHA-256 before and after: **zero changes**, including ten historical neural checkpoints and every existing Phase 4C DQN artifact. No DQN implementation, agent factory, public API/UI, production dependency, hosting or provider integration changed. No commit, push, merge or deployment was performed.

## Exact experiment configuration

| Setting | Declared value |
| --- | --- |
| Model | Fresh canonical Connect4Net; unchanged 326,026-parameter architecture |
| Encoding / values | `connect4-current-player-1x6x7-v1`; actor-relative W/D/L |
| Seeds | Python 42; NumPy 42; PyTorch 42; independent search Random(42); independent sampling Random(42) |
| Diagnostics | Separate Random(42) per fixture, reset identically before/after; does not advance collection RNG |
| Search | 32 simulations per move; PUCT coefficient 1.41; temperature 1.0 throughout |
| Targets | Existing `root-visits-temperature-v1`; exactly the sampled-action distribution |
| Disabled | Tactical guard, root Dirichlet noise, horizontal augmentation |
| Training | Persistent Adam, learning rate 0.001; equal policy/value head losses; batch size 32 |
| Sampling | Uniform indices without replacement within each batch, from all completed examples; repeated batches allowed |
| Schedule | Games 1–2: zero updates. After each later complete game: up to ten updates, checking every global limit first |
| Global limits | 20 completed games; 840 actual plies; 200 updates; 900 training seconds; first limit stops new work |
| Runtime | Native macOS ARM64; Python 3.11.17; torch 2.10.0; NumPy 1.26.4 |
| Execution | CPU float32; one intra-op thread; one inter-op thread; deterministic algorithms enabled |
| Collection | Bounded to the ply cap, completed validated games only; no eviction/augmentation |

The existing optional `/tmp/board-game-phase4c1-venv` environment was reused. Backend verification used `.venv`. No dependencies were installed or changed.

The single actual launch was:

```sh
PYTHONHASHSEED=42 /tmp/board-game-phase4c1-venv/bin/python \
  -m games.connect4.neural_self_play \
  --output experiment-output/phase4d2-neural-seed42-20261004
```

This is a record of the completed launch, not authorization to run it again. The output directory was new and already ignored by `/experiment-output/`. The driver rejects an existing run directory. Limits can only be reduced in synthetic tests; the CLI exposes no tuning or budget-extension options.

`rng-initial.json` records separate full RNG states before model initialization and before collection; `rng-final.json` records their end states. `config.json` records configuration, runtime, source identity, fresh initialization weight digest and `checkpoint_initialization: null`. The tracked source diff and new source/test bytes were archived before collection and hashed in the manifest. Deterministic replay is intended within this source/runtime and fixed operation order; cross-runtime bitwise reproduction was not tested. **Exact interrupted training resumability is not implemented or claimed.**

## Verification gate before the actual launch

All provider interactions remained mocked/blocked by the existing autouse HTTPS prohibition. Synthetic controlled-history tests use fabricated root distributions, small discarded in-memory updates and temporary synthetic checkpoints; none constitutes a preliminary real self-play training run.

| Gate | Command | Result |
| --- | --- | --- |
| Existing neural correctness + new driver/writer | Optional environment: `python -m pytest -q tests/test_neural_self_play.py tests/test_connect4_neural_mcts.py` | **206 passed**, 2.97 s (175 existing + 31 new) |
| Existing DQN regressions | Optional environment: `python -m pytest -q tests/test_connect4_dqn.py tests/test_dqn_experiment.py tests/test_dqn_diagnostics.py tests/test_dqn_symmetry.py tests/test_dqn_validation.py tests/test_dqn_replication.py` | **466 passed**, 5.75 s |
| Backend regressions | `.venv/bin/python -m pytest -q tests` | **354 passed, 8 skipped**, 6.73 s |
| Explicit neural and DQN API isolation | `.venv/bin/python -m pytest -q tests/test_dqn_import_isolation.py tests/test_neural_mcts_import_isolation.py` | **2 passed**, 0.72 s |
| Whitespace | `git diff --check` | Passed |

Backend skips are optional PyTorch-dependent test modules, including the new driver tests; their corresponding gates ran in the optional environment. No correctly configured verification test failed. New tests cover both winners and draws, pre-move replay alignment, capture corruption/mutation, illegal/nonfinite policy data, visits/guards/labels, persistent optimizer/sampling order, schedule/limits, deadline/interruption quarantine, nonfinite-gradient abort, diagnostics and minimal copied checkpoint schema/reload/overwrite protection.

## Actual budget and smoke gate

| Measurement | Actual |
| --- | --- |
| Stop status / reason | `bounded_stop` / `max_games` |
| Started / completed games | 20 / 20 |
| Applied / collected plies | 382 / 382 |
| Optimizer updates | 170 |
| Collection + optimization duration | 7.471 s |
| End-to-end driver duration | 8.415 s |
| Partial games | None |
| Numerical / mathematical failures | None |

The timer for the 900-second training limit covers collection, validation, event writing and optimization. End-to-end timing also includes initialization, source archiving, both diagnostic phases and checkpoint writing/reload, but starts after module imports. Bounds are cooperative: an already-running search/update can finish beyond a deadline; no new operation begins afterward. The synthetic deadline test verifies an expired search cannot then apply a move or optimize.

**Smoke gate passed after games 1–2, with 34 labeled plies and zero optimizer updates.** Both were actual O wins (18 and 16 plies). X received 17 loss targets; O received 17 win targets. Adam had no optimizer state at the gate. For each game the driver replayed its actual legal history and verified:

- Every captured observation and actor matched the pre-move state; capture happened before exactly one validated move.
- Policies were finite, normalized, exactly the existing visit-target calculation, and zero on illegal columns.
- Root visits, actual root-child visits and recorded visits each summed to 32.
- Both actors were represented; every resolved target matched that completed game's actual winner.
- Fingerprints remained unchanged after each subsequent move; finalization preserved observations and policies.
- Every search reported `tactical_guard=False` and `guard_applied="disabled"`.

These checks continued for all later games. Unfinished games cannot enter the collection; no update occurs during an episode. Updates ran in fixed blocks of ten after games 3–19. At game 20 the game limit was reached before its update block, giving 170 updates rather than 180 or 200. The final game's 31 examples are validated and collected but were never sampled for optimization; the last update used a collection of 351 examples.

## Outcomes and completed labels

| Outcome | Games |
| --- | ---: |
| X win | 16 |
| O win | 4 |
| Draw | 0 |

Game plies in order: `18, 16, 11, 13, 9, 9, 26, 23, 21, 25, 17, 25, 21, 17, 19, 25, 22, 27, 7, 31`. O won games 1, 2, 7 and 17. All other games were X wins. Mean game length was 19.1 plies; the range was 7–31.

| Captured actor | Win target +1 / class 0 | Draw target 0 / class 1 | Loss target −1 / class 2 | Total |
| --- | ---: | ---: | ---: | ---: |
| X | 158 | 0 | 41 | 199 |
| O | 41 | 0 | 142 | 183 |
| Both | 199 | 0 | 183 | 382 |

These are actor-relative labels, not winner IDs inferred from a changed board. The example-count imbalance follows odd-length X wins. The absence of draws is observed data coverage, not a labeling defect: synthetic draw tests passed. The X-heavy self-play outcomes are not evidence of strength against an independent opponent.

## Loss and numerical behavior

All losses below are **pre-update sampled-batch means** from the authoritative `NeuralTrainer`, using raw logits. No loss was computed from diagnostic tactical labels. All 170 updates had finite gradients and finite post-update parameters; gradient L2 norms ranged from 0.474 to 9.887. There was no clipping or added loss term.

| Metric | First update | Last update | First 10 mean | Last 10 mean |
| --- | ---: | ---: | ---: | ---: |
| Policy loss | 1.9454 | 1.3422 | 1.9384 | 1.4880 |
| Value loss | 1.0523 | 0.0642 | 0.8158 | 0.0961 |
| Combined loss | 2.9977 | 1.4064 | 2.7542 | 1.5841 |
| Batch policy-target entropy, nats | 1.9049 | 1.1219 | 1.8734 | 1.2239 |

Losses fluctuated as examples accumulated. For example, mean combined loss after game 5 was 2.0565, after game 6 was 2.1713, after game 16 was 1.5421, after game 17 was 1.6824, and after game 19 was 1.5841. No schedule or configuration changed in response. These changing minibatches and moving self-play targets do not provide a controlled generalization-loss comparison.

Across all 382 policy targets, entropy averaged 1.1601 nats (range 0–1.9400). It averaged 1.8740 in the smoke games, 1.2319 in games 3–10 and 0.9984 in games 11–20. More concentrated search targets and lower fitting losses establish functioning updates, not improved strategic decisions.

## Fixed NN Only versus raw NN + MCTS diagnostics

Fourteen fixed, legally replayed nonterminal positions covered both players, opening/middlegame, immediate wins for X/O, unique forced blocks for X/O, full columns and mirrored tactical/full-column cases. Exact histories, complete raw/legal policies, visit distributions, selected actions, concentration, entropy and W/D/L probabilities are in the initial/final diagnostic JSON files. They were never training fixtures.

NN Only chooses the lowest-index maximum of the legally masked policy. MCTS uses the same shared network, 32 raw simulations, coefficient 1.41, temperature 1.0 and a separate seeded RNG; its reported selected action is sampled. Both have the tactical guard disabled. The separate modal MCTS action is the lowest-index maximum-visit action, included to distinguish search concentration from the sampled choice. Engine tactical scans annotate results afterward; they never select moves or filter search candidates.

All action numbers below are zero-based physical columns. Each arrow means initial → final.

| Position (actor) | Required tactic | NN Only action | MCTS sampled action | MCTS modal action |
| --- | --- | --- | --- | --- |
| Opening (X) | — | 6 → 3 | 1 → 2 | 1 → 3 |
| Opening after column 3 (O) | — | 6 → 3 | 1 → 2 | 1 → 2 |
| Middlegame (X) | — | 6 → 4 | 1 → 2 | 1 → 2 |
| Middlegame (O) | — | 6 → 2 | 1 → 2 | 1 → 2 |
| Immediate win (X) | Win at 0 | 6 → 2 | 0 → 2 | 0 → 0 |
| Immediate win (O) | Win at 1 | 6 → 3 | 1 → 1 | 1 → 1 |
| Forced block (X) | Block at 1 | 6 → 3 | 1 → 3 | 1 → 1 |
| Forced block (O) | Block at 0 | 6 → 3 | 1 → 3 | 1 → 3 |
| Full column 0 (X) | Legal action only | 6 → 2 | 5 → 3 | 3 → 3 |
| Mirrored immediate win (X) | Win at 6 | 6 → 3 | 6 → 3 | 6 → 3 |
| Mirrored immediate win (O) | Win at 5 | 6 → 6 | 5 → 5 | 5 → 5 |
| Mirrored forced block (X) | Block at 5 | 6 → 4 | 1 → 3 | 1 → 0 |
| Mirrored forced block (O) | Block at 6 | 6 → 6 | 3 → 0 | 6 → 6 |
| Full column 6 mirror (X) | Legal action only | 4 → 0 | 4 → 3 | 1 → 3 |

| Preliminary diagnostic aggregate | Initial | Final |
| --- | ---: | ---: |
| NN Only correct required tactical actions | 2 / 8 | 1 / 8 |
| MCTS sampled correct required tactical actions | 5 / 8 | 2 / 8 |
| MCTS modal correct required tactical actions | 6 / 8 | 5 / 8 |
| NN Only mean maximum legal probability | 0.1492 | 0.4090 |
| MCTS mean maximum visit probability | 0.3839 | 0.6406 |
| NN Only mean legal-policy entropy, nats | 1.9237 | 1.5644 |
| MCTS mean visit-policy entropy, nats | 1.5450 | 0.9495 |

No playing-strength claim follows from this small selected fixture set. Observed tactical correctness did **not** improve. In the final X-win fixture, the modal search action remained the win at 0 with probability 0.8125, but temperature-1 sampling chose 2; the mirrored X-win search's modal action itself missed the win. These illustrate both exploration during sampled play and weaknesses in the learned priors/value estimates at this budget.

Initial W/D/L predictions were close across all positions: win 0.3319–0.3327, draw 0.3011–0.3019, loss 0.3654–0.3669. Final ranges were win 0.00654–0.99993, draw approximately 8.9e−9–0.000296, loss 0.0000732–0.99346. Every prediction remained finite and normalized. Selected final distributions:

| Position | P(win) | P(draw) | P(loss) |
| --- | ---: | ---: | ---: |
| Opening X | 0.732197 | 0.000296 | 0.267507 |
| Opening O | 0.015055 | 0.000013 | 0.984931 |
| Immediate win X | 0.066577 | ~0.000001 | 0.933422 |
| Mirrored immediate win X | 0.999927 | <0.000001 | 0.000073 |
| Immediate win O | 0.014205 | <0.000001 | 0.985795 |
| Forced block X | 0.989922 | ~0.000001 | 0.010077 |
| Mirrored forced block O | 0.006537 | ~0.000001 | 0.993462 |

This shows poor tactical value calibration, strong reflection asymmetry and suppression of the draw class. The latter is consistent with no draw targets; it is not evidence that actual draws are impossible. Full-column raw policy mass fell from 0.1414 to 0.0316 for column 0 and from 0.1453 to 0.0182 for mirrored column 6. Legal masking yielded exactly zero mass on those columns before move choice throughout. Learning does not replace that legality contract.

## Whole-search timing and memory

The 382 measured self-play searches include root copying/initialization, all 32 traversals, leaf inference, expansion, backup and action sampling. They exclude the additional root-prediction measurement, example validation, JSONL writing and training.

| Native ARM64 CPU measurement | Value |
| --- | ---: |
| Whole-search mean | 12.745 ms |
| Whole-search median | 13.270 ms |
| Whole-search p95 (nearest rank) | 14.205 ms |
| Whole-search minimum / maximum | 1.992 / 59.669 ms |
| Total time in self-play searches | 4.869 s |
| Initial / final diagnostic mean whole search | 10.768 / 10.399 ms |
| Fresh model initialization | 4.909 ms |
| Checkpoint writing | 6.363 ms |
| Safe checkpoint loading | 2.768 ms |
| Peak process memory at initialization | 267.766 MiB |
| Maximum observed process peak memory | 301.781 MiB |

Memory uses macOS `resource.ru_maxrss` in bytes converted to MiB: a process peak, not current RSS or system memory. Per-ply/per-update peak observations are retained. Trees are released after capture rather than kept in the collection. This short local run does not establish long-lived memory stability, concurrent serving performance or performance on other hardware/hosting. No deployment was attempted.

## Checkpoint and evidence

Local evidence directory: [experiment-output/phase4d2-neural-seed42-20261004](../experiment-output/phase4d2-neural-seed42-20261004/).

The candidate [candidate.pt](../experiment-output/phase4d2-neural-seed42-20261004/candidate.pt) is **1,309,427 bytes**. Its SHA-256 is:

```text
ddca8f8ec49bd3cf74a6ddb05c6c7c0b4a2068ffe3169ac227dc2ad40435fe54
```

The artifact contains exactly `contract` and `model_state_dict`. The contract remains format version 1, architecture `connect4-conv64-128-128-policy7-wdl3-v1`, canonical encoding, physical action order `[0,1,2,3,4,5,6]`, value order `[win,draw,loss]`, current-player perspective and logits outputs. The state dictionary contains validated independently copied CPU float32 tensors. No experiment metadata, replay, optimizer state or RNG object is embedded.

Reload used the unchanged canonical `load_checkpoint()` with `weights_only=True` and `map_location="cpu"`. **All weight tensors were exactly equal. All logits and softmax predictions were exactly equal on all fourteen fixed legal positions, finite, normalized and correctly tagged with canonical representation metadata.** Writer tests also prove payload tensors do not alias model storage and existing files cannot be overwritten.

The external [manifest.json](../experiment-output/phase4d2-neural-seed42-20261004/manifest.json) hashes the candidate and provenance files. `config.json`, `events.jsonl`, `report.json`, both diagnostic JSON files, initial/final RNG JSON files, `source.patch` and archived new source/test files preserve the actual run. External post-run verification and preservation records supplement that evidence. Manifest file hashes were independently checked after the run. Historical artifact hashes were independently rechecked unchanged.

The candidate is a local research artifact. There is no accepted playing-weight designation, public integration or exact-resume artifact.

## Remaining concerns and proposed next milestone

The corrected Phase 4D.1 PUCT, value perspective, policy target and outcome-label contracts were preserved; no ambiguity required a mathematical redesign. No unresolved numerical or pipeline invariant defect was observed in this run. Cooperative time bounds and lack of transactional rollback after a failed post-update finite check remain explicit implementation limits; an invariant failure aborts without retry.

Learning concerns remain substantial: only 20 games, only one seed, no draws, a strong X/O outcome imbalance, repeated reuse of a small growing collection, finite-budget exploration without root noise, increasingly concentrated targets, poor tactical value estimates, mirror asymmetry and no observed tactical improvement. Search sampling at temperature 1.0 intentionally continues throughout; guard success cannot explain results because the guard was always off. Lower fitting loss and valid checkpoints establish **successful pipeline execution**, not learned Connect 4 strength.

A proposed **separately authorized** next milestone is one fresh seed-42 run capped at 200 completed games, 8,400 plies, 1,000 updates and 900 training seconds, using the same canonical architecture, 32 simulations, raw-search settings, learning rate and batch size. Predeclare a ten-update block after each post-smoke completed game with the same first-limit stop rule; freeze the same diagnostics before launch and add a small independent-opponent gameplay protocol to assess whether strength improves. Review coverage, calibration, tactical/mirror behavior and whole-search resources before any integration or further increase. Noise, augmentation or a temperature schedule would each require an explicit new protocol rather than being silently introduced.

**That proposal was not implemented or launched. Work stopped after this one experiment and its handoff.**
