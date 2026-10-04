# Phase 4C.1 — Correct DQN foundation

Implemented October 3, 2026 on local branch `phase4c1-dqn-foundation`. Preflight found a clean tree and fetched `origin/main` at `ddfd25640f6c6311ff89483ee0b34ff498b6fbbe`. That commit differs from the requested reference `fd6b8444ce3932bfe82fba72b2c3b21ffc16b6db` only by the archived Phase 4B report/evidence. No commits, pushes, deployment, self-play experiments, trained artifacts, API/UI integration, or production dependency changes are part of this phase.

The historical `Connect 4/DQN.ipynb` was read directly from `cfc2a6f`, without executing cells. The [Phase 4B audit](phase4b/audit.md) is already durable in the project; its local filesystem links now refer to repository sources or the existing evidence archive. The historical evidence ZIP is unchanged.

## Representation and mathematics

The MLP preserves the notebook's **42 → 128 ReLU → 128 ReLU → 7 raw Q-values**, with 22,919 parameters. There is no softmax. Every forward accepts a nonempty `(N, 42)` float32 tensor with values exactly in `{-1, 0, +1}`. Outputs must be finite `(N, 7)` float32 values.

The engine represents X as player `0`, O as player `1`, and an empty cell as a space. `encode_game` maps the **current player's** pieces to `+1`, the opponent's to `-1`, and spaces to `0`. Flattening is top row to bottom row, each row left to right. Thus index `7*r+c` is engine square `[r][c]`. Action index `c` is physical column `c`, from leftmost `0` through rightmost `6`; it is unrelated to the engine's center-first legal-move enumeration. Invalid shapes, symbols, player IDs, inconsistent `piece`/player metadata, and invalid numerical values are rejected.

Both inference and training import this same encoder. `collect_transition(game, action)` is the explicitly **mutating** collection boundary: capture actor and state, validate the action, advance exactly one ply, then encode for the **next player**. It uses `make_move`, not the legacy `step` method with its fixed-X reward. The engine flips players even after a winning move.

For actor-relative reward `r`, discount `gamma`, and next-player state `s_next`:

```text
terminal:     y = r
nonterminal:  y = r - gamma * max_{a in legal(s_next)} Q_target(s_next, a)
```

Either actor's immediate win earns `+1`; a draw or ongoing move earns `0`. A valid one-ply move cannot produce an opponent win, so there is no immediate `-1` reward in this convention. A losing move acquires a negative value through the opponent's positive continuation. For example, with `gamma=0.9` and legal next Q-values whose maximum is `0.5`, the target is `-0.45`, for either X or O. A full column with Q=999 cannot change this target. If the best legal opponent continuation is `-2`, the controlled numerical target is `+1.8`; this deliberately artificial test checks sign and masking, not realistic trained value calibration.

Terminal rows never call the target network, including terminal draws. Their stored legal masks are all false even when a winning board has open columns. Nonterminal masks must match the next board's top-row availability and contain at least one legal action. Missing legal actions are an invariant error. Mixed batches evaluate only nonterminal successors, in eval mode under `no_grad`.

`Transition` stores immutable copied tuples, validates shapes/values/actions/reward domain/masks, and checks that the successor is exactly one gravity drop followed by a perspective negation. This catches an unflipped successor before it reaches replay. `collect_transition` supplies engine-derived terminal/outcome labels; callers constructing low-level `Transition` objects themselves remain responsible for accurate outcome labels and valid reachable positions. Shape validation is not a complete arbitrary-board reachability proof.

## Training and inference boundaries

`ReplayMemory` is a bounded deque with independent seeded sampling. Insertion reconstructs validated transition data; tensors are detached and copied, and live engine rows, arrays, and caller masks cannot mutate stored samples. Samples themselves are immutable.

`DQNTrainer` creates a seeded online network, a synchronized independent target network with gradients disabled, and one persistent Adam optimizer. Model initialization preserves the caller's CPU torch RNG state. Exploration and replay use separate local Python RNG instances. No NumPy randomness is used. Configuration validates integer capacities/batch/synchronization counts, seed, finite positive learning rate, discounts, epsilon bounds, and their relationships.

Each successful `optimize()` samples one batch, computes separate detached scalar targets, gathers only `Q(s,a)`, and takes one optimizer step on mean Smooth L1 loss (beta=1). Unselected action outputs receive zero direct loss gradient. This is PyTorch's [Smooth L1 contract](https://docs.pytorch.org/docs/2.10/generated/torch.nn.SmoothL1Loss.html). Nonfinite outputs, loss, or gradients fail explicitly. Shared hidden weights may still change other action predictions after an update.

Epsilon decays **once per successful optimizer step**, clamped to its configured floor; insufficient replay leaves counters and epsilon unchanged. Hard target synchronization occurs every `target_sync_interval` successful steps (default 100), and `sync_target()` also supports an explicit copy. Defaults retain the historical replay capacity 10,000, batch 64, gamma 0.99, Adam lr 0.001, epsilon start 1.0 / minimum 0.01 / decay 0.995. These are initial configuration values, not validated experiment hyperparameters.

`DQNAgent(preloaded_network).choose_move(game)` only performs greedy inference. It has no environment binding, memory, optimizer, epsilon, or trainer import. It encodes the current player, rejects terminal/no-legal states, enters eval and `torch.inference_mode`, masks full columns, and chooses the lowest numbered legal column on a tie. It never changes the game. A compatible preloaded test module is accepted as well as the versioned MLP; arbitrary in-memory model semantics cannot be inferred from tensor shapes, so the caller must supply the documented contract. Persisted models should come through `load_checkpoint`. The caller owns the model, and concurrent training of that model during inference is unsupported.

The current implementation is explicitly CPU/float32. There is no episode runner, training CLI, automatic checkpoint saving, or import-time training. A future offline driver can explicitly call `choose_move`, `collect_transition`, `memory.append`, and `optimize` within separately approved bounds.

## Differences from the notebook

| Historical behavior | Corrected foundation |
|---|---|
| Absolute X/O encoding and fixed-X reward | Shared current-player encoding and actor-relative reward |
| Adds ordinary single-agent bootstrap over all columns | Subtracts opponent's best **legal** next Q-value |
| Mutable inputs accepted by `remember` | Detached independent immutable transition snapshots |
| One update for each of 64 replay examples | One vectorized optimizer update for a sampled batch |
| Overwrites a prediction tensor and compares all seven outputs | Separate targets and selected-action mean Huber loss |
| Target forward builds unnecessary graphs | `no_grad`, target parameters frozen, terminal rows excluded |
| Environment-bound agent mixes exploration and inference | Separate deterministic `choose_move(game)` adapter |
| Unseeded NumPy/Python exploration | Configurable local RNGs and reproducible CPU initialization |
| Episode target copies and epsilon can undershoot floor | Explicit update-count target schedule and clamped epsilon |
| No artifact contract | Versioned metadata, restricted load, strict tensor validation |

## Checkpoint contract v1

A torch-saved dictionary has **exactly** these top-level fields:

| Field | Required value |
|---|---|
| `format_version` | integer `1` (not boolean) |
| `architecture` | `connect4-dqn-42-128-128-7-relu-v1` |
| `state_encoding` | `connect4-current-player-row-major-v1` |
| `action_order` | integer list `[0, 1, 2, 3, 4, 5, 6]` |
| `value_convention` | `actor-reward-next-player-q-negamax-one-ply-v1` |
| `model_state_dict` | exact six `fc1`/`fc2`/`fc3` weight and bias tensors |
| `training_metadata` | finite JSON-compatible dictionary; empty if unavailable |

Tensor shapes are respectively `(128,42)`, `(128,)`, `(128,128)`, `(128,)`, `(7,128)`, `(7,)`. Loading explicitly uses `torch.load(..., weights_only=True, map_location='cpu')`, checks exact keys/shapes/float32 dtype/dense layout/finiteness, then calls `load_state_dict(strict=True)`. Saving validates against a fresh canonical architecture and copies detached CPU weights. Incompatible versions, unknown fields, malformed tensors, and historical MCTS-NN checkpoint schemas are rejected. There is no fallback to unrestricted pickle or custom deserialization allowlist. See [PyTorch's restricted loading documentation](https://docs.pytorch.org/docs/2.10/generated/torch.load.html); this is a trusted local artifact format, not a sandbox for arbitrary hostile files.

`load_checkpoint(path)` returns `(eval_model, training_metadata)`. The future experiment should supply metadata including seed, `dataclasses.asdict(config)`, completed games/plies/updates, epsilon, source commit/diff identity, runtime/device versions, and validation results. This minimal format is for inference weights and provenance, **not exact training resumption**: optimizer, target-network, replay and RNG states are not saved. A separate versioned resume artifact must be designed and verified before promising resumable experiments. No trained checkpoint was created; checkpoint tests use temporary untrained models only.

## Files changed

- `games/connect4/dqn/__init__.py`: inert optional package boundary.
- `games/connect4/dqn/encoding.py`: canonical state and legal-mask contract.
- `games/connect4/dqn/network.py`: historical MLP and numerical validation.
- `games/connect4/dqn/training.py`: transition collection, replay, signed targets, loss, configuration and update primitives.
- `games/connect4/dqn/checkpoint.py`: versioned safe save/load contract.
- `games/connect4/agents/dqn_agent.py`: independent greedy inference adapter.
- `requirements-dqn.txt`: optional isolated ML test/development dependencies.
- `tests/test_connect4_dqn.py`: controlled DQN correctness tests.
- `tests/test_dqn_import_isolation.py`: API startup with neural imports blocked.
- `docs/phase4b/audit.md`: portable links in the existing audit report.
- `docs/phase4c1-dqn-foundation.md`: this handoff.

The engine, Random, Negamax, MCTS, MCTS-NN, VictorAgent, production factory, API, explanations, UI, existing requirements, hosting files, credentials and historical weights are unchanged.

## Verification

Tested natively on macOS ARM64 with Python 3.11.17. A disposable virtual environment outside the repository used torch 2.10.0, NumPy 1.26.4 and pytest 8.4.2. The existing backend environment remains free of torch. Reproduce with separate environments:

```sh
# Disposable ML environment; not the API environment.
python3 -m venv /tmp/board-game-dqn-tests
/tmp/board-game-dqn-tests/bin/python -m pip install -r requirements-dqn.txt
/tmp/board-game-dqn-tests/bin/python -m pytest -q tests/test_connect4_dqn.py

# Existing backend development environment, installed from requirements-dev.txt.
.venv/bin/python -m pytest -q tests
git diff --check
```

Results: **72 DQN tests passed**. Existing backend verification first passed all **352 existing tests** independently. Final backend discovery passed **353 tests with one optional DQN module skipped**, including the new import-isolation check. That subprocess actively blocks torch/torchvision/torchaudio and DQN implementation imports, imports the API, creates an app, and verifies health returns 200. Importing the inert DQN package itself requires no torch. `git diff --check` passed.

Coverage includes both encodings and actors, explicit numerical signed targets, mixed terminal/nonterminal batches, win/draw targets without target calls, full-column masking with a dominant illegal Q-value, immutable replay, exact selected-action gradients, independent/frozen target parameters, deterministic sampling/exploration/initialization, epsilon floor and synchronization timing, stable greedy inference, unchanged caller games, malformed states/outputs/configuration, checkpoint round-trip and metadata/schema/tensor rejection. Tiny fixed fixture batches perform only a few optimizer steps; loss and gradients are finite, gradients are nonzero, and the fixed terminal batch loss decreases. This is a correctness test, not an agent training experiment or evidence of playing strength.

All backend provider interactions remain mocked, with the existing autouse HTTPS guard active. No paid requests or real explanation calls occurred.

## Recommended bounded Phase 4C.2 procedure (not executed)

1. Obtain explicit approval for one offline run and its fixed seed, CPU thread settings, game/ply/update/wall-time limits and output location. A conservative initial proposal is at most **200 games, 8,400 plies, 8,400 optimizer steps or 10 minutes**, stopping at whichever limit is reached first. This is a proposed smoke experiment, not a promise of useful playing strength; no automatic extension or sweep.
2. Train from scratch with this encoder/collector. Record code identity, runtime, complete configuration and initial seed before starting. Decide epsilon schedule against the finite update budget explicitly; retain one optimizer across the run. Keep the production API uninvolved.
3. Log games, plies, update count, elapsed time, epsilon, replay size, sampled loss, gradient health, legal Q-value ranges and target-sync events. Abort on an invariant error or nonfinite value. Use held-out legally replayed X/O win/block/draw/full-column fixtures for numerical and action checks. Do not select solely by decreasing loss.
4. Save bounded candidate inference checkpoints with provenance and an external SHA-256 manifest. Reload using this contract and verify predictions and legal actions. If exact interruption/resume is required, first add and verify the separate full training-state artifact; the present checkpoint must not be represented as resumable.
5. Review fixed diagnostic positions and a bounded manual-play sample for collapse/obvious regressions. Record limitations without claiming strength from a short run or tactical examples. Stop for review before more training, API integration, or any hosting decision.

Remaining limits are unmeasured learned strength, no selected trained artifact, no training driver/resume machinery, no GPU/MPS support or cross-device reproducibility guarantee, and no serving integration or runtime performance claims. Those are subsequent assignments, not unfinished Phase 4C.1 actions.
