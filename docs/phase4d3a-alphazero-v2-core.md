# Phase 4D.3A — AlphaZero v2 core implementation (Milestone 1)

Completed October 5, 2026. **This phase is correctness work only.** It ran no research training and no self-play campaign. It produced no learned v2 candidate, no research checkpoint, no strength tournament, no frozen evaluation set, no public API/UI integration and no deployment. **It makes no strength claim.** The only optimizer updates were discarded updates on tiny synthetic data inside unit tests.

Fetched `origin/main`. It matched the expected `f77801ebe9bb252f44bdc2dfa9bdaf0f9daec1e9`, and the working tree was clean. Created local branch `phase4d3a-alphazero-v2-core` from that reference. There was no commit, push, merge, paid request, dependency change or production configuration change.

The design authority is [Phase 4D.3 review](phase4d3-alphazero-v2-review.md) §10–11, Milestone 1. Implementation did not expose a contradiction in the specification, so the design was not changed. Interpretations the specification left open are labelled **(choice)** below.

Historical v1 records are preserved:

- Nothing under `experiment-output/` or `models/` is newer than the review document.
- No v1 module or report was edited, except the engine fix (§2) and one stronger isolation assertion (§9).
- v1 W/D/L checkpoints are not readable as v2 artifacts, and v2 artifacts are not readable by v1 (§7).

## 1. Source organization

New package [`games/connect4/alphazero_v2/`](../games/connect4/alphazero_v2/). It is inert on import, has no CLI, and nothing imports it from the agent factory, API or UI.

| Module | Torch | Responsibility |
| --- | --- | --- |
| [`config.py`](../games/connect4/alphazero_v2/config.py) | no | Contract constants, `V2Config` defaults and validation, `ceil(4M/128)` update budget, ply temperature schedule. |
| [`network.py`](../games/connect4/alphazero_v2/network.py) | yes | `AlphaZeroV2Net`, output validation, legal-logit masking, `V2Inference`, the minimal inference artifact, weight digests, frozen snapshots. |
| [`search.py`](../games/connect4/alphazero_v2/search.py) | yes | `PUCTSearch` (reuses the v1 `Node`/`backup`/`terminal_value`), `V2RootNoise`, `select_action`, `SelfPlayer`, `EvaluationAgent`. |
| [`data.py`](../games/connect4/alphazero_v2/data.py) | no | `V2Example`, actor-relative outcomes, finalization by full history replay, reflection, `ReflectionAugmenter`, `GenerationReplay`. |
| [`training.py`](../games/connect4/alphazero_v2/training.py) | yes | Batch tensors, policy CE + value MSE, AdamW parameter groups, `V2Trainer` with clipping. |
| [`selfplay.py`](../games/connect4/alphazero_v2/selfplay.py) | yes | Plays complete games and collects exactly N complete games. |
| [`generation.py`](../games/connect4/alphazero_v2/generation.py) | yes | `GenerationRunner` lifecycle, resume boundary save/load, runtime and source identity. |
| [`artifacts.py`](../games/connect4/alphazero_v2/artifacts.py) | yes | Atomic publication: temporary file, then validation, then a no-overwrite hard link. |

Tests are in [`tests/test_alphazero_v2.py`](../tests/test_alphazero_v2.py) (104 tests) and [`tests/test_connect4_engine.py`](../tests/test_connect4_engine.py) (23 tests, torch-free).

## 2. Engine boundary fixes

Fixed in [`connect4.py`](../games/connect4/connect4.py):

- `is_valid_move(col)` now requires an `Integral` column that is not `bool`, in physical range `0 ≤ col < 7`, and not full.
  - Negative Python indices are rejected. Before this fix, `-1` silently played column 6.
  - Bools are rejected. Before, `True` played column 1. This matches the API, which already rejects non-`int`/bool columns.
  - Floats, strings and `None` now return `False`. Before, they raised `TypeError`.
  - NumPy integers are still accepted.
- `make_move(col)` returns `False` and leaves state unchanged for an invalid column and for **any move after an X win, an O win or a draw**.
  - The terminal check, `_has_terminated()`, scans the 69 four-cell windows directly and then checks for a full board. It is an exact equivalent of `is_game_over()`, tested on 300 random legal games and 300 arbitrary boards.
  - It costs about 6.5 µs, against about 33 µs for `is_game_over()`, so production MCTS rollouts and Negamax barely slow down.
- `step()` now raises `ValueError` whenever `make_move` rejects a move.
- **(choice)** `is_valid_move`/`get_valid_moves` keep their historical meaning of physical column availability on finished boards. DQN, the API snapshot and other callers already check `is_game_over()` first. The engine now enforces termination itself, at `make_move`.

Legal gravity, turn alternation, winner detection and step rewards are unchanged and covered by tests. The review's separate Negamax defects (unbounded cache entries, heuristic terminal scores) are **not** fixed in this phase (§12).

## 3. Exact v2 neural contract

- **Encoding:** unchanged from v1 (`connect4-current-player-1x6x7-v1`). CPU float32 `(N,1,6,7)`: current player +1, opponent −1, empty 0, top row first, physical columns 0–6. v1's validated `encode_current_player` and `validate_input` are reused.
- **Architecture:** `connect4-conv64-128-128-policy7-scalar-tanh-v2`, with **325,896 parameters** in 16 tensors.
  - Trunk: padded 3×3 convolutions 1→64→128→128, ReLU after each.
  - Policy head: 1×1 convolution 128→32, ReLU, flatten, Linear 1344→7, giving **raw logits**.
  - Value head: 1×1 convolution 128→32, ReLU, flatten, Linear 1344→64, ReLU, Linear 64→1, **tanh**. Output shape is `(N,1)`, in [−1,1].
  - There is no draw logit and no W/D/L head.
- **Value meaning:** the expected final outcome for the player to move. This is the same perspective as terminal leaves in search.
- **Model identity:** `AlphaZeroV2Net` declares `alphazero_v2_contract` and deliberately has **no** v1 `representation_version`. As a result, v1 `NeuralInference` and `NeuralTrainer` reject it, and `V2Inference`/`V2Trainer` reject v1 `Connect4Net`.
- **Initialization:** fresh only. `GenerationRunner` seeds a domain-separated torch seed inside `fork_rng`. Nothing in v2 reads v1 weights.

**Legal inference.** `legal_policy_from_logits` checks for seven finite raw logits and non-empty, distinct `int` legal moves. It then takes a float64 softmax over the **legal logits only**, after subtracting their maximum. Illegal probabilities are exactly 0, and there is no uniform repair path. `V2Inference.predict` rejects terminal and no-legal-move states before inference, and requires the model to be in `eval()`.

The reviewed pathological case is a regression test. With logits `[1000,0,1,2,3,4,5]` and column 0 full:

- v2 gives ≈ `[.00427,.01161,.03155,.08576,.23312,.63369]` on columns 1–6.
- The v1 path (softmax first, then mask) collapses to uniform 1/6. The test asserts both results.

Training never uses masked logits. The policy CE is computed over all seven finite raw logits, and illegal visit targets are exactly 0.

## 4. Scalar value targets and loss

`outcome_for_actor(winner, actor)` gives `z = +1` when the completed-game winner is the pre-move actor, `0` for an actual draw, and `−1` otherwise.

`finalize_game` replays the entire history from an empty board. For each example it checks the pre-move observation, actor, ply and action. It rejects any history that continues after termination and any game that is unfinished. Only then does it label the examples.

There is no anchoring, tactical override, discount, root-Q substitution or truncated-game draw label. `tactical_value`, `train_mcts_nn` and `ValueTargetAnchorer` are never imported by the v2 path; a subprocess test asserts this.

```
policy_loss = mean_i( −Σ_a π_ia · log_softmax(raw_logits_i)_a )
value_mse   = mean_i( (tanh_value_i − z_i)² )
combined    = policy_loss + value_mse        # both coefficients 1
```

`training_loss` returns all three values separately. It rejects non-finite logits (including −inf), targets that are not normalized, and `z ∉ {−1,0,+1}`. Tests cover a hand-computed loss, X/O actors, wins, losses and draws.

## 5. Search (corrected v1 semantics, unchanged)

`PUCTSearch` reuses the v1 single-parent `Node`, `backup` and `terminal_value`:

- **Selection score:** `score = −Q(child) + 1.41·P(parent,a)·√max(1,N(parent)) / (1+N(child))`.
- **Q perspective:** each node's Q is for that node's player to move.
- **Backup:** alternates sign and is undiscounted.
- **Terminal values:** come from the engine. After a winning move, the child holds −1, so the parent sees +1.
- **Root accounting:** root expansion is uncounted and its value is discarded. Exactly B root-edge traversals occur, checked against both `root.visits` and the sum of child visits.
- **What is absent:** no tactical guard, transposition or tree reuse, and no fixed 32-simulation assumption anywhere in v2.
- **RNG:** ties use the injected search RNG.

Tests cover:

- Selection by `−Q`.
- Sign alternation at leaves for both players.
- Terminal wins for X and O: child Q is exactly −1 and the winning edge has the most visits.
- A terminal draw contributes 0.
- Exact visit accounting for budgets 1, 2, 7 and 33, including positions with a full column.
- Deterministic results from identical RNGs.

## 6. Policy target versus action temperature; root noise

**Separate objects:**

- `V2SearchResult.visits` and `.visit_target`. The visit target is `π(a) = N(a)/ΣN`, always computed from raw visit counts.
- `ActionSelection` holds `move`, `temperature` and the execution `distribution`, which is never used as a training target.
- `V2Example` stores `visits`, `action` and `action_temperature`. Its `policy_target` is *derived* from `visits`, so no temperature can reach it.

Tests check that:

- Identical visits give identical targets at τ = 0, 1 and 0.25.
- Self-play examples at ply 7 and at ply 8 both store `visits/B`.

**Schedule:** `action_temperature(ply, 8)`.

- Plies 0–7 (the number of pieces on the board before the move) use τ=1, which samples in proportion to visits.
- Ply 8 onward uses τ=0: uniform among the maximum-visit actions, drawn with a seeded action RNG.
- `EvaluationAgent` always uses τ=0.
- The boundary at ply 7 versus ply 8 is tested directly and inside self-play.

**Root noise.** `V2RootNoise` defaults to ε=0.25 and α=1.0. The v1 α=0.30 is not reused.

- Each self-play search makes one Dirichlet draw over legal root actions only: `P' = 0.75P + 0.25η`.
- Noise uses a private PCG64 stream in the domain `connect4-alphazero-v2-self-play-root-dirichlet`. State can be saved and restored.
- Only `SelfPlayer` can pass noise to the search. `EvaluationAgent` has no noise parameter, so evaluation and any future inference cannot turn noise on.
- Tests check that:
  - noise is zero on a full column;
  - the mixing is exact;
  - priors below the root equal the clean legal network policy;
  - evaluation results contain no noise and do not consume noise draws.

## 7. Artifacts

**Minimal inference checkpoint.** `save_inference_checkpoint` / `load_inference_checkpoint` use the payload `{"contract": INFERENCE_CONTRACT, "model_state_dict": …}` and nothing else: no optimizer, replay, RNG or provenance.

- Loading is `torch.load(..., weights_only=True)` on CPU. It checks exact keys, shapes, dtype, device and finiteness.
- Tensors and predictions round-trip exactly.
- An existing file is never overwritten.

Contract identifiers:

- `INFERENCE_CONTRACT.format = "connect4-alphazero-v2-inference"`, `format_version = 2`, with the architecture, encoding, raw-logit policy, legal-mask inference rule and tanh value perspective/semantics spelled out.
- `RESUME_CONTRACT.format = "connect4-alphazero-v2-resume"`, `format_version = 1`.

**Cross-rejection (tested).**

- The v1 `load_checkpoint` and the historical research loader reject v2 files.
- The v2 inference loader rejects v1 checkpoints.
- The resume loader rejects both inference formats.
- The inference loader rejects resume files, because their keys differ.

**Atomic publication.** `atomic_torch_save` writes a temporary file in the same directory and fsyncs it. It then reloads and validates the file, publishes it with `os.link`, which atomically refuses an existing path, and fsyncs the directory. The temporary file is always removed. A test confirms that a failed validation publishes nothing and leaves no temporary file.

**Resume boundary schema** (`generation-NNNN.resume.pt`, a weights-only-loadable dict):

| Key | Content |
| --- | --- |
| `contract` | `RESUME_CONTRACT` (embeds `INFERENCE_CONTRACT`, target conventions, replay schema) |
| `config` | `V2Config.to_dict()` |
| `counters` | completed generations, total games/positions, trainer steps |
| `model_state_dict` | learner tensors |
| `optimizer_state_dict` | full AdamW state (moments, per-parameter step, groups) |
| `replay` | per retained generation: games with `index`, `winner`, `moves`, per-ply `visits`, `action_temperatures`; plus content `digest` |
| `rng` | global Python/NumPy/torch; search, action, sampling, augmentation (`random.Random`); root noise (PCG64, big ints as strings, draw count) |
| `augmentation_counts` | reflected / unreflected samples |
| `history` | per-generation summaries |
| `learner_weights_sha256`, `state_sha256` | weight digest; canonical digest of everything determining continuation |
| `runtime`, `source` | Python/torch/NumPy/platform/machine/thread counts/deterministic flag; git commit + dirty flag |

**Loading a boundary** follows these steps:

1. Rebuild every replay example by legal replay. Each example's visit total must equal `self_play_simulations`.
2. Re-finalize the outcomes.
3. Verify the replay digest, the weight digest and `state_sha256`.
4. Restore the optimizer and check its hyperparameters against `config`.
5. By default, require the same Python, torch and NumPy versions, the same machine and the same intra-op thread count.
6. Restore the process-global Python, NumPy and torch RNG states. A flag can turn this off.

`save_boundary(dir)` writes both `generation-NNNN.inference.pt` and `generation-NNNN.resume.pt`.

## 8. Generation lifecycle, replay and optimizer

`GenerationRunner.run_generation()` runs one generation:

1. **(A) Freeze.** Record the learner's weight digest and step count, then make a `frozen_copy`: a deep copy in `eval()` with `requires_grad=False`.
2. **(B, C) Collect.** Play exactly `games_per_generation` complete games from empty boards. That one snapshot plays both sides. Each game is finalized by replay. The runner then checks that neither the learner nor the snapshot changed and that no trainer step occurred.
3. **(D) Add to replay.** Add the whole generation and evict whole oldest generations.
4. **(E) Train.** Run `ceil(samples_per_new_position · M_new / batch_size)` updates. Each update samples a batch of distinct positions uniformly from the entire retained window, reflects each one independently with p = 0.5, and calls `V2Trainer.step`.
5. **(F) Save.** Optionally write the boundary (`run_generation(dir)`).
6. **(G) Continue.** The trained learner collects the next generation.

The runner refuses to run beyond `max_generations`.

If an exception or interrupt occurs during a generation, that generation is discarded. Nothing partial enters the replay. The runner is marked `failed`, refuses further generations and refuses boundary writes. Recovery is from the last valid boundary.

**Defaults** (`V2Config`, review §10):

- Simulations: self-play 256, evaluation 512, interactive 512.
- Root noise: ε .25, α 1.0. Exploratory plies: 8.
- Collection: 256 games per generation. Replay: the last 8 generations, at most 2,048 games.
- Training: batch 128, 4 samples per new position, reflection probability 0.5.
- AdamW: lr 3e-4, betas (.9, .999), eps 1e-8, weight decay 1e-4 on conv/linear weights and 0 on biases, global clip norm 5.0.
- Campaign bound: `max_generations` 20.

**Replay.** `GenerationReplay` accepts only non-empty, fully labeled `CompletedGame`s with the matching generation id, strictly increasing ids, and game indices `0..n−1`.

- An insertion that would exceed `max_games` is rejected before any change is made.
- Positions are addressed as `(generation, game, ply)`.
- `counts()` reports games and positions per generation.
- `digest()` is a content identity.
- The replay does not import diagnostic or tactical fixtures.

**Trainer.** `V2Trainer` creates one AdamW, and its identity is checked on every update.

- Each step: train mode, combined loss, backward, then a check that every gradient is finite.
- `clip_grad_norm_(…, 5.0)` records the unclipped norm, the clipped norm and whether clipping happened.
- After the step, parameters are checked for finiteness. A non-finite parameter marks the trainer failed.
- Gradients are always cleared afterwards, and the model is returned to `eval()`, including on failure.

The generation summary records:

- outcomes, new and retained positions, and evicted generations;
- updates and sampled positions;
- reflected samples and root-noise draws;
- mean policy/value/combined loss, maximum gradient norm and the number of clipped updates;
- the weight digests before and after training.

## 9. v1 isolation and production isolation

- v1 modules, tests, checkpoints and reports are unchanged, except the engine fix. All v1 tests pass.
- `ValueTargetAnchorer`, tactical proofs, `neural_self_play`, `neural_evaluation` and the value audit stay available as v1 research and diagnostic tools. A subprocess test proves that importing the v2 runner loads none of them.
- `config.py` and `data.py` import with torch blocked.
- [`test_neural_mcts_import_isolation.py`](../tests/test_neural_mcts_import_isolation.py) now also blocks `games.connect4.alphazero_v2*`, so API startup is proven not to reach v2.
- No requirements file changed. PyTorch is not a production dependency.

## 10. Verification

| Check | Environment | Result |
| --- | --- | --- |
| v2 focused (`test_alphazero_v2.py`) + engine (`test_connect4_engine.py`) | `/tmp/board-game-phase4c1-venv` (Python 3.11.17, torch 2.10.0, NumPy 1.26.4) | **127 passed**, 7.3 s |
| v2 + engine + all neural (`test_connect4_neural_mcts`, root noise, symmetry, scaled, self-play, anchoring, value audit) + all DQN + standalone MCTS | same | **1004 passed**, 77.8 s |
| Full `tests/` backend suite | `.venv` (no torch) | **410 passed, 14 skipped** (torch-dependent modules) in 7.1 s |
| Both API import-isolation gates | `.venv` | **2 passed** |
| Explicit API startup | `.venv` | health 200; no `torch*`, neural or `alphazero*` module loaded |
| `git diff --check` | — | clean |

Mutation checks (each applied, run and reverted):

| Mutation | Tests failed |
| --- | --- |
| Noise applied below the root | 1 |
| Sampling RNG not restored on resume | 4 |
| One-hot target at τ=0 | 3 |

**Resume determinism.**

- **Same process:** an uninterrupted runner and a runner saved and reloaded after generation 2 produce identical generation-3 summaries, games, weights, optimizer state and `state_sha256`.
- **Fresh process:** a subprocess loads the boundary and runs the next generation. Its summary and state digest match the uninterrupted runner exactly.

All runner tests use tiny synthetic settings: 2–6 simulations, 2 games per generation, batch 8. Every model, artifact and update was created in pytest temporary directories and discarded.

## 11. Known limitations

- **Exact continuation is limited.** It is promised only from a completed boundary, in the same runtime: same Python/torch/NumPy versions, machine and intra-op threads. Inter-op threads and the deterministic-algorithms flag are recorded but not enforced. There is no mid-game or mid-generation recovery. Interrupted generations are rerun from the boundary, and RNG restoration makes the rerun identical.
- **The runner does not set the process runtime.** It does not set torch threads or `use_deterministic_algorithms`. A campaign launcher must do that and record it.
- **Loading changes global state.** Loading a boundary restores process-global RNGs by default.
- **Retention and provenance are minimal.** There is no retention or pruning policy (such as keeping the latest boundary plus its predecessor), no separate manifest file, and no time, ply or resource budget accounting. Source identity is commit + dirty flag only, with no patch or untracked-file archive.
- **Diagnostics are limited.** Per-generation output covers losses and counts only. Policy entropy, illegal raw mass, visit coverage, value saturation, contradiction rates, timing and memory are not yet recorded.
- **Scale is untested.** Resume file size and load time at full replay size (≈86k positions) are not measured. Search timing at 256/512 simulations is not benchmarked.
- **Engine scope.** `is_valid_move`/`get_valid_moves` intentionally still report physical availability on finished boards; only `make_move`/`step` enforce termination. Rejections still print, as before.
- **Not addressed in this phase:** standalone Negamax's cache-bound and terminal-scoring defects, the solved-position oracle and the evaluation methodology.

## 12. Exact remaining work for Phase 4D.3B (evaluation and preflight)

1. **Corrected reference Negamax**, separately versioned: EXACT/LOWER/UPPER cache entries or no cache, dominant exact terminal win/loss scores, and 0 for draws. Keep the legacy agent as a labelled historical opponent.
2. **Independently verified solved-position oracle.** Use an exact or bounded exhaustive solver cross-checked by a second method, with every history replayed through the engine. Record exact outcomes for every legal action.
3. **Sealed acceptance packages with a development/final split.**
   - 400 tactical base states plus mirrors.
   - 300 solved base states (100 each of win, draw, loss) plus mirrors.
   - Deduplicate by board + actor, including transpositions and reflections. Exclude every previously inspected fixture. Store content hashes.
   - Define the overlap-exclusion sensitivity rule.
4. **Opening banks** (development and sealed): 100 pairs per opponent, made of 20 empty-board pairs and 80 prefixes of 2–8 plies, balanced in reflection and mover.
5. **Paired arena and statistics.**
   - Side-swapped pairs, scored with an opening-cluster bootstrap (10,000 resamples, fixed seed) and W/D/L per color.
   - Ladder: Random, corrected depths 1/2/4, guarded UCT at 800, untrained v2 at 512, and retained 4D.2f at 512. The last one needs an explicit v1-agent evaluation adapter.
   - NN Only versus Random.
   - Champion gate at generations 5/10/15/20.
6. **Evaluation metrics.**
   - Raw-head value metrics on solved states: MSE/MAE, sign, wrong-sign saturation, draw MAE.
   - Behavioral-calibration protocol: 256 held-out games, binning, game-cluster intervals.
   - Contradiction measurements.
7. **Campaign launcher** with explicit authorization, which must:
   - write to a new output directory and never reuse one;
   - set threads to 1/1 and deterministic algorithms;
   - enforce the per-run (8 h) and campaign (24 h) caps and the evaluation-game ceiling;
   - handle signals cooperatively at game/generation boundaries;
   - write a manifest with file hashes and archive the source patch and untracked files;
   - retain the latest and previous resume boundaries plus every generation's inference snapshot;
   - account for abandoned attempts against the budgets.
8. **Per-generation diagnostics** listed in review §11: unique board families, reuse by age, entropy, illegal raw mass, visit coverage and depth, raw/root values, saturation, timing and memory.
9. **Preflight measurements:** a small inference-only benchmark of 256/512-simulation searches on the M4 (whole-search mean/p95, memory), and a resume round-trip at realistic replay size (bytes, save/load seconds).
10. **Freeze the campaign declaration:** seeds 42 (primary) and 314159 (replication), 20 generations, every `V2Config` value, opening banks and sealed quotas. All of this must happen before any authorized Milestone 3 run.

**No research training, learned v2 checkpoint or strength claim was produced in Phase 4D.3A.**
