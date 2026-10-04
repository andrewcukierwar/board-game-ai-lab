# Phase 4D.2b — Scaled neural baseline: launch failure before training

October 4, 2026. **The requested 200-game experiment is incomplete. One CLI launch was attempted; it exited before self-play, with zero games, zero applied plies and zero optimizer updates. No retry was attempted.** The failure was diagnostic JSON serialization, not a numerical or mathematical failure. The serialization defect is repaired and verified. A new launch requires separate authorization because the request explicitly prohibited automatic retries.

## Preflight and preservation

Fetched `origin/main`, matching `5ab044b4e836dc374509cfe639b5e7a60dfe01f5`. The starting working tree was clean. Created local branch `phase4d2b-neural-scaled` from that reference. Reviewed both Phase 4D foundation/training documents and the driver, shared neural contract, neural search and trainer. No applicable AGENTS.md was found. No mathematical ambiguity was found; no PUCT, value-perspective, encoding, architecture or loss/target contract changed.

Before launch, hashed all **181 existing files** under `models/` and `experiment-output/`. All remain byte-for-byte unchanged, including ten `models/connect4/connect4_model_iter_*.pt` historical neural checkpoints and every prior DQN experiment/checkpoint. The Phase 4D.2 candidate retains SHA-256 `ddca8f8ec49bd3cf74a6ddb05c6c7c0b4a2068ffe3169ac227dc2ad40435fe54`.

No DQN implementation, public API/UI, production dependency or hosting configuration changed. No paid requests, dependency installations, additional training seeds, commits, pushes, merges or deployments occurred.

## Implemented configuration and measurement protocol

[Driver](../games/connect4/neural_self_play.py): `Bounds()` retains the pilot limits and existing tests. Explicit `--profile scaled` selects `Bounds.scaled()` with **200 completed games, 8,400 applied plies, 2,000 optimizer updates and 900 training seconds**. Limits are checked before search, applying a move and each optimizer update. First reached limit stops new training work. Time limits remain cooperative: an in-flight operation may complete beyond a deadline. Intermediate measurements count toward the scaled training timer.

All original settings are preserved: fresh canonical 326,026-parameter Connect4Net, seed 42 for Python/NumPy/PyTorch and separate search/batch RNGs; CPU float32; one intra-op and inter-op thread; deterministic algorithms; 32 simulations; PUCT 1.41; temperature 1 throughout self-play; no tactical guard, root noise, symmetry augmentation or temperature schedule; batch size 32; persistent Adam at 0.001; equally weighted original policy and W/D/L losses. There is no checkpoint reader on the initialization path.

Games 1–2 remain a collection-only smoke gate with empty Adam state. After each later completed game, up to ten updates sample from completed examples only. At the 200th completed game, the global game limit skips that game's entire update block. A synthetic full schedule proves **1,970 updates**, including continuing beyond game 103, and snapshots after games **50/100/150 at 480/980/1,480 updates**. These are synthetic schedule results, not actual training measurements.

[Inference-only measurements](../games/connect4/neural_evaluation.py) add predetermined initial, game-50, game-100, game-150 and final snapshots. They include separate recent policy/value losses, entropy, full W/D/L distributions, draw probability, value saturation, NN Only greedy actions, raw MCTS modal and temperature-1 sampled actions, engine-annotated tactics, mirror distributions/actions and action concentration. Full per-update losses remain in the driver event/report schema. Snapshots do not change training settings or advance collection/batch RNGs.

The original fourteen legal probes remain unchanged. Supplemental diagnostics read the already inspected frozen research fixtures directly as JSON, without importing or modifying DQN implementations: 60 original, 96 validation and 96 replication positions. SHA-256 checks freeze all three suites. Tactical labels only annotate results; they never enter examples, losses or action selection. Immediate wins alone provide proven WDL labels for the added Brier/NLL summaries; a forced block does not establish a game-theoretic outcome. These fixtures are **previously inspected research positions, not blind holdouts**.

The bounded opponent protocol predeclares six explicit legal varied opening prefixes (three reflection pairs, even and odd lengths), both X/O sides, both initial/final models, NN Only and raw neural MCTS, and Random/Negamax depth 1/depth 2. It schedules **144 games** under **one combined 180-second deadline**. Opponent seeds and search settings match across model comparisons. Random wraps the existing agent with isolated per-game RNG state; Negamax uses fresh per-game agents and their existing cache/tie behavior. No tactical guards, exploration noise or learning occur. Completed histories are replay-validated, and W/L/D is counted by opponent/model/configuration/side. Partial games are excluded and reported; the deadline never extends automatically.

Self-play and sampled diagnostic MCTS use **temperature 1**. Playing-strength evaluation uses **temperature 0**, preserving the foundation's seeded random selection among maximum-visit ties. The separate diagnostic modal action selects the lowest-index maximum for historical comparability.

## Verification and uncovered failure

Used the existing optional `/tmp/board-game-phase4c1-venv` PyTorch environment and separate backend `.venv`. Provider requests remained mocked/blocked by the test suite. No synthetic test launches the real self-play training CLI.

| Gate | Before launch | After repair |
| --- | --- | --- |
| Neural correctness + pilot driver + scaled tests | 215 passed, 17.84 s | 215 passed, 17.05 s |
| All six existing DQN regression modules | 466 passed, 5.44 s | 466 passed, 5.85 s |
| Backend `pytest -q tests` | 354 passed, 9 skipped, 6.61 s | 354 passed, 9 skipped, 6.99 s |
| Explicit DQN + neural API isolation | 2 passed, 0.70 s | 2 passed, 0.84 s |
| `git diff --check` | Passed | Passed |

Neural commands cover `test_connect4_neural_mcts.py`, `test_neural_self_play.py` and `test_neural_scaled.py`. DQN commands cover `test_connect4_dqn.py`, `test_dqn_experiment.py`, `test_dqn_diagnostics.py`, `test_dqn_symmetry.py`, `test_dqn_validation.py` and `test_dqn_replication.py`. Backend skips are optional PyTorch-dependent modules; those tests ran in the optional environment. The neural API isolation test additionally blocks the new evaluation module.

Focused synthetic tests cover scaled ceilings, unchanged pilot limits, the two-game smoke gate, exact 200-game/1,970-update scheduling, intermediate snapshot timing, first-limit update/ply stopping and partial quarantine, immutable diagnostic/model/RNG behavior, frozen suites, balanced/matched evaluation settings, full synthetic game-history/count validation, illegal-action abort, Random RNG isolation and evaluation deadline quarantine. Existing driver tests continue to cover time/interruption and numerical-abort behavior.

The original new snapshot test checked its contents but missed JSON serialization. During launch, summing NumPy probability comparisons produced a `numpy.int64` saturation count; `json.dumps(..., allow_nan=False)` rejected it. The repair converts each comparison to a native boolean before summation. The snapshot test now serializes the entire nested payload, including every frozen-suite row and summary. Synthetic evaluation reports also undergo full JSON serialization. This was a test-coverage gap; passing the original gate did not establish complete CLI execution.

The single attempted command was:

```sh
PYTHONHASHSEED=42 /tmp/board-game-phase4c1-venv/bin/python \
  -m games.connect4.neural_self_play --profile scaled \
  --output experiment-output/phase4d2b-neural-scaled-seed42-20261004
```

**This records an unsuccessful launch and is not a rerun instruction.** The new, Git-ignored directory is retained and cannot be reused by the driver. The failure occurred while writing `snapshot-initial.json`, before opening `events.jsonl` or calling `run_training`. No automatic relaunch occurred after repair.

## Actual experiment measurements

| Requested measurement | Observed status |
| --- | --- |
| Completed / started self-play games | 0 / 0 |
| Applied / collected self-play plies | 0 / 0 |
| Optimizer updates | 0 |
| Training time | Training loop never entered; zero collection/optimization execution |
| Smoke gate | Not reached |
| X wins / O wins / draws | 0 / 0 / 0; no gameplay |
| Win / draw / loss labels | 0 / 0 / 0; no examples |
| Policy / value loss progression | Unavailable; no updates |
| Intermediate / final snapshots | Not reached |
| Opponent evaluation | Not started; zero complete games |
| Final candidate | Not created |

Fresh initialization completed in **5.711 ms**; initial process peak memory was **268.375 MiB**. Fourteen original untrained diagnostic searches averaged **10.622 ms** for the entire 32-simulation search. These are initialization/probe measurements, not self-play throughput or scaled training resource measurements. The full failed-launch duration and peak memory after initialization were not captured.

The saved original probes have untrained W/D/L ranges approximately **win 0.331866–0.332742, draw 0.301100–0.301897, loss 0.365361–0.366878**. Complete untrained policy/visit/action/tactical/mirror records survive in `diagnostics-initial.json`. The additional initial snapshot was computed in memory but was not saved when serialization failed. No learned model exists for a comparison.

Consequently this attempt provides **no evidence** about calibration improvement versus saturation, NN Only versus learned MCTS, tactical improvement, learned mirror behavior or playing strength against existing agents. No competitive-strength claim can be made. The completed Phase 4D.2 pilot remains the latest actual neural learning evidence.

## Integrity and evidence

[Failed launch evidence](../experiment-output/phase4d2b-neural-scaled-seed42-20261004/) preserves the attempted configuration, runtime, initial RNG states, initial fixed diagnostics, original launch source patch and archived untracked source/test bytes, plus the full traceback. `report.json` explicitly records the pre-training failure; `preservation.json` records all prior artifact checks. The external manifest hashes all retained evidence. A separately identified `repaired-source/` archive preserves the repaired source, rather than replacing the original launch provenance.

The initialized weight-byte digest was `559620773060542d4569b25eb92f2e60c862f44a45299dca00c666cef9d4033e`. This digest is not an inference checkpoint: the failed launch saved no model weights. There is therefore **no new candidate to safe-load, compare for exact weights/predictions or promote**. Existing synthetic minimal-checkpoint writer/CPU-loader/equality tests pass; actual trained-candidate integrity remains unperformed. Manifest hashes and all 181 historical artifact hashes were independently checked. Exact interrupted training resumability is neither implemented nor claimed.

## Interpretation and next step

- **Mathematical correctness:** existing PUCT, alternating value backup, representation and training contracts remain unchanged and their correctness tests pass.
- **Successful optimization:** not measured in this attempt; no optimizer step occurred.
- **Tactical behavior:** only initial untrained probes survive; no learned comparison exists.
- **Measured playing strength:** no opponent games ran.

The next action is a separately authorized fresh launch of the repaired bounded baseline in a new output directory. The next learning-method change should be decided after its scaled evidence exists. Based solely on the earlier pilot's mirror asymmetry, horizontal symmetry augmentation is a reasonable subsequent isolated experiment, but it is not justified by new scaled results and must remain disabled for this baseline. Do not alter architecture, root noise, temperatures or budgets to compensate for this failed launch.
