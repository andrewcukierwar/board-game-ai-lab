# Phase 4B — Neural foundations and checkpoint audit

Audit date: October 3, 2026. Archived in the repository after the audit; local source links were made portable in Phase 4C.1. Audit only; no repository file changes, training, new checkpoints, commits, pushes, workflows, paid requests or deployment.

**Executive finding:** Preserve the historical artifacts, but do not deploy their weights. All ten load and produce finite normalized predictions; the inspected predictions and verified training/search defects provide no defensible basis for calling them useful learned opponents. Build a small, correct DQN vertical slice first, then corrected neural MCTS with fresh training. PyTorch CPU inference fits in the completed local constrained-memory experiment. Preserve Connect4Net’s architecture while repairing correctness; do not redesign it around Render Free. The new feasibility targets are native Mac mini M4 / 16 GB and, secondarily, Cloud Run 1 vCPU / 1 GiB; neither has been deployed or measured in this audit.

**Updated scope:** This report incorporates the user’s revised hosting assumptions. Existing Render production remains unchanged. No dependencies were reinstalled and no experiments were rerun after resuming. The formerly queued 0.1-CPU probes had already completed before the update; their files are preserved but excluded from the decision tables and architectural recommendations. No Cloud Run or Cloudflare Tunnel setup was performed.

**Evidence labels:** VERIFIED = inspected source/artifact or directly reproduced behavior (the text identifies which). INFERRED = consequence supported by analysis, not an observed historical training outcome or production result. RECOMMENDED = future implementation proposal.

## 1. Repository and artifact inventory

**VERIFIED:** Fetched `origin/main`; it and HEAD both resolve to `fd6b8444ce3932bfe82fba72b2c3b21ffc16b6db`. Branch remains `phase4a2-mcts-integration`. Working tree was clean before and after the audit. Existing local branches were preserved; no checkout/reset/merge was performed. Fetch updated Git fetch metadata only. No applicable AGENTS.md was found in the repository or inspected ancestor directories.

All 14 requested source/configuration/documentation files were read, alongside relevant tests, API initialization, historical engine/board code and Git history. README/project_plan/Phase 4A.2 documents still describe pending deployment; the user's confirmation supersedes that status. This audit did not re-inspect the public deployment.

`models/connect4/` contains exactly ten tracked `.pt` files, iterations 10–100, totaling **13,099,080 bytes**. No extra files were present there. Each is a 20-member PyTorch ZIP archive, with 1,305,771 bytes of uncompressed members. All have precisely `model_state_dict` and `iteration` as top-level keys. The iteration metadata is zero-based (9–99); filenames are one-based counts (10–100), consistent with the save loop. There is no optimizer state, RNG state, training seed, loss history, data identity, architecture/encoding version, code commit or validation result.

Recovered `Connect 4/DQN.ipynb` from `cfc2a6f` into this temporary audit directory, without executing its cells. SHA-256: `8e486c4ba606fd1f17fb4e58929b30c1a6da871f720421bc0178707c5aa76b1a`. Its kernel metadata says Python 3.12.3. Stored output reaches episode 1000 and epsilon 0.0100; that is evidence of historical notebook output, not reproducible training provenance or retained weights. No DQN model file appears in current tracked checkpoints, the inspected historical tree, or `connect4.zip` (which contains another DQN notebook). Private/untracked historical artifacts elsewhere cannot be ruled out.

## 2. Checkpoint compatibility and numerical health

**VERIFIED:** CPU-only PyTorch **2.10.0+cpu**, Python **3.11.17**, NumPy **1.26.4**. Every load explicitly used `weights_only=True, map_location="cpu"`, then checked exact keys, shapes, dtypes and CPU tensors and used `load_state_dict(..., strict=True)`. No unsafe fallback, custom deserialization allowlist or historical loader was used. Checkpoint inspection ran in a network-disabled, read-only-root container with read-only source/model mounts, a 512 MiB cap, no swap and external timeout. We did not test the old `torch==2.2.2` training dependency pin.

All ten pass: **16 parameter tensors, 326,026 float32 parameters, 1,304,104 raw parameter bytes (~1.244 MiB)** each. All parameters are finite and nonzero. All pairwise weight vectors differ. Adjacent checkpoints change 130,068–183,369 parameters, with L2 differences 6.692–10.594. Nontrivial weights and changing predictions demonstrate artifact evolution, not beneficial learning.

| Filename iteration | Metadata | Bytes | SHA-256 | Strict load / finite |
|---|---:|---:|---|---|
| 10 | 9 | 1,309,906 | `e08d1a7070a277da13b1d03b9bb982a94abd7843f194e6215825338bf5f9f510` | Pass / pass |
| 20 | 19 | 1,309,906 | `63b3172aafc6c67a5244ff21346fed823ccb87e16fc2f1441afcaed3397a9461` | Pass / pass |
| 30 | 29 | 1,309,906 | `1fc34df8763ba01a405b360d97b86e96829e2e7506f7c480654812d8d4b683ca` | Pass / pass |
| 40 | 39 | 1,309,906 | `160eb5d557460e17666aa21d58212037d475fd7085d44ed76b94c8459887e0e9` | Pass / pass |
| 50 | 49 | 1,309,906 | `c05c7ac08d0e15a03b37d9933ae57f6597f3fca4cc2b2aeb683b3c0faa610505` | Pass / pass |
| 60 | 59 | 1,309,906 | `2efabcfa994472554b5cbe3153c39a1d79441e60ccd2558e67675470b045619b` | Pass / pass |
| 70 | 69 | 1,309,906 | `12489484e7e1d761d26ef22d3d5b84d702a53566a972b220d1a9e3587525cb66` | Pass / pass |
| 80 | 79 | 1,309,906 | `c0a9220717d33e27866ad2c73d84e5b99f4dacc47ac86df1b892ceb14703492c` | Pass / pass |
| 90 | 89 | 1,309,906 | `cb2c97028656e5d176cfd7a211f39ce0e460ee0972b21ef8b61ca680131c4953` | Pass / pass |
| 100 | 99 | 1,309,926 | `8ced8d56469faffbcee2fa34da9780a9285567f531bcaf60b7f1036aeebb225a` | Pass / pass |

The following shapes/counts apply identically to every checkpoint; each weight and bias is float32. Full per-checkpoint/per-tensor min/max/std/nonzero/finite statistics are in `checkpoints.json` in the [audit evidence archive](evidence.zip).

| Layer | Weight shape | Bias shape | Combined parameters |
|---|---|---|---:|
| conv1 | [64, 1, 3, 3] | [64] | 640 |
| conv2 | [128, 64, 3, 3] | [128] | 73,856 |
| conv3 | [128, 128, 3, 3] | [128] | 147,584 |
| policy_conv | [32, 128, 1, 1] | [32] | 4,128 |
| policy_fc | [7, 1344] | [7] | 9,415 |
| value_conv | [32, 128, 1, 1] | [32] | 4,128 |
| value_fc1 | [64, 1344] | [64] | 86,080 |
| value_fc2 | [3, 64] | [3] | 195 |

PyTorch's restricted loader supports tensor state dictionaries without executing the general-purpose pickle surface; this is a risk reduction, not a proof that arbitrary serialized input is harmless. See [serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization).

## 3. Neural inference validation

**VERIFIED — architecture and representation:** Three padded 3×3 convolutions (1→64→128→128), ReLU activations, a 128→32 1×1 policy head followed by 1344→7, and a 128→32 1×1 value head followed by 1344→64→3. Outputs are softmax probabilities, with value classes `[win, draw, loss]`; the search uses `P(win) − P(loss)`.

Training expects `(N,1,6,7)` float32. `_get_state_tensor` actually returns `(1,6,7)`, a supported *unbatched* Conv2d input; its comment calling this a batch dimension is misleading. Both that path and explicit `(1,1,6,7)` produced matching outputs in this runtime. Batched outputs were `(11,7)` and `(11,3)`; individual outputs were `(1,7)` and `(1,3)`. `_predict` returns a length-7 policy and scalar tensor. The contract should nevertheless be made explicitly batched and validated in future work.

Encoding is fixed absolute color: **X=+1, O=−1, empty=0**, top row first. Both colors encode correctly. No turn plane or perspective flip exists; changing only current_player/piece leaves the tensor unchanged (diagnostic metadata perturbation, not a legal alternative turn). For reachable X-first games, turn is inferable from piece counts, so lack of an explicit turn plane alone does not make these legal states non-Markov. It does require a consistently defined target perspective, which this trainer lacks. Unexpected cell symbols silently become zero.

Eleven positions were legally replayed with validation before every move: empty; openings after 1 and 2 plies; middlegames after 18 and 19; a filled column; late positions after 38 and 41; a full-board draw; X win; O win. Both movers and terminal outcomes were covered. Move sequences, boards and outputs are recorded in `checkpoints.json`; late-game fixtures reuse the existing deterministic draw sequence. These fixtures are diagnostic examples, not a representative strength sample.

**VERIFIED — numerical results:** All 110 checkpoint/position combinations have finite nonnegative policy/value probabilities. The maximum probability-sum error is **1.1921×10⁻⁷**. No terminal/nearly-full forward crashed. Policies change across positions and checkpoints, but often saturate, with exact zero/one probabilities in some files. The raw network assigns probability to full columns; legal masking belongs in the agent. Iteration 100 assigns 2.787% to the full column in the six-piece fixture and about 77.045% total to unavailable columns in the 41-ply fixture.

| Iteration | Empty-board P(column 6), zero-based | Scalar value range across 11 fixtures |
|---|---:|---|
| 10 | 0.41684413 | -1.00000000 to -1.00000000 |
| 20 | 0.97136432 | -0.99930245 to 0.46132383 |
| 30 | 0.99190277 | -1.00000000 to -1.00000000 |
| 40 | 0.99999607 | -0.99999982 to -0.97708815 |
| 50 | 0.99980193 | -1.00000000 to -0.99999875 |
| 60 | 0.99999881 | -1.00000000 to -0.99996710 |
| 70 | 1.00000000 | -1.00000000 to -0.99998569 |
| 80 | 1.00000000 | -1.00000000 to -0.99999690 |
| 90 | 1.00000000 | -1.00000000 to 0.95738930 |
| 100 | 0.99997783 | -1.00000000 to -1.00000000 |

Iteration 100's scalar is exactly −1 in all eleven cases, including the terminal draw. Its empty-board policy is approximately `[0,0,0.0000222,0,0,0,0.9999778]`; at 18 plies it is approximately `[0,0,0.09648,0.00013,0,0,0.90340]`; at 41 plies `[0.01665,0.02881,0.24651,0.28603,0.18242,0.01003,0.22955]`. Policy variation is measurable. Useful strategic discrimination is not established. Iteration 20 has varying values; iteration 90's positive values occur only on terminal win fixtures in this sample. Neither observation identifies a deployable “best” checkpoint. Terminal outcomes should come from the engine, irrespective of network prediction.

**A: loads successfully — yes, all ten. B: numerically valid predictions on inspected fixtures — yes, all ten. C: strategically useful behavior — unproven; observations raise substantial concerns.** No tournaments or playing-strength estimates were produced.

## 4. Neural MCTS search defects, ranked

| Severity | Finding and evidence | Consequence / disposition |
|---|---|---|
| Critical | Child Q uses child-to-move value, while the parent maximizes it. Terminal leaves use the *next* mover after a winning move and therefore get −1; alternating backup preserves each node's own-to-move convention. VERIFIED source; controlled selector chose the +1 child. | INFERRED: each parent prefers favorable evaluations for its opponent. Choose one explicit convention and test both players. Either use −child.Q with current-player node values, or store parent/previous-player edge values. |
| Critical | Transposition table reuses whole mutable Node objects carrying one parent, one incoming move/prior and shared visits/Q. VERIFIED at 96 simulations with a constant fake network: 556 nodes, 117 multiply referenced nodes and 117 incoming edges whose parent pointer points elsewhere. | Backprop follows the original parent path, not the traversed path; exploration reads the wrong parent's visit count, priors belong to another edge, and final visit counts cease to represent root-edge traversals reliably. Replace with a plain tree initially. Later cache immutable network evaluations only, or design explicit edge statistics/path-based backup. |
| High | Opponent-block check changes `current_player` but leaves `piece` unchanged. VERIFIED: both existing X/O blocking fixtures return None. | Simulates the current player's piece when trying to test the opponent. Replace this guard with the Phase 4A actual-successor/reply scan. Immediate own-win logic can be retained conceptually. |
| High | No initial terminal rejection; forced-move scan runs first. VERIFIED: a finished X-win board returned column 3. | Agent can produce a move on a finished game. Reject terminal/no-legal states before tactical/search work. |
| High | No simulation/temperature validation or robust zero-visit path. VERIFIED: budget 1, temperature 0 returns column 3 with seven NaNs in its stored policy. | First simulation expands the root without visiting children; normalization divides by zero. Positive-temperature selection would receive invalid probabilities; budget 0 has no candidate children. Add validated bounds and an explicit low-budget fallback/initialization. |
| High | Always records training examples during inference; copies only outer board list. VERIFIED direct mutation probe; details below. | Input search boards are detached, but saved observations mutate with caller play. Long-lived agents accumulate unnecessary examples; separate inference and collection. |
| Medium | PUCT-like score is Q + 1.41×P×sqrt(parent.N)/(1+N), but N=0 returns infinity irrespective of prior. Ties use deterministic dict order `[3,2,4,1,5,0,6]`. | Every unvisited child is explored first, independent of NN prior; small budgets and training targets inherit move-order bias. Decide and test explicit PUCT/tie behavior. This is not UCT merely because it is named ucb1. |
| Medium | Legal policy masking and renormalization exist; zero-sum fallback is uniform. No finite/shape checks; temperature exponentiation can overflow/underflow. | Preserve masking mathematics; validate inputs/outputs, nonempty legal support and normalization. Use stable temperature handling. |
| Medium | Factory ignores NN configuration and defaults to 1000 simulations / temperature 0.1, constructs/loads a fresh network, uses a CWD-relative path. Loader omits explicit weights_only. | Replace loader/dispatch contract before integration. No production use of the old loader. |
| Medium | Per-search mutable table/examples on the agent, redundant deep copies and eval() on every prediction, synchronous unbatched inference. | Do not share search-agent state concurrently. Share a read-only model, allocate search state per request, use inference_mode and explicit thread limits. |

**Terminal evaluation nuance:** The neural agent correctly identifies draws as zero *after* checking terminal status and treats a finished win as a loss for the player now to move. The defect is how that value is consumed by parent selection, not simply “all terminal values are wrong.”

**Comparison to Phase 4A:** Standalone MCTS explicitly stores previous-player reward (`player_just_moved`), so maximizing child UCB favors the parent's mover. Its terminal guards, independent copied tree states, validated budgets, RNG injection, robust-child selection and actual-successor tactical checks provide sound patterns to reuse. The neural implementation has no consistent equivalent contract.

**Transposition nuance:** Board-only keys omit turn metadata. For valid X-first Connect 4 states, parity determines turn, so omission alone does not demonstrate a collision between legal opposite-turn states. Sharing edge-specific mutable statistics is the demonstrated bug. Occupancy increases at each move; this audit did not find or claim a cycle/deadlock. Existing “remove illegal child and continue” recovery is not a correctness solution for invalid search invariants.

**Search isolation:** The bounded probe left the caller board unchanged; root and successor game copies detach normal search mutation. Saved training examples do not share that protection. The table is rebuilt for a normal search, but the forced-move early return can retain the prior tree. Root visit-count choice is the right general idea, provided counts truly belong to edges and zero totals/temperature are handled.

No historical default-budget `choose_move` was executed. Runtime search probes used a constant fake model, explicit budgets 1, 2 or 96, a disposable subprocess/container and a 30-second external timeout. They were invariant checks, not agent comparisons.

## 5. Training pipeline and implications for weights

**VERIFIED source and direct helper probes:**

1. **Stored states mutate.** `Board` subclasses list; `board.copy()` shares inner rows. A stored empty board acquired X at column 3 after the caller made a move. During self-play, prior examples therefore converge to that game's final board while retaining earlier policies. This destroys state-policy alignment.
2. **Winner targets are wrong.** `example.board == 1` / `== 2` are list-to-integer comparisons, both False; the sum test always chooses player 1. The engine numbers X/O as 0/1, whereas this expression assumes 1/2. Every example gets −1 for an X win, +1 for an O win, zero for a draw, regardless of mover. Both agents run the same conversion. Merely changing the container to ndarray would not solve string/numeric and indexing mismatches. Store mover explicitly before the move.
3. **Search-derived policy targets inherit the search bugs.** Broken value selection, transposition accounting and blocking contaminate supervision; forced moves produce one-hot policies. Visit distributions are stored without temperature transformation even though actions use temperature-adjusted visits. That can be an intentional policy-target choice, but it needs an explicit contract. One-simulation NaNs must never enter data.
4. **Softmax is fed into CrossEntropyLoss twice.** Both heads already emit probabilities, then CE treats them as logits and applies log-softmax. Soft probability targets themselves are supported; the defect is passing probabilities as inputs. Use raw logits with CE or explicit negative target-weighted log-probability. [PyTorch CE input contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). The wrong objective compresses effective class contrast and can saturate gradients; it does not mean every gradient is zero.
5. **Training lifecycle/provenance are weak.** Adam is recreated each iteration, losing its running moments. No shuffle, seeds, replay across iterations, validation split, evaluation gate, RNG/optimizer checkpoint or resume path. Ten fresh games per iteration, ten epochs, batch size 32, lr 0.001; two agents share one model. Logged sum-of-batch-mean losses divided by example count is incorrectly scaled as a per-example average.
6. **Mode/import mismatch.** Prediction sets eval(); training never calls train(). There are no dropout/batchnorm layers, so this is not a present numerical mode bug, but it is fragile. `from agents...` combined with `.connect4` makes normal package/direct invocation inconsistent. Checkpoints are saved in CWD while the loader expects `models/connect4/`. No claim is made that the notebook-era invocation matched today's script layout.

**INFERRED:** If these code paths produced the supplied weights, board/label/search corruption could materially compromise both heads. Extreme value saturation and edge-column concentration are consistent with bad supervision; they do not prove the exact training history or that every learned feature is worthless. Metadata cannot reconstruct the actual run, seed, data or code. Iteration metadata proves no particular number of valid games or optimizer steps.

**RECOMMENDED:** Fresh training is necessary before deployment of a corrected neural MCTS agent. Retain all ten files as historical fixtures/forensic artifacts. Do not select iteration 100 simply because it is newest, or designate iteration 20 “best” because its values vary. Fine-tuning old weights is an optional later experiment, not the default recovery path or a substitute for corrected supervision.

## 6. Historical DQN assessment

**VERIFIED source:** MLP **42→128→128→7**, ReLU/ReLU/linear, **22,919 parameters** (~89.5 KiB float32). Fixed absolute X=+1/O=−1/empty=0 flattened board, no explicit player feature. A deque stores up to 10,000 transitions; batches of 64 are randomly sampled. The training loop calls `flatten()`, which creates independent arrays, so it does not exhibit the neural trainer's shallow-row mutation bug in that path. `remember` itself does not enforce copying.

Online and target networks start synchronized; target weights are copied after every episode. Adam lr 0.001, gamma 0.99. Epsilon starts at 1, decays by 0.995 after each replay call once enough data exists, minimum test 0.01 (multiplication can undershoot slightly). Both random exploration and greedy action selection use the live environment's legal moves. Exploration schedule is per move/replay, not per episode.

**Critical target defects:** For a nonterminal transition, `reward + gamma * max Q_target(next_state)` considers all seven next actions, including full columns. Each transition is one ply, so the next state belongs to the opponent. Meanwhile `Connect4.step` gives rewards from X's fixed perspective: +1 for X wins and −1 for O wins. Both colors choose argmax. This is neither a correct current-player negamax target nor fixed-X minimax Q-learning. A legal current-player formulation would canonicalize own/opponent pieces and use `r_actor − gamma * max_legal Q_target(next_state)`, with no bootstrap at terminal; alternatively use an explicitly documented two-ply transition. Exact terminal/draw/no-action handling is required.

**Loss nuance:** `target_f = model(state); target_f[action] = target; mse_loss(target_f, model(state))` is awkward but not automatically a zero-gradient failure. For this deterministic MLP, unselected predictions coincide and contribute zero error; CopySlices replaces the selected target with a constant, leaving the predicted selected Q trainable. The mean over seven outputs scales the selected-action loss by 1/7. Replace with batched gather of Q(s,a) against detached scalar targets; use no_grad for targets and inference. The current loop does 64 separate optimizer steps per sampled minibatch and constructs unnecessary target graphs before `.item()` detaches the Bellman scalar. That is inefficient, not proof that nothing trained.

**Missing pieces:** No weight save/load, no versioned artifact, no evaluation-mode/epsilon-zero `choose_move(game)` adapter, no independent state-based masking contract, no reproducible training CLI or checkpoint selection, no API/UI integration. Stored notebook episode output is not a deployable checkpoint.

**Minimum recovery:** Preserve the MLP and replay/target-network concepts; extract a small module with one perspective convention, copied transitions plus next-state legal mask, correct signed Bellman targets, batched selected-action loss, bounded/seeding-aware training, target-sync and epsilon schedules, manifest-bearing checkpoint save/load, and legal deterministic inference. Add focused tests for both colors, full columns, terminal/draw transitions, hand-computed Bellman targets and nonzero selected-action gradients. Train only in a separately authorized future phase, offline from the serving process.

## 7. CPU, memory, latency and image feasibility

**VERIFIED environment:** Linux amd64 containers on Docker Desktop's ARM64 Linux VM, hosted on macOS arm64; Python 3.11.17 / glibc 2.41 / kernel 6.12.76-linuxkit, torch 2.10.0+cpu, NumPy 1.26.4, Flask 3.1.0, Gunicorn 22.0.0. This is **emulated x86-64**, not native Render hardware. NNPACK reported unsupported hardware and fell back. No macOS PyTorch timings are presented. All inference runs had network disabled, read-only application/model mounts, 512 MiB RAM and swap disabled. CPU cgroup quotas were explicitly verified. Model loading always remained CPU-only.

PyTorch 2.10.0 was a deliberately pinned audit runtime, not a recommendation to update production to that version blindly. Neither torchvision nor torchaudio was installed; imports, loads and predictions all succeeded. No inspected neural inference code needs either package. Pillow and tqdm are likewise unnecessary for serving this model.

**Single process staged RSS, 0.5 CPU / one intra-op and one inter-op thread:**

| Stage | Current RSS MiB |
|---|---:|
| python | 24.18 |
| api_import | 64.95 |
| torch_import | 289.72 |
| constructed | 293.40 |
| loaded | 295.51 |
| first_forward | 302.46 |
| after_10 | 302.46 |
| after_100 | 302.46 |
| after_300 | 302.46 |

The model's raw parameters are only ~1.24 MiB; PyTorch import accounts for ~225 MiB additional RSS. Construction/loading/first inference add comparatively little. After the first forward, RSS was unchanged through 300 more forwards. This is a short stability observation, not a leak proof for search/session traffic. Cgroup memory and RSS are different accounting measures and should not be subtracted interchangeably.

**Actual local Gunicorn baseline:** One master + one worker configured for four request threads; a local health request verified startup. Baseline master RSS 38.72 MiB, worker 67.20 MiB; cgroup current/peak 102.98/104.19 MiB. With one model loaded and warmed via a temporary worker hook: master 33.53 MiB, worker 299.54 MiB; cgroup current/peak 264.68/264.96 MiB. The cgroup includes the small monitoring Python process, timeout wrapper and container accounting; process RSS sums also double-count shared pages. These are idle/health observations, with no sessions, concurrent gameplay or explanation load. They establish local fit, not a production capacity guarantee.

**Forward-only latency (300 calls per row; cold = first forward after import/load):**

| CPU quota | Intra/inter threads | Cold ms | Warm median ms | p95 ms | Max ms | Total 300 ms |
|---|---|---:|---:|---:|---:|---:|
| 0.5 | 10/10 | 178.62 | 6.968 | 97.851 | 192.086 | 13293.26 |
| 0.5 | 1/1 | 65.84 | 1.103 | 1.356 | 53.963 | 687.48 |
| 0.5 | 2/1 | 97.64 | 0.471 | 0.664 | 86.822 | 600.91 |

The completed 0.5-CPU runs used 512 MiB, disabled swap, Linux amd64 under emulation, and 300 batch-size-one forwards each. Default torch threading detected 10/10 threads and reached 19 OS threads. Torch 1/1 still left NumPy/BLAS threads created by earlier import. Two threads reduced the median in this small sample, but throttling affected tails; neither configuration is established as optimal for the Mac mini or Cloud Run. Torch import took approximately 1.95–2.10 seconds and strict checkpoint loading 8.8–12.5 ms in the recorded one-thread/default processes. These import/load/forward times measure different stages and exclude container startup.

**Measurement boundary:** These are not native Mac mini timings and not Cloud Run estimates. NNPACK fallback and CPU quotas affect the results; do not linearly extrapolate them to M4 or 1 vCPU. No corrected whole-search latency exists to measure yet. The archived `memory-free-*.json` files reflect already-completed earlier work, not an ongoing Render Free acceptance requirement.

**Image size:** The disposable image adds only torch CPU and its dependencies to the existing amd64 API image. Docker history reports an added installed layer of approximately **902 MB**; Docker image inspect's size field grows from **81,349,817 to 314,105,200 bytes** (+232,755,383 bytes, ~222 MiB in that metadata measure). Docker Desktop's image listing reports **366 MB → 1.5 GB**, reflecting its local image storage accounting rather than process RAM. The installed `/opt/venv` occupies 991,944 KiB (~969 MiB) after installation. Thus budget roughly **0.9 GB additional unpacked runtime and ~0.23 GB additional image-content transfer/storage estimate**, plus one ~1.31 MB checkpoint. Exact registry compressed transfer size was not measured; no image was pushed. Packaging all ten checkpoints is unnecessary. Smaller networks reduce inference work but do not remove PyTorch's dominant runtime footprint.

### Updated target assessment

| Target | Feasibility assessment | Remaining evidence |
|---|---|---|
| Native Mac mini M4, 16 GB unified memory | INFERRED: comfortable memory headroom for this ~1.24 MiB model and a single PyTorch API worker; CPU inference is a sensible default. No architecture reduction justified. | Actual native ARM64 cold/warm full-search latency, memory with active sessions, thread settings, and operational availability. |
| Cloud Run, 1 vCPU / 1 GiB | INFERRED: sensible initial CPU inference configuration, given the ~300 MiB local worker and ~265 MiB measured test cgroup; corrected search/concurrency can add memory. | Native Linux runtime plus cold initialization, whole-request latency, repeated games, concurrent requests and session ownership. |
| Existing Render Free | Remains the current production host. | Future neural compatibility is optional; it is no longer a design or acceptance constraint. |

**Mac mini — RECOMMENDED:** Use native macOS ARM64 Python/PyTorch for the first CPU baseline, one serving worker and one reused model. The 16 GB is whole-system unified memory, not a per-process allocation; still measure application growth. An always-on process can load and warm the approved artifact at worker startup, then retain it, avoiding import/model initialization on every move. A locked lazy loader remains useful if neural play is optional, but its first request pays that cost. Do not load one model per game or increase Gunicorn workers while sessions remain process-local. Start with low explicit numerical-library thread counts, then compare a few native settings only after search correctness is established; more M4 cores do not automatically make small sequential forwards faster. Linux-emulation RSS is context, not a native macOS memory prediction.

**MPS — VERIFIED platform capability, untested here:** PyTorch exposes Apple GPU execution through its macOS `mps` device; models and tensors must be placed on that device. [PyTorch MPS documentation](https://docs.pytorch.org/docs/2.14/notes/mps.html). **INFERRED:** Per-leaf batch-size-one GPU dispatch, scalar extraction and host synchronization can outweigh compute savings for this network. GPU use is optional; MPS does not imply Apple Neural Engine execution. Native CPU is the simplest initial path. A future MPS comparison should synchronize timing and test output/move parity; it is more likely worth investigating for batched offline training or batched leaf evaluation than as an immediate requirement. The current agent creates CPU tensors and calls `.numpy()` directly, so MPS is not a drop-in switch: a future device adapter would need explicit placement and CPU conversion boundaries. No MPS loading, training or inference experiment was run. A native macOS MPS experiment would be separate from the existing Linux Docker experiment.

**Mac hosting boundary:** Cloudflare Tunnel can connect an HTTP origin using an outbound `cloudflared` connection without a publicly routable origin IP. [Cloudflare Tunnel overview](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/). **INFERRED:** The Mac, network connection and application process would still need to remain available; a tunnel does not persist sessions or make the origin highly available. **RECOMMENDED:** Keep any future tunnel/service startup, restart recovery, exact-origin CORS and trusted proxy/client-IP handling in a separate hosting assignment. No tunnel, credentials, DNS or host settings were changed.

**Cloud Run — RECOMMENDED:** Begin eventual profiling with CPU-only PyTorch, one model per worker, torch intra/inter threads 1/1 and a bounded heavy-search reservation. **VERIFIED documentation:** 1 vCPU supports 1 GiB; image size is distinct from the instance’s RAM allocation, though process allocations and filesystem writes count toward memory. [Cloud Run memory limits](https://docs.cloud.google.com/run/docs/configuring/services/memory-limits). **INFERRED:** The measured runtime leaves useful initial headroom at 1 GiB, but sessions, the corrected tree, imports and concurrency must be measured together. No new 1-vCPU container or cloud benchmark is needed to justify retaining this network during correctness work.

Bundle one approved checkpoint; initialize it once during worker startup and report neural readiness after warmup, or explicitly budget lazy first-request latency. Import overhead dominates this tiny file's load time in the local experiment. Keep CPU-only dependencies and avoid torchvision/torchaudio; image/package growth still matters operationally, but optimizing it is not a prerequisite to fixing the agents. **VERIFIED documentation:** Cloud Run can scale to zero and provides startup CPU; startup CPU boost is an optional facility. [Container runtime contract](https://docs.cloud.google.com/run/docs/container-contract). **RECOMMENDED:** Decide minimum instances, cold-start tolerance and any boost only in a separately authorized hosting plan; no spending/configuration choice has been made here.

Set an explicit low request-concurrency policy instead of treating four Gunicorn threads as four independent CPU search slots. A future inference-only service can start with concurrency one; a full API needs responsive health/gameplay requests and one admitted heavy search, so its service concurrency should be tested separately. [Cloud Run concurrency guidance](https://docs.cloud.google.com/run/docs/about-concurrency). These settings are recommendations for future experiments, not configuration changes.

**Cloud Run migration blocker separate from inference:** This API owns games, locks, quota counters and reservations in process memory. **VERIFIED documentation:** Cloud Run session affinity is best effort and can break on replacement, saturation or routing changes; it does not guarantee return to the same instance. [Session-affinity contract](https://docs.cloud.google.com/run/docs/configuring/session-affinity). **INFERRED:** Merely deploying the existing stateful API with autoscaling would make game routing/reliability incorrect. One worker per instance does not solve multiple-instance ownership, and a one-instance cap is not durable session storage. **RECOMMENDED:** Before a full API migration, explicitly choose shared/durable session ownership, an accepted restart/session-loss model with routing constraints, or a stateless inference service receiving detached board states while the existing API retains ownership. Do not add that infrastructure to Phase 4C.1.

**Architecture decision — RECOMMENDED:** Preserve the current 326,026-parameter Connect4Net convolutional trunk and dual-head dimensions for corrected neural MCTS. Change training/output contracts as needed (logits for loss, softmax for inference) and version the state/value semantics; do not shrink the network to meet Render Free. Preserve DQN's 42→128→128→7 MLP because it is a straightforward prototype baseline, not because a tiny network is mandated by hosting. Both require new trained weights under their corrected conventions. Neither MPS, ONNX, quantization nor a smaller CNN is warranted before correctness and native whole-agent profiling reveal a need.

For either host, share a read-only model and allocate search state per request; use eval/inference_mode, explicit trusted artifact paths, schema/hash checks, bounded simulations/nodes and no serving-time training examples. Model reuse and simple correctness boundaries remain worthwhile independently of hardware size.

## 8. Reuse-versus-rebuild decision matrix

| Component | Decision | Reason/status |
|---|---|---|
| Historical ten NN checkpoints | Retain as forensic/load fixtures; do not deploy or use as default training initialization | VERIFIED healthy serialization; strategic quality unproven, strong saturation and supervision concerns |
| Connect4Net layer definitions | Preserve as reference architecture | VERIFIED compatible and small parameter storage; training should consume logits, inference apply softmax |
| Absolute historical encoder | Preserve for reproducing old artifacts only | VERIFIED historical contract; new training should make perspective explicit/canonical and version it |
| Corrected Phase 4A terminal/tactical/isolation patterns | Reuse | VERIFIED source/test contracts; adapt reward conventions deliberately |
| Neural selection/backup and root handling | Targeted rewrite with explicit convention | Critical sign, invalid budget and terminal defects |
| Mutable-node transposition table | Remove initially | VERIFIED wrong-parent edges; complexity unnecessary for bounded first version |
| Legal-mask/normalization math | Retain with validation | Correct basic shape, needs finite/zero/no-legal safeguards |
| TrainingExample and self-play label pipeline | Replace storage/label contract | VERIFIED mutation and player-label errors |
| Training loop scaffolding | Retain conceptual self-play → optimize → save; repair execution and reproducibility | Optimizer reset, double softmax, missing manifests and validation |
| DQN MLP/replay/target-network ideas | Reuse architecture/concepts, rewrite transitions/targets/inference interface | Prototype exists; no recoverable deployable weights |
| Production runtime/configuration | Leave unchanged until acceptance gates pass | Local torch fit supports both proposed starting targets; native whole-agent capacity and session routing remain unverified |

## 9. Recommended Phase 4C / 4D milestones and acceptance gates

**RECOMMENDED Phase 4C — correct DQN first.** It has a smaller implementation surface (state/target/loss/inference contracts), needs no tree/transposition repair, and fills the missing learned-agent path. This recommendation remains after removing Render Free as a constraint: bounded correctness work and a clear training/inference boundary are the reasons. One forward per move is a useful secondary benefit, not an architecture mandate or a strength claim.

1. **4C.1 correctness-only:** Specify canonical input/actor reward and signed legal Bellman targets; extract MLP, replay/trainer and independent deterministic inference adapter. Test hand-computed transitions for X/O, wins/losses/draws, full columns and immutable replay. No web exposure or training in that implementation step.
2. **4C.2 bounded offline training:** A separately scoped run with explicit seed, game/step/time budget, artifact manifests and retained diagnostics. Train from scratch. Select a checkpoint through correctness, numerical health, noncollapse and manual-play review; no tournament framework required. Do not claim strength from training loss alone.
3. **4C.3 deployment readiness:** Measure the real artifact/runtime on the selected target (native Mac ARM64 first; native Linux/Cloud Run only if selected); only then add narrowly scoped API/UI/package integration. Review local container/browser behavior before an independently authorized rollout.

**RECOMMENDED Phase 4D — corrected neural MCTS, then fresh self-play.**

1. **4D.1 search/training contracts:** Plain tree, explicit value convention, legal PUCT, terminal/root/budget correctness, safe root guards, immutable examples with stored mover, logits-compatible losses and persistent optimizer. Use fake/oracle networks for deterministic search tests before any training.
2. **4D.2 bounded fresh training:** Keep Connect4Net initially to isolate algorithm fixes; use recorded seeds/config/code and model/data versioning. Consider a smaller network only if profiling warrants it. Historical checkpoints remain load fixtures, not accepted deployed artifacts.
3. **4D.3 resource-gated integration:** Choose explicit simulation presets from native whole-request measurements, model reuse and process-wide concurrency limits. Validate gameplay/history/explanation boundaries; paid provider requests are not required for these checks.

**Specific gates:**

- **Correctness:** Legal moves for both players, terminal rejection, exact draw/win/loss perspective, stable finite normalized masked policies, simulation budgets including minimum/invalid values, no caller/replay mutation, request isolation and reproducible seeded behavior. Neural tree edge visits/backups must follow the sampled path; no shared mutable parents. DQN terminal targets never bootstrap; next-state max excludes illegal actions and includes opponent sign.
- **Checkpoint:** CPU weights-only load, exact schema/keys/shapes/dtypes, all finite, versioned encoding/action order/value perspective, SHA-256 manifest, code/config/seed/training-step identity, and repeatable save/load predictions. Retain optimizer/RNG states for training resume separately from the minimal inference artifact. Publish only files that pass the new manifest contract.
- **Learning sanity:** Value/policy or Q outputs respond to held-out legal states without unexplained universal saturation; check labeled terminal/tactical examples and manual play without attributing tactical-guard success to learned strength. Loss improvement alone and A/B compatibility alone are insufficient.
- **Training:** Before authorizing a real run, require immutable pre-move samples, explicit mover labels, hand-calculated terminal/nonterminal targets, legal finite normalized policy targets, finite nonzero gradients on a tiny synthetic batch, and loss reduction on a small fixed diagnostic set. NN losses consume logits, Adam state survives intended training iterations, and model train/eval mode is explicit. Seed Python/NumPy/torch and record device/runtime/config; require repeatable CPU runs within stated tolerance, not universal cross-device bitwise identity. An actual run needs a fixed game/step/wall-time budget, progress/loss diagnostics, held-out sanity positions and checkpoint/resume validation. Those future tests/runs were not launched in this audit.
- **Deployment:** Preserve the current deployment until a separate integration/hosting assignment. Validate one reused model, per-request tree isolation, atomic session updates, invalid-output/load-error recovery, restart/session behavior and continued operation of existing agents. Measure cold initialization and warm *whole moves* on the chosen native target, plus repeated complete games and overlapping requests. Proposed initial UX targets: warm move p95 ≤2 seconds for DQN and ≤5 seconds for neural MCTS at a documented preset/load; these are proposed engineering targets, not measured guarantees. Record tails, sample size, CPU settings and health responsiveness. On Mac, require stable memory and no sustained swap pressure under the agreed load; on a selected 1-GiB Cloud Run configuration, propose peak total memory ≤800 MiB to retain headroom, with native confirmation and no OOM. No 0.1-CPU or 512-MiB neural acceptance gate remains. CPU must work first; MPS, if later enabled, needs device-parity/tolerance checks. Cloud Run additionally needs an explicit correct session-ownership plan; a Mac/tunnel option needs a separately verified origin/proxy/restart plan.

**Roadmap changes required:** “Recover and expose old neural checkpoints” must become “repair semantics and train a new artifact before exposure.” DQN is source recovery plus new training, not checkpoint recovery. Sequence DQN first for bounded implementation scope, then neural MCTS; retain Connect4Net's existing scale. Replace Render-Free-specific neural gates with selected-target readiness checks. Treat any Mac/tunnel migration or Cloud Run session redesign as separate authorized work; neither is a dependency of the immediate algorithm foundation assignment. The repository's plan does not yet codify detailed 4B/4C/4D milestones and still has stale 4A.2 rollout text; update it in a later authorized documentation change. This audit leaves it unchanged.

## 10. Unresolved questions and experimental limits

- Which exact code/data/seed produced each old checkpoint? Metadata cannot recover this. Could a backbone help after correction? Possible, untested, and not a reason to deploy it.
- Does any earlier checkpoint have useful play despite defects? No strength experiment was run; the small position sample cannot answer. Iteration 20's numerical variation is not selection evidence.
- What are native M4 CPU cold/warm whole-agent latency, optimal low thread counts, memory with populated sessions and health responsiveness? Would a synchronized MPS comparison improve batch-one search or only batched training? No native M4/MPS experiment was needed to settle the architecture decision here.
- If Cloud Run is selected, do 1 vCPU / 1 GiB meet the agreed cold/warm latency and memory gates, and how will game ownership survive multiple/replaced instances? The Linux emulation experiment cannot answer either.
- What is the corrected MCTS tree's per-request node/memory budget and actual end-to-end latency? Implementation must exist before that experiment.
- Would ONNX Runtime, quantization or a smaller network materially improve the complete service? No export/runtime comparison was done, and network shrinkage alone will not remove torch import cost.
- What bounded training budget produces an acceptable, manually playable DQN/neural MCTS checkpoint? Neither training nor a sweep was launched.
- Does a private historical DQN checkpoint exist outside inspected repository/history/archive paths? None was found here.

## Executive recommendation

The best next assignment is **Phase 4C.1: implement and test a perspective-correct DQN training/inference foundation, with no self-play run, production dependencies, packaging, API/UI exposure or hosting changes**. Keep the 42→128→128→7 network; define canonical current-player input, actor-relative rewards, signed legal Bellman targets, copied replay samples, batched selected-action loss, target-network scheduling, deterministic `choose_move(game)`, and versioned safe checkpoint loading/saving. Verify them with focused synthetic/tactical cases and fake or in-memory untrained models; do not deliver random weights as a trained opponent. Stop after local implementation, tests and a reviewable diff. Follow with a separately bounded offline training assignment, then target-specific readiness. Phase 4D should retain Connect4Net’s current architecture, simplify neural MCTS to a correct tree, repair supervision and train fresh weights before release. The old checkpoint set is historical evidence, not a release candidate; the choice of future host does not alter that conclusion.

### Source locations for implementation follow-up

- [Network, prediction, search and loader](../../games/connect4/agents/mcts_nn_agent.py): network 17–52; selection 75–80/303–309; block guard 131–150; tree reuse 234–269; examples/labels 311–337; loader 339–352.
- [Training pipeline](../../games/connect4/train_mcts_nn.py): collection 8–29; losses/optimizer 39–80; iterations/saving 82–112.
- [Corrected standalone MCTS](../../games/connect4/agents/mcts_agent.py), [engine reward/turn handling](../../games/connect4/connect4.py), [list-backed Board](../../games/connect4/board.py).
- `DQN-recovered.txt` in the [audit evidence archive](evidence.zip), `checkpoints.json` in the [audit evidence archive](evidence.zip), `defects.log` in the [audit evidence archive](evidence.zip).

The original audit used a temporary directory outside the repository. This report is now preserved here; diagnostic scripts, recovered notebook and JSON measurements are preserved in the [audit evidence archive](evidence.zip). Containers used `--rm`; the temporary audit image was local only. No application source, production dependencies, Dockerfiles, workflows, model files or credentials were modified.
