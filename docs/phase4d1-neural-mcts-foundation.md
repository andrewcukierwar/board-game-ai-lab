# Phase 4D.1 — Correct neural MCTS foundation

Completed October 4, 2026. **Correctness-only local implementation and verification; no self-play training, new saved neural-MCTS weights, strength experiment, or production integration.**

Fetched `origin/main`, which matched the expected `85692e28668fdc34a2e53bcf2cb72a2fe462848c`. The initial working tree was clean. Created local branch `phase4d1-neural-mcts-foundation` from that reference. No commit, push, merge, deployment, paid request, dependency installation, or production configuration change occurred.

Read the [Phase 4B audit](phase4b/audit.md), [completed DQN replication](phase4c2e-dqn-replication.md), historical neural search/trainer, and Phase 4A search, engine and test contracts. Phase 4C remains closed: its code, fixtures, checkpoints and experiment outputs were preserved. Before/after SHA-256 comparison found **zero changes among 167 model/experiment artifact files**, including all ten historical neural-MCTS checkpoints. Those ten files remain forensic artifacts and were not used for inference or optimization in this phase.

## Implementation boundaries

- [Shared neural contract](../games/connect4/neural_mcts.py): architecture, versioned encoding, logits validation, policy/value conversion, legal masking, inference and checkpoint readers.
- [Neural search](../games/connect4/agents/mcts_nn_agent.py): plain tree, PUCT, alternating backup, exact visit accounting, stable temperature and optional root tactics. `Connect4Net` remains importable from its historical module path.
- [Training primitives](../games/connect4/train_mcts_nn.py): immutable examples, completed-game labeling, logits-compatible loss and persistent optimizer. The old self-play loop, implicit collection and checkpoint writer are removed. Module execution exits with a foundation-only message.
- [Correctness tests](../tests/test_connect4_neural_mcts.py) and [additional API isolation test](../tests/test_neural_mcts_import_isolation.py).

The agent factory, historical battle script, public API, React UI and dependency files are unchanged. The old research factory/battle path still attempts a default historical load; that loader now rejects missing explicit canonical checkpoints. This is an intentional retirement of implicit historical-weight use, not a supported integration path. No public NN Only agent has been added.

## Architecture, encoding and inference

Connect4Net retains all 16 parameter tensors, their names/shapes, and **326,026 float32 parameters**:

| Component | Dimensions |
| --- | --- |
| Shared padded 3×3 convolutions + ReLU | 1 → 64 → 128 → 128 |
| Policy 1×1 convolution + ReLU, linear | 128 → 32, 1344 → 7 |
| Value 1×1 convolution + ReLU, linear + ReLU, linear | 128 → 32, 1344 → 64 → 3 |

The new representation is **`connect4-current-player-1x6x7-v1`**. Input is explicitly CPU float32 `(N,1,6,7)`, with a nonempty batch, top row first. Cells are +1 for the current player's pieces, −1 for the opponent's, and 0 for empty. Actions are physical columns 0–6, left to right; there is no reflection or action remapping. Both heads concern the current player. Encoding validates board size/symbols, player IDs X=0/O=1, and consistency of `current_player` with `piece`. It makes detached input tensors. It assumes a reachable engine state and does not solve arbitrary-board reachability.

`Connect4Net.forward` returns **raw `(N,7)` policy logits and `(N,3)` value logits**, with finite, shape, dtype and device checks. `NeuralInference.predict` applies softmax separately to both heads. Value classes are `[win, draw, loss]`, and the scalar is

\[
v(s)=P(\mathrm{win}\mid s)-P(\mathrm{loss}\mid s)\in[-1,1].
\]

`predict_legal` rejects terminal/no-action states and masks illegal policy entries. Remaining mass is normalized; if every legal probability is zero (including softmax underflow), legal moves receive a uniform prior. Scaling by the maximum legal probability before normalization handles subnormal legal mass. Probability boundaries reject nonfinite/negative values, incorrect dimensions, and sums differing from one by more than an absolute `1e-6` tolerance.

`NeuralInference` requires an already-evaluating model, uses `torch.inference_mode`, and does not change model modes, weights, gradients or buffers. Callers own sequential training/inference phases; concurrent model training is unsupported. Independent agents or searches can share the same read-only inference object. This is the exact policy/value interface intended for a future NN Only agent; it has no tree, tactical guard or sample collection.

### Historical compatibility and checkpoint separation

The absolute historical encoder is explicitly named `encode_historical_absolute`, with tag **`connect4-absolute-x-o-v0`**. The separate `load_historical_network_for_research` accepts the old `model_state_dict`/`iteration` envelope and tags the resulting model historical. Researchers must explicitly use that absolute encoder and softmax the returned logits. Historical value semantics and playing strength remain unestablished. Canonical inference and the trainer reject the historical tag.

`load_checkpoint` instead requires exactly `contract` and `model_state_dict`. The contract records format version 1, the unchanged architecture, canonical encoding, physical action order, W/D/L order, current-player value perspective and logits outputs. It validates exact tensor keys, shapes, float32 CPU dtype/device and finiteness. Both readers use `torch.load(..., map_location="cpu", weights_only=True)` with no unsafe fallback. There is no default canonical artifact and no writer in this phase. Checkpoint tests inject synthetic, in-memory state dictionaries; they do not load historical learned parameters or save files.

An in-memory module must explicitly declare the canonical `representation_version`; its semantic honesty cannot be inferred from tensor shapes. Manually copying untagged historical parameters into a fresh model would bypass provenance and is unsupported. Phase 4D.2 must add provenance and a separately defined resumable training artifact before any run.

## Tree, PUCT and value mathematics

Every search creates a new root and plain tree. Each child has exactly one parent and a detached successor game. There is no transposition table, shared mutable node, evaluation cache, tree reuse or agent-held last root. `SearchResult` returns its own diagnostic tree; retaining results retains those trees, while `choose_move` returns only the action.

For a node \(s\), \(N(s)\) counts visits, \(W(s)\) sums values **for the player to move at that node**, and \(Q(s)=W(s)/N(s)\), with unvisited Q=0. The parent selects the child maximizing

\[
-Q(s_a) + c_{\mathrm{puct}}P(s,a)
\frac{\sqrt{\max(1,N(s))}}{1+N(s_a)}.
\]

Default `exploration` is 1.41; it must be finite and nonnegative. Zero explicitly disables exploration. Crucially, an unvisited action has its finite prior-dependent score, not unconditional infinity. `max(1,N(s))` lets neural priors guide the first traversal when root visits are zero. Equal selection scores use the injected `random.Random`-compatible RNG.

At a neural leaf, the returned scalar concerns the leaf's mover. Backup first increments that leaf's visits/value sum, negates the value for its parent, and repeats at **every** ply, including the root. When a parent evaluates a child, it therefore uses **−child.Q**. No discount is applied.

Engine terminal results override the network. `Connect4.make_move` switches both player and piece even after a winning move, so a reached winning leaf has value **−1 for the now-to-move player**, and a draw has value 0. Its parent receives +1 for the winning mover. Tests establish this independently for X and O, together with multi-ply sign alternation and nonterminal known-value backups.

### Simulation budget and action/target distribution

`simulation_limit` must be a positive Python integer; bools, floating values, strings, zero and negatives are rejected. A **simulation** means one traversal from the already-expanded root through a root edge to an unexpanded or terminal leaf, leaf evaluation/expansion where needed, and backup through the entire path. Root expansion is initialization outside the budget; its value prediction is discarded. A terminal leaf is never expanded or sent to the network.

Thus with budget B:

- Root visits and the sum of root-child visits are exactly B.
- A budget of one visits one legal root edge and produces a valid one-hot target.
- Root initialization uses one network evaluation; at most B further evaluations occur, with terminal traversals requiring none.
- Each expansion creates all permitted children, so memory grows with the tree; simulation budget is not a node-count limit.

Move choice uses root-edge visits, not Q, PUCT scores or raw priors. At positive temperature,

\[
\pi_\tau(a)=\frac{N(s_a)^{1/\tau}}{\sum_b N(s_b)^{1/\tau}}.
\]

The implementation subtracts the maximum log-count before dividing by temperature, preserving zero-visit support and stability at tiny/huge positive temperatures. At **temperature zero**, the distribution is uniform among maximum-visit actions and zero elsewhere; the RNG chooses among those ties. Temperature must be finite and nonnegative. The returned seven-action policy is exactly the action-sampling distribution and future policy target. Invalid/empty/nonfinite visit distributions are rejected rather than repaired into misleading targets.

Full game-state copies protect caller board rows and metadata. All mutable search statistics are local to the call. A per-agent RNG intentionally advances across calls; use separately seeded agents or `search(game, rng=Random(seed))` to replay a search. Concurrent calls should supply independent per-call RNGs. Concurrent tests verify repeatable results and separate trees with one shared read-only model.

## Optional root tactics

Raw neural MCTS is the default: **`tactical_guard=False`**. Enabling it reuses Phase 4A's actual-successor/reply checks via the standalone MCTS helpers:

1. If any immediate winning moves exist, restrict root actions to those wins.
2. Otherwise, if safe responses exist, exclude moves allowing an immediate opponent winning reply.
3. If every move loses immediately, preserve all legal moves.

Terminal draw successors are safe. The guard never changes `current_player` independently of `piece`, never fabricates an opponent turn, and never applies below the root. It deterministically filters candidates; search/RNG still choose among candidates. Even immediate wins retain the full declared simulation budget and valid visit targets. A saturated prior outside the retained candidates falls back to uniform retained priors.

Results identify whether the guard was enabled and whether `immediate_win`, `safe_responses`, `none`, or `disabled` applied. Examples record guard configuration. This keeps future comparisons of NN Only, NN + MCTS, and NN + MCTS + Tactical Guard distinct. Guard success is not evidence of learned tactical strength.

## Immutable examples and training loss

Normal inference has no example buffer. Explicit `capture_example(result)` copies the detached search root's **pre-move** canonical board and acting player; it remains correct even if the caller has since played the selected move. Observations are nested tuples, policies are tuples, and the dataclass is frozen. Construction copies mutable input containers and validates cells, legal policy support, probability normalization, actor and convention metadata.

Each example records:

| Field | Contract |
| --- | --- |
| `observation` | Immutable canonical 6×7 pre-move cells |
| `acting_player` | Captured X=0 or O=1, before the move |
| `policy` | Finite normalized seven-action root visit target |
| `policy_target` | `root-visits-temperature-v1` |
| `temperature` | Actual target/action-sampling temperature, including zero-tie convention |
| `tactical_guard` | Whether root tactical filtering was enabled |
| `encoding` | Canonical v1 representation |
| `outcome` | Pending `None`, then actor-relative +1/0/−1 |

`finalize_examples(examples, completed_game)` requires engine termination, obtains winner ID X=0/O=1/draw=−1, and returns new examples. An actor matching the winner gets +1, the other actor −1, and either actor gets 0 for a draw. Already labeled examples are rejected. No outcome depends on board/list comparisons, reconstructed move parity, or the player's identity after a move. The caller must associate examples with the correct completed game; game provenance orchestration is future work.

Batches use float32 `(N,1,6,7)` states, float32 `(N,7)` policy targets, and int64 W/D/L class labels: actor-relative +1→class 0, 0→class 1, −1→class 2. Winner IDs and value classes are deliberately distinct.

With policy logits p and value logits u, the implemented equally weighted objective is

\[
L=\operatorname{mean}\left[-\sum_a\pi(a)\log\operatorname{softmax}(p)_a\right]
 +\operatorname{mean}\left[-\log\operatorname{softmax}(u)_{y}\right].
\]

There is no softmax before the loss and no regularization term in this foundation. `training_loss` validates target dimensions, probabilities, class IDs and finite logits/loss. `NeuralTrainer` owns one persistent Adam (default learning rate 0.001) across successive `step` calls; future iterations must reuse this object. It enters train mode for each update and eval mode on return, including failures. It checks every trainable parameter for a present, finite gradient before stepping, and checks parameter finiteness afterward. A post-update nonfinite-parameter error requires discarding the trainer; transactional optimizer rollback is not implemented.

The synthetic learning test uses three controlled examples spanning all W/D/L classes, seed 4101 and **eight updates**. It verifies final fixed-batch loss is below 90% of initial loss, finite nonzero gradients, unchanged optimizer identity, Adam step counters of eight, and correct train/eval boundaries. A separate injected-NaN-gradient test proves no optimizer step occurs on invalid gradients. These are discarded in-memory correctness parameters, not learned artifacts or a self-play experiment.

## Differences from the historical implementation

| Historical behavior | Corrected contract |
| --- | --- |
| Absolute X/O input with ambiguous supervision perspective | Explicit canonical current-player v1; separate historical encoder/loader |
| Forward returned softmax probabilities used as loss logits | Forward returns logits; softmax only at inference |
| Parent maximized child's own-player Q | Parent maximizes negative child Q plus prior exploration |
| Mutable transposition nodes with conflicting parent pointers | Single-parent plain tree with path-consistent backup |
| Infinity for every unvisited action | Finite prior-sensitive PUCT from the first traversal |
| First simulation only expanded root; zero-visit targets possible | Root initialization precedes B actual root-edge traversals |
| Unsafe temperature exponentiation and implicit tie order | Stable log scaling and seeded RNG ties |
| Terminal roots could return moves | Early terminal/no-legal rejection |
| Guard changed player without changing piece | Optional actual-successor/reply filtering |
| Inference accumulated shallow copied observations | Explicit collection of immutable pre-move examples |
| Outcome labels inferred incorrectly from list comparisons | Stored acting player plus completed-game winner IDs |
| Adam recreated each iteration; missing mode boundaries | Persistent trainer with validated logits/loss/gradients and explicit modes |
| Implicit default historical checkpoint loading | Explicit versioned canonical artifact required |

## Verification

Native macOS ARM64, Python 3.11.17, torch 2.10.0, NumPy 1.26.4 and pytest 8.4.2 in the existing optional DQN environment. Neural correctness tests temporarily use one intra-op thread and restore the previous setting. Backend tests use the separate existing `.venv`; optional neural tests skip cleanly there. No neural dependency was added to `requirements-api.txt` or any other dependency file.

| Gate | Command | Result |
| --- | --- | --- |
| Neural-MCTS correctness | `/tmp/board-game-phase4c1-venv/bin/python -m pytest -q tests/test_connect4_neural_mcts.py` | **175 passed**, 1.73 s |
| Existing DQN regressions | `/tmp/board-game-phase4c1-venv/bin/python -m pytest -q tests/test_connect4_dqn.py tests/test_dqn_experiment.py tests/test_dqn_diagnostics.py tests/test_dqn_symmetry.py tests/test_dqn_validation.py tests/test_dqn_replication.py` | **466 passed**, 6.04 s |
| Backend suite | `.venv/bin/python -m pytest -q tests` | **354 passed, 7 skipped**, 7.14 s |
| Explicit API isolation | `.venv/bin/python -m pytest -q tests/test_dqn_import_isolation.py tests/test_neural_mcts_import_isolation.py` | **2 passed**, 1.01 s |
| Whitespace | `git diff --check` | Passed |
| Preserved artifact hashes | Before/after SHA-256 checks of model/experiment files | **167 unchanged**, including ten historical neural-MCTS files |

The new isolation test blocks torch/torchvision/torchaudio, DQN implementation imports and every new neural-MCTS implementation module, then creates the API and verifies its health route returns 200. The existing DQN isolation test is unchanged.

Controlled fake logits establish sign/PUCT/tactical behavior rather than relying on random-network playing strength. Untrained Connect4Net parameters are used only for architecture, shared-model isolation, synthetic checkpoint validation and tiny-gradient tests. The bounded gameplay test runs the fake neural agent as X and as O against an opponent always choosing the lowest legal column. Both games terminate legally in at most 42 plies; they collect no training data and assert no strength claim.

## Remaining limitations

This establishes an AlphaZero-inspired correctness foundation, not a full AlphaZero training system or accepted playing model. There is no self-play driver, replay management, root Dirichlet noise, symmetry augmentation, temperature schedule, data/provenance writer, checkpoint writer/resume mechanism, evaluation campaign or learned weight acceptance gate. Exact-zero priors may leave actions unexplored at finite budgets; no exploration floor is silently added. Extremely small search budgets are mathematically valid but do not promise strong play.

Search is synchronous CPU float32, with unbatched leaf inference and full game copies. There are no transpositions, performance optimizations or throughput claims. The diagnostic result tree remains mutable by its owner; immutable captured examples do not depend on later tree mutations. Board validation checks representation and engine metadata, not arbitrary-game reachability. Public-serving concurrency limits and model lifecycle integration are deferred. There is no MPS, ONNX, GPU, Render, Cloud Run or deployment work.

## Recommended separately authorized Phase 4D.2 plan

1. Freeze the reviewed foundation and define a bounded runner before launching it. Start a **fresh** Connect4Net; never initialize from historical neural or DQN checkpoints. Record code identity, encoding/target/checkpoint versions, Python/NumPy/torch seeds, runtime, CPU/thread settings, optimizer configuration and all stop limits.
2. Add a reproducible collector that captures `SearchResult` examples before moves, labels only completed games, drops or explicitly quarantines partial games, and reuses one trainer/Adam across iterations. Keep raw MCTS and guard-enabled data separate. Use guard **off** for the initial learning experiment so deterministic rules cannot masquerade as learned improvements.
3. Proposed initial authorization: one seed, **at most 20 completed games, 840 plies, 200 optimizer updates and 15 minutes**, stopping at the first reached limit; 32 simulations/move, PUCT coefficient 1.41, batch size 32, Adam 0.001, and temperature 1 throughout this first diagnostic run. Start with a two-game collection-only smoke check within those same limits, then inspect labels/visits before enabling the bounded updates. Keep noise and augmentation off initially to isolate this foundation; consider them only under a later recorded protocol.
4. Before the run, implement a minimal versioned inference writer plus a separate resume artifact containing optimizer/RNG state, counters and replay identity. Verify round-trip predictions and a controlled interrupted/resumed synthetic update. Write only to a new experiment directory with hashes/manifests, preserving every existing artifact. A saved diagnostic candidate is not automatically an approved playing weight.
5. Log policy/value losses separately, gradient norms, policy entropy, legal probability mass before masking, value-class distributions, visits and game outcomes. Stop on nonfinite data, failed invariant or exceeded budget. Freeze held-out legal/tactical positions independently of DQN fixtures; inspect both players and draws, action concentration and value calibration. Bounded manual/automated gameplay should compare the same candidate's NN Only and raw MCTS, with guard-on results labeled separately.
6. Choose any later budget increase only after reviewing the bounded report. Loss reduction alone does not establish playing strength. Native CPU on the Mac mini M4 is the likely future inference target; measure complete moves before setting serving presets. Cloud Run remains an alternative requiring separate readiness work. No Render Free CPU constraint is used to shape this foundation.

These numbers are a proposal for the next authorization, not a launched run. Stop at this local implementation and verified handoff; actual self-play training remains a separate phase.
