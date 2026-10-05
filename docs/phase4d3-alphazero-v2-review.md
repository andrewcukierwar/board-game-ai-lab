# Phase 4D.3 — AlphaZero v2 architecture and methodology review

Review date: October 5, 2026. Reviewed **`df8ed9b882b3a4df2008c7d4422bb05918760864`**. The checkout, local `origin/main`, and live `refs/heads/main` returned by `git ls-remote` all matched. The working tree was initially clean. This assignment changes only this document: no training, optimizer updates, checkpoint creation, code changes, commits, pushes, merges, or deployment.

Evidence terminology: **verified** means inspected implementation or a stated direct check; **historical** means a result reported in the retained experiments; **inference** means an architectural interpretation; **proposed** means future work, not an observed result or authorization to execute it.

## Executive assessment

**The corrected neural-search core is a suitable foundation. The present experimental training system is not a suitable default for developing a strong agent.** Keep its perspective contracts, plain PUCT tree, immutable examples, symmetry augmentation, and provenance discipline. Replace its tiny, rapidly changing data-generation/training loop with a coherent policy-iteration system. Do not continue the series of 200-game, single-change experiments by default.

The evidence does not identify insufficient network capacity as the main bottleneck. It identifies weak search teachers, exploratory mistakes throughout games, repeated fitting to very little experience, and an evaluation process that is useful for diagnosis but too small and repeatedly inspected for acceptance. Increasing fitting effort against the same kind of targets is unlikely to resolve that combination.

Recommended v2: retain the three-convolution Connect4Net trunk and policy head; replace W/D/L with a **scalar tanh value trained by MSE on actual actor-relative final outcomes**. Use **256 simulations per self-play move**, **512 for evaluation and initial interactive play**, legal-logit masking, root noise with **epsilon .25 / alpha 1.0**, and **temperature 1 for the first eight plies, then 0**. Store normalized visits as the policy target throughout, independently of action temperature. Collect **256 games with frozen weights**, train from the **last eight generations** at **four sampled training positions per newly generated position**, and repeat. Use batch 128 and persistent AdamW at .0003. Keep exact tactical proofs as diagnostics, with anchoring and tactical guards disabled.

This is a defensible AlphaZero-inspired design, not a guarantee of strength. Its first campaign should test the complete design at meaningful scale, with two predeclared seeds and independent acceptance evidence. Historical checkpoints remain research artifacts.

## 1. Remaining correctness review

### Independent perspective and search derivation

The engine switches `current_player` and `piece` after **every** successful move, including a winning move. Let `t(s)` be the mover in state `s`. Network value is for `t(s)`, not fixed X and not the player who entered the node.

In [`mcts_nn_agent.py`](../games/connect4/agents/mcts_nn_agent.py), `Node.value_sum` and `q_value` use that node's mover. Selection uses

\[
U(s,a)=-Q(s_a)+cP(s,a)\frac{\sqrt{\max(1,N(s))}}{1+N(s_a)}.
\]

For an immediate X win, the resulting leaf has O to move: terminal value is −1, its backup contributes +1 to X's parent, and the parent evaluates that edge using `−child.Q = +1`. An O win is identical with colors exchanged. At a nonterminal leaf with value +.6, its parent receives −.6, grandparent +.6. A terminal draw contributes zero at every level. **The current signs are correct.** Fixed-X encoding or maximizing positive child Q would break this contract; neither is present.

Root expansion precedes the simulation loop and discards its network value. Every simulation traverses a root edge and backs up exactly once along its actual parent chain. Thus root visits and summed root-edge visits equal the budget. Nonroot expanded nodes include their initial leaf evaluation, so their outgoing visits can sum to `node.visits − 1`; this is intentional, not lost accounting. Independent children, terminal-leaf handling, and randomized exact ties are sound. No historical mutable-node transposition defect remains.

### Findings and classification

| Area / location | Classification | Finding and consequence |
| --- | --- | --- |
| Search selection, terminal values, backup | Verified correct in reviewed paths | Current-player encoding, negated child Q and alternating backup agree. Terminal values bypass the network. |
| Root initialization and `max(1,N)` | Legitimate AlphaZero variant | Prior-sensitive first traversal and B actual root-edge simulations are internally consistent. Discarding the root value does not bias the backed-up signs. |
| Zero initial Q; fixed `c=1.41`; fresh tree per move | Legitimate variants / hyperparameters | First-play value zero and no tree reuse are simple defensible choices. They do not guarantee exploration of every action. Keep initially. |
| Legal policy: `neural_mcts.py:128–165` | Verified numerical weakness, not a sign bug | Softmax happens over all seven logits before illegal actions are removed. A dominant illegal logit can underflow every legal probability, invoking a uniform fallback and discarding valid relative legal logits. Mask logits first in v2. Existing fallback correctly implements the declared v1 contract, but cannot recover lost information. |
| Extremely low or zero legal priors | Finite-budget limitation | Even correct PUCT can miss a winning action. Noise is not a coverage guarantee; increasing simulations is not a proof of exhaustive search. Do not confuse this with the old infinity-for-unvisited bug. |
| Root Dirichlet noise | Verified correct for sequential self-play | One legal-root-only draw after masking, convex mixing, normalized support, independent stream; no noise below root. Evaluation disables it. |
| Noise-enabled shared agent concurrency | Interface limitation | Per-call search RNG does not isolate the agent-owned noise RNG. The concurrency test covers noise OFF. Use separate collection-agent streams; serving keeps noise OFF. No concurrent training/search is supported. |
| `visit_policy` and captured policy | Verified correct; coupling is a design choice | Stable visit powers, zero support, and uniform maximum-count ties at tau=0. Currently action distribution and policy target are the same object. This is legitimate, but v2 should separate them before introducing a temperature schedule. |
| Completed-game labels | Verified correct | Pre-move actor is captured explicitly; +1/0/−1 comes from the actual completed winner. Terminal-only finalization and full history replay protect alignment. No winner-ID/class-ID reversal remains. |
| Immutable capture / augmentation | Verified correct | Tuple ownership prevents board mutation. Horizontal reflection reverses columns and policy, preserving actor and outcome. No vertical reflection, rotation or perspective negation belongs here. |
| Trainer / loss | Verified correct mathematics | Raw policy logits use soft-target cross-entropy; raw W/D/L logits use class CE. Equal head weights, persistent Adam, mode transitions, and finite checks are consistent. No double-softmax defect remains. |
| Entire-game tau=1, 32 simulations, Adam .001 without regularization | Questionable combined design; individual hyperparameters | These are not mathematical violations. Their combination with tiny replay and frequent updates makes a poor default teacher/learner regime. |
| Growing collection and ten updates after each game | Questionable methodology | No data expiration, highly unequal lifetime reuse, rapidly changing teacher, and no exact resume. This is the main component to replace. |
| Final game's skipped updates | Deliberate bounded-run behavior | The game cap is checked before optimization, so game 200 is collected but never trained. Reports/tests correctly disclose it. V2 should bound generations explicitly, including their training phase. |
| Conservative tactical proofs | Verified correct within stated scope | +1 is an immediate win; −1 requires a winning opponent reply after every legal action and no own immediate win. A safe block or draw escape is unknown, not zero. No exact draw proof is implemented. |
| Value anchoring | Legitimate hybrid, questionable default | Changes temporary value targets only; immutable behavioral outcomes remain intact. It combines minimax targets on a subset with behavioral outcomes elsewhere. Correct implementation does not establish beneficial learning. |
| Diagnostics and arena | Correct bookkeeping, insufficient acceptance design | Legal replay, actor-relative results, disabled evaluation noise, separate sampled/modal actions, and restricted known-win scores are appropriate. Twelve games per opponent/mode and repeated tactical suites do not establish reliable superiority or general calibration. |
| `Connect4.make_move`, engine boundary | Verified validation defect | It accepts negative columns through Python indexing and allows moves after termination. Current neural paths protect themselves by checking terminal status and legal actions. Fix the engine boundary before expanded reuse; this does not show corruption in retained games. |
| Standalone Negamax cache | Verified correctness bug | Cutoff results are cached as exact values without alpha/beta bound types. Reusing those results under another window can change full-window scores/actions. Do not scale this implementation to become an authoritative deep benchmark. |
| Standalone Negamax terminal scoring | Questionable evaluator / benchmark defect | Terminal wins use the ordinary window heuristic, not an exact dominant win/loss score. Search stops at wins but does not explicitly prioritize outcome above every nonterminal heuristic. Repair before defining a stronger ladder. |

**Direct reproductions, without training:** logits `[1000,0,1,2,3,4,5]` with full column 0 produce uniform mass `1/6` on the remaining actions through the current probability-mask path. Softmax on the six legal logits instead gives approximately `[.00427,.01161,.03155,.08576,.23312,.63369]`. This establishes a reachable numerical failure mode for arbitrary finite model outputs, not that it caused the historical regressions.

For legal history `[5,4,3,6,2,4]`, a depth-3 Negamax call with window `[0,1]`, followed by a full-window call on the same agent, returned `(move=4, score=9)`; a fresh full-window agent returned `(move=3, score=10)`. The narrow result was `(3,5)`. This directly demonstrates unsafe cache reuse. It does **not** invalidate the recorded depth-1/2 match scores: those remain results against that exact legacy opponent, and this reproduction uses a different query pattern. Depth 1/2 remain useful weak comparators after clearly versioning a corrected opponent.

On an empty engine, `make_move(-1)` returned true and placed X in physical column 6. After completed history `[0,1,0,1,0,2,0]`, another `make_move(2)` also returned true. Current search/collector guards prevent both cases. Winner detection and gravity on legally replayed histories showed no remaining defect in the reviewed paths. The engine's `step` returns fixed-X reward and a mutable board reference; the neural collector does not consume it. Do not substitute it for the explicit v2 actor-relative contract.

The standalone UCT agent correctly uses the **incoming mover's** reward, with win/draw/loss 1/.5/0 and no alternating numeric sign in backup. Its children therefore already score for the selecting parent's mover. This differs consistently from neural node values. Its always-enabled root tactical checks must be disclosed in any ladder comparison.

### Test evidence and limits

Read the neural foundation, self-play, scaled, symmetry, noise, value-audit, anchoring and import-isolation tests, plus standalone MCTS tests. Ran this deliberately restricted existing subset:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/board-game-phase4c1-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_connect4_neural_mcts.py tests/test_neural_root_noise.py \
  tests/test_neural_value_target_audit.py tests/test_neural_value_anchoring.py \
  -k 'not fixed_tiny_batch_loss_decreases and not nonfinite_gradient_aborts and not no_noise_diagnostics_or_strength_evaluation and not freezer and not driver_anchors'
```

**276 passed, 5 deselected, 9.70 seconds.** This selection performs no optimizer updates and writes no checkpoints; full training/writer suites were read rather than executed. It includes proof checks against the locally retained 10,608 audited states. Passing tests support existing contracts, not learned strength. They do not test the proposed scalar head, generation lifecycle, full resume, pre-softmax masking, or corrected Negamax. Those require implementation-time tests. No learned checkpoint was loaded in this review.

## 2. What is genuinely AlphaZero-style?

The defining mechanism is already present: a shared policy/value network guides adversarial tree search; search visits supervise policy; completed self-play outcomes supervise value; learning changes future search. Canonical player-relative representation, exact rule-based terminal scoring, and legal actions fit that mechanism. W/D/L CE is a reasonable distributional variant because `P(win) − P(loss)` estimates the same expected reward that search needs.

The original AlphaZero formulation estimates scalar expected outcome, trains with value MSE plus policy cross-entropy and regularization, and updates a single continuing network. It explicitly differs from AlphaGo Zero's best-player selection. Therefore neither continuous updates nor absence of promotion gates makes this repository “not AlphaZero.” [Silver et al., AlphaZero preprint, pp. 2–4](https://arxiv.org/pdf/1712.01815).

AlphaGo Zero used early visit-proportional sampling followed by greedy visits, root Dirichlet noise, recent-game replay, and a separate best-player evaluator. Its published system used a residual network and a scalar tanh head. Those are useful reference choices, not requirements to copy at Go scale. [Silver et al., AlphaGo Zero, Methods](https://ai6034.mit.edu/wiki/images/Nature24270_AlphaGoZero.pdf).

The later AlphaZero report documents 800 training simulations per move and PUCT with a slowly growing exploration coefficient. Neither 800 nor that coefficient is intrinsically correct for a seven-action game. [Silver et al., 2018, Search and Table S3](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf).

| Dimension | Current implementation | Why the difference matters here |
| --- | --- | --- |
| Value | W/D/L CE; scalar expectation used by PUCT | Valid, but an unobserved draw category is not delivering useful extra information. Scalar regression is simpler. |
| Policy target | Temperature-adjusted visits, also used for action sampling | Valid. Separating the teacher distribution from execution temperature preserves softer supervision late in v2 games. |
| Outcome target | Actual final result, with optional tactical replacement | Final outcomes are standard supervision, even when play is imperfect. Replacement changes the learning objective. |
| Root noise | Implemented correctly; opt-in | Keep for self-play diversity, disable for evaluation. Tune scale to seven actions rather than borrowing chess/Go concentration mechanically. |
| Temperature | 1 at every self-play ply | Keeps sampling avoidable tactical errors through the ending. An early-only schedule is more appropriate for short Connect 4 games. |
| Search | 32 simulations, seven root actions, fresh tree | Correct but often too weak to improve a bad learned prior. Seven actions permit more useful search at modest local cost. |
| Replay / updates | All examples, no expiration; 320 sample appearances per post-smoke game | Tiny early datasets are fitted repeatedly; old weak behavior remains influential. A bounded recent window and explicit data ratio matter more than paper-sized buffers. |
| Model lifecycle | Frozen within a game; updated between games | No within-game race or moving model. Staging several hundred games improves auditability and data diversity per update phase. |
| Promotion | No trained champion gate | Not a correctness defect. Separate continuing learning from conservative release selection. |
| Architecture | Three-layer CNN, dense heads | Adequate spatial representation for 42 cells; no demonstrated need for a large residual tower. |

The project should claim **AlphaZero-inspired self-play reinforcement learning**, with its exact departures documented. It should not claim faithful large-scale reproduction or solved play.

## 3. Value-head decision: choose B, scalar tanh with outcome MSE

For every pre-move state, store `actor = current_player` and assign after legal termination:

\[
z_t=\begin{cases}0&\text{draw}\\+1&\text{winner = actor}_t\\-1&\text{otherwise.}\end{cases}
\qquad v_\theta(s_t)=\tanh(h_\theta(s_t)).
\]

Train `mean((v − z)^2)` with no discount, no class weighting, no rescaling to winner IDs, no tactical override, no root-Q substitution, and no loss on unfinished games. Both network and terminal leaf values use the same current-player perspective. The value target is an estimate of expected final reward under the generating search/behavior policy, averaged over replay's recent policies; improvement toward minimax is an aspiration of policy iteration, not a property guaranteed by the labels.

Why prefer this over W/D/L? Search only consumes the expectation. The existing three-class output expends a separate probability dimension on draws for which virtually no independent supervision exists. A scalar explicitly models the quantity being optimized and avoids presenting poorly supported draw probabilities as meaningful. The change removes only 130 parameters; its purpose is semantic simplicity, not compression or a claim of increased capacity.

W/D/L is **not mathematically inferior**. With sufficient data, CE can estimate a full outcome distribution and its expectation correctly. Nor does tanh cure saturation: `d(v−z)^2/dh = 2(v−z)(1−v^2)`, which can become small at wrong extreme outputs. The current CE has a strong corrective logit gradient on confident wrong classes. Monitor scalar saturation and wrong-sign extremes, use less aggressive fitting and regularization, and do not attribute a future gain automatically to the head switch. The stronger case for scalar is a direct objective and simpler diagnostics, coupled with the redesigned data regime.

In deterministic perfect-information Connect 4, a fixed state's minimax outcome is one of −1/0/+1. Standard 7×6 Connect 4 is a first-player win from the empty board; it is not a game whose perfect opening outcome requires many draws. Drawn positions still exist and must be preserved correctly. [Allis, 1988 thesis, abstract](https://tromp.github.io/c4/connect4_thesis.pdf); [Tromp's solved-position database description](https://tromp.github.io/c4/c4.html).

Consequently, sparse self-play draws are a coverage limitation, not evidence that an otherwise correct value formulation must fail. Scalar zero can mean a forced draw **or** a balance of wins and losses under imperfect play. `(v+1)/2` is expected game score, not generally win probability. Do not invent a draw probability from it. Test proven draws independently; do not label unknown positions draw or oversample the one historical drawn game into apparent coverage.

Neither scalar regression nor W/D/L fixes behavioral/minimax disagreement. A target changed from −1 to +1 changes the training objective under either head. The mixed anchoring results therefore do not settle the head comparison. Other formulations—dual behavioral/minimax heads, TD/root-value mixtures, score margins or distance-to-win—add target and weighting decisions without evidence that they are necessary. Omit them from v2.

## 4. Self-play exploration and policy targets

Use zero-based pre-move occupancy `ply`: **tau_action=1 for ply 0–7, then 0 for ply 8–41**. At zero, sample uniformly among maximum-visit ties using a recorded RNG. This is deterministic away from ties, not lowest-column tie bias. Evaluation always uses tau_action=0. The schedule counts plies, not full turns or generation number.

Eight plies allow opening diversity, then prioritize execution as immediate threats become common. A schedule copied as “first 30 moves” would leave most historical Connect 4 games exploratory almost to termination. Eight is a justified starting default, not an empirically optimized breakpoint. Some earliest wins occur before the cutoff, so the schedule cannot eliminate every exploratory mistake.

**Separate target from action:** always store `pi_target[a] = N(root,a)/B` (tau_target=1), including after ply eight. Use the action schedule only to choose the move. Also store visits, action, action temperature and target convention separately. Late one-hot maximum-visit targets would be a legitimate alternative, but throw away useful relative search allocation at a finite budget. Do not accidentally get that alternative merely by changing `agent.temperature` in the old collector. Cross-entropy need not supervise the exact distribution used to execute an action; it distills the search teacher.

Keep one fresh root-noise draw per self-play move throughout the game:

\[
\eta\sim\operatorname{Dirichlet}(1.0,\ldots,1.0)\text{ on legal actions},\quad
P'=0.75P+0.25\eta.
\]

With seven actions, total concentration is 7 rather than the historical 2.1. This supplies less spiky broad exploration than alpha .3 while retaining state-dependent randomness. Noise mass has mean `.25/K` per legal action; alpha changes variance, not that mean. Keep alpha 1.0 as legal count falls, for simplicity. It is a heuristic Connect 4 default, not an experimentally proven optimum. Noise affects search allocation; late greedy visits still let search reject poor noisy suggestions. Disable it in every strength/calibration deployment-mode search. Do not add noise directly to training policies after search.

**Will the schedule reduce contradictions? Plausibly, but not completely.** A read-only calculation on retained 4D.2d audit rows found 314 immediate-win states at one-based ply ≥9: actual sampling took the win in 183, while the historical lowest-index modal action won in 233. Among 66 eventual-outcome contradictions in that subset, **26 had every maximum-visit action immediately winning**. Changing just those recorded decisions to tau=0 would immediately win from those states, under any maximum-visit tie. Another modal case depended on tie choice. These are local counterfactual decisions, not a replayed alternative campaign or a predicted percentage reduction: earlier decisions change all subsequent states, searches, labels and training.

Other contradictions arise because search never visits the win, ranks it below another action, misses a deeper tactic, or the opponent fails to exploit a losing state. Root noise remains active and can affect modal choices. Better search and later exploitation address complementary causes. Retain contradiction measurements with separate denominators: proven states and all states. Zero contradictions is not a training correctness requirement.

## 5. Search budget

| Use | Proposed simulations | Interpretation |
| --- | ---: | --- |
| Training self-play | **256** | Eight times the old teacher budget; enough to make deeper comparisons materially more plausible without expensive infrastructure. |
| Acceptance evaluation and first interactive preset | **512** | A fixed stronger deployment-like budget; same convention and raw search, noise OFF, tau=0. |
| Unit tests / historical comparison | 1–32 as appropriate | Keep low budgets for invariants and historical reproduction, not as the new strength target. |

At the supplied native-M4 measurement of roughly 12 ms for 32 simulations, simple proportional estimates are **96 ms at 256** and **192 ms at 512**. These are planning estimates, not benchmarks: terminal revisits, depth, state copying, validation and CPU scheduling change costs. Root initialization also has fixed overhead. Measure actual whole searches and p95 before integration. The minimal head change should not materially alter compute, but that is still an inference.

Thirty-two is sufficient for legality tests and detecting pipeline defects, but the retained zero-visit winning actions show that it is a poor sole training/evaluation budget here. The full width of two seven-action plies already contains 49 action sequences; simulation count is not depth or exhaustive coverage, but 32 offers little room to correct a misleading policy. A bigger budget can also reinforce bad values, so strength must improve relative to an **untrained network at the same budget**, not merely relative to a 32-simulation predecessor.

Do not add 1,024/2,048 presets or a simulation sweep to the first campaign. No Render Free constraint informs these choices. Retain the simple CPU tree; optimize copying or batch inference only if measured campaign/runtime cost makes it necessary.

## 6. Training loop and replay lifecycle

Replace `complete game → ten updates → repeat` with the following serial generation lifecycle:

1. Freeze the current learner's weights and their identity. Generate 256 complete games from empty boards, using that same snapshot for both sides and the declared self-play settings. No optimizer updates during collection.
2. Validate histories, finalize outcomes, append complete games, and evict generations older than the latest eight. Keep original immutable examples; reflect only sampled minibatches.
3. Let `M_new` be newly collected positions. Perform **K = ceil(4 M_new / 128)** updates, sampling 128 positions uniformly from the current replay, without replacement within a batch and independently across batches. Do not run four epochs over the entire growing window.
4. Validate the candidate and save an atomic resumable generation boundary. The candidate becomes the next self-play model if numerical/contracts checks pass. Run development measurements each generation and periodic strength/champion arenas as specified below.

The first generation supplies thousands of positions before any fitting, avoiding a two-game bootstrapping bottleneck. At steady state with eight similarly sized generations, four appearances per new position means roughly half a replay-window pass per generation and about four lifetime appearances per ordinary example. Warm-up examples can receive more while the window fills; log actual exposures and age. Uniform position sampling weights longer games more, intentionally and transparently. Avoid prioritized sampling or synthetic class balancing in v2.

**Verified historical reuse:** 4D.2b used 63,040 sample appearances over 3,673 collected positions, averaging 17.16 appearances; its first two games' 34 positions received 2,698 appearances, averaging **79.35** each. 4D.2f averaged 17.59 overall, but **87.90** for its first two games. These counts were independently recomputed from retained `sampled_indices`. The terminal game's examples received none. “Uniform sampling” within each update therefore did not mean uniform lifetime influence. This supports lowering and measuring the data ratio rather than prescribing more epochs.

The replay window is **eight generations / 2,048 games**, typically tens of thousands of positions, with an absolute 86,016-position bound at 42 plies/game. Expire whole generations oldest first. Do not preserve weak early experience indefinitely. Persist archived game records outside the active replay if desired; archival retention and optimization eligibility are different concepts.

Use **AdamW**, learning rate **3e−4**, betas **(.9,.999)**, epsilon **1e−8**, decoupled weight decay **1e−4 on convolution/linear weights**, zero decay on biases; batch **128**; global gradient clipping **5.0** after checking unclipped gradients are finite. Keep equal policy/value loss weights. Use a constant learning rate for this bounded campaign and record gradient norm/clipping frequency. Lower LR and mild decay aim to reduce abrupt fitting and extreme logits; they are proposed defaults, not proven cures. No scheduler, optimizer reset, or loss-weight adaptation per generation.

The optimizer continues with the learner across generations. A separate “champion” is an evaluation/release candidate, not the owner of the optimizer or a mandatory self-play gate. Rejecting champion promotion must not silently roll back learner weights while retaining incompatible Adam moments. A failed numerical update stops the run; recover only from a valid atomic boundary, restoring model **and** optimizer **and** replay **and** RNG state.

For this portfolio project, essential pieces are fixed generation identities, bounded recent replay, measured data ratio, outcome integrity, independent evaluation, and reproducible resume. Distributed actors, a parameter server, GPU queues, league populations, prioritized replay, tree reuse, transposition tables, automated hyperparameter tuning and gating every generation are unnecessary initial complexity. Serial frozen generations are chosen for clarity, not because continuously updated AlphaZero training is invalid.

### Checkpoints and resume

Create a new explicit scalar architecture/output contract; reject v1 W/D/L checkpoints on the v2 path. Fresh initialization only. Keep the historical readers/artifacts sufficient for research reproduction, without making compatibility constrain v2.

Separate a small inference artifact from a resume artifact. A resume boundary must bind model tensors, full optimizer state/step, generation/phase counters, exact ordered replay or content-addressed replay shards, pending collected-generation records where applicable, all Python/NumPy/torch/search/noise/sampling/augmentation RNG states, config, code identity, runtime/thread versions, data/target schemas, and manifest hashes. Record remaining cumulative game/ply/update/time budgets so resuming cannot reset campaign limits. Do not serialize live trees or Python model objects.

Write temporary artifacts then atomically publish their completed manifest; never overwrite historical experiment directories. Keep completed generation inference snapshots, and the latest valid resume boundary plus its predecessor. An interrupted generation may restart from its last complete boundary, with abandoned attempts retained and counted against resource limits; exact continuation is claimed only from a complete boundary in the same runtime. Test uninterrupted versus interrupted/resumed trajectories and optimizer state during implementation. JSON RNG provenance alone is not exact resumability.

## 7. Network architecture decision

**Minimally change the value head; retain the trunk.** The existing 1→64→128→128 padded 3×3 convolutions with ReLU have a 7×7 receptive field after three layers. Dense heads can combine spatial locations over the full 6×7 board. This is enough structural reach to represent lines, gravity-dependent patterns and interactions; it is not proof that all such concepts will be learned, but there is no demonstrated information bottleneck.

The v2 model has **325,896 parameters**: 326,026 minus the difference between 64→3 and 64→1 final value layers (130 parameters). Policy head remains 128→32 by 1×1 convolution, ReLU, flatten 1,344→7 logits. Value head remains 128→32 by 1×1 convolution, ReLU, flatten 1,344→64, ReLU, then 64→1 and tanh. No batch normalization or dropout; existing train/eval discipline still applies.

A small residual tower could improve optimization and compositional feature learning, and need not be larger. However, the current trunk is only three layers deep; there is no deep-gradient problem demonstrated here, and no capacity/generalization comparison under a credible data regime. Replacing it now adds normalization, initialization and latency decisions without addressing the clearest failure mechanisms. A Connect 4-specific MLP or column network similarly lacks a stronger evidence basis. Retention follows a fresh assessment of adequacy and simplicity, not checkpoint compatibility or sunk cost.

Keep the signed **(N,1,6,7)** canonical board: own +1, opponent −1, empty 0, top row first, physical action columns 0–6. It contains all game-relevant information for valid Connect 4 states. There are no repetition, castling, hidden-state or history-dependent rules requiring history planes. Splitting own/opponent occupancy into two binary planes is reasonable but not essential; no extra actor plane is necessary with mover-relative semantics. Preserve explicit physical actor metadata for outcome validation.

## 8. Lessons from Phases 4D.2b–f: keep / change / remove

The earlier reports were appropriately cautious about single-seed results. Their recommendations were local next experiments, not permanent constraints on this redesign.

| Component or intervention | V2 decision | Evidence / reason |
| --- | --- | --- |
| Corrected current-player PUCT and immutable examples | **Keep** | Independent sign derivation and tests support the repaired foundation. |
| CNN trunk and policy head | **Keep** | Modest size, sufficient spatial reach; insufficient evidence of a capacity bottleneck. |
| W/D/L head and CE value objective | **Change** | Scalar tanh + outcome MSE directly expresses the quantity used by search. No claim that CE was broken. |
| Post-softmax legality masking | **Change** | Mask logits before softmax; retain raw logits separately for training/diagnostics. |
| Horizontal augmentation probability .5 | **Keep** | 4D.2c reduced pooled legal-policy mirror L1 from 1.2853 to .4773 and improved modal tactics 123→167/240 versus its learned baseline. Valid symmetry independent of those scores. |
| Root Dirichlet noise | **Keep, change alpha to 1.0** | 4D.2d broadened root coverage; mean visited actions rose 4.72→5.82. Tactical/strength gains were mixed. Keep exploration, use a less spiky seven-action default. |
| All-game temperature 1 | **Change** | Early exploration followed by visit-greedy execution reduces an identifiable source of avoidable late mistakes. |
| Action-temperature-dependent policy target | **Change** | Explicit normalized-visit teacher throughout; separate execution distribution. |
| 32-simulation training / acceptance | **Change** | 256 / 512; preserve 32 only for cheap checks and historical context. |
| Engine-proven anchoring in default training | **Remove from v2 path** | Correct hybrid intervention, but 4D.2f improved exact-loss prediction while exact-win prediction worsened; depth-2 raw MCTS remained 1 win/12. |
| Tactical root guard in learned agent | **Keep OFF** | Learned strength must be demonstrated without a rule layer masking missed tactics. Existing guarded UCT remains a labeled opponent. |
| Conservative exact-value proofs and contradiction audit | **Keep as diagnostics** | Separates behavior from minimax; add solved draws and deeper positions outside optimizer supervision. |
| Ten updates per game / unbounded growing replay | **Replace** | Frozen generations, expiring window and explicit data ratio address unequal reuse and unstable teacher changes. |
| Adam .001 without decay | **Change** | Persistent AdamW .0003, mild weight decay and bounded gradients; record rather than assume benefit. |
| Repeated old “blind” tactical suites | **Keep as development regressions** | Once inspected repeatedly they cannot supply fresh acceptance evidence. |
| Six-prefix, 12-game-per-opponent comparisons | **Replace for acceptance** | Broader paired openings, uncertainty estimates, stronger audited opponents and independent seeds. |
| Minimal inference format and provenance | **Keep principles; version and extend** | New scalar schema plus genuine resumable training boundaries. |
| Old experiment files and reports | **Keep intact** | Historical evidence must remain independently interpretable. No relabeling or checkpoint conversion. |

Anchoring should be regarded primarily as a **diagnostic and optional hybrid-training mechanism**, not as a necessary repair to AlphaZero. It addresses only an easily proved tactical subset and can leave policy targets encouraging actions inconsistent with the substituted value. In 4D.2f only 878/63,040 optimizer appearances actually changed targets; aggregate primary Brier improved .7277→.5574, while proven-win Brier worsened .5523→.6376. The result neither establishes that anchoring is harmful nor supports a permanent default. It also supplies no draws.

Constant exploratory self-play is one reason anchoring appeared attractive, but not the only possible source of disagreement. Do not assert that a temperature schedule will make exact supervision universally unnecessary. If a future solver-supervised agent is desired, define it explicitly as a hybrid with consistent targets and separate evidence. That is outside this v2 campaign.

## 9. Cleaner evaluation and acceptance methodology

Treat four questions independently: does the implementation obey its contracts; can it execute tactics; are its values useful for the stated target; and does it win games? Never substitute lower fitting loss, target agreement after anchoring, mirror agreement, or successful serialization for strength.

### A. Training correctness

Implementation-time gates must cover both actors and mirrors, multi-ply sign traces, exact terminal draws, legality under extreme logits, root-only noise, independent action/target temperatures, immutable completed examples, generation freeze, exact update counts and eviction, optimizer persistence, and interrupted/resumed equivalence. Include loss/gradient and tiny-fit tests under the future implementation assignment. They were not run as training in this review.

Keep history replay and hashes, but distinguish essential assertions from expensive diagnostics. Rechecking every prior example after every ply and duplicating root inference are useful audit scaffolding; v2 can validate immutable capture at boundaries and avoid redundant prediction calls. Do not weaken outcome/legality/finite checks to gain speed.

### B. Tactical competence and solved-state evaluation

Use all previously inspected suites as **development** sets. Before training, freeze a new sealed acceptance package using model-blind legal histories, independently checked labels and content hashes. Exclude all previously inspected diagnostic/probe boards and their reflections, and keep its base situations disjoint from the new development package. Deduplicate by board plus actor, including transposed histories. Proposed package:

* **400 base tactical states**, half immediate wins and half unique immediate-safe responses, balanced by actor and stratified by game stage/direction, plus their mirrors (800 rows).
* **300 base fully solved states**, 100 each exact win/draw/loss, balanced as feasible by actor, plus mirrors (600 rows). Include multi-ply wins, forks, losing traps, safe-looking but losing blocks, and late draw-preserving decisions. Record every legal action's exact outcome, not just one favored move.

Use a separately validated exact solver or bounded exhaustive endgame solver, not the present heuristic Negamax as the oracle. Compare independent solution methods on a subset and validate every history against the engine. Pascal Pons's solver is an available reference implementation; its signed solution scores need explicit conversion to actor-relative W/D/L, and arbitrary shortest-win preferences should not be confused with preserving optimal outcome. [Pons, Connect 4 solver](https://connect4.gamesolver.org/en/).

Complete and freeze quotas before the campaign; inability to construct/verify them is a preflight issue, not permission to replace them after seeing models. Cluster mirrors and transposed equivalent boards as one base situation. Check direct/reflected overlap with training records and report both the complete set and the prespecified overlap-excluded sensitivity. Incidental self-play overlap is not deliberate leakage, but cannot count as independent evidence. This matters because 4D.2f already encountered one exact-holdout pair naturally.

At 512 simulations, noise OFF, guard OFF, tau=0, report immediate-win rate, safe-response rate, solved optimal-action preservation, avoidable outcome loss, and zero-visit winning actions. Report NN Only separately. Use search/tie seeds 0, 1, 2 and 3 for each tactical/solved row and average within base state rather than treating each repeat as independent. A unique safe response proves reply safety, not the eventual game value.

### C. Value accuracy versus behavioral calibration

On the solved set, score **raw scalar values** with MSE/MAE against exact outcome; report win, draw, loss, actor and stage separately, wrong sign and wrong-sign saturation (`abs(v)≥.95`). Score search root means separately: they are search averages, not the same estimator as the raw head. For draws, report `mean(abs(v))` and whether search preserves drawing actions. Do not reuse W/D/L NLL/Brier with fabricated scalar-derived class probabilities.

For behavioral calibration, freeze the chosen model, generate **256 new held-out games per seed** with the training search/exploration schedule and no updates, and compare pre-search values to actual eventual outcomes. Report MSE versus the zero predictor and a development-estimated constant predictor, plus mean predicted versus mean realized score in five predeclared equal-width bins. Use game-cluster bootstrap intervals and disclose sparse bins; positions within a game are not independent. This measures the declared behavior, not optimal-play calibration or human win probability. Deployment-mode arena values/outcomes can be reported separately because opponents and exploration differ.

### D. Playing strength, opponent ladder and champion selection

Use score `(wins + .5*draws)/games`. Always report W/D/L, X and O separately, opening identity, search budget, model hash, timing and complete histories. Do not estimate Elo from the old tiny repeated matches.

The ladder is **Random; corrected Negamax depth 1; corrected Negamax depth 2; corrected Negamax depth 4; standalone UCT at 800 simulations with its root guards disclosed; initial untrained v2 at 512; retained 4D.2f at 512**. The last two provide equal-search-budget learned baselines; they prevent crediting a simulation increase entirely to learning. Keep old 32-simulation results as historical context only. Also measure NN Only versus Random. A fresh cache does not repair the Negamax bound bug: use correct EXACT/LOWER/UPPER entries or omit caching in the reference opponent, and use exact dominant terminal scores with zero for draws.

Depth 1 remains a sanity check, depth 2 an important required floor, and depth 4/guarded UCT provide headroom. Beating depth 2 does not establish strong general play. Fully solved positions measure strategic errors that a single opponent style can miss. A perfect solver need not be beaten as a release requirement; its role is the oracle and upper reference.

For each final ladder opponent, play **100 opening pairs / 200 games**, swapping agent sides on the identical position in each pair. Include 20 empty-board pairs with distinct seeds and 80 distinct nonterminal prefixes of length 2–8, selected before model evaluation, balanced in reflection and mover. Use separate development and sealed-final opening banks. The exact solver can record opening outcomes to expose unavoidable disadvantages; report empty-board and prefix strata separately. Side swapping does not change whose turn the board specifies—it changes which agent owns each color.

Use a paired, opening-cluster bootstrap (10,000 resamples, fixed analysis seed) for score and paired differences. Treat reflection families together, and the repeated empty-board results conservatively as a shared opening family; disclose that the effective independent sample size is smaller than 200. No post hoc exclusion of difficult openings or incomplete games. A deadline-limited arena is incomplete evidence; do not promote from its biased prefix. Fixed simulations measure algorithmic quality; report wall time as well, without calling unequal opponent algorithms compute-matched.

**Training progression is ungated. Release selection is gated.** Every numerically valid learner generation proceeds. At generations 5/10/15/20, play a 200-game development arena against the current champion at equal 512 budgets. Initially champion is untrained v2. Promote only with score ≥.55 and the lower end of the two-sided 95% opening-cluster interval >.50, all correctness checks passing, and no greater than two percentage points of regression in either immediate-win or safe-response accuracy against the champion on the fixed development tactics. Ties/inconclusive results retain champion. Also run 40 development games each against Random and corrected depth 1/2 at those four checkpoints. These are selection measurements, not independent final claims. No transitive “best-ever” strength guarantee follows from winning one arena.

After training, choose each seed's champion using development evidence only, then evaluate it once on the sealed package/ladder. Do not use sealed results to pick a different generation or to tune the configuration. Multiple development promotion opportunities introduce selection bias; the separate final evaluation is the protection. A later attempt informed by final failures needs a newly declared acceptance protocol, not relabeling the same data blind.

## 10. Exact proposed AlphaZero v2 configuration

This table is the single recommended implementation baseline. Values are engineering defaults to validate, not claimed optima.

| Setting | V2 specification |
| --- | --- |
| Input | CPU float32 `(N,1,6,7)`, own +1/opponent −1/empty 0; physical columns; validated reachable histories |
| Trunk | Existing padded 3×3 Conv 1→64→128→128, ReLU after each |
| Policy head | Existing Conv1×1 128→32, ReLU, flatten, Linear1344→7 raw logits |
| Policy inference | Mask illegal logits before stable softmax; no legal actions means terminal/reject, not uniform repair |
| Policy target / loss | `pi=N_a/B` at every ply; mean `−sum(pi * log_softmax(raw_logits))` over all seven outputs; illegal target entries zero |
| Value head | Existing Conv1×1 128→32 + ReLU; Linear1344→64 + ReLU; Linear64→1 + tanh |
| Value target / loss | Actual final +1/0/−1 for pre-move actor; mean squared error; coefficient 1 |
| Total data loss | Policy CE + value MSE, each minibatch mean; AdamW decay separate, not a second L2 term |
| Search statistics | Node-own-mover Q; unvisited Q=0; alternating backup without discount |
| PUCT | `−Q(child) + 1.41 P(s,a) sqrt(max(1,N(parent))) / (1+N(child))` |
| Budget accounting | Root expansion uncounted; exactly B root-edge traversals; terminal leaves use engine value |
| Search structure | Fresh single-parent tree per move; no mutable transpositions/cache/tree reuse |
| Simulations | Self-play 256; acceptance/interactive 512 |
| Root noise | Self-play only, epsilon .25, legal-action Dirichlet alpha 1.0; one independent draw per move |
| Action temperature | 1 at pre-move ply 0–7; 0 thereafter; evaluation always 0; seeded uniform max-visit ties |
| Symmetry | Each sampled example reflected horizontally with probability .5; no inference ensemble |
| Rules in training | Exact terminal outcomes only; tactical guard OFF; value anchoring OFF; no resignation/truncated-game draw labels |
| Collection | 256 complete empty-start self-play games per frozen generation |
| Replay | Last eight complete generations, ≤2,048 games / 86,016 positions; uniform position sampling |
| Updates | Batch 128; `ceil(4 * new_positions / 128)` steps per generation, including generation 1 and final generation |
| Optimizer | Persistent AdamW lr .0003, betas .9/.999, eps 1e−8; weight decay .0001 on weights, none on biases; clip global gradient norm at 5 |
| Runtime | Native M4 CPU float32; initially one intra-op/inter-op thread, deterministic algorithms; no new device backend |
| Model progression | Latest valid learner generates next generation; no per-generation win gate |
| Champion gate | Every five generations; 200 paired development games, score ≥.55 and 95% lower bound >.50; no failed correctness gate |
| Artifacts | New scalar contract; atomic generation inference + resumable boundaries; hashes/config/runtime/replay/RNG/optimizer identities |

All-seven-output policy CE intentionally penalizes illegal raw mass while inference conditions on legality. That is a valid supervised policy objective. Do not multiply zero target mass by negative-infinity masked log-probabilities in the loss, which can create NaNs. Use finite raw logits for training and the separate legal-logit path for search.

## 11. Short migration plan

**Milestone 1 — coherent v2 contracts and runner, no research training.** Add the scalar head/schema and stable legal-logit inference; separate visit targets from action sampling; make search budgets and ply schedule explicit. Reuse the corrected tree and encoders. Implement generation replay/expiration, data-ratio updates, persistent optimizer and atomic resume. Harden engine move validation. Keep v1 research compatibility isolated, preserve its files and histories, and avoid rewriting experiment reports. Remove fixed-32/tau-1/200-game assumptions from the new path rather than patching every old audit function into a general trainer. Verify scalar perspective/loss, generation freeze/eviction and resume in focused synthetic tests.

**Milestone 2 — credible evaluation and preflight.** Provide a corrected reference Negamax, independently verified solved-position oracle, development/final split, paired arena scoring and confidence intervals. Freeze all campaign settings, opening banks and sealed quotas. Benchmark a small inference-only batch of 256/512 searches on the M4 and verify resource/deadline handling. Round-trip the new artifact/resume contract. Do not tune against the sealed package. This milestone must make the campaign executable and reviewable as one bounded job.

**Milestone 3 — one bounded v2 campaign and decision.** Run exactly the declared two seeds after separate authorization, preserve all generations, select champions on development data, then execute sealed acceptance once. Report both successful and failed criteria. If v2 fails, stop with a diagnosis of search-teacher quality, coverage, value fitting and arena behavior; do not automatically append another 200-game intervention. If it passes, prepare a separate public-integration assignment using the approved immutable artifact and preset.

### First campaign: meaningful but bounded

Two independent fresh runs, seeds **42 and 314159**, same full configuration, no ablations or checkpoint warm starts. Seed 42 is the predeclared primary; seed 314159 is replication, not a second chance to select a luckier outcome. Each run has **20 generations × 256 games = 5,120 training games**, at most **215,040 training plies**, and at most **6,720 optimizer steps** (20 × 336, derived from the maximum new positions). The actual update count follows `M_new`, not the ceiling.

Per run, cap collection plus optimization at **8 hours** and the combined two-run campaign, including predeclared evaluation, at **24 hours**, whichever applicable limit occurs first. Interrupted attempts consume those budgets; no automatic extensions, extra seeds or changed hyperparameters. The maximum training root-edge work is 55,050,240 simulations per run, plus root initialization. Partial games receive no outcome and no replay entry; a partial generation is not silently treated as a full one.

At 20–30 plies/game and the proportional 96-ms search estimate, self-play alone is about **2.7–4.1 hours per seed**. Training, logging, longer games and evaluation add time. These estimates make an hours-scale campaign plausible on the M4; they do not promise completion within the cap. The 200-game campaigns lasting roughly 85–133 seconds tested functioning learning plumbing, not the ceiling of this architecture.

Bound evaluation to **6,400 complete nontraining games combined**: per seed the four scheduled development arenas/ladders use 1,280 games; final seven-opponent ladder uses 1,400; NN Only versus Random uses 200; behavioral-calibration self-play uses 256. Total planned = **6,272** across both seeds, with unused ceiling not authorizing extra comparisons. Tactical/solved inference uses only the frozen packages and declared tie repeats, no adaptive searches for favorable fixtures. If any required evaluation is incomplete at the time cap, acceptance is incomplete.

Record per generation: new/replay positions, unique board/reflection families, game lengths and outcomes, draw games (not just draw rows), effective sample reuse by age, losses, finite/clipped gradients, policy entropy, illegal raw mass, visit coverage/depth, raw/root values, wrong saturation on development positions, tactical contradiction rates by ply/actor, whole-search timing and memory. Generation changes will deliberately combine several improvements; this campaign tests whether the coherent v2 system works. It does not isolate the causal contribution of each change.

### Acceptance criteria before proceeding to public integration

Freeze these proposed thresholds before training. They are project quality gates, not published AlphaZero standards. Apply the primary criteria to **both** seeds; do not average away a failed run or choose a seed based on the sealed test. Publish intervals and sample sizes alongside thresholds.

| Dimension | Required evidence |
| --- | --- |
| Correctness / integrity | All v2 contract and resume checks pass; zero illegal moves, corrupted histories, mislabeled completed outcomes or nonfinite artifacts; every required evaluation completes. |
| Raw-search immediate tactics | At 512, ≥99% immediate wins and ≥95% unique safe responses on the sealed tactical set; each actor separately ≥97% / ≥90%. Average predetermined tie repeats within each base; mirrors are correlated. |
| Deeper decision quality | ≥80% optimal-outcome-preserving actions on the sealed solved set; ≤5% avoidable losses from states whose exact value is win or draw. Report categories separately. |
| Exact value usefulness | Class-balanced scalar MSE <.60 (zero baseline =2/3 on equally balanced W/D/L), ≥85% correct sign in each decisive class, draw MAE ≤.35, wrong-sign saturated predictions ≤5% of decisive states. Raw-head metrics, not terminal engine answers. |
| Behavioral calibration | Held-out scalar MSE improves over zero with game-cluster uncertainty reported; disclose comparison to the frozen development constant. Occupied calibration bins with ≥30 independent games contributing have absolute mean-score gap ≤.15; sparse bins are unestablished, not passing evidence. No general win-probability claim. |
| Random | Raw-search score ≥.95 and 95% lower bound ≥.90. |
| Corrected Negamax depth 1 | Score ≥.80 and 95% lower bound >.70. |
| Corrected Negamax depth 2 | Score ≥.65 and 95% lower bound >.55; neither side's observed score below .50. This is the central shallow-search strength gate. |
| Evidence of learning | Equal-budget score ≥.60 and 95% lower bound >.50 versus initial untrained v2 and versus retained 4D.2f; NN Only score ≥.85 versus Random with lower bound >.75. |
| Stronger references | Complete/report depth-4 and guarded-UCT ladder results and solved-error rates. Winning these matchups is a stretch goal, not required to demonstrate superiority to shallow Negamax. |
| Local readiness | On the approved artifact at 512, measured warm whole-move p95 ≤500 ms on native M4 over ≥500 varied legal positions; bounded memory across repeated games and no caller-state mutation. This is a proposed target, not the linear estimate reported as fact. |

Simultaneous satisfaction of several metrics is a conservative release policy, not a claim that each 95% interval is a familywise statistical guarantee. The frozen primary depth-2 and equal-budget initial-model comparisons carry the main strength/learning claims; other results constrain quality. If draw/value gates fail while play succeeds, describe a useful but incompletely validated research agent and defer public acceptance under this protocol. Do not silently relax the thresholds after inspecting results.

Passing these gates authorizes a recommendation to begin integration, not deployment. Public integration still needs an immutable approved checkpoint manifest, optional neural dependency isolation, shared read-only model with per-request trees/RNG, explicit budget/deadline behavior, safe handling of terminal or malformed states, and local concurrency/session tests. The public label should state the search preset and that the agent learned from self-play. Any later tactical safety layer must be separately disclosed and evaluated; it cannot retroactively satisfy the raw learned-agent gates.

## Review coverage and evidence boundary

Read the requested [Phase 4B audit](phase4b/audit.md), [4D.1](phase4d1-neural-mcts-foundation.md), [4D.2](phase4d2-neural-training.md), [4D.2b](phase4d2b-neural-scaled.md), [4D.2c](phase4d2c-neural-symmetry.md), [4D.2d](phase4d2d-neural-root-noise.md), [4D.2e](phase4d2e-value-target-audit.md), and [4D.2f](phase4d2f-neural-value-anchoring.md). Historical numbers above are from those reports except the explicitly identified read-only recomputations of reuse and modal-win decisions.

Implementation coverage includes [`neural_mcts.py`](../games/connect4/neural_mcts.py), [`mcts_nn_agent.py`](../games/connect4/agents/mcts_nn_agent.py), [`train_mcts_nn.py`](../games/connect4/train_mcts_nn.py), [`neural_self_play.py`](../games/connect4/neural_self_play.py), [`tactical_value.py`](../games/connect4/tactical_value.py), all four neural evaluation/symmetry/exact-value/value-audit modules, engine/Board, standalone Negamax/UCT, and the relevant tests described above. Primary papers and solver references were consulted for conceptual claims, not used as authority for project-specific hyperparameters.

This review does not establish the causes of historical regressions by controlled ablation, claim exhaustive absence of bugs, rerun learned-model tournaments, or prove v2 will pass its gates. It identifies a sound reusable core, concrete remaining defects, and one complete next architecture with a bounded way to judge it. **Work stops at this document.**
