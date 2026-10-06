# Phase 4D.3B final launch-readiness review

**Verdict: CONDITIONAL GO — fix B1–B4, test the repaired boundaries, and re-freeze the declaration before authorization.**

> **Status:** the token `2741314399…67d8` reviewed here is **REJECTED / NOT AUTHORIZED**, as are the later Phase 4D.3B.1 (`8a8a52b100…`), 4D.3B.2 (`ed641aea…`) and 4D.3B.3 (`34b4d899…`) tokens. The current declaration is described in [phase4d3b4-complete-schema-fix.md](phase4d3b4-complete-schema-fix.md).

The current repository is **not ready to execute the campaign under declaration token `2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8`**. That token is not approved for launch. The learning architecture and inspected exact labels do not require redesign. The blockers concern executable identity, interruption recovery, deadline completion, and durable budget accounting.

Reviewed October 5, 2026: local branch `phase4d3b-alphazero-v2-preflight`, HEAD `a66345440f5e42063511bda7ba8e0ad2e8bc4773`, initially clean. Read all three requested design/handoff documents completely, every v2 Python module, reused engine/neural/UCT primitives, legacy and corrected Negamax, all three v2 test modules, frozen artifacts, and the mutation harnesses/results. The handoff's description of uncommitted implementation work is historical; this checkout contains that work committed.

Only this review is added to the repository. No source edits, frozen-package regeneration, real campaign, research training, learned-model strength evaluation, commit, push, merge, or deployment occurred. Existing correctness tests and additional discarded synthetic tests used temporary directories. No model was run on the actual frozen sealed positions. The retained 4D.2f checkpoint was hashed, not evaluated.

## Findings and minimum conditions for authorization

| ID | Classification | Finding |
| --- | --- | --- |
| B1 | **BLOCKER** | The declaration does not bind executable source. Fresh launches and final evaluation do not enforce the reviewed source/runtime identity across the campaign. |
| B2 | **BLOCKER** | Generation publication is not a recoverable transaction: interruption can strand an archive or silently skip required development diagnostics. |
| B3 | **BLOCKER** | Final calibration omits its per-move stop callback; completion can be published after the deadline or a stop request. Other last-operation deadline edges also lack a completion check. |
| B4 | **BLOCKER** | A hard crash can erase evaluation-game consumption, in addition to the disclosed time gap, allowing the supposedly cumulative ceiling to be exceeded. |
| S1 | **SHOULD FIX BEFORE CAMPAIGN** | Preserve partial evaluation evidence and accurate failure status; several paths discard evidence already computed. |
| S2 | **SHOULD FIX BEFORE CAMPAIGN** | Finish the one-time evaluation reporting/measurement path: retain row-level evidence, stage/search-value summaries, and a declared learned-artifact timing measurement. |
| S3 | **SHOULD FIX BEFORE CAMPAIGN** | Public exact-oracle action helpers accept already-won histories. Current package paths exclude them, but the helper contract is unsafe outside those paths. |
| S4 | **SHOULD FIX BEFORE CAMPAIGN** | Make preflight a real launch check, including available CPU identity and the retained checkpoint; distinguish it from its current informational output. |
| L1 | **NONBLOCKING LIMITATION** | Solved positions start at 12 pieces; independent exhaustive coverage is late-game only; development lacks vertical immediate wins and has thin late solved coverage. |
| L2 | **NONBLOCKING LIMITATION** | Opening outcomes are unknown, empty-board games share one cluster, and timings come from an untrained model. These constrain interpretation, not core validity. |

The minimum repairs are:

1. **B1:** Put the execution inventory/digest into the declaration and enforce it before a fresh run, resume, and final evaluation. Carry a consistent recorded runtime identity through evaluation as well as training. Exercise changed source in a fresh interpreter with the old token, and changed source/runtime between training and final evaluation. A runtime with unavailable identifying fields must not receive an unqualified exact-continuation claim.
2. **B2:** Give each generation a durable commit/recovery state covering its archive, inference artifact, resume artifact, manifest entries, summary, and development diagnostics. Recovery must recognize verified existing outputs, finish pending work, and refuse inconsistent outputs. Test interruption between each publication step. Do not require deleting evidence by hand to resume.
3. **B3:** Forward deadline/stop checks through calibration and other unchecked evaluation work; recheck at phase completion before publishing completed evidence or selection. Distinguish permission to start another game from checking elapsed time after the last allowed game, so exactly reaching the game ceiling is handled correctly. A completed primitive may overrun a cooperative deadline, but the campaign must then be reported incomplete.
4. **B4:** Durably reserve/account for evaluation games and define conservative recovery of unclosed attempts. Persist completed/abandoned work without waiting for a periodic heartbeat. Either conservatively charge uncertain crash intervals or stop as incomplete instead of silently recovering budget. Cover torn ledger/publication records and enforce one writer per campaign, or explicitly reject concurrent invocation.

After these changes, re-run the focused and adversarial checks and issue a **new declaration hash**. No evidence found here justifies regenerating or rebalancing the frozen position packages. Preserve their current bytes unless a subsequently demonstrated label defect requires a separately reviewed correction. S1–S4 should be addressed in the same pre-launch repair, without changing training settings, acceptance thresholds, seeds, or fixtures.

## 1. AlphaZero-v2 core — NO ISSUE

The scalar is consistently for the player to move. Encoding uses own pieces +1, opponent −1; final targets use the captured pre-move actor. The engine switches mover after a winning move, so a winning child is −1 for its new mover, the parent receives +1, and selection correctly maximizes `-child.Q + U`. Backup alternates signs without discount. Draws contribute zero. Root expansion is outside the simulation budget; every one of B simulations traverses a root edge, and both root visits and summed edge visits equal B.

Legal inference softmaxes only finite legal logits after subtracting their legal maximum. The pathological illegal logit 1000 cannot flatten valid legal preferences. Training instead uses finite, unmasked seven-action logits for soft-target cross-entropy, plus scalar outcome MSE. This avoids `0 * -inf` and penalizes illegal raw mass.

`V2Example.policy_target` derives only from raw visits. Execution temperature is 1 at plies 0–7 and 0 thereafter; late ties are sampled uniformly. It cannot turn the teacher into a one-hot target. Reflection reverses board columns, visits, and action while preserving actor/outcome. The immutable original examples remain intact.

Each generation collects exactly 256 complete games from a deep-copied frozen learner; learner and snapshot hashes and optimizer-step counts are checked. Whole generations expire after eight, and minibatches sample distinct positions uniformly within a batch. Persistent AdamW, weight-only decay, finite checks, and clipping match the declaration. The update count is `ceil(4 * new_positions / 128)`, including the last generation. No champion decision rolls back the learner or optimizer. No anchoring or tactical guard enters the training objective.

Evidence: direct inspection of `network.py`, `search.py`, `data.py`, `training.py`, `selfplay.py`, `generation.py`, and reused `Node`/backup/terminal/visit-policy primitives; passing core tests and the additional mid-update recovery check below.

## 2. Runtime and source reproducibility — NO ISSUE at a valid runner boundary; BLOCKER B1 at campaign scope

The Milestone-1 loader defects really were repaired. `runtime_differences` compares the complete key union, including missing/added keys. Strict load enforces Python implementation/version, torch version/git/build-config digest, NumPy version, platform, machine, CPU model/capability, intra/inter-op threads, deterministic and warn-only modes, matmul precision, mkldnn, and five native-library thread environment variables. All recorded fields are tested individually. A runner checks runtime/source at generation entry and boundary creation; source is also checked on load. A persistent drift is refused before a boundary can certify it.

The content inventory covers the current execution closure: **25 files**, consisting of all 19 v2 `.py` files and these six external dependencies:

```text
games/connect4/__init__.py
games/connect4/agents/mcts_agent.py
games/connect4/agents/mcts_nn_agent.py
games/connect4/board.py
games/connect4/connect4.py
games/connect4/neural_mcts.py
```

The v2 inventory is `__init__`, `arena`, `artifacts`, `campaign`, `config`, `data`, `diagnostics`, `evaluation`, `generation`, `network`, `oracle`, `packages`, `profiling`, `provenance`, `reference_negamax`, `search`, `selfplay`, `statistics`, and `training`. Inspection of lazy opponent imports and a loaded-module inventory including campaign/evaluation found no missing repository module. The existing fresh-process training/resume closure test also passed. This establishes coverage of current paths, not every possible future import.

Current execution digest:

```text
fae266e4ee08d31ac4d5028aa42b94b60c126ffc1fe58588b2f281aaef1245c4
```

**B1 reproduction.** A temporary in-memory override of `Path.read_bytes` appended a comment to the bytes read for `search.py`; no repository file changed. The execution digest became `3dba3c1428b21a8d5e7d5933338f37cc22d7445d370a94eb0130bb9fef39ab66`. `campaign.preflight` still accepted the same declaration and returned the original authorization token. This follows directly from [declaration validation](../games/connect4/alphazero_v2/campaign.py#L128): it validates packages/configuration but never compares an expected execution identity. A new `GenerationRunner` adopts the current process identity. Existing strict resumes reject source changes, but that is not a source freeze for the first run or for a second seed starting fresh.

Final evaluation has a separate gap: it loads minimal inference artifacts, whose schemas intentionally contain no runtime/source provenance, and never loads or compares a resume identity. It configures threads/determinism but does not compare to the training runtime/source. `Budget.check` is not an identity check. Thus source changes between training and final evaluation, and runtime drift within development/final evaluation, are not refused by the campaign as claimed.

**Exactness actually supported:** deterministic logical continuation from a valid completed boundary, with identical recorded runtime and covered source, valid hashes/replay/state, and restored owned RNG/optimizer state. Git commit/dirty status is provenance only, correctly. Non-strict loads record differences and withdraw the exactness claim. There is no mid-tree/update continuation, cross-runtime numerical guarantee, deterministic wall-clock stopping point, or guarantee of identical artifact bytes/timings. Third-party versions/build metadata are not hashes of all installed binaries or a hermetic runtime. Hardware model is not a unique physical-machine identifier.

**S4:** In this sandbox the CPU probe returned `null`; a separate permitted `sysctl` read confirmed **Apple M5**. Two unidentified CPUs could both compare equal as `null`. Also, CLI `preflight` merely reports runtime: with thread environment pinned it exited successfully while reporting inter-op=10 and deterministic=false. `run` subsequently pins 1/1/true, so this is not evidence that actual training would use those wrong settings; it means a successful preflight is not a successful runtime gate. Preflight also does not verify availability/hash of the retained 4D.2f artifact; construction of that opponent checks it much later.

## 3. RNG ownership and resume — NO ISSUE in trajectory restoration; BLOCKERS B2/B4 in orchestration

| Stream | Ownership and continuation |
| --- | --- |
| Initialization | SHA-256 domain-separated seed, torch `fork_rng`, explicit manual seed; fresh model only. |
| Search ties | Runner-owned `random.Random`, domain `connect4-alphazero-v2-search-ties`. |
| Action sampling | Separate runner-owned `random.Random`, action-selection domain. |
| Dirichlet noise | Separate PCG64 generator seeded from its own domain; complete generator state and draw counter serialized. |
| Replay sampling | Separate sampling-domain `random.Random`; restored, not borrowed from search. |
| Augmentation | Separate reflection-domain `random.Random`, restored with exposure counts. |
| Arena | Derived from namespace/opening ID/game index/role/seed; independent streams for the agents. |
| Position evaluation | Derived from evaluation namespace/row ID/tie seed; intentionally shared between search and action ties within that evaluation search. |
| Calibration | Fresh calibration-domain search/action/noise streams, indexed by seed and held-out game; no runner stream used. |
| Bootstrap / package generation | Private fixed analysis RNG; separate model-blind builder seeds. Campaign openings are enumerated from frozen banks, not adaptively selected. |

Exactly one `dirichlet` call occurs at the root of each declared noisy self-play search, with legal support only. Nonroot priors are clean. The exact reference-PCG64-state test passed, which is stronger evidence than the draw counter alone. Epsilon zero consumes no noise draw, but the frozen campaign uses .25. Evaluation agents cannot enable noise. Held-out behavioral calibration intentionally uses the training noise/temperature schedule.

Boundary payloads restore all owned streams, model, optimizer moments/steps, ordered replay, counters, augmentation counts, and history. Global Python/NumPy/torch states are also stored/restored, although they do not determine this runner's continuation and are excluded from its logical state digest. The legacy v1 checkpoint loader initializes a module and can consume global torch RNG; it is used only in final evaluation, and no training stream depends on that global state. Do not generalize the v2 evaluation RNG test to every legacy loader.

Additional discarded checks:

- Interrupted a genuine tiny generation **after one optimizer update**, invalidated that runner, restored its prior boundary, and reran. Logical state, including optimizer/RNG/replay/history, matched uninterrupted execution exactly (`c54805413c994bbf5e12456704f6a49ead90020c2ec2dfc3b1eeab0b85228686`; 44 total tiny-run updates).
- Interrupted a tiny development champion check **after one completed arena game**. Resume reran the incomplete check, and final weights, both decisions, and selection matched the uninterrupted run. The extra work remains an attempt cost, not additional accepted evidence.
- Existing same-process/fresh-process boundary tests and cooperative SIGINT tests passed.

The existing campaign scenario described as “mid-training” actually triggers its second stop at the outer training check before generation 2 starts. Its first evaluation stop is before any game. It is useful orchestration evidence, but does not alone establish the stronger interruption claims; the extra checks above cover actual progress.

**B2 reproduction:** [run_seed publication order](../games/connect4/alphazero_v2/campaign.py#L447) archives games before saving/registering the new boundary. With a fake generation runner (no model/update), injecting failure immediately after archival left the previous boundary current. Resume reran the generation and failed with `FileExistsError` on `games/generation-0001.jsonl`. Separately, failure in `_raw_value_metrics` after a registered boundary resumed to `completed` with **zero diagnostic calls and zero generation summary files**. Pending recovery tracks champion checks, not missing per-generation diagnostics. Inference and resume manifest entries are also appended separately, so a crash between them can leave duplicate inference registrations after retry; `_inference_record` requires exactly one.

These are defects in campaign transactions, not in the mathematical restoration of a committed learner. Individual atomic `.pt` files do not make the multi-file generation atomic. Incomplete games/generations are correctly excluded from training replay, but a campaign cannot yet promise no skipped evidence or reliably recover every interruption boundary.

## 4. Corrected Negamax — NO ISSUE

Independently reran the documented legacy reproduction through the existing test: history `[5,4,3,6,2,4]`, depth 3, window `[0,1]` returns `(3,5)`; reusing that cache for a full window returns `(4,9)`, versus fresh `(3,10)`. Legacy entries omit bound type. Its terminal evaluation also uses ordinary window weights, including a four worth 81, without a dominant outcome contract.

The new `connect4-reference-negamax-v1` uses board/mover/depth keys and EXACT/LOWER/UPPER entries. Bound tightening, cutoff storage, full-window per-action scoring, and state undo are sound in inspected paths. Terminal magnitude `1,000,000 + remaining_depth` dominates nonterminal magnitude bounded by `69 * 9 = 621`; draws are zero. Depth preference therefore selects faster forced wins and delays forced losses without letting heuristic advantages outweigh outcomes.

Beyond the existing tests, **160 independently sampled legal nonterminal histories**, cycling depths 1–4, passed **960 window-sequence checks and 1,078 per-action comparisons** with plain unpruned minimax. This tests alpha-beta/cache correctness, not unlimited-depth strength. Fast-win, O-block, X/O terminal, draw, symmetry/tie, and caller-state tests passed.

The legacy source has no diff against the Milestone-1 base `a58abccb…`; historical files/results were not edited or reinterpreted as results against the corrected opponent. No depth-limited Negamax score is used as an exact solved label.

## 5. Exact solver — NO ISSUE for frozen nonterminal rows; SHOULD FIX S3 for terminal API misuse

**Why method A is exact:** it searches finite acyclic Connect 4 continuations with W/D/L terminal rules and no heuristic leaf cutoff. Node-budget exhaustion raises, never supplies an approximate label. For states without an immediate own win, two playable opponent threats prove loss; one forces a block; filling the square below an opponent winning cell immediately loses. Those are exhaustive tactical deductions, not depth-limited evaluations. Threat counts only order remaining moves. Lower/upper transposition bounds preserve alpha-beta semantics.

The bitboard key `position + mask` is injective for legal gravity columns: a height-h mask plus own stones occupies the disjoint integer interval `[2^h - 1, 2^(h+1) - 2]`; the seven-bit column stride prevents carry into the next column. Recursion swaps mover ownership and negates values. Winning actions are answered before entering their terminal children. For full-window W/D/L searches, a cutoff at +1 or −1 is already an exact extreme outcome.

**What method B independently checks:** column-stack representation, a separate four-in-a-row implementation, exhaustive memoized minimax over every legal continuation, no alpha-beta, threat pruning, or A-table logic. Memoized results are exact. Its action wrapper shares grid/history helpers with tactical scanning, but shares neither A's bitboards nor its pruning. Agreement is meaningful for representation, threat pruning, backups, and cache bounds. Both still share the game's specification and input-history convention; agreement is not an external proof of all positions.

All **2,100 frozen rows** were legally replayed and structurally checked. Independent solver spot checks in this review comprised:

- Method A recomputation of **31 bases**, one selected from each available split/value/actor/stage cell (highest occupancy in each cell), including every legal action, state maximum, and optimal-action set. Thirty fit a 200,000-node audit budget; `sealed-D-o-018-base` at 17 plies matched after a separate 3,000,000-node allowance, using 763,963 nodes / about 3.8 s.
- **27 late package bases** checked by an additional engine-backed minimax written for this audit, using engine terminal outcomes and every legal child, memoization only. All action values agreed.
- **15 of those bases** additionally recomputed with method B. Existing A/B random late-position and reused-window tests also passed.

The package metadata records B agreement on **128 sealed + 30 development bases**, every base at ≥24 pieces. I verified those counts; I did not re-solve all 158 with B or all 900 rows with A during this audit. The handoff's all-900 A recomputation is retained prior evidence, not a new result claimed here. No external Pons binary was used.

Values/action labels are actor-relative for both X and O, not fixed-X. All actions tied for the maximum W/D/L are optimal. A/B intentionally have **no shortest-win preference**; a slower win is not mislabeled a mistake. The reference opponent's separate depth bonus is not ground truth. Draw-preserving moves, multiple optimal actions, mirrors, immediate wins, and terminal draw behavior are covered by inspection/tests.

**S3 reproduction:** `BitboardSolver.value` rejects an already-won history, but `action_values([0,1,0,1,0,1,0])` returns a seven-action map, including `1: +1`, after X has already won. `exhaustive_action_values` also lacks a terminal-root guard. Low-level history conversion does not itself replay the engine's post-win prohibition. Fix or explicitly enforce the legal-nonterminal precondition at public solver boundaries. Current builder/verification/evaluation rows are nonterminal, so this does **not** invalidate the frozen labels.

## 6. Frozen evaluation packages — NO ISSUE; coverage limitations L1

All six file hashes and row hashes match, all histories replay legally, both tactical labelers agree, and exclusions reconstructed from the actual retained sources match the frozen **296 family keys and source hashes**. The copied original probes and historical openings match their v1 definitions. The builder imports no model/inference/training machinery; reference Negamax chooses candidate histories, while exact solvers supply solved labels. Source inspection, matching builder hashes for `packages.py`/`oracle.py`/`reference_negamax.py`, and torch-blocked tests support model-blind construction. Code/evidence cannot prove that no person ever ran an unrecorded external command.

| Package | Rows / bases | Independently checked composition |
| --- | --- | --- |
| Tactical sealed | 800 / 400 | 100 per category × actor; early/middle/late 110/152/138. Win H/V/up/down 63/40/51/46; safe 57/55/47/41. |
| Tactical development | 200 / 100 | 25 per category × actor; stages 21/40/39. Win H/V/up/down 25/0/15/10; safe 18/14/10/8. |
| Solved sealed | 600 / 300 | 50 per W/D/L × actor. Stage counts W 38/36/26, D 35/34/31, L 38/38/24. |
| Solved development | 300 / 150 | 25 per W/D/L × actor. Stage counts W 35/15/0, D 24/19/7, L 28/22/0. |
| Each opening bank | 100 entries | 20 empty entries + 40 distinct prefix families and mirrors; base movers 20 X / 20 O; lengths 2–8. |

Every nonempty family appears exactly twice, with matching base/mirror histories; no unexpected repeated family exists within or across these packages. Nonempty development/sealed families are disjoint by board+actor/reflection, including transpositions, and exclude historical families. **The required empty-board openings are an intentional exception:** both banks contain the same empty board 20 times, also present in historical probes. The verifier explicitly exempts empty histories. Therefore the handoff's unqualified “every family is disjoint” should be read with this predeclared exception, not as literal complete opening-bank disjointness.

There are **23 sealed forced-block-loses bases / 46 rows**, rather than 23 rows as stated once in the handoff. That wording error does not change package content.

The 12-piece solved minimum, absent development vertical immediate wins, and only seven late development solved bases (all O-to-move draws) are **NONBLOCKING LIMITATIONS**, not grounds to regenerate inspected packages. The sealed quotas, actor balance, directions, and broad stages meet the acceptance design. Development blind spots reduce the sensitivity of promotion regressions; late-only B checks limit independent solver coverage. Neither is a hidden label change or a reason to reopen settled architecture.

## 7. Sealed-test isolation — NO ISSUE in current model call paths

Mechanically traced: `run_seed` obtains development solved rows; `_champion_check` obtains development openings/tactics; learner replay comes from completed self-play. The only actual sealed inference path is `_final_seed`, reached by `final_evaluate` after both `selection.json` files exist and after publishing `final/started.json`. Loading declarations deserializes/checks sealed package content, but runs no model and supplies no sealed score to a selection decision. Diagnostics use rules, not sealed labels.

Final results do not update `champion.json`, `selection.json`, learner weights, or hyperparameters. A second final invocation is refused by the no-overwrite marker. Readable labels are not code-path leakage. This assessment applies to current executable content; B1 must bind it to the authorized launch. One-time execution is correctly conservative after interruption, but makes preserving partial/raw evidence especially important (S1/S2).

## 8. Arena and statistics — NO ISSUE in current frozen estimators/gates

Each opening entry produces an identical-position pair: evaluated agent owns X, then O. The board's mover remains determined by the prefix. Each agent gets a detached board and a role-specific RNG. Winner IDs become W/D/L relative to the evaluated agent; score is `(W + .5D)/completed_games`. Abandoned games have no result and are excluded from scored records; incomplete evidence cannot pass a gate.

The 10,000-resample percentile bootstrap uses seed 4,303,009 and resamples opening-family `(points,games)` totals. Each sampled total contributes its full denominator, so the reported score stays game-weighted despite unequal cluster sizes. Mirrors share one family and all 40 empty-board games share one family: **41 clusters per 200-game arena**, not 200 independent observations. Paired differences resample common families. The helper checks family membership, not full per-opening pair equality; current generated complete banks satisfy the stronger pairing contract. Generic callers should not treat its family-set check as proof of matched records.

Position metrics average tie seeds within row, orientations within family, then families. Avoidable losses use only exact-win/draw positions as denominator. All actions in a forced-loss position preserve its outcome, as intended. Wrong-sign saturation is row-weighted; every frozen family has two rows, and sensitivity removes whole families, so it equals family weighting here. Calibration uses game-cluster resampling, position-weighted MSE, `(v+1)/2` expected score, five fixed bins, and independent-game counts for bin coverage. Sparse bins are unestablished.

Independent hand checks:

- Families `[W,L]` and `[D,W]`: W/D/L=2/1/1; score `2.5/4=.625`; X=.75, O=.50. Resampling two whole families gives .50/.625/.75; the implementation's 95% interval is **[.50,.75]**.
- All wins/all draws give degenerate scores/intervals 1/.5; two perfectly correlated all-win/all-loss families give interval [0,1], as tests require.
- Tactical family accuracies .75 and 1 average to .875, regardless of repeats; safe-family accuracies 1 and .5 give .75. A .020 regression passes and .021 fails. Lower bound exactly .50 fails promotion.
- Forty games with `v=.5,z=1`: MSE=.25 versus zero=1; improvement=.75 with [.75,.75] interval; predicted score=.75 versus realized=1, gap=.25, so the populated-bin gate fails. The implementation matched these values.

The champion gate correctly requires complete evidence, score ≥.55, lower bound >.50, correctness, and ≤.02 regression in each tactical category. `campaign` currently calls the statistics default gate rather than passing the declaration's gate object; their frozen numbers match exactly. Future declaration changes must not be presumed executable without validation/wiring. No denominator or pseudoreplication defect was found in the present complete-bank path.

## 9. Champion lifecycle — NO ISSUE, subject to transaction repairs

The learner always advances after a numerically valid generation. Only generations 5/10/15/20 can promote, using development arenas/tactics. Champion 0 is the seed's own untrained initialization. Rejection leaves learner/AdamW untouched. Pending checks run before collecting another generation. Final sealed evidence cannot retroactively choose another generation.

Seeds 42 and 314159 have separate run directories, initialization and training streams, optimizer/replay, champion histories, and final results. Seed 42 is explicitly primary; there is no better-seed selection code. Both must be reported under the same acceptance protocol. Repeating an interrupted development check is an explicit replay of incomplete evidence, with costs charged; it is not permission to rerun a completed seed or to add seeds/settings. B2/B4 qualify the durability of these claims after hard failures.

## 10. Campaign limits — arithmetic correct; BLOCKERS B3/B4

The configured logical run is 20 × 256 = **5,120 accepted complete games/seed**, at most 215,040 positions, 336 updates/generation and 6,720 updates/seed. Replay caps at 2,048 games / 86,016 positions. Actual updates follow new positions. Partial generations cannot be treated as complete replay, and restarting one repeats work from the last boundary. These are accepted-trajectory limits, not a promise that interruptions can never cause extra discarded physical games/updates.

Evaluation arithmetic is correct:

```text
per seed: 4 * (200 + 3 * 40) + 7 * 200 + 200 + 256 = 3,136
two seeds: 6,272; ceiling: 6,400; retry headroom: 128 games total
```

Eight hours is 28,800 seconds of collection/optimization per seed; 24 hours is 86,400 cumulative active invocation seconds including evaluation. Time between invocations is not counted. Normal stop/failure paths append attempt-end counters. Existing fake-clock tests correctly refuse work at exact training/campaign/evaluation limits and carry consumption to another attempt; another seed has its own training allowance. There is no built-in settings extension or extra-seed loop. Partial/abandoned arena attempts consume evaluation slots conservatively, although the review's ceiling was phrased in complete games.

**B3 direct reproduction:** exercised the actual `final_evaluate`/`_final_seed` orchestration with synthetic model/metric adapters and a fake clock, with one calibration game and a one-second cap. The last calibration operation advanced time to two seconds and requested a stop. Output was **`completed`**, and `final/seed-42.json` was written. The callback supplied to calibration was **absent**. At [the calibration call](../games/connect4/alphazero_v2/campaign.py#L675), `calibration_games(..., check=check)` is missing; after the last game neither `_final_seed` nor `final_evaluate` checks elapsed time or the stop flag. This is a false completed result, not merely harmless bounded overshoot.

Related unchecked work includes NN-only/raw-value position batches and post-search statistics. Training checks occur before updates, not after the last update; exceeding the per-run training cap on that last update can proceed to evaluation, whose check does not enforce the training cap. The final champion decision/selection similarly lacks a final deadline check. Cooperative checking is appropriate, but completion must validate that the required work finished within the declared budget.

**B4 direct reproduction:** forced a heartbeat at time zero, counted five evaluation games without advancing to the 30-second heartbeat threshold, then recovered consumption solely from the ledger. In-memory count was **5**, recoverable count **0**. [`count_evaluation_game`](../games/connect4/alphazero_v2/campaign.py#L273) increments memory then calls the throttled heartbeat. After a hard crash, a new attempt regains those slots. Repetition can exceed the 6,400 ceiling. `training_games` also counts only full successful generations, omitting completed games discarded with interrupted generations; it is not an attempted-game audit.

The handoff discloses lost time since the last heartbeat, but not lost evaluation-game counts. Even the time gap is not unconditionally ≤30 seconds: checks do not run inside every long operation. JSONL readers also reject a torn trailing record rather than implementing recovery. These facts prevent an exact cumulative-cap guarantee after arbitrary interruption. A campaign lock is absent, so simultaneous invocations can independently read the same prior totals; the prescribed sequential shell commands avoid this, but the launcher should enforce that prerequisite.

## 11. Frozen declaration — hash verified; source commitment absent (B1)

Both independent `shasum -a 256` and Python/file-loader recomputation returned:

```text
2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8
```

It commits to the exact JSON bytes containing name/kind/format, seeds/primary seed, generation count, every serialized `V2Config` setting, budgets/planned games, thread/determinism/strict-resume declarations, evaluation simulations/tie seeds/noise/guard/temperature, champion schedule/gate/baseline row IDs/opponents, final ladder/calibration/one-time flag, thresholds, retention, notes, and these package identities:

| Package | SHA-256 |
| --- | --- |
| openings-development | `3990bc230084bd5c9293c15a44eae05f35749d2a3a18e66566461dae98f82f72` |
| openings-sealed | `d30683aa300fb7fe095da3f867c8bb9b301d387532aebe751660b90fb8aa34eb` |
| solved-development | `41ffd4dfcdd85b939c8afaea692566339e612224f16d2a62797c2960192f37b9` |
| solved-sealed | `2c5f70dcbb9bd0e3242a760e240eec3ca52cd075a066eedaf5fa39472376394b` |
| tactical-development | `106cc315c7e8d9b44a260df1a3407578ce1197411a6e614be3a7c38a5b9e6d9d` |
| tactical-sealed | `cb4e2e1e4ef1b877d8ecce840e04b04e8522df08747ba765c0d5608134af6f47` |

It also commits the retained checkpoint path and SHA-256 `78276912bdef5f1b1b53ac6b2ec3ede2eed20a8ebb5796a9bdcbb57e4cb28c51`, independently matched against the local file. Package bytes indirectly bind their metadata, rows, source-exclusion records, and label-version strings.

It commits to **no executable source digest or inventory**, no concrete M5/Python/torch/NumPy/platform identity, and no hashes of `manifest.json`, `exclusions.json`, or `build-provenance.json` as separate artifacts. The latter records are currently verified independently, but are not transitively protected merely because they sit next to the declaration. Constants such as PUCT 1.41, bootstrap seed/resample count, RNG domain strings, architecture/encoding implementation, and exact solver code live in source rather than explicit declaration fields; without a source commitment their executable meaning is unfrozen.

Thus the handoff's statement that editing any v2 `.py` changes the execution digest is true. The stronger statement that this **invalidates this declaration/token is false**. Recording today's digest in this review does not repair the launcher.

## 12. Performance and resource preflight — NONBLOCKING LIMITATION L2; SHOULD FIX S2/S4

Read the retained M5 search/artifact JSON reports directly. They substantiate 256 mean/p95 **87.8/133.9 ms**, 512 **172.2/240.9 ms**, approximately **5.8/6.9 MB** resume files, **6.6/10.3 s** save including reload validation, **5.7/8.9 s** load, and **367/429 MiB** peak RSS during artifact profiling. The six untrained self-play games report .0836 seconds/ply and flat 281 MiB peak RSS. CPU identification independently confirmed M5; no M4 measurement is claimed.

At .084 seconds/ply, 5,120 games × 20–30 plies is **2.39–3.58 hours/seed** for collection; 42-ply games would be about 5.02 hours. At the handoff's .042 seconds/update, the absolute 6,720-update ceiling adds about 4.7 minutes. The 8-hour cap is credible for the measured regime. The two-seed **13–17-hour** estimate is plausible, with uncertain game lengths, learned search shapes, evaluation/UCT cost, diagnostics, and repeated interrupted work. It is not a completion guarantee under 24 hours. Fixing cap enforcement remains necessary regardless of apparent headroom.

Two evidence qualifications: the retained search profile has execution digest `7a6645…`, different from the current `fae266…` (the artifact profile matches current); and its `position_plies.p0=41` is inconsistent with median 10. The current nearest-rank quantile helper fixes that minimum-index defect and has a passing regression test, but the old JSON does not itself establish the claimed minimum zero. The handoff also undercounts sealed tactical searches in its estimate: **800 × 4 = 3,200**, not 2,400; solved searches are 600 × 4 = 2,400. The additional 800 searches/seed are roughly 2.3 minutes/seed at the measured 512 mean, not enough alone to invalidate the hours-scale estimate.

**S2:** Current `profiling.profile_search` always constructs the untrained seed-42 model. The final evaluator does retain per-move arena timings, but does not produce the declared warmed ≥500-varied-position learned-artifact p95 readiness result. Before launch, wire or explicitly predeclare that measurement for each selected artifact, including model hash, sample histories, warm-up convention, timing/memory summary, and its charge to the combined budget. Do not claim that 241 ms establishes learned readiness.

Final results also discard per-row choices/visits/raw values, searched root values, and calibration rows after aggregation; no solved stage summary is emitted, NN-only tactical results lack the overlap-excluded sensitivity, and development baseline histories are discarded. Some predeclared reports therefore cannot be reconstructed from saved output without rerunning sealed inference. Preserve enough evidence during the single evaluation to report the complete protocol. Existing aggregate gate helpers need an explicit complete/incomplete reporting layer; completion of the CLI is not acceptance of a trained agent.

**S1:** A stop during a final ladder or position pass can throw away all partial results for that seed because the JSON is written only after `_final_seed` returns. Tactical-stage stops in champion checks do not use `_incomplete_check`; baseline stops save only that baseline's records. Unexpected exceptions in `final_evaluate` append `attempt_end` with status still `running`. Persist phase-level incomplete/failed evidence and completed records while retaining the one-time sealed marker and prohibition on selecting a different champion.

## 13. Mutation/testing evidence and review limits

**NO ISSUE in the reported interpretation when limited to the listed mutants.** Read the 30-item first sweep, four survivors, nine provenance mutations, the 24-item second sweep, first-pass/chunked results, and follow-up notes. The source-rewriting harnesses were **not run**, honoring the no-source-modification constraint. The claimed two equivalent mutants are credible: a terminal full-board draw's open-window heuristic is zero, and removing the double-threat early return leaves a losing forced response that search can still prove. A killed mutant only means some selected test failed; the harness accepts any nonzero test exit and is not a correctness proof or a complete coverage metric.

High-impact behavior not pinned by those sweeps includes declaration-to-source binding, evaluation runtime/source continuity, missing final calibration callbacks, final-operation deadline overruns, hard-crash game accounting, multi-file commit windows, mandatory diagnostic recovery, partial-final evidence persistence, and oracle terminal-action misuse. The new probes exposed failures while the existing focused suite remained green. Adding more mutations of already-covered signs would not address these gaps.

New verification in this review:

| Check | Result |
| --- | --- |
| `test_alphazero_v2.py`, `test_alphazero_v2_evaluation_core.py`, `test_alphazero_v2_campaign.py`, engine tests | **216 passed, 1 deselected**, 127.14 s. Excluded only `phase4d2f_adapter_is_hash_pinned` to avoid running the retained learned model. |
| API import isolation, `.venv` | **2 passed**, .46 s. |
| Frozen structural verification, independent composition/reflection/exclusion reconstruction | All 2,100 rows passed; 296 exclusion families matched. |
| Independent Negamax expansion | 160 positions; 960 windows; 1,078 action scores matched minimax. |
| Exact package spot checks | 31 A bases, 27 additional engine-minimax bases, 15 B bases; no disagreement. |
| Actual mid-update / progressed mid-check recovery | Equal logical state / equal weights, decisions, and selection. |
| Virtual source edit / publication fault injection / fake deadlines / crash accounting | Reproduced B1–B4 as described. No production source mutation or actual sealed-model inference. |
| Declaration preflight and independent hashes | Expected token and 6,272 planned games; informational-runtime caveat above. |

Focused test command used `/tmp/board-game-phase4c1-venv/bin/python`, with `PYTHONDONTWRITEBYTECODE=1`, all five thread environment variables set to 1, `pytest -q -p no:cacheprovider`, a dedicated `/tmp/phase4d3b-launch-pytest` base directory, and `-k 'not phase4d2f_adapter_is_hash_pinned'`. Extra audit harnesses/results were placed under `/tmp/phase4d3b-launch-audit/`; synthetic campaign/model artifacts were temporary/discarded. This document contains the relevant conditions and results without making those scratch files part of the frozen protocol.

The prior 1,094-test / 451-test sweeps and all-900-row re-solve remain historical evidence from the handoff; they were not repeated or presented as new results. Tests do not establish learned strength, all-position solver correctness, or completion within measured time estimates. The required follow-up is the bounded implementation repair and re-freeze above, followed by separate launch authorization. This review stops here.
