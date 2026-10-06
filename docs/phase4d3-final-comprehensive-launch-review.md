**Board Game AI Lab — Phase 4D.3 Final Comprehensive Launch Review**

**Verdict: GO**

Reviewed October 6, 2026. Branch `phase4d3b-alphazero-v2-preflight`, commit `5182fe97c8c98ffb48b9a96aad9c564b45f25565`, and the initially clean working tree matched the request. The candidate declaration is `2e39713f6e2791c735406c3bec7def0d336adb6add1cd43d49319e83aac486ab`.

This review found no fundamental algorithmic defect and no reproduced, plausible path by which the unmodified frozen campaign can omit required scientific work or produce scientifically invalid official COMPLETE evidence. The decision uses the supplied threat model: coherent deliberate forgery is outside scope; a failure that refuses evidence or ends INCOMPLETE is not a false completion.

No execution code was changed, no frozen package was regenerated, and no research campaign was launched. Existing automated tests used temporary synthetic packages and tiny test campaigns. Additional review probes lived in temporary files. This document is the only repository addition. Approval establishes launch readiness, not learned strength or satisfaction of the eventual scientific acceptance thresholds.

**1. AlphaZero mathematical correctness**

Traced [network.py](../games/connect4/alphazero_v2/network.py), [search.py](../games/connect4/alphazero_v2/search.py), [data.py](../games/connect4/alphazero_v2/data.py), [training.py](../games/connect4/alphazero_v2/training.py), and the reused encoder, node, backup, and engine operations.

The canonical observation is a CPU float32 `1×1×6×7` board: the player to move is +1, the opponent −1, and empty cells 0. For X this retains the absolute encoding; for O it reverses its sign. Policy columns remain physical columns. Inference returns seven finite raw logits and one finite scalar tanh value for that mover. Illegal logits are excluded before a stable softmax over legal actions, leaving illegal probabilities exactly zero. Terminal roots are rejected; terminal leaves bypass inference.

Each node stores Q for its own player to move. A child represents the opponent of its parent, so the implemented selection score is correct:

`−Q(child) + 1.41 × P(child) × sqrt(max(1, N(parent))) / (1 + N(child))`.

A nonterminal leaf contributes its mover-relative network value; backup alternates signs on every edge. The engine switches the mover even after a winning move. Thus an X winning child has O to move and value −1, which becomes +1 at the X parent. An O winning child similarly has X to move and backs up +1 to O. Draws contribute 0 throughout. These cases and deeper alternating sign traces are covered by the rerun tests.

Root expansion is uncounted; each simulation traverses a root edge and backs up exactly once. Both the root count and summed child visits must equal the requested simulations. Training policy targets always use `N(a)/sum(N)`. Action temperature only controls execution. Examples are captured before the move; full legal replay must reach termination before labels are assigned as +1 for the winning pre-move actor, −1 for the loser, or 0 for a draw.

Training minimizes minibatch-mean policy cross-entropy on all seven finite logits plus scalar value MSE, with equal weights. Zero illegal target mass and finite training logits avoid `0 × −infinity`; penalizing illegal raw policy mass is coherent with conditioning on legality at inference. Scalar tanh output and targets in {−1, 0, +1} have matching perspective and range.

**2. Generation and training correctness**

[GenerationRunner.run_generation](../games/connect4/alphazero_v2/generation.py) creates an independent eval-mode snapshot with gradients disabled, then serially collects exactly 256 completed games using that snapshot for both sides. It verifies snapshot/learner weight hashes and unchanged optimizer step count after collection. Training begins only after collection returns and finalized games enter replay. There is no concurrent learner update path.

Replay retains eight complete generations, at most 2,048 games, expiring the oldest generation as a whole. Each minibatch samples 128 distinct positions uniformly from all retained positions; later batches may reuse positions. Reflection is sampled independently at probability 0.5, reversing observation columns, visits, and action while preserving actor, ply, and outcome. This preserves both policy and value semantics.

The integer update calculation is exactly `ceil(4 × M_new / 128)`, based on newly collected positions rather than replay size. One AdamW instance persists across generations, including its moments and step counts; weight decay applies to weights, not biases. Finite loss and gradients are checked before stepping; gradients are clipped at norm 5, and parameter finiteness is checked after stepping. Numerical failure aborts the official campaign.

Generation zero is initialization only. The loop trains generations 1 through 20 inclusive, giving 5,120 completed self-play games per seed. The next generation collects from the continuing learner regardless of champion promotion. Seeds 42 and 314159 construct separate models, trainers, replay windows, and domain-separated RNG streams. Evaluation/loading preserve training RNG behavior. Seed 42 remains primary and 314159 replication; sealed results cannot choose a preferred seed.

**3. Search and exploration protocol**

The frozen settings match the executed path:

| Setting | Verified behavior |
| --- | --- |
| Self-play | 256 root-edge simulations per move |
| Deployment evaluation | 512 simulations, noise off, temperature 0, seeded uniform maximum-visit ties |
| Exploration noise | Legal-action Dirichlet alpha 1.0, epsilon 0.25, mixed into root priors only |
| Action schedule | Temperature 1 at pre-move plies 0–7; temperature 0 from ply 8 |
| Policy supervision | Raw normalized root visits at every ply, including temperature-zero moves |
| Tactical intervention | No tactical guard in v2 self-play or deployment search |
| Tree lifecycle | Fresh single-parent tree per search; no tree reuse or transposition reuse |

Final behavioral calibration deliberately uses held-out self-play under the frozen training schedule, including 256 simulations, noise, and the ply-dependent temperature. This is the specified calibration workload; it is separate from the 512-simulation deployment evaluations. The guarded UCT ladder opponent retains its explicitly declared guards.

**4. Development lifecycle and champion selection**

[Campaign._run_seed, _champion_check, and _select](../games/connect4/alphazero_v2/campaign.py) perform champion checks only at generations 5, 10, 15, and 20. Each candidate is that generation's saved inference model; the incumbent is the latest promoted checkpoint, or generation zero. Promotion never replaces the learner, optimizer, or replay.

Every arena opening is used twice, with the candidate owning X then O. The position's actual mover is preserved when ownership swaps. Per-game RNGs are separately derived by seed, namespace, opening, game index, and role. Each check runs 200 candidate/incumbent games and 40 games against each of Random and corrected Negamax depths 1 and 2. Baseline results are reported; the declared promotion decision uses the candidate/incumbent arena and development tactics.

Promotion requires a complete arena, score at least .55, opening-family bootstrap lower bound strictly above .50, successful correctness checks, and no more than .02 regression in either immediate-win or safe-response accuracy. The tiny tolerance at the regression boundary handles floating-point representation. Incumbent tactical evidence is reused only from the same owner's evaluation of that frozen generation. Missing scheduled checks, generation artifacts, or required diagnostics prevent selection.

**5. Sealed-test isolation**

Mechanically traced the call chain: both `_run_seed` calls finish and publish their selections; `DEVELOPMENT_SELECTION_COMPLETE` records both selections; `_final` publishes validated `started.json`; the durable `RUNNING_FINAL_EVALUATION` transition records its hash and those selections; only then does `_final_seed_units` invoke sealed searches, arenas, or calibration.

Before that transition, model evaluations use `solved-development`, `tactical-development`, and `openings-development`. Launch validation and package copying read sealed labels but do not evaluate models or feed labels into training or selection. Loading agents to describe them before the transition likewise performs no sealed inference.

After the transition, there is no training or promotion call. Final agents load the selected, hash-checked checkpoints. Both seeds use the same declared row sets, simulation counts, tie seeds, opponents, paired openings, and calibration count. Independent streams vary by seed as intended. Selection copies in the two history entries, started record, and results must agree at acceptance. No path feeds sealed outcomes back into the learner or model choice.

**6. Reference evaluation and statistics**

Read [reference_negamax.py](../games/connect4/alphazero_v2/reference_negamax.py), [oracle.py](../games/connect4/alphazero_v2/oracle.py), [arena.py](../games/connect4/alphazero_v2/arena.py), [evaluation.py](../games/connect4/alphazero_v2/evaluation.py), and [statistics.py](../games/connect4/alphazero_v2/statistics.py).

Corrected Negamax uses mover-relative scores, dominant terminal values, exact draws, depth-aware keys, and EXACT/LOWER/UPPER cache semantics. Root actions receive full-window depth-limited scores. The bitboard solver's immediate wins, forced replies, threat pruning, alternating values, and stored bounds agree with the independent late-position minimax checks. A solver budget exhaustion raises rather than manufacturing a label. Frozen package verification checked legal positions, tactical labels, solved-label consistency, exclusions, and development/sealed family separation.

Additional independent review code used an immutable 42-cell board, its own 69 winning-window scan, and unpruned minimax, without production engine/oracle transition helpers. All legal action outcomes matched both the bitboard solver and stored labels on 30 sealed solved base rows: the five latest positions in each outcome/actor stratum. A separately written depth-limited minimax matched every corrected Negamax root score on 19 nonterminal X/O positions at depths 1, 2, and 4, including immediate-win examples. Existing tests additionally cover arbitrary cache-window sequences, mirrored positions, terminal draws, and two independent solver implementations.

Arena outcomes belong to the tested agent's assigned color, with win/draw/loss points 1/.5/0. Abandoned games are not scored and cannot complete an official unit. Confidence intervals resample opening families, retaining paired games and mirrors together; empty-board pairs share one family. The 10,000-resample percentile calculation and fixed analysis seed are deterministic. An independent calculation with unequal family sizes matched the score and interval exactly.

Tactical metrics average tie seeds within rows, rows within families, then families, including actor-specific metrics. Solved metrics compare action values with the mover's optimal outcome; avoidable losses are restricted to recoverable positions. Raw value metrics use exact mover-relative labels, class-balanced MSE, decisive signs, draw error, and wrong-sign saturation. Calibration compares raw pre-search values with completed-game actor outcomes, resamples whole games, and computes score-space bin gaps with distinct-game support. Independent arithmetic checks matched its MSE, zero baseline, game/position counts, and endpoint bin coverage.

No mathematical or evaluation defect was found. COMPLETE certifies completion and evidence integrity; a scientifically complete campaign can legitimately fail strength or value gates. It does not automatically certify successful research outcomes or public-release readiness.

**7. Official evidence architecture**

Read all 1,147 lines of [official_evidence.py](../games/connect4/alphazero_v2/official_evidence.py). The architectural invariant holds:

`raw evidence + frozen declaration + frozen package copies + started selections/descriptions → derive_seed_result`.

`derive_seed_result` is the sole sealed-result producer. `seed_result_problems` first checks exact raw-evidence structure and declaration-derived coverage, then requires the entire stored result to equal the JSON-normalized derivation, with type-strict comparison. Booleans cannot substitute for integers; missing/extra fields and list-length differences fail. Summaries, gates, copied selections/descriptions, and overlap sensitivity are re-derived rather than trusted because they carry a digest.

Acceptance requires exactly seeds 42 and 314159; all 800 sealed tactical rows and 600 sealed solved rows per seed; four declared search repetitions per row; all seven ladder opponents; and exactly the 200 ordered paired games per opponent and NN-only comparison from the 100 sealed openings. Search visits must be legal, sum to 512, and support a maximum-visit chosen action. Seed-independent root values must agree, including the separately recorded solved raw value.

Arena records are independently replayed through bitboard rules, must extend the specified opening to legal termination, and must agree on winner, agent-relative result, mover/action decisions, and pairing identity. Calibration requires games 0–255 exactly once and in order, contiguous plies from zero, actor parity, coherent alternating outcomes, finite values, and terminal-length/parity constraints supported by the recorded fields.

All declared package copies in the campaign directory are hash-bound to the token-certified declaration. The same result validator runs before each result is published, again on files read from disk before the completion barrier, and independently in `official_results`. Started-record validation likewise occurs at all three stages. Acceptance returns the documents it actually validated. Unexpected validator failures refuse acceptance.

The known calibration limitation is nonblocking under the stated threat model. `calibration_games` gets its values from an observer called exactly once per applied move; the corresponding examples come from that same move loop. `play_game` returns only after legal terminal replay and outcome finalization. An exception or stop prevents it from returning a partial game, and `_unit` rechecks limits after computation. Although `zip` would truncate unequal arrays supplied by changed code, the unmodified observer/example construction has equal lengths. No natural incomplete-record path was found. Deliberately removing an even number of plies and re-deriving every dependent summary remains the explicitly excluded forgery.

The known `overlap_families` limitation is also nonblocking. The producer scans every hash-checked archived training-game file and all pre-move positions, canonicalizing reflections. Acceptance validates the supplied families and recomputes sensitivity summaries without independently repeating that archive scan. The full-set primary metrics do not depend on this diagnostic list. No unmodified-campaign path to a false primary conclusion was found.

**8. Fail-closed campaign control**

The official path has one serial owner, one exclusive `flock`, and a fresh directory. It never loads published resume boundaries or partial evidence to continue a campaign. Generation boundary reloads during temporary-file validation do not resume scientific work. Crashes leave either a terminal record or a nonterminal record that cannot be accepted; a later invocation marks the interrupted campaign INCOMPLETE and refuses continuation.

COMPLETE and INCOMPLETE have no outgoing transitions. Result publication, flushes, validation, identity checks, and workload checks precede one completion barrier. Its single monotonic reading is the authoritative endpoint. An already handled stop or exceeded limit fails the barrier. A later signal is recorded as late and cannot retroactively invalidate completed science. A failure before COMPLETE becomes visible ends INCOMPLETE; a failure after the rename adopts the visible COMPLETE and cannot overwrite it through the error path. Independent `official_results` validation remains necessary even when a state string says COMPLETE.

The rerun control tests exercised real SIGTERM handling during publication, stops immediately before the barrier, deadline boundaries, failures before and after terminal rename, owner-lock exclusion, identity drift, and hard process exits at training, selection, and sealed-evaluation boundaries. These support the inspected one-way terminal semantics.

**9. Budgets and required scientific work**

The declaration binds 28,800 seconds of collection plus optimization per seed, 86,400 seconds overall, and a ceiling of 6,400 started evaluation games. Cooperative checks precede new work; post-work checks reject overruns. Training time accumulates across generations, and the final barrier checks both seed totals and overall elapsed time. Games are reserved before their first move. No retry/resume path resets these accounts.

The independently recomputed workload is:

| Evidence unit | Per seed | Both seeds |
| --- | ---: | ---: |
| Development candidate/incumbent games | 800 | 1,600 |
| Development baseline games | 480 | 960 |
| Development tactical rows | 1,000 | 2,000 |
| Final tactical rows | 800 | 1,600 |
| Final solved rows | 600 | 1,200 |
| Final ladder games | 1,400 | 2,800 |
| Final NN-only games | 200 | 400 |
| Calibration games | 256 | 512 |

The game rows total 3,136 per seed and **6,272 combined**, leaving 128 ceiling slots that authorize no extra experiments. Every evidence kind must have `started = completed = declaration-derived count`; aggregates and copied budget quantities must agree. Training completion requires exactly 20 generations and 5,120 games per seed. Optimizer work is mandatory in the sequential generation loop and its update count is tested against the exact formula; counters cannot cause the owner to skip it. Missing work, timeout, or ceiling exhaustion cannot take this execution path to official COMPLETE.

**10. Declaration and execution binding**

Independently SHA-256-hashed the declaration bytes, every declared source file, package, frozen record, and retained checkpoint. Recomputed source/group digests directly from sorted JSON path-to-file-hash maps, independently of the provenance digest helper. Reconstructed the declaration in memory from the current builder, manifest, frozen-record entries, freeze notes, and configured runtime; serialized bytes were identical. No freeze or package-build command was run on the frozen assets.

| Binding | Verified value |
| --- | --- |
| Declaration, 17,440 bytes | `2e39713f6e2791c735406c3bec7def0d336adb6add1cd43d49319e83aac486ab` |
| Complete execution closure, 27 files | `bbc74893998e1971dcb245611b5ac94d6266fcf9b339f3750aa3c545fc03fa29` |
| Training group, 18 files | `22678c4b1c056f4b2b61a94fdc2eaaa0cd67de884a82c558ab15324406899e35` |
| Evaluation/launch group, 9 files | `f6444ff21dfa4e6215766038c8177eecc95bf96fbafdebe47668a8f092708a70` |
| Retained 4D.2f checkpoint | `78276912bdef5f1b1b53ac6b2ec3ede2eed20a8ebb5796a9bdcbb57e4cb28c51` |

The closure covers the training modules, shared encoder/engine/search dependencies, evaluation agents and references, statistics, package loader, provenance, launch control, and acceptance module. The launcher imports the closure before work and repeatedly checks its content and runtime identity. The declaration also binds scientific settings, six packages, three frozen records, retained checkpoint, and launch-control semantics. The actual preflight passed all five gates on Apple M5, arm64, macOS 26.6.2, CPython 3.11.17, torch 2.10.0, NumPy 1.26.4, deterministic algorithms, and the declared single-thread settings/build identity.

Every training-group file was additionally compared byte-for-byte with Git commit `85dd988`, before B.5. All 18 are unchanged, as is the training-group digest. Inspection of the campaign refactor confirmed evidence production/acceptance changes without changes to collection, optimization, model progression, or champion methodology.

**11. Verification and meaning of the systematic tests**

Tests ran with bytecode/cache writing disabled. Torch checks used `/tmp/board-game-phase4c1-venv/bin/python` and all five declared thread variables set to 1; no-torch checks used `.venv/bin/python`.

| Check rerun for this review | Result |
| --- | --- |
| Campaign and launch-control suites, nothing deselected | 186 passed |
| Remaining torch backend suites | 1,100 passed |
| Complete no-torch backend suite, including API/import-isolation coverage | 482 passed, 15 skipped |
| Generated artifact-schema mutations in the campaign suite | 5,121; no incorrect acceptance/refusal |
| Additional exhaustive concrete-path seed-result sweep | 17,212; zero accepted, zero validator exceptions |
| Frozen package verification | Six packages, 2,100 rows; exit 0 |
| Actual bound-runtime preflight | Five gates passed; `ok: true`; 6,272 planned games |
| Independent minimax/statistical probes | All comparisons passed |
| Declaration reconstruction and independent hashes | Exact matches |

The torch backend split excludes the five API modules requiring the separate API environment and the two import-isolation gates; those execute in the complete no-torch run. An initial torch collection attempt exposed the missing `dotenv` dependency in an API module; correcting the test split required no dependency or repository changes. The 15 no-torch skips are the expected unavailable-optional-backend coverage, supplied by the torch runs.

The 5,121 count is 1,951 and 1,945 seed-result mutations, 395 state mutations, 286 started-record mutations, and 544 declaration mutations. These delete and retype generated schema paths, substitute booleans for integers, and add unexpected object members. Repeated list structures are sampled at first/last occurrences; seed, opponent, and row identities remain distinct. Only declared optional phase-timing diagnostics and valid free declaration items are expected to survive the relevant mutations. Package-copy hash failures are tested separately.

The additional exhaustive sweep removed the repeated-list sampling reduction: 8,243 mutations for seed 42 and 8,969 for seed 314159. It used the genuine COMPLETE two-seed synthetic fixture produced during this review and the unchanged validator. Deterministic derivations were memoized by their complete inputs to avoid repeating identical bootstraps. Every mutation was refused through explicit validation, not an exception backstop.

These counts are evidence of broad structural coverage, not proof of every scientific property. Their significance is the shared producer/validator architecture plus tests for declaration-derived workloads, coherent unit reductions, semantic game replay, summaries inconsistent with raw evidence, and validation at publication/completion/acceptance. Independent mathematics, source tracing, runtime binding, and interruption tests supply the complementary checks. No new field-specific patch or discretionary hardening is required by this review.

**12. Launch approval**

The frozen declaration
2e39713f6e2791c735406c3bec7def0d336adb6add1cd43d49319e83aac486ab
is approved for the separately authorized Phase 4D.3 Milestone-3 campaign on
the bound Apple M5/runtime.

Work stops at this review. The research campaign has not been launched.
