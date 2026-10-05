# Phase 4D.3B — Milestone 1 hostile review and AlphaZero v2 evaluation/preflight (Milestone 2)

> **Superseded launch instructions — read first.** The declaration token
> `2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8` given below is **REJECTED / NOT
> AUTHORIZED**. The [launch-readiness review](phase4d3b-launch-readiness-review.md) found blockers B1–B4. Phase
> 4D.3B.1 fixed them and re-froze the declaration (format 2) with a new token: see
> [phase4d3b1-launch-control-fixes.md](phase4d3b1-launch-control-fixes.md). The current launcher refuses the
> old token and the old declaration bytes. Do not use the §10 token or the §13 commands as written.

Completed October 5, 2026. **No research training was run.** This phase produced no learned v2 candidate, no research checkpoint, no strength evaluation, no API/UI exposure and no deployment. **It makes no strength claim.** The only optimizer updates were discarded updates on tiny synthetic data in unit tests and temporary directories, plus one discarded synthetic update used to size a full-scale resume artifact (§9).

Starting point: `origin/main` = `a58abccb2f62c4308b074ac123cd6f3ad798c2c0` (Phase 4D.3A), clean tree. Work is on local branch `phase4d3b-alphazero-v2-preflight`, uncommitted. There was no commit, push, merge, dependency change or production-configuration change. Historical evidence under `experiment-output/` and `models/` was only read.

Design authority: [Phase 4D.3 review](phase4d3-alphazero-v2-review.md) §9–11. Milestone 1 contracts: [Phase 4D.3A](phase4d3a-alphazero-v2-core.md). Interpretations the review left open are marked **(choice)**.

## Summary

**Milestone 1 review.** Six of the eight areas held up; two were correct but needed hardening. Correct: scalar perspective, PUCT signs, target/action separation, noise scope, frozen generations, replay and the persistent optimizer. **Both concerns named in the assignment were real provenance defects**, and both are fixed:

- the exact-resume claim did not enforce the runtime it depends on (thread count measurably changes the trajectory here);
- it did not bind source identity at all.

A 30-mutation sweep left 4 contracts untested. All four, plus 9 mutations aimed at the new provenance checks, are now killed by new tests.

**Milestone 2 is implemented, and the campaign is frozen but not run:**

- a corrected, versioned Negamax;
- an exact oracle with two independent solvers;
- frozen model-blind development and sealed packages (tactical 400+100 bases, solved 300+150 bases, opening banks 100+100 rows);
- a paired arena with opening-family bootstrap;
- frozen acceptance thresholds and champion gate;
- per-generation diagnostics;
- a bounded, authorization-gated launcher with budgets, attempts, resume, retention, source snapshots and a one-time sealed evaluation;
- a deterministic runtime;
- 256/512 search and full-scale artifact profiling.

A second mutation sweep (24 mutations) ends with 22 killed and 2 equivalent.

**The frozen declaration's SHA-256 is `2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8`.** It is the authorization token a Milestone 3 run requires.

## 1. Milestone 1 hostile review

### Method

Every v2 module and the reused v1 primitives were re-read against the review and the 4D.3A handoff, with the eight areas named in the assignment. I did not take passing tests as evidence that an area was correct. Instead I ran a **mutation sweep**: each item is a deliberate, plausible defect applied to the source, followed by the v2 test file, then restored. A surviving mutation means the tests do not pin that contract.

The sweep script and logs are not in the repository. The mutations are listed below so they can be reproduced.

### First sweep against the 4D.3A tests (30 mutations)

26 were killed. Four survived:

| Surviving mutation | What it showed |
| --- | --- |
| `STRICT_RUNTIME_KEYS` without `intra_op_threads` | Concern 1 confirmed: no test enforced any runtime field except a faked torch version. |
| Inference loader ignores `contract` equality | Only the key set was tested. A same-shaped file with a changed contract (e.g. `value_perspective`) would load. |
| Replay sampling draws from the *search* RNG | Stream ownership was not pinned. Coupled streams stay deterministic, so resume tests could not see it. |
| Root noise consumes an extra RNG draw per search | "One Dirichlet draw per search" was checked by a counter, not by the stream position. |

Killed mutations covered: PUCT sign, backup alternation, terminal sign, outcome sign, schedule boundary (`<` vs `<=`), target derived from action temperature, global-NumPy noise, snapshot aliasing the learner, replay window +1, sampling with replacement, newest-generation-only sampling, Adam moment reset, bias decay, clipping removed, update floor instead of ceil, softmax-then-mask, reflection without visit reversal, resume skipping optimizer / noise / augmentation / step counters, constant stored action temperature, τ=1 evaluation, labelling by the final mover, partial generation insertion and ignored ε.

### Area verdicts

| Area | Verdict | Evidence |
| --- | --- | --- |
| Current-player scalar perspective and PUCT signs | Correct | Re-derived. `−Q(child)` selection, alternating backup, terminal −1 for the side to move after a win; all sign mutations killed. |
| Policy target vs action temperature | Correct; **hardened** | Target derived only from raw visits. The resume loader now also rejects stored records whose action temperature violates the ply schedule, or whose τ=0 action is not a maximum-visit action. |
| Root-noise scope and RNG ownership | Correct; **tests strengthened** | Root-only, legal-only, private PCG64. New tests: an exact reference-stream reproduction (kills the extra-draw mutant), and phase ownership — collection advances only search/action/noise, training only sampling/augmentation (kills the shared-stream mutant). A generation also consumes no process-global RNG (tested), so excluding global RNGs from `state_sha256` is sound. |
| Frozen-generation lifecycle | Correct | Snapshot deep-copied and checked unchanged; no trainer step during collection. |
| Replay sampling/eviction | Correct | Whole-generation expiry, uniform without replacement within a batch; all mutants killed. |
| Persistent optimizer | Correct; **hardened** | One AdamW for the run. On load, every per-parameter Adam `step` must now equal the trainer step counter. |
| Inference/resume isolation | Correct; **tests strengthened** | Any contract-field change is rejected (new test). Inference and resume artifacts of one boundary hold identical weights (new test). |
| Deterministic boundary resume | **Defective provenance** (below); trajectory logic correct | Fixed. |

### Concern 1 — runtime identity was recorded, not enforced (confirmed defect, fixed)

`STRICT_RUNTIME_KEYS` covered `python, torch, numpy, machine, intra_op_threads`. `inter_op_threads`, `deterministic_algorithms` and `platform` were recorded but not compared. Several numerics-relevant facts were not recorded at all.

Direct measurement on this machine (Apple M5, torch 2.10.0, `BLAS_INFO=accelerate`), using a tiny 3-generation runner with 64-sample batches:

| intra / inter / deterministic | Generation-1 and -2 weight hashes |
| --- | --- |
| 1 / 1 / off | `37c6c476eff0`, `287640716ba9` |
| 1 / 1 / on | identical |
| 1 / 4 / off | identical |
| 4 / 1 / off | identical |
| **8 / 1 / off** | `37c6c476eff0`, **`e832208cd1a9`** |

The thread count therefore changes the trajectory. In this probe, inter-op threads and the deterministic flag did not, but that is absence of evidence, not a guarantee. Two further facts matter:

- torch links Apple **Accelerate**, so the operating-system build can change BLAS numerics.
- `machine` records only `arm64`, which cannot tell an M4 from the **Apple M5** this work actually ran on. The review assumed an M4.

**Fix** ([provenance.py](../games/connect4/alphazero_v2/provenance.py)):

- `runtime_identity()` records the fields below, and **every one is enforced**. `runtime_differences()` reports any changed, added or missing key. A parametrized test changes each key in turn and requires the load to be rejected.
  - Python implementation and version; torch version, git version and a hash of `torch.__config__.show()`; NumPy version
  - full platform string; machine; CPU model (via `sysctl`); torch CPU capability
  - intra- and inter-op threads; deterministic flag and warn-only flag
  - float32 matmul precision; mkldnn flag
  - the library thread environment (`OMP/MKL/OPENBLAS/VECLIB/NUMEXPR_NUM_THREADS`)
- A runner pins its runtime at construction or load. It refuses to run a generation or write a boundary if the runtime drifts while it is alive (e.g. `torch.set_num_threads(2)` between generations; tested). Without this, a boundary could claim a runtime under which earlier generations did not run.
- `configure_deterministic_runtime(threads)` pins intra/inter threads and deterministic algorithms, and verifies they took effect.

### Concern 2 — source identity was recorded, not enforced (confirmed defect, fixed)

The boundary recorded `git rev-parse HEAD` and a dirty flag, and nothing compared them on load. The commit is the wrong identity in both directions:

- **Not sufficient.** Under one commit, a dirty tree can contain any execution change. The dirty flag also counted any untracked file anywhere in the repository.
- **Not necessary.** A documentation-only commit leaves execution unchanged.

**What must be bound** for the exact-resume claim to be defensible is the *content* of every source file whose code can affect the trajectory. That is the **execution closure**:

- every `*.py` in `games/connect4/alphazero_v2/` **(choice: the whole package, conservatively, including evaluation and launcher modules)**;
- `games/connect4/{__init__,board,connect4,neural_mcts}.py` and `agents/{mcts_agent,mcts_nn_agent}.py`.

Third-party code is bound by the runtime identity above.

Fix:

- The boundary stores per-file SHA-256s plus a combined digest. Loading **rejects any difference** by default and names the differing files.
- Git commit, tracked/untracked dirt and the diff of execution files from HEAD are still recorded as provenance, never enforced. A test shows that a different commit with identical execution content still claims exact continuation.
- The process digest is taken when `generation` is imported. Disk is re-hashed whenever a runner starts, runs a generation or writes or loads a boundary, so editing a source file mid-campaign stops the runner rather than silently mixing code.
- A fresh-interpreter test runs, saves, resumes and continues a generation. It then asserts that every repository module actually loaded lies inside the declared closure; it caught that `board.py` must be listed (mutation killed).

### Resulting exact-resume contract (resume format_version 2)

Exact continuation is claimed **only** when:

- the boundary file matches its expected hash (optional `expected_sha256`; the launcher always passes it);
- every runtime field is identical;
- every execution-closure file is identical;
- the boundary's replay, weight and state digests verify.

`strict_runtime=False` / `strict_source=False` loads remain possible. They append a **lineage** record with the differences and `exact_continuation_claimed: false`. Lineage is saved in later boundaries, excluded from `state_sha256`, and checked by test.

The resume format moved to `format_version: 2` because the schema changed. No resume artifact had ever been produced outside tests.

### Milestone-1 fixes at a glance

[generation.py](../games/connect4/alphazero_v2/generation.py) (runtime/source enforcement, drift refusal, lineage, schedule validation, optimizer-step cross-check, `expected_sha256`, `initial_model`), new [provenance.py](../games/connect4/alphazero_v2/provenance.py), [config.py](../games/connect4/alphazero_v2/config.py) (contract version). Tests: 16 new test functions (34 collected cases, including one per runtime-identity field) new regressions in [test_alphazero_v2.py](../tests/test_alphazero_v2.py). After the fixes, the four survivors and nine new mutations aimed at the fixes themselves were all killed. Those nine: an ignored inter-op field, a dropped deterministic field, disabled runtime and source gates, removed drift and disk checks, a removed schedule check, lineage not saved, and a closure missing `board.py`.

## 2. Milestone 2 source organization

All new code is in [`games/connect4/alphazero_v2/`](../games/connect4/alphazero_v2/). The package stays inert on import, and the API import-isolation gate still proves API startup never reaches it.

| Module | Torch | Responsibility |
| --- | --- | --- |
| [provenance.py](../games/connect4/alphazero_v2/provenance.py) | yes | Runtime identity, execution-closure digest, Git provenance, deterministic runtime setup (§1) |
| [oracle.py](../games/connect4/alphazero_v2/oracle.py) | no | Engine replay, immediate-tactics scans, exact solver A (bitboard α-β), exact solver B (exhaustive), family keys |
| [reference_negamax.py](../games/connect4/alphazero_v2/reference_negamax.py) | no | Corrected, versioned depth-limited Negamax |
| [packages.py](../games/connect4/alphazero_v2/packages.py) | no | Model-blind package generation, exclusions, dev/sealed split, freeze, verification (CLI) |
| [statistics.py](../games/connect4/alphazero_v2/statistics.py) | no | Opening-cluster bootstrap, paired differences, tactical/solved/value/calibration metrics, champion gate, frozen acceptance thresholds |
| [arena.py](../games/connect4/alphazero_v2/arena.py) | no | Paired side-swapped arena; Random, reference Negamax and guarded-UCT opponents |
| [evaluation.py](../games/connect4/alphazero_v2/evaluation.py) | yes | v2 search / NN-only arena agents, hash-pinned 4D.2f adapter, tactical/solved/raw-value runs, held-out calibration games |
| [diagnostics.py](../games/connect4/alphazero_v2/diagnostics.py) | numpy | Per-generation search, data, sampling, gradient and resource diagnostics |
| [campaign.py](../games/connect4/alphazero_v2/campaign.py) | yes | Declaration, authorization, budgets/ledger, attempts/resume, retention, source snapshots, champion gate, one-time sealed evaluation (CLI) |
| [profiling.py](../games/connect4/alphazero_v2/profiling.py) | yes | 256/512 search profiling; full-scale resume artifact profiling (CLI) |
| [frozen/](../games/connect4/alphazero_v2/frozen/) | — | Frozen packages, exclusions, manifest, build provenance, campaign declaration |

Milestone-1 modules changed only as §1 describes, plus hooks: `run_generation(check=...)` and `collect_generation(check=..., observer=...)`. Search results gained deterministic tree statistics (`max_depth`, `leaf_depth_sum`, `terminal_leaves`).

## 3. Corrected reference Negamax (`connect4-reference-negamax-v1`)

The legacy [`NegamaxAgent`](../games/connect4/agents/negamax_agent.py) is **unchanged**, and a test asserts it has no diff from HEAD. It stays a labelled historical opponent: old depth-1/2 results remain results against that exact agent.

The reference agent fixes both verified defects:

- **Cache semantics.** The transposition table stores `EXACT/LOWER/UPPER` bounds keyed by (board, mover, remaining depth). It applies standard bound tightening and never returns a cut-off value as exact. The table is fresh per decision.
- **Dominant terminal scoring.** A completed four scores exactly `±(1,000,000 + remaining depth)`, so faster wins and slower losses are preferred. A full board without four scores exactly 0. The nonterminal heuristic is the legacy open-window score (1/3/9), from the mover's perspective. Its magnitude is below 69×9, so it can never rival a terminal score.
- **Root and ties.** Every legal root move gets its exact depth-limited score under a full window, so tie sets are exact. Ties are broken uniformly by an injected per-game RNG, or center-first without one.

Evidence:

- The review's reproduction: the legacy narrow call returns `(3,5)`, reused full-window `(4,9)`, fresh `(3,10)`. Under the same reused-table sequence the reference returns 10, equal to plain unpruned minimax.
- **Property test.** On random positions at depths 1–4, one shared table answers random window sequences. Every answer respects fail-low/exact/fail-high semantics, and the final full-window value equals plain minimax.
- Per-move root scores equal minimax. The agent takes the fastest win, blocks at depth 2, does not mutate the caller's game, and spreads uniformly over exact ties on a symmetric board.
- Cost: depth 4 takes about 5 ms per decision (Python, this machine).

## 4. Independent solved-position oracle

[oracle.py](../games/connect4/alphazero_v2/oracle.py) values are game-theoretic outcomes for the player to move (+1/0/−1, no distance). Every history is first replayed through the real engine.

- **Method A** (`connect4-bitboard-alphabeta-wdl-v1`) — bitboard negamax over the three-valued outcome.
  - The transposition table stores explicit lower/upper bounds.
  - It prunes forced blocks, unblockable double threats and moves under an opponent threat, and orders moves by threats created. Ordering never changes values.
  - A node budget bounds the work; exceeding it raises and the candidate is skipped and counted.
- **Method B** (`connect4-column-stack-exhaustive-minimax-v1`) — a plain memoized minimax over a different board representation. It has no pruning, ordering or threat logic and shares no search code with A. It is feasible only late in the game.

Evidence:

- On random late positions (28–37 pieces), A and B agree on every action's value.
- A's table is sound under window sequences and reuse: one solver, windows (0,1), (−1,0), (0,1), (−1,1) on 24–31-piece positions, checked against B. This test kills a mutation that stores every result as exact.
- Mirror equivariance holds, and immediate-win scans agree across the engine, the grid scan and the bitboard on 300 random positions.

A third, external check is recommended for the reviewer: Pascal Pons's C++ solver. Its signed score converts to actor-relative W/D/L by sign (positive = mover wins). Columns are 1-based in its notation. Its shortest-win preference does not matter for W/D/L.

## 5. Frozen model-blind packages

Built once with `python -m games.connect4.alphazero_v2.packages build` (seed 4,303,002) from the exact source in `frozen/build-provenance.json`, then verified and copied to [`frozen/`](../games/connect4/alphazero_v2/frozen/).

**Generator (choice).** Each seeded game picks a style:

- **tactical** — a uniform random move with p=.4, otherwise reference Negamax depth 1/2;
- **positional** — random with p=.1, otherwise depth 2/4, which gives long, drawish games.

Negamax ties are seeded. No model is consulted, and self-symmetric boards are excluded so that every base forms a two-row mirror family.

**Deduplication.** Rows are keyed by reflection family of (board, actor); transpositions share a board.

**Exclusions.** All previously inspected positions and their reflections are excluded: 

| Source | Rows | Families | SHA-256 (prefix) |
| --- | ---: | ---: | --- |
| `dqn/diagnostic_positions.json` | 60 | 30 | `ec99e0ee…` (= `FROZEN_SUITES`) |
| `dqn/validation_positions.json` | 96 | 48 | `1c79db45…` (= `FROZEN_SUITES`) |
| `dqn/replication_positions.json` | 96 | 48 | `8e63fd8b…` (= `FROZEN_SUITES`) |
| 4D.2c blind (`preflight/blind-positions.json`) | 96 | 48 | `ba312684…` (= the 4D.2c recorded `fixture_sha256`) |
| 4D.2d blind (`blind-positions.json`) | 96 | 48 | `53775d75…` |
| 4D.2f exact-value blind (`exact-value-blind.json`) | 128 | 64 | `53a8bf0d…` |
| Original neural probes (`neural_self_play`) | 9 bases | 9 | copied verbatim |
| Historical arena `OPENINGS` | 6 | 3 | copied verbatim |

. Three of the sources are gitignored local files, so `frozen/exclusions.json` commits their SHA-256s and the 296 derived family keys.

**Split.** Sealed quotas are filled first from the shuffled pool, then development. Opening families are excluded from the position packages, and every family is disjoint across splits.

| Package | Rows | Bases | Composition (bases) |
| --- | ---: | ---: | --- |
| `tactical-sealed` | 800 | 400 | 100 each unique win X/O and unique safe X/O; directions win h/v/d↑/d↓ 63/40/51/46, safe 57/55/47/41; stages early/middle/late 110/152/138 |
| `tactical-development` | 200 | 100 | 25 each category×actor; **no vertical immediate wins** (the cell was exhausted by the sealed split); stages 21/40/39 |
| `solved-sealed` | 600 | 300 | 50 each W/D/L × X/O; stages 12–19/20–27/28–41: W 38/36/26, D 35/34/31, L 38/38/24; 23 "forced block loses" rows |
| `solved-development` | 300 | 150 | 25 each W/D/L × X/O; **late stage thin** (only 7 late draws, no late W/L) |
| `openings-sealed` | 100 | 20 empty pairs + 40 families | 200 games per opponent |
| `openings-development` | 100 | 20 empty pairs + 40 families | 200-game champion arena; 20-row baseline subset |

Generation examined 557 games / 894 tactical candidates, and 3,332 games / 2,698 solved candidates. Of the solved candidates, 18 were skipped at the 3M-node budget and 527 rejected as trivial; draws were the binding quota. Every quota was met with no fallback. Balance is "as feasible", and the imbalances above are reported, not hidden.

**Tactical labels.** Each tactical row needs **both** labelers to agree on both orientations: engine successor states (`make_move` + `check_winner`) and an independent grid scan. A *unique immediate win* has exactly one winning column. A *unique safe response* means:

- the mover has no own immediate win;
- the opponent has an immediate threat;
- exactly one column leaves no opponent immediate win.

This is one-reply safety, not a game value. Rows are stratified by direction of the line completed or blocked × stage (early 6–13, middle 14–23, late 24–41), with round-robin balance.

**Solved labels.** Each solved row records every legal action's exact outcome from method A, and the state value is cross-derived (`value == max(action values)`). Method B independently re-solved every base with ≥24 pieces that fit its budget: **128 of 300** sealed bases and **30 of 150** development bases (every base with ≥24 pieces), with full agreement on every action. Mirrors were re-solved by A and checked for equivariance.

Selection rules **(choice)**:

- no immediate win for the mover (those belong to the tactical package);
- a win needs at least one non-winning action;
- a draw needs at least one losing action ("late draw-preserving decisions");
- a loss must not be provable by a one-reply scan.

Rows are stratified by stage: 12–19, 20–27, 28–41 pieces.

**Openings.** Each bank has 20 empty-board pairs plus 40 prefix families × (base + mirror), with lengths 2–8. Bases are 20 even-length (X to move) and 20 odd-length (O to move). Every prefix is nonterminal, with no immediate win for either side, and is not self-symmetric. Historical arena openings are excluded.

**Verification.** `packages verify --directory frozen --resolve` re-hashes every file, replays every history through the engine, re-labels every tactical row with both labelers, re-solves every solved row from scratch with method A, and checks exclusions and dev/sealed disjointness: **passed for all 2,100 rows (every one of the 900 solved rows re-solved from scratch and identical), exit 0**.

**Limitations (stated, not hidden).**

- Positions solvable by Python method A within its budget come mostly from 12+ pieces, so the solved package under-represents openings.
- Method B covers only the late subset.
- Opening exact values are not computed.
- Draw rows depend on the positional style.
- The sealed package is sealed by protocol, not cryptography. Its labels are visible in the repository, but no model prediction has ever been computed on it. The launcher only evaluates it once, after champion selection (§7).

## 6. Paired arena and statistics

- **Arena.** For each opening row, game 0 gives the agent X and game 1 gives it O from the identical position.
  - Each agent call has its own RNG, derived from (namespace, opening id, game, role), so results do not depend on execution order (tested).
  - Agents receive a copy of the game. An illegal move raises a correctness error.
  - A cooperative stop between games marks the arena incomplete. A stop mid-game records an abandoned game with no outcome that is never scored (tested).
- **Score and intervals.** Score is `(W + .5D)/games`, with W/D/L, per color and per stratum.
  - The 95% interval is a percentile bootstrap that resamples **opening families**: a prefix and its mirror form one family, and **all empty-board pairs form a single shared family**. It uses 10,000 resamples and analysis seed 4,303,009.
  - A test kills a mutation that resamples games instead of families.
  - Paired differences resample shared families.
- **Positions.** Tactical and solved metrics average over search seeds within a row, then over rows within a family, then over families.
  - Value metrics are class-balanced MSE, per-class MAE and sign, draw MAE, and wrong-sign saturation (|v|≥.95).
  - Behavioral calibration uses game clusters, MSE against the zero predictor (and a development constant, if declared), and five equal-width bins with contributing-game counts.
- **Frozen thresholds** ([statistics.py](../games/connect4/alphazero_v2/statistics.py) `ACCEPTANCE_THRESHOLDS`, `CHAMPION_GATE`) match review §11 exactly, and a test pins every number. An incomplete arena never passes a gate.

## 7. Bounded campaign launcher

```sh
# Every run command must be started with the library thread environment pinned:
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m games.connect4.alphazero_v2.campaign preflight --declaration games/connect4/alphazero_v2/frozen/campaign-declaration.json
# Milestone 3 only, after separate authorization (the token is the declaration SHA-256):
#   ... campaign run --declaration D --campaign-dir NEW_DIR --seed 42 --authorize <sha256>
#   ... campaign run --declaration D --campaign-dir NEW_DIR --seed 314159 --authorize <sha256>
#   ... campaign final-evaluate --declaration D --campaign-dir NEW_DIR --authorize <sha256>
```

**Refusals.**

- The authorization token must equal the declaration SHA-256.
- The library thread environment must be pinned (`configure_deterministic_runtime(1)`).
- A campaign directory can only be reused with the same declaration. All published files are never overwritten.
- A seed cannot be rerun after its selection is written.
- The sealed evaluation is one-time: a second attempt is refused once `final/started.json` exists. It cannot start before both seeds have development selections.

**Budgets** ([`Budget`](../games/connect4/alphazero_v2/campaign.py)) are cumulative across attempts through an append-only, fsynced `ledger.jsonl`:

- per-run collection + optimization seconds;
- campaign wall-clock seconds, including evaluation;
- the evaluation-game ceiling.

A check runs before every self-play search, update, arena game, arena move and tactical search. Heartbeats are written at least every 30 s and at every phase end. Abandoned work counts. Time between the last heartbeat and a hard crash is lost, and is disclosed rather than estimated. SIGINT/SIGTERM request a cooperative stop: the CLI exits 3 with `stopped: signal N`, which is tested with a real SIGINT.

**Attempts and resume.**

- Each invocation creates `attempt-NNN/` with a source snapshot: tracked patch, untracked files (≤100 MB or refused), Git provenance and execution digest.
- It resumes from the newest unpruned resume boundary via `load_resume_boundary(..., expected_sha256=<artifacts.jsonl hash>)`, strict on runtime and source.
- A stopped generation is discarded and rerun.
- A champion check that was stopped is recorded as incomplete evidence, never used for a decision, and rerun on resume.

**Test evidence.** A tiny campaign was interrupted mid-champion-check, then mid-training, and resumed across three attempts. It produced byte-identical generation-2 weights, the same champion decisions and the same selection as the uninterrupted campaign.

**Retention.**

- Every generation keeps its inference snapshot, starting from generation 0 (the untrained initial champion).
- Only the latest two resume boundaries are kept. Older ones are deleted after a hash check, and the deletion is recorded in `artifacts.jsonl` and the ledger.
- Every generation's games (moves and winner) are archived outside the active replay. The sealed evaluation's overlap sensitivity uses that archive.

**Champion gate** (development evidence only), at generations 5/10/15/20:

- a 200-game paired arena against the current champion on the development opening bank;
- 40 games each against Random and corrected depth 1/2 (the predeclared 20-row subset `baseline_rows`);
- development tactics at 512 simulations with seeds 0–3.

Promotion requires a complete arena, score ≥.55, a lower bound >.50, and at most 2 points of regression in immediate-win or safe-response accuracy against the champion. Correctness failures abort the run. Rejecting promotion never touches the learner or its optimizer.

**Sealed final evaluation** (one time, after both selections):

- sealed tactics and solved rows (search 512, seeds 0–3), plus NN-only tactics;
- raw-value metrics;
- the 7-opponent ladder on the sealed bank (100 rows / 200 games each);
- NN Only versus Random;
- 256 held-out calibration games with the training schedule.

Every position metric is also reported with the **prespecified overlap-excluded sensitivity**: any package family seen in any archived training position of that seed, in either orientation, is dropped.

**Planned evaluation games** are computed from the declaration and must equal it: 4×(200+3×40) + 7×200 + 200 + 256 = 3,136 per seed, **6,272** total, against a ceiling of 6,400.

**Per-generation logging.** Deterministic diagnostics live in the runner history and therefore take part in resume identity:

- new/replay positions;
- unique positions and reflection families (new and retained);
- game length, outcomes and draw games;
- samples by source generation and by age, and unique sampled positions;
- gradient-norm quantiles and clipped fraction;
- mean root prior and visit-target entropy;
- illegal raw policy mass;
- raw and searched root values;
- root visit coverage, maximum and mean leaf depth, terminal-leaf fraction;
- tactical contradictions by actor/value/ply (proven states and all states).

Resource measurements are returned beside the history, never in resume state: search-time quantiles, collection/training seconds, peak RSS. The campaign adds development raw-value metrics per generation: MSE by class, sign, draw MAE, wrong-sign saturation.

## 8. Deterministic runtime for the campaign

The declaration requires 1 thread and deterministic algorithms. The launcher refuses to start unless `OMP/MKL/OPENBLAS/VECLIB/NUMEXPR_NUM_THREADS=1` are set **before** Python starts, because native libraries read them at load time. It then pins intra/inter-op threads to 1 and enables `torch.use_deterministic_algorithms(True)`, and verifies both. The whole runtime identity goes into every boundary and is enforced on resume. The tiny campaign tests run under exactly this configuration.

## 9. Preflight measurements (Apple M5, 1 thread, deterministic algorithms; untrained seed-42 network)

The reports are in `experiment-output/phase4d3b-preflight-20261005/{search,artifacts}/*.json` (gitignored, new directory). The sizing `.pt` files (an untrained network plus one discarded synthetic update) were deleted after measurement, so no v2 weights exist outside tests.

**Whole-search latency** (`profiling search`): 500 model-blind positions with 0–41 pieces (median 10), after 20 warm-up searches. Root noise was off and τ=0.

| Budget | Mean | p50 | p95 | p99 | Max | Max depth (p50/max) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 87.8 ms | 95.4 ms | 133.9 ms | 150.1 ms | 343.5 ms | 3 / 11 |
| 512 | 172.2 ms | 191.0 ms | 240.9 ms | 276.8 ms | 353.0 ms | 4 / 17 |

These are close to the review's proportional estimates (96/192 ms). **512 p95 is 241 ms**, against the ≤500 ms readiness target. That figure is for an untrained network, whose diffuse priors give shallow trees. A learned network's deeper trees and terminal revisits may differ, so readiness must be re-measured on the approved artifact.

Other costs:

- **Self-play at 256:** 0.084 s per ply over 6 games (14–41 plies). Peak RSS stayed flat at 281 MiB across games.
- **Training:** 42 ms per AdamW step at batch 128, so at most 336 steps ≈ 14 s per generation.
- **Opponents:** guarded UCT-800 takes 0.44 s per move (mean); reference Negamax depth 4 takes 4 ms.

**Full-scale resume artifacts** (`profiling artifacts`; 8 generations × 256 synthetic, legal, schedule-conforming games; one discarded synthetic update so AdamW moments exist):

| Replay | Positions | Resume file | Save (incl. validation reload) | Load (full replay rebuild + checks) | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Typical generated lengths | 51,988 | 5.8 MB | 6.6 s | 5.7 s | 367 MiB |
| Worst case (all 42-ply) | 86,016 | 6.9 MB | 10.3 s | 8.9 s | 429 MiB |

The inference artifact is 1.31 MB. Retaining two resume boundaries plus 21 inference snapshots per seed is about 41 MB. Size and resume time are not constraints.

**Budget plausibility (planning arithmetic, not a promise).**

- Collection: 256 games × 20–30 plies × 0.084 s ≈ 7–11 min per generation, so **2.4–3.6 h per seed**, plus about 5 min of training. That is inside the 8 h per-run cap.
- Development evaluation: each scheduled check is about 20–25 min (200 v2-vs-v2 games at 512, 120 baseline games, development tactics), so about 1.5 h per seed.
- Sealed evaluation per seed: the 1,400-game ladder (UCT-800 is the slowest opponent), 2,400 tactical and 2,400 solved searches, NN Only and 256 calibration games — a few hours.
- Two seeds total roughly 13–17 h against the 24 h campaign cap. Learned networks and longer games could exceed these estimates, and the launcher enforces the caps regardless.

## 10. Frozen campaign declaration

> **REJECTED (Phase 4D.3B.1):** the declaration and token described in this section must never authorize a campaign. See [phase4d3b1-launch-control-fixes.md](phase4d3b1-launch-control-fixes.md) §6.

[`frozen/campaign-declaration.json`](../games/connect4/alphazero_v2/frozen/campaign-declaration.json), SHA-256 `2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8`. It is validated on every load: exact key set, config round-trip per seed, schedule within generations, package hashes, and the planned-games arithmetic.

- **Seeds** 42 (primary) and 314159 (replication); **20 generations**.
- **Config:** every `V2Config` default from review §10 (256/512 simulations; noise .25/1.0; 8 exploratory plies; 256 games per generation; last 8 generations / 2,048 games; batch 128; 4 samples per new position; AdamW 3e-4, betas .9/.999, eps 1e-8, decay 1e-4, clip 5; reflection .5).
- **Budgets:** per-run collection+optimization 28,800 s; campaign 86,400 s; evaluation ceiling 6,400 (planned 6,272).
- **Runtime:** 1 thread, deterministic algorithms, strict resume on runtime and source.
- **Evaluation:** 512 simulations, tactical seeds 0–3, noise off, guard off, τ=0, seeded ties.
- **Champion gate:** as in §7. **Final ladder:** Random, corrected Negamax 1/2/4, guarded UCT-800, initial v2 at 512, 4D.2f at 512, all on the sealed bank; 256 calibration games. No calibration constant is declared, so only the zero-predictor comparison applies.
- **Thresholds:** `statistics.ACCEPTANCE_THRESHOLDS` (review §11). **Retention:** 2 resume boundaries, all inference snapshots, all archived games.
- **4D.2f checkpoint:** `experiment-output/phase4d2f-…/candidate.pt`, SHA-256 `78276912…c28c51` (matches its own manifest and was verified here).
- **Package hashes:** in [`frozen/manifest.json`](../games/connect4/alphazero_v2/frozen/manifest.json). Builder provenance (seed, command, builder-source hashes identical to the current `packages.py`/`oracle.py`/`reference_negamax.py`) is in `frozen/build-provenance.json`.

`campaign preflight` on this declaration reports the token above, 6,272 planned games, the runtime (Apple M5, macOS 26.6.2, Python 3.11.17, torch 2.10.0, NumPy 1.26.4), the execution digest, and Git provenance (HEAD `a58abcc`, tracked tree dirty because this work is uncommitted). **Committing this work changes Git provenance but not the execution digest. Any further edit to a v2 `.py` file changes the execution digest; that is intended.**

## 11. Verification

| Check | Environment | Result |
| --- | --- | --- |
| v2 core (`test_alphazero_v2.py`) | `/tmp/board-game-phase4c1-venv` (Python 3.11.17, torch 2.10.0, NumPy 1.26.4) | 138 passed (104 from 4D.3A + 34 new cases) |
| Evaluation core (`test_alphazero_v2_evaluation_core.py`, torch-free) | same, and `.venv` without torch | 41 passed in both; a subprocess test blocks every torch import |
| Campaign/evaluation/diagnostics (`test_alphazero_v2_campaign.py`) | torch venv | 15 passed. Includes subprocess campaigns under the pinned thread environment, interrupted-vs-uninterrupted equivalence, and a real SIGINT |
| All v2 + engine + every neural, DQN and standalone-MCTS suite | torch venv | **1094 passed**. The two API-startup isolation gates cannot import `dotenv` in this venv; they pass in `.venv` (next row) |
| Full `tests/` backend suite | `.venv` (no torch) | **451 passed, 15 skipped** (torch-dependent modules), including both API import-isolation gates |
| Frozen packages | — | structural verify passed; **passed for all 2,100 rows (every one of the 900 solved rows re-solved from scratch and identical), exit 0** |
| `git diff --check` | — | clean |
| Campaign `preflight` on the frozen declaration | pinned thread environment | token and planned games as in §10 |

Mutation evidence is in [`research/phase4d3b-mutation/`](../research/phase4d3b-mutation/): harness, lists and results. Milestone 1: 30 mutations, 26 killed and 4 survivors, all now killed; plus 9 provenance mutations, all killed. Milestone 2: 24 mutations. Five survivors led to new tests and are now killed. Two are equivalent mutants: a full-board heuristic that is always 0, and a pruning-only shortcut.

All optimizer updates in tests and profiling used synthetic data in temporary directories, and their artifacts were discarded.

## 12. Limitations and open questions for the independent review

1. **Hardware differs from the review.** The review planned for a native M4; this machine reports **Apple M5**. Profiling and any future campaign bind the M5 runtime. A campaign moved to another machine is refused by the strict resume, by design.
2. **Exact resume** is claimed only under identical runtime identity and execution content. A non-strict load is allowed, recorded in lineage, and makes no exactness claim. Time between the last heartbeat (≤30 s) and a hard crash is not counted.
3. **Untrained-network profiling.** Learned networks change tree shape; re-measure the 512 p95 readiness target on the approved artifact.
4. **Solved-package coverage.**
   - The Python method-A oracle reliably solves ≥12-piece positions, so openings are absent from the solved set.
   - Method B independently confirms only the ≥24-piece subset (158 bases). An external Pons-solver check of all 450 solved bases would strengthen independence; the conversion rule is in §4.
   - Development late-stage rows are thin, and development tactics lack vertical immediate wins.
5. **Selection rules for solved rows are project choices (§5).** Excluding one-reply-provable losses deliberately makes the loss class harder.
6. **Sealing is procedural.** The sealed labels are committed and readable; no model output has ever been computed on them. The launcher evaluates them once, after development-only selection, and records the start marker before any computation. A reviewer should confirm that nobody runs `evaluation.search_rows` on sealed rows before then.
7. **Empty-board pairs share one bootstrap cluster** (conservative, as the review requires). The effective sample is about 41 families, not 200 games.
8. **The champion gate's correctness input** is implicit. Any correctness failure raises and aborts the attempt, so a decision is only reached when correctness held.
9. **Behavioral calibration** has no development constant declared, so only the zero-predictor comparison applies (the review says "disclose", not "gate on").
10. **The Docker image would include the inert v2 package** and its about 1 MB of frozen JSON, because the Dockerfile copies `games/`. Nothing imports it, and the API isolation gate proves it. Excluding it via `.dockerignore` is a possible production-config change, deliberately not made here.
11. **The 4D.2f checkpoint is local and gitignored.** The ladder needs that exact file, which is verified by hash at construction.
12. **Diagnostics cost.** Per-generation contradiction scans and replay family counts add an estimated ~2 s per generation at full scale (not measured at scale).

## 13. Exact steps for Milestone 3 (not authorized here)

> **Stale (Phase 4D.3B.1):** these commands use the rejected token and the format-1 declaration. Use the steps in [phase4d3b1-launch-control-fixes.md](phase4d3b1-launch-control-fixes.md) §11 instead.

1. Independent Astra/Codex review of this document, the code, the frozen packages and the declaration. Optionally, add an external Pons cross-check of the solved packages.
2. Commit this work. The execution digest is unchanged by committing; any reviewer-requested code change changes it and requires refreezing the declaration (and the packages only if their builder changes).
3. With separate authorization, on the same machine and runtime, run:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
D=games/connect4/alphazero_v2/frozen/campaign-declaration.json
T=2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8
C=experiment-output/phase4d3c-alphazero-v2-campaign-<date>      # must not exist
python -m games.connect4.alphazero_v2.campaign preflight --declaration $D
python -m games.connect4.alphazero_v2.campaign run --declaration $D --campaign-dir $C --seed 42 --authorize $T
python -m games.connect4.alphazero_v2.campaign run --declaration $D --campaign-dir $C --seed 314159 --authorize $T
python -m games.connect4.alphazero_v2.campaign final-evaluate --declaration $D --campaign-dir $C --authorize $T
```

   A stopped `run` is resumed by rerunning the same command. Budgets carry over through the ledger. A budget-exhausted run is a final outcome, not a reason to extend.
4. Report every acceptance criterion against the declared thresholds for **both** seeds, with intervals and the overlap-excluded sensitivity. Do not relax thresholds, add seeds or retune after seeing results.

**No research training, learned v2 checkpoint, strength evaluation, API/UI exposure or deployment occurred in Phase 4D.3B.**

