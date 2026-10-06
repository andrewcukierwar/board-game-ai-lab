# Phase 4D.3B.1 focused final launch review

**Verdict: NO GO**

> **Status:** the token `8a8a52b100…5e39` reviewed here is **REJECTED / NOT AUTHORIZED**, as are the Phase 4D.3B.2 token `ed641aea…0bbb` and the Phase 4D.3B.3 token `34b4d899…c390`. The current declaration is described in [phase4d3b4-complete-schema-fix.md](phase4d3b4-complete-schema-fix.md).

The frozen declaration `8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39` is **not approved** for the Milestone-3 campaign. Its bytes and identities are correct, and the generation transaction fixes the previous duplicate inference-registration inconsistency. However, the five launch-blocking findings below prevent the required identity, recovery, deadline and accounting guarantees.

Reviewed October 5, 2026, local HEAD `c302c181c4ab96d0ab4e3c917c56a95ccb5bc526`, initially clean. Read the prior launch-readiness review, launch-control fixes, superseded preflight notices, implementation, relevant tests, and the 42-defect mutation harness/results. This review adds only this document. No source changes, research training, research campaign launch, frozen-package regeneration, retained learned-model evaluation, commit, push, merge or deployment occurred. Synthetic test games/updates and test-only fixtures used temporary directories; no inference ran on the actual frozen sealed positions. The candidate's constructor was exercised only through PRELAUNCH, with no attempt or work started.

## Launch-blocking findings

### R1 — Runtime-drifted sealed evidence survives refusal and is accepted on resume (B1; final-resume contract)

**Locations:** `campaign.py:462–483` (`_unit`), `:833–848` (historical identity check), `:867–875` (final seed completion).

`_unit` persists `unit_complete` without verifying execution identity. The later final-seed identity check refuses a drifted runtime, but leaves those completed units reusable. Historical verification checks the identity recorded at **attempt start**, not the identity under which its evidence was computed. The next invocation configures the declared runtime and reuses that evidence.

**Independent reproduction:** completed a tiny synthetic seed under the real pinned runtime. Immediately before the actual final calibration call, changed `torch.use_deterministic_algorithms(False)` and ran calibration. Final publication raised `LaunchRefused` naming `deterministic_algorithms`, but the journal already contained a completed calibration unit. Reopened normally, which restored deterministic algorithms. Replaced calibration with an assertion that it must never execute again. Final evaluation returned `completed` and published `COMPLETE`, using the prior drifted evidence. No declaration, source file, or historical identity record was altered.

This is not merely a missing warning: evidence computed outside the authorized identity becomes accepted sealed evidence. Similar exposure exists for development units saved before a later identity refusal. `_diagnostics` and `_select` also do not independently call `verify_identity`, despite the handoff's every-boundary claim.

**Required for launch:** completed reusable evidence must certify the bound identity, and an identity failure must not leave uncertified evidence eligible for later successful publication. Preserve the prohibition on reselection and rerunning already accepted sealed evidence.

### R2 — A second crash during torn-tail recovery permanently blocks automatic recovery (B2)

**Location:** `launch_control.py:213–227`, `Journal.repair`.

Repair first preserves the damaged suffix with `write_once(journal.jsonl.torn-<offset>, ...)`, then truncates the journal. If the process dies after preservation but before truncation, the next repair attempts the same write-once path and raises `FileExistsError`. It never reaches truncation or normal recovery.

**Independent reproduction:** created a valid checksummed journal with an incomplete trailing record; interrupted immediately after the real `write_once` had durably saved that suffix; reopened from disk. The next `repair()` raised `FileExistsError: Refusing to overwrite ...journal.jsonl.torn-<offset>`. The original tail remained unrepaired. This models a process crash, not power loss or malicious modification.

The ordinary torn-tail test and 30 crash scenarios do not interrupt recovery itself. The campaign can still require manual intervention after an allowed interruption.

**Required for launch:** recovery must itself tolerate interruption, including recognition and verification of an already preserved identical torn suffix.

### R3 — Terminal success is not consistent with the final durable deadline/accounting transitions (B3/B4)

**Locations:** `campaign.py:877–883` (`campaign_complete`), `:444–450` (later settlement), `:833–836` (outcome shortcut); `launch_control.py:322–325`, `:605–627` (crash charge).

Two independently reproduced windows remain:

1. **Before the terminal commit:** with a 10-second synthetic campaign cap, the last calibration unit finished at fake time 9.9. The last completion check passed. Advanced the clock by only 0.2 seconds at the existing `before_campaign_complete` transition hook. The journal recorded `campaign_complete`, the command returned `completed`, and both `outcome.json` and settled `charged_seconds=10.1` reported success beyond the 10-second cap. The hook models scheduling/serialization/publication latency after the last check; no long operation or lease expiry is required.
2. **After the terminal commit, before settlement:** in a fresh subprocess, called the real journal append for `campaign_complete`, then used `os._exit(94)` before `attempt_end`. On reopening, recovery charged the outstanding 300-second lease against the synthetic 10-second cap. `final_evaluate()` took its existing-outcome shortcut and published `COMPLETE` anyway. The same inconsistency applies near the actual 86,400-second limit whenever the final unclosed lease exceeds the remaining allowance.

The second case is conservative charging rather than proof that physical computation exceeded its cap. It still violates the declared rule that successful completion respects **every charged limit**. The authoritative terminal outcome and authoritative resource account disagree, and recovery publishes that disagreement as success.

**Required for launch:** reconcile terminal completion with durable final charges and deadline checks, including crashes immediately before and after the completion record. An over-limit authoritative account cannot bypass enforcement through the terminal-outcome shortcut.

### R4 — Finite leases do not guarantee no undercharge after an unchecked interval (B4)

**Locations:** `launch_control.py:542–555` (`lease`), `:571–580` (`check`), `:605–627` (recovery).

**Independent fake-clock reproduction:** reserve campaign and training time through 300 seconds at t=0. Check at t=150: exactly 150 seconds remain, so no renewal occurs. Let an unchecked operation/stall last 151 seconds and simulate a hard crash at t=301. Reopening and recovering charges **300 seconds** of both campaign and training time. There is no expiry refusal or unresolved-time state. Larger excess intervals disappear in the same way.

The fixes document explicitly discloses this assumption in §10.3. Measured typical operation durations do not enforce it. There is no mechanism proving the 150-second maximum for arbitrary interruption, a stall, or an unchecked sequence of operations. Consequently the requested no-undercharge guarantee is not met; repeated restarts can recover consumed allowance. This finding does not depend on the separate disclosed macOS power-loss/fsync limitation.

**Required for launch:** uncertain elapsed consumption must remain conservatively charged or prevent continuation as successful within the original limits. A measured expected operation duration cannot establish the required hard bound.

### R5 — Training work can disappear before its post-work accounting callback (B4)

**Locations:** `generation.py:208–224`, `selfplay.py:44–49`, `campaign.py:685–693`; `launch_control.py:370–378`, `:484–487`.

Self-play games and optimizer updates are journaled **after** execution. Unlike evaluation games, they have no durable per-unit begin/reservation. Their counters therefore omit work when a crash occurs between execution and the callback.

**Independent real-process reproductions using tiny settings:**

- Wrapped the real `V2Trainer.step`; after it returned with `steps == 1`, wrote an external test witness and called `os._exit(92)` before `progress("update")`. Recovery correctly discarded the generation, but reported `optimizer_steps_attempted == 0`.
- Wrapped the real self-play `play_game`; after a complete **19-ply** game returned, wrote a witness and called `os._exit(93)` before `on_game`. Recovery reported `selfplay_games_completed == 0` and `training_plies_completed == 0`.

These are completed physical operations, not invented journal edits. Replaying the discarded generation correctly restores the accepted trajectory, but its earlier physical work disappears from the purported attempted-work audit. Partial games have no per-game attempt record either. The existing crash matrix stops at `progress:game` and `progress:update`, **after** the missing window has already closed.

Time remains conservatively leased for these short examples; this is not a claim that the committed optimizer state is corrupted or that their wall time is free. It is a failure of the explicitly required durable training-resource accounting and distinction between attempted, completed and accepted work.

**Required for launch:** retain durable accounting for begun/uncertain training operations and their restart costs, without treating them as accepted examples or updates.

## Source, runtime and declaration verification

Independently read and hashed each declared source, package, frozen record and the retained checkpoint. Reconstructed the declaration **in memory**, using the current defaults, actual configured runtime, manifest, frozen records and `FREEZE_NOTES`, then serialized with `json.dumps(..., indent=1, sort_keys=True, allow_nan=False) + "\n"`. The bytes were identical to the frozen file; nothing was regenerated on disk.

```text
Declaration SHA-256
8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39

26-file execution SHA-256
86a4b8502b1fb708e3c3e97614985dad98bc71ab5cbe46a236f5550874855886

18-file training group
22678c4b1c056f4b2b61a94fdc2eaaa0cd67de884a82c558ab15324406899e35

8-file evaluation_and_launch group
4c32d8e42049c2ee280552557164deae88c76978f63cc087333b3c6385f77bda
```

The inventory is:

| Group | Files |
| --- | --- |
| Training: v2 modules | `__init__.py`, `artifacts.py`, `config.py`, `data.py`, `diagnostics.py`, `generation.py`, `network.py`, `oracle.py`, `provenance.py`, `search.py`, `selfplay.py`, `training.py`, all under `games/connect4/alphazero_v2/` |
| Training: reused modules | `games/connect4/__init__.py`, `games/connect4/agents/mcts_agent.py`, `games/connect4/agents/mcts_nn_agent.py`, `games/connect4/board.py`, `games/connect4/connect4.py`, `games/connect4/neural_mcts.py` |
| Evaluation/launch | `arena.py`, `campaign.py`, `evaluation.py`, `launch_control.py`, `packages.py`, `profiling.py`, `reference_negamax.py`, `statistics.py`, all under `games/connect4/alphazero_v2/` |

After importing the execution closure in a separate interpreter, all 26 loaded repository implementation modules were covered, with no extra loaded repository source outside the inventory. Inspected lazy opponent imports are included. The fresh-process training/load test also passed and found only training-group modules. The existing fresh-interpreter copied-tree tests rejected one-byte source edits, while docs/tests/non-closure edits remained permitted. Git commit, dirty status and snapshots are provenance; content identity remains independently mandatory.

All **17 runtime fields** matched an independently assembled runtime dictionary:

| Field(s) | Bound value |
| --- | --- |
| `python_implementation`, `python` | CPython, 3.11.17 |
| `torch` | 2.10.0 |
| `torch_git_version` | `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` |
| `torch_build_sha256` | `ae49aeb898c01a3cd7bdb6311dce7ccea64c05a02c023ba09c707938191c6fea` |
| `numpy` | 1.26.4 |
| `platform`, `machine` | `macOS-26.6.2-arm64-arm-64bit`, `arm64` |
| `cpu_model`, `cpu_capability` | Apple M5, DEFAULT |
| `intra_op_threads`, `inter_op_threads` | 1, 1 |
| `deterministic_algorithms`, `deterministic_warn_only` | true, false |
| `float32_matmul_precision`, `mkldnn_enabled` | highest, true |
| `thread_environment` | OMP, MKL, OPENBLAS, VECLIB_MAXIMUM and NUMEXPR thread variables all `1` |

This binds the OS version/platform string and torch build metadata, not every OS/library binary or a unique physical CPU. Static metadata is cached; mutable torch/thread settings are queried live. Missing declared keys fail exact-key validation; missing current keys fail union comparison; unavailable `None` values fail closed. The 17-field launch/resume rejection tests passed. These correct entry gates do not close R1's saved-evidence hole.

The token binds all 22 top-level declaration fields: format/version, name/kind, seeds/primary seed, complete scientific config and generation count, budgets, runtime settings and concrete identity, evaluation settings, champion schedule/gate/baselines, final ladder/calibration, packages, thresholds, retention, retained checkpoint, notes, execution identities, frozen records and launch-control semantics. Source hashes also bind constants not repeated in JSON, including RNG domains, PUCT and statistics implementations. Package bytes bind their rows, labels and metadata. No missing current campaign input was found in this inventory; the defects are in enforcement and state transitions.

Verified separately:

- All six package hashes and all three frozen-record hashes match. The retained checkpoint hash is `78276912bdef5f1b1b53ac6b2ec3ede2eed20a8ebb5796a9bdcbb57e4cb28c51`.
- Compared the previous declaration restored with `git show a663454:games/connect4/alphazero_v2/frozen/campaign-declaration.json`: among its original fields, only `format_version` and `notes` changed. Scientific settings, budgets, seeds and package identities are unchanged.
- Old authorization token `2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8` is refused before creating a campaign directory. Restored old declaration bytes are also refused, including when supplied with the new token. Existing CLI rejection tests passed.
- New declaration preflight exited 0 with every gate passing and 6,272 planned evaluation games. Independent PRELAUNCH construction produced only `campaign_created`, no attempts, units or seed work, in a discarded temporary directory.
- An appended single whitespace byte to a copy of the actual candidate declaration, with its bound files accessible, was refused for token mismatch before directory creation. Existing tests additionally exercised every top-level declaration field and representative source, package, frozen-record and checkpoint changes.

## State-machine and budget audit

The generation repair is substantive. Inference, resume and archived games are staged and hashed; the single durable `generation_committed` record is the official boundary. Recovery verifies or installs that record's artifacts. Uncommitted staging is never adopted, and each attempt uses a distinct staging path. There is no longer a separate inference-registration append to duplicate. Completed generation numbers cannot be committed twice, strict resume checks their state/hash, and mandatory diagnostics and champion checks are finished before the next generation. The prior latent `_inference_record` inconsistency is removed at its cause.

Pruning is journal-first and hash checked. Authoritative views are write-once and verified against journal content; `status.json` is explicitly replaceable. The journal verifies sequence numbers and per-record checksums and refuses internal corruption. The `flock` tests establish exclusion both within and across processes and release on death. These mechanisms passed the 30 fresh-process crash scenarios, including generation commit/install, diagnostics/pruning, champion decision/selection, seed transition, final units and final seed publication. R2 concerns a missing recovery-interruption scenario.

The fold is not, by itself, a complete protocol validator: caller sequencing supplies additional constraints. In the inspected campaign paths, `_finish_pending` preserves champion order, `_select` checks all required generations/diagnostics/decisions, and final evaluation requires every declared seed selection. A selected seed cannot run again. The two declared seeds remain separate, and there is no best-seed substitution.

| Resource | Durable treatment and result |
| --- | --- |
| Generations | Begin/commit/discard records distinguish attempted and accepted generations; incomplete generations restart from the last committed boundary. |
| Training games, plies, optimizer work | Accepted totals derive from committed summaries. Post-work progress records count discarded work only when they reached disk; R5 covers the missing windows. |
| Development, final ladder, NN-only and calibration games | A fsynced `unit_begin` permanently charges one slot before play. Abandonment/restart keeps the earlier charge and reserves a new one. |
| Position rows | Begin/complete/abandon records; incomplete rows are not evidence. Completed rows return journaled evidence without inference. |
| Evaluation ceiling | 6,272 planned, 6,400 charged slots. Exactly reaching the ceiling permits non-game work; needing another game yields terminal INCOMPLETE. |
| Time | 28,800 training seconds per seed, 86,400 combined active seconds. Clean attempts settle measured time; hard interruptions retain their last lease. R3/R4 qualify enforcement. |

Before searches, arena moves and optimizer updates, checks refuse starts at `>=` the time limit. Final calibration now receives its callback. Last-update and last-calibration overruns, cooperative stops, exact game-ceiling completion and lost-slot exhaustion all passed existing tests. Normal exhaustion produces terminal `INCOMPLETE`, not PASS. The launcher uses COMPLETE for execution completeness, with scientific acceptance separately reported. R3 demonstrates exceptions at terminal durable transitions.

Downtime between attempts is excluded by starting a new monotonic interval and adding prior durable charges. Ordinary crashes do not reset that history. A lease normally overcharges by roughly 150–300 seconds, at most about 300 seconds per crashed attempt; this is documented. A live status reservation can decrease when a clean attempt settles to measured time, so the displayed reservation is not itself a numerically monotonic consumption counter. Evaluation reservations and completed evidence counts are monotonic. R4 identifies why the finite reservation is not an unconditional upper bound on consumed time.

## Final-evaluation resume and deferred items

Both selections and model hashes are fixed in `final_started` before sealed inference. Declaration/source binding fixes iteration order, opponents, simulations, tie seeds, calibration indexing and stopping rules. Resumption uses these same selections, refuses mismatches, visits the same ordered units and skips completed ones. It does not use scores to select a different model, reorder work, alter settings or stop early for performance. Completed games/rows are not rerun; an interrupted unit restarts with its prior game charge retained. The 30-scenario comparison verified identical model/state, decisions and deterministic evidence against uninterrupted tiny runs, excluding timings/attempt metadata.

Thus the new resume design preserves the scientific selection barrier on its ordinary path. It cannot yet be approved because R1 allows evidence from an unauthorized live runtime, and R3/R4 undermine completion within the original charged budget.

The three requested nonblockers remain deferred:

- **Selected-model 512-simulation p95 latency:** measure after training and before public integration, on the selected artifact under the stated readiness procedure. The untrained measurements do not establish learned-model readiness. Its absence is not a launch blocker in this focused review.
- **Searched-root-value diagnostic:** supplementary; raw scalar-head value evaluation remains the frozen primary metric. No search or metric change is requested here.
- **`exhaustive_action_values` terminal guard:** all current package callers are constrained to legal nonterminal candidates or verified frozen rows; evaluation consumes frozen labels rather than calling this helper. Traced candidate filtering, mirror construction, cross-check and verification call sites. Independently replayed all **2,100** frozen histories as legal and nonterminal and reran structural verification. The builder/oracle/reference source hashes and frozen bytes are unchanged. The generic helper guard remains outside this campaign's blocking issues.

## Verification evidence and limits

| Check newly run | Result |
| --- | --- |
| Four v2 suites: core, evaluation/solver, campaign, launch control | **287 passed, 1 deselected**, 185.83 s; includes all 30 hard-crash scenarios |
| Engine, standalone MCTS, neural MCTS suites | **269 passed**, 1.97 s |
| Independent fault probes | **7 distinct scenarios reproduced**: R1; R2; both R3 windows; R4; optimizer and self-play variants of R5 |
| Declaration reconstruction, independent hashes/runtime, module inventory | Matched candidate bytes, 26 sources and 17 runtime fields |
| Frozen packages | Structural verification passed; all 2,100 histories independently legal/nonterminal; no package regeneration or new sealed inference |
| Candidate/old-token boundary checks | Candidate PRELAUNCH/PREFLIGHT only; rejected token/old bytes/one-byte candidate mutation refused as described |

Tests used `/tmp/board-game-phase4c1-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1` and all five thread variables pinned to 1. The deselected test was `phase4d2f_adapter_is_hash_pinned`, avoiding inference with the retained learned model. Synthetic fixtures and discarded updates are test evidence, not research training or a learned-strength evaluation.

Independent scratch harnesses are `/tmp/phase4d3b1_independent_review.py`, `/tmp/phase4d3b1_identity_review.py` and `/tmp/phase4d3b1_token_boundary_review.py`; results use `/tmp/phase4d3b1-*-results.*`, with the main suite log at `/tmp/phase4d3b1-review-tests.log`. The reproduction descriptions above preserve the pertinent transitions and observed outcomes without making scratch artifacts part of the declaration.

Reviewed the 42 recorded mutations and their first-failure results; did **not** run the source-rewriting mutation harness. Its 42 kills establish sensitivity to those edits, not coverage of the additional crash windows, drifted evidence reuse or lease assumption. Architecture, hyperparameters, accepted Milestone-2 estimators and frozen labels were not reopened. Approval remains withheld solely for R1–R5.
