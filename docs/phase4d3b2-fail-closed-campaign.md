# Phase 4D.3B.2 — Fail-closed, non-resumable official campaign

Completed October 5, 2026 on local branch `phase4d3b-alphazero-v2-preflight`, starting from clean HEAD `1f6ae8f`. **No campaign was launched.**

Not done in this phase:

- no research training, learned research checkpoint or strength evaluation;
- no package regeneration, architecture or hyperparameter change;
- no public integration, deployment, commit, push or merge.

Every campaign run in this phase was a tiny synthetic run in a temporary directory.

> **REJECTED / NOT AUTHORIZED (both permanently):**
> `2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8` (Phase 4D.3B) and
> `8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39` (Phase 4D.3B.1, NO GO).
>
> **New frozen declaration token (format 3):**
> **`ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb`**
> It is valid only with this tree's exact execution sources and runtime. Launching still needs separate authorization.

Authorities:

- [4D.3B.1 final launch review](phase4d3b1-final-launch-review.md): NO GO, R1–R5;
- [4D.3B.1 launch-control fixes](phase4d3b1-launch-control-fixes.md), now superseded;
- [launch-readiness review](phase4d3b-launch-readiness-review.md);
- [4D.3B preflight](phase4d3b-alphazero-v2-preflight.md);
- [4D.3 design review](phase4d3-alphazero-v2-review.md).

## Summary

The resumable controller is gone from the official path. It had a journal, journal repair, time leases, crash charging, unit-level evidence reuse and three separate invocations. The replacement has these parts:

- **One owning process.** It holds an exclusive lock. It runs seed 42, then seed 314159, then the development selection, then the sealed final evaluation, all under one declaration.
- **One authoritative file.** `state.json` is replaced atomically on each transition. Its states are linear, and its two terminal states never transition again.
- **Write-once artifacts.** They are official only if a `COMPLETE` state certifies their hashes.
- **Fail-closed endings.** Any abnormal ending makes the campaign `INCOMPLETE`: a crash, kill, reboot, signal, deadline, game ceiling, identity drift or error. A later invocation never repairs, resumes, retries or reuses anything.

| | Before (4D.3B.1) | After (4D.3B.2) |
| --- | --- | --- |
| `launch_control.py` | 634 lines: journal, fold, repair, leases, recovery | 350 lines: atomic state file, lock, monotonic budget |
| `campaign.py` | 1,256 lines: `run --seed`, `final-evaluate`, recovery, evidence replay | 1,144 lines: one `launch` command, in-process evidence |
| Official resume paths | journal repair, `recover_open_attempts`, `load_resume_boundary`, completed-unit reuse, outcome shortcut | none |
| Training source group (18 files) | `22678c4b…9e35` | **unchanged** `22678c4b…9e35` |

Only `campaign.py` and `launch_control.py` changed among the 26 execution files. `generation.py` and every other training-trajectory file are byte-identical.

## 1. Simplified campaign lifecycle

```text
CREATED ─► RUNNING_SEED_42 ─► RUNNING_SEED_314159 ─► DEVELOPMENT_SELECTION_COMPLETE ─► RUNNING_FINAL_EVALUATION ─► COMPLETE
   │              │                    │                          │                               │
   └──────────────┴────────────────────┴──────────────────────────┴───────────────────────────────┴──► INCOMPLETE
```

- **Order.** States are entered only in this order ([`state_sequence`](../games/connect4/alphazero_v2/launch_control.py#L50)). `INCOMPLETE` may follow any non-terminal state. `COMPLETE` and `INCOMPLETE` are terminal: [`CampaignStateFile.transition`](../games/connect4/alphazero_v2/launch_control.py#L244) refuses every transition and counter rewrite after them.
- **Atomic writes.** Each transition writes the whole document to a temporary file, then fsyncs it (plus `F_FULLFSYNC` on macOS), renames it over `state.json`, and fsyncs the directory ([`atomic_replace`](../games/connect4/alphazero_v2/launch_control.py#L146)). A crash leaves the previous or the next complete state.
- **Validation on load.** [`validate`](../games/connect4/alphazero_v2/launch_control.py#L216) rejects a history that is not a prefix of the line, a state that is not the last history entry, and an outcome present without a terminal state (or missing with one).
- **Contents.** `state.json` holds the declaration hash, seeds, owner (pid, host, start time), transition history, the last counter snapshot and the terminal outcome.

[`Campaign.run`](../games/connect4/alphazero_v2/campaign.py#L451) runs the work in this order:

1. **Launch gates.** The constructor applies them before any directory exists: token, declaration validation, runtime configuration, full source/runtime/package/record/checkpoint identity, then import of the whole execution closure.
2. **Fresh directory.** The directory must not exist. The launcher creates it, takes `flock` on `campaign.lock`, writes `declaration.json` once, and creates `state.json` in `CREATED`.
3. **Source snapshot.** Recorded for provenance, then identity is re-verified.
4. **Each seed, in declared order.** The campaign enters `RUNNING_SEED_<seed>`. Generation 0 is written. Each of the 20 generations then runs, in order:
   1. training;
   2. writing inference, resume (diagnostic only) and games once;
   3. diagnostics;
   4. pruning of old resume files;
   5. the scheduled champion check.

   Then comes the development selection (`selection.json`).
5. **`DEVELOPMENT_SELECTION_COMPLETE`.** This transition records both selections.
6. **Sealed final evaluation.** `final/started.json` is written once. Then `RUNNING_FINAL_EVALUATION` records both selections and that file's hash, **before any sealed inference**. Each seed's sealed units follow, and `final/seed-S.json` is written once.
7. **`COMPLETE`.** The terminal record lists the hashes of both `final/seed-S.json` files.

Evidence lives only in the owning process. Nothing is read back from disk to continue. The one exception is re-hashing the generation artifacts this process wrote, before loading them ([`_artifact`](../games/connect4/alphazero_v2/campaign.py#L561)). [`official_results`](../games/connect4/alphazero_v2/campaign.py#L973) returns sealed results only under three conditions:

- `state.json` is `COMPLETE`;
- `declaration.json` matches its hash;
- every certified `final/seed-S.json` matches.

Otherwise it raises `NotOfficialEvidence`. Partial artifacts are forensic only.

## 2. Exact failure semantics

| Event | Where it is detected | Result |
| --- | --- | --- |
| Cooperative stop: SIGINT/SIGTERM | next `check()`, before every unit, search, move and optimizer update | `INCOMPLETE` (reason `signal N`), exit 4 |
| Any limit reached at a start check (`>=`) | `Budget.check`, `Budget.start_game` | `INCOMPLETE`, exit 4 |
| Work that finishes past a limit (`>`) | `completion_violation` before a generation, unit, diagnostic, decision, selection or seed result is accepted | the work is never accepted; `INCOMPLETE`, exit 4 |
| Source or runtime identity differs after launch | `verify_identity` around every generation and every evaluation unit, and before every transition and publication | `IdentityDrift`; `INCOMPLETE`, exit 4 |
| Any other exception, including an artifact hash mismatch or a runner self-check | `run` | `INCOMPLETE` is recorded, then the exception propagates (traceback, nonzero exit) |
| Hard death (crash, `kill -9`, reboot, power loss) | nothing in-process; `state.json` stays non-terminal | the next invocation records `INCOMPLETE` and prints the notice below, exit 4 |
| A second process while the owner lives | `flock` | `CampaignLocked`, exit 2; nothing is written to state or artifacts |
| Existing directory without `state.json` (unrelated, empty, or a pre-4D.3B.2 journal campaign) | constructor | refused, exit 2, nothing written |
| Existing campaign for another declaration | constructor | refused, exit 2; state and artifacts untouched |
| `COMPLETE` or `INCOMPLETE` campaign invoked again | constructor | `TerminalCampaign`, exit 2; state and artifacts untouched |
| `state.json` unreadable or invalid | constructor, `status`, `official_results` | refused as not resumable; never `COMPLETE` |

On a later invocation against an interrupted campaign, the launcher prints exactly:

```text
Existing interrupted official campaign is INCOMPLETE and cannot be resumed.
```

and exits with status 4. Before printing, it makes one durable transition to `INCOMPLETE`, with the reason `interrupted: owner pid <pid> ended while <state>; official campaigns are non-resumable`. Apart from rewriting the pid line in `campaign.lock`, that transition is the only write. The invocation never touches evidence or artifacts, never repairs anything, and never starts work ([`_refuse_existing`](../games/connect4/alphazero_v2/campaign.py#L421)). A power-loss that drops a recent write can only leave an earlier non-terminal state, which becomes `INCOMPLETE`. A lost write can never produce `COMPLETE`, because `COMPLETE` is the last write and `official_results` re-verifies the hashes it certifies.

Exit codes: 0 `COMPLETE`, 2 refused, 4 `INCOMPLETE`. The old 3 ("stopped, rerun to resume") no longer exists.

## 3. Deadline semantics

Time is `time.monotonic()` inside the owning process, measured from construction ([`Budget`](../games/connect4/alphazero_v2/launch_control.py#L272)). Nothing is reconstructed across processes, so there are no leases, no settlement and no crash charging.

| Limit | Declared | Measured as |
| --- | --- | --- |
| Per-seed collection + optimization | 28,800 s (8 h) | time inside `run_generation`, accumulated per seed |
| Whole campaign: training, development evaluation, selection and sealed evaluation | 86,400 s (24 h) | elapsed since the owner started |
| Evaluation games | ceiling 6,400 (6,272 planned) | games reserved before their first move |

- **Start checks (`>=`).** `check()` runs before every generation, evaluation unit, diagnostic, search, move and optimizer update. The existing cooperative checks inside self-play, arena games, tactical searches and calibration are all retained. No new major unit starts after its deadline.
- **Acceptance checks (`>`).** An in-flight primitive may finish past a cooperative check. Its result is accepted only if every limit still holds. Otherwise it is discarded and the campaign ends `INCOMPLETE`.
- **Terminal check.** [`_complete`](../games/connect4/alphazero_v2/campaign.py#L794) takes **one** counter snapshot, checks every limit against that same snapshot, and writes it into the `COMPLETE` record. The authoritative account therefore always satisfies the limits it certifies. The only time not included is the few milliseconds of the atomic write itself.
- **Exhaustion.** Reaching any limit means `INCOMPLETE`, no `PASS`, and no extension.

Phase timings are recorded in `counters.time.phase_seconds`: `setup`, `seed-S:setup`, `seed-S:training`, `seed-S:development` and `final`.

## 4. Why resume is intentionally forbidden

The 4D.3B.1 review showed that each resumable mechanism opened its own gap. Leases undercharged time across restarts. Journal repair could itself be interrupted. Completed units outlived identity refusals. Terminal completion and settlement could disagree. Post-work counters could vanish. Fixing each gap would add more crash windows to the controller.

A non-resumable campaign removes these gaps instead:

- every accepted result was computed in **one** process, under **one** continuously verified identity, within **one** monotonic time account;
- an interruption cannot create free work, reuse evidence from a different runtime, or let a partial sealed result influence any later choice;
- the cost is that an interrupted campaign must be relaunched under a **new** declaration and directory, from scratch.

That cost is accepted for the Milestone-3 scientific run. Generic generation-boundary resume in `generation.py` is unchanged and still tested for development and debugging, but the official launcher never calls it:

- **Static check.** `campaign.py` contains no `load_resume_boundary`, `Journal`, `repair`, `lease`, `run_seed` or `final_evaluate` identifier (test `test_official_launcher_has_no_resume_code_path`).
- **Dynamic check.** A full in-process campaign completes with `load_resume_boundary` patched to fail on any published boundary (`test_normal_campaign_completes_without_any_resume_path`). The only loads are the writer's validation of each new resume file's unpublished temporary copy.
- **Real crash check.** A real crash after a valid, loadable generation-1 resume boundary exists still ends `INCOMPLETE` (`test_valid_resume_boundary_does_not_permit_official_continuation`).

## 5. Reassessment of the five NO-GO findings

### R1: runtime-drifted evidence reuse

**Removed path:** `_unit` returning journaled evidence, and the historical attempt-identity check in `final_evaluate`. Both are deleted. Unit evidence is no longer persisted for reuse, and the constructor never reads evidence.

**Mechanism now:**

- [`_unit`](../games/connect4/alphazero_v2/campaign.py#L524) verifies identity before computing and again after computing, before accepting the evidence.
- Generations, diagnostics, decisions, selections, sealed seed results and every state transition are also gated by `verify_identity`.
- Drift raises `IdentityDrift`, and the campaign becomes terminal `INCOMPLETE`.
- No later invocation can continue it, even after the runtime is restored.

**Reviewer's scenario:** `deterministic_algorithms=False` during the last calibration game.

- The unit's evidence is computed but never accepted (`calibration_game: started 1, completed 0`).
- `final/seed-42.json` is never written and `official_results` refuses.
- Relaunching with the declared runtime restored is refused as `TerminalCampaign`, and the directory is unchanged.

Test: `test_runtime_drift_during_sealed_evaluation_leaves_no_acceptable_evidence`, plus five runtime fields and source drift mid-training.

### R2: interrupted journal repair

**Removed path:** `Journal`, `Journal.repair` and the torn-tail `write_once` side file. The official path has no journal.

**Mechanism now:**

- `state.json` is replaced atomically, so it has no torn tail to repair.
- Unreadable or invalid state is refused as not resumable and never becomes `COMPLETE`.
- The only write a later invocation makes is the single `INCOMPLETE` transition. If that write is interrupted, the state remains non-terminal or becomes `INCOMPLETE`, and the next invocation repeats the same idempotent decision.
- There is no write-once side file that could block it.

Tests: `test_state_validation_rejects_forged_or_damaged_documents`, `test_atomic_replace_leaves_old_or_new_bytes`, `test_existing_directories_are_never_adopted` (torn `state.json`), and the 11-point crash matrix.

### R3: terminal deadline/accounting gaps

**Removed paths:**

- the lease settlement after `campaign_complete`;
- recovery charging a crashed final attempt;
- the existing-outcome shortcut that returned `COMPLETE`.

There is no post-terminal accounting, and the constructor refuses every existing directory before any shortcut could run.

**Mechanism now:** `_complete` takes one snapshot, checks it against every limit, and writes it into `COMPLETE`. The reviewer's two windows now behave as follows:

1. **Delay just before the terminal write.** `clock 9.9 → 10.1` at `before_campaign_complete` yields `INCOMPLETE` with `elapsed_seconds = 10.1`. Exactly 10.0 is accepted as `COMPLETE` with 10.0 recorded.
2. **Crash right after the terminal write.** The written `COMPLETE` already carries the checked account. Nothing settles afterwards, so nothing can disagree. A crash just before the write leaves a non-terminal state, which becomes `INCOMPLETE`.

Tests: `test_overrun_just_before_the_terminal_write_is_incomplete`, `test_complete_is_written_only_with_an_account_within_every_limit`, crash point `before_campaign_complete`.

### R4: lease undercharging

**Removed path:** `LEASE_SECONDS`, `RENEW_BELOW_SECONDS`, `Budget.lease/settle`, `recover_open_attempts` and cross-process time accounting. They are absent from the module, the declaration and the state file. Tests assert their absence by attribute and by code identifier.

**Mechanism now:** Time is one monotonic interval inside the one process that does all the work. An unchecked stall is measured, because the clock keeps running. It is charged at the next check and at every acceptance check. A crash cannot recover allowance, because a crashed campaign can never continue.

Tests: `test_no_lease_or_journal_accounting_exists`, `test_start_checks_refuse_at_limits_and_completion_rechecks_overruns`, `test_no_unit_starts_at_the_deadline`.

### R5: lost training-work counters

**Removed path:** journaled post-work `selfplay_game` and `optimizer_step` records, which fed a durable attempted-work budget used by later invocations.

**Mechanism now:**

- Counters are live, in-process provenance. They are snapshotted into `state.json` at transitions and after each generation, champion check and sealed seed.
- An interrupted collection also counts its in-progress game as attempted.
- If the process dies before a snapshot, the recorded counters are explicitly lower bounds. The campaign is `INCOMPLETE`, and no later process consults them to authorize work.

**Reviewer's window:** a crash right after a completed self-play game. The snapshot shows 0 games (a lower bound; one game physically ran). The next invocation records `INCOMPLETE` and does no work.

Tests: `test_lost_counters_cannot_authorize_continuation`, `test_counters_record_every_unit_of_live_work`.

## 6. Sealed evaluation

- **Selections fixed first.** Before the first sealed unit, `state.json` is already `RUNNING_FINAL_EVALUATION`. Its history entry holds both selections, equal to each seed's `selection.json`, and `final/started.json` exists. This is tested by observing the state at the first `final_tactical_row`.
- **Interruption.** A crash or stop during sealed evaluation, at any point from `RUNNING_FINAL_EVALUATION` to just before `COMPLETE`, ends `INCOMPLETE`. Sealed evaluation is never restarted under this declaration. Every certified-result check refuses, even when both `final/seed-*.json` files were already written.
- **No reselection.** Selections are made once from development evidence. No path re-enters selection after the sealed phase starts, so partial sealed results cannot influence another model or configuration choice.

## 7. Scientific protocol: unchanged

Field by field against the rejected 4D.3B.1 declaration, these are **identical**:

- `config` (every `V2Config` value: 256 games per generation, 256 and 512 simulations, PUCT, replay, update formula, network, optimizer, root noise, temperature schedule);
- `seeds` 42 / 314159, `primary_seed`, `generations` 20;
- `budgets` (28,800 s / 86,400 s / ceiling 6,400 / planned 6,272);
- `evaluation`, `champion` (schedule 5/10/15/20, gate, baselines), `final` (ladder, calibration), `thresholds` and `retention`;
- `packages` (all six hashes), `frozen_records` and `phase4d2f_checkpoint`;
- `runtime_identity`, `name` and `kind`.

No package was regenerated, and the structural verify passed. The training source group hash is unchanged.

## 8. New declaration

`games/connect4/alphazero_v2/frozen/campaign-declaration.json`, regenerated from scratch with `campaign freeze` in the pinned runtime:

**SHA-256 / authorization token: `ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb`**

That hash was confirmed three ways: `shasum -a 256` of the file, the `freeze` output, and an independent in-memory reconstruction from `build_declaration`, the configured runtime, the manifest, the frozen records and `FREEZE_NOTES`, which produced byte-identical output.

Changes from the 4D.3B.1 declaration (all 22 keys retained):

| Field | Change |
| --- | --- |
| `format_version` | 2 → **3**. Format-2 declarations no longer validate. |
| `runtime` | `{threads: 1, deterministic_algorithms: true}`. The `strict_resume_runtime` and `strict_resume_source` flags are removed, because the official campaign never resumes. Validation now requires exactly these two keys. |
| `launch_control` | `connect4-alphazero-v2-launch-control-v2-fail-closed`: state-file format; `resume` ("Forbidden…"); `single_owner`; `states`; `interruption`; `identity`; `deadlines`; `evaluation_games`; `final_evaluation`; `counters`; both rejected tokens. The `journal`, `lease_seconds`, `renew_below_seconds`, `recovery` and `time_accounting` entries are removed. Validation requires equality with the launcher's own constants. |
| `execution_source` | **26 files**, combined `08df122b199eb6eaf6ca5fa8714be89e984b8fa3cb0e2a6c69c0937a84a787ca`. `training` (18 files) `22678c4b1c056f4b2b61a94fdc2eaaa0cd67de884a82c558ab15324406899e35` is **unchanged**. `evaluation_and_launch` (8 files) `f6e2646160f27038333ed2b32bbf339a3560e74d6b158ddd2254cd62209571f5`; only `campaign.py` and `launch_control.py` differ. |
| `notes` | Phase 4D.3B.2, format 3, both rejected tokens, non-resumable rule. |

Bound values carried over unchanged:

- **`runtime_identity`:**
  - CPython 3.11.17;
  - torch 2.10.0 (git `449b1768…`, build-config `ae49aeb8…`);
  - NumPy 1.26.4;
  - `macOS-26.6.2-arm64-arm-64bit`, arm64, Apple M5, CPU capability DEFAULT;
  - intra/inter-op threads 1/1;
  - deterministic true, warn-only false;
  - matmul precision highest, mkldnn true;
  - OMP/MKL/OPENBLAS/VECLIB/NUMEXPR threads 1.
- **`frozen_records`:**
  - `manifest.json` `d1f0d4df…06db`;
  - `exclusions.json` `73c0f5a5…e044`;
  - `build-provenance.json` `09e352db…7a77`.
- **`packages`:**
  - openings-development `3990bc23…2f72`, openings-sealed `d30683aa…34eb`;
  - solved-development `41ffd4df…37b9`, solved-sealed `2c5f70dc…394b`;
  - tactical-development `106cc315…6d9d`, tactical-sealed `cb4e2e1e…6f47`.
- **4D.2f checkpoint:** `78276912bdef5f1b1b53ac6b2ec3ede2eed20a8ebb5796a9bdcbb57e4cb28c51`.

**Preflight on the new declaration** (pinned environment, this machine): `"ok": true`, exit 0. All five gates passed (declaration, thread environment, configured runtime identity, execution source, 4D.2f checkpoint), with 6,272 planned games.

### Old-token rejection

Both tokens are in `launch_control.REJECTED_DECLARATION_TOKENS` and in the new declaration's `launch_control.rejected_declaration_tokens`. The launcher refuses in three places:

- either token as `--authorize`, before any directory exists;
- any declaration file hashing to either token. This was tested with the exact bytes restored from git commits `a663454` and `c302c18`: refused in-process, through `preflight`, and through the CLI `launch` in a fresh interpreter (exit 2, no directory created);
- a format-2 declaration, because the format version is now 3.

The new token is required: another declaration's hash, an arbitrary hash or a re-cased token is refused (`test_the_new_token_is_required`). Mutating any one of the 22 declaration fields changes the hash, so the original token no longer authorizes it (`test_every_declaration_field_is_bound_by_the_token`). The old `run --seed` / `final-evaluate` commands no longer exist.

## 9. Tests

New or rewritten files:

- [`tests/test_alphazero_v2_launch_control.py`](../tests/test_alphazero_v2_launch_control.py): 17 torch-free tests.
- The launcher section of [`tests/test_alphazero_v2_campaign.py`](../tests/test_alphazero_v2_campaign.py). The file has 85 tests in total, including the unchanged evaluation, diagnostics and profiling tests.

All use tiny synthetic configs (2 simulations, 2 games, 2 generations, batch 8), temporary directories and fake clocks.

| Requirement | Tests |
| --- | --- |
| Interrupted official campaign cannot resume | `test_interrupted_official_campaign_can_never_resume` × **11 hard-crash points** in real fresh interpreters (`os._exit`). The points run from `RUNNING_SEED_42` through mid self-play, mid optimization, after a generation write, mid development arena, between seeds, after selection, after `RUNNING_FINAL_EVALUATION`, mid sealed ladder game, mid calibration, and before `COMPLETE`. Each one checks four things: (1) the crash leaves a non-terminal state; (2) the CLI relaunch exits 4, prints the exact notice and records `INCOMPLETE`; (3) the file tree is unchanged apart from `state.json` and the lock file's pid line; (4) a third relaunch exits 2 and certified results are refused. Also `test_second_process_cannot_advance_a_live_campaign`. |
| Valid resume checkpoint does not permit continuation | `test_valid_resume_boundary_does_not_permit_official_continuation` (the boundary loads successfully, yet the relaunch is `INCOMPLETE`); `test_official_launcher_has_no_resume_code_path`; `test_normal_campaign_completes_without_any_resume_path` |
| Process/runtime/source mismatch → `INCOMPLETE` | `test_source_drift_mid_campaign_is_incomplete_and_never_continues`; `test_runtime_drift_mid_campaign_is_incomplete` × 5 fields; `test_runtime_drift_during_sealed_evaluation_leaves_no_acceptable_evidence`. Launch-time refusal: `test_every_runtime_identity_field_refuses_launch` × 17 fields; `test_changed_execution_source_refuses_launch` × 6; fresh-interpreter one-byte edit; packages, records and checkpoint |
| Terminal states never reopen | `test_terminal_states_never_reopen` × 2; `COMPLETE` and `INCOMPLETE` relaunches refused with byte-identical state |
| Deadline → `INCOMPLETE` | `test_deadline_during_development_is_incomplete`, `test_no_unit_starts_at_the_deadline`, `test_training_overrun_is_never_accepted`, `test_last_calibration_overrun_is_incomplete`, `test_overrun_just_before_the_terminal_write_is_incomplete`, `test_complete_is_written_only_with_an_account_within_every_limit`, `test_exact_game_ceiling_completes_and_one_less_is_incomplete`; budget unit tests |
| Sealed interruption → `INCOMPLETE` | crash points after `RUNNING_FINAL_EVALUATION`, mid ladder, mid calibration and before `COMPLETE`; `test_cooperative_stop_during_sealed_evaluation_is_incomplete`; `test_selections_are_fixed_before_any_sealed_inference` |
| Partial evidence cannot be accepted | `test_partial_sealed_results_are_never_official` (both seed results exist, yet the evidence is not official); `test_forged_complete_state_without_matching_results_is_not_official`; every `INCOMPLETE` test calls `official_results` |
| No lease accounting in the official launch | `test_no_lease_or_journal_accounting_exists` (module attributes, `Budget` methods, code identifiers of both modules); declaration test asserts no `lease*` launch-control keys; the normal campaign's directory has no journal |
| Old tokens rejected | `test_both_old_tokens_and_their_declarations_never_authorize` × 2 commits; `test_both_previous_tokens_are_rejected` |
| New token required | `test_the_new_token_is_required`; `test_frozen_declaration_is_the_new_format_3_declaration`; `test_old_multi_command_workflow_no_longer_exists` |
| Normal tiny campaign completes identically to the uninterrupted expected result | `test_uninterrupted_tiny_campaign_completes_identically_to_the_expected_result`. A CLI `launch` of a two-seed campaign completes with exit 0 and certified results. An independent second uninterrupted run has an identical deterministic fingerprint (generation summaries, champion checks, selections, sealed seed results, timings excluded). Every generation's `state_sha256` equals a standalone `GenerationRunner` trajectory with no campaign around it. Also `test_counters_record_every_unit_of_live_work`, `test_signal_stops_the_cli_and_the_campaign_is_incomplete`. |

## 10. Verification

| Check | Environment | Result |
| --- | --- | --- |
| AlphaZero v2 suites: `test_alphazero_v2.py` 138, `_evaluation_core.py` 41, `_campaign.py` 85, `_launch_control.py` 17. Also engine 23, MCTS 71, neural MCTS 175, and every DQN, neural self-play/symmetry/root-noise/value-target/anchoring suite | `/tmp/board-game-phase4c1-venv` (CPython 3.11.17, torch 2.10.0, NumPy 1.26.4), pinned thread env, `PYTHONDONTWRITEBYTECODE=1` | **1,201 passed, 3 deselected**, 235 s. Deselected: `phase4d2f_adapter_is_hash_pinned` (avoids running the retained learned model, as in prior reviews) and the two API-startup isolation gates. The five API modules were not collected (`--ignore`), because this venv lacks `dotenv`, a known limitation; they pass in `.venv` (next rows). |
| Full backend `tests/` | `.venv` (no torch) | **468 passed, 15 skipped** (torch-dependent modules). Includes the 17 torch-free launch-control tests. |
| API import isolation (`test_neural_mcts_import_isolation.py`, `test_dqn_import_isolation.py`) | `.venv` | **2 passed** |
| Solver and frozen packages: `packages verify` | torch venv | exit 0. All six packages verified structurally; package, manifest, exclusions and build-provenance bytes unchanged (`git diff` touches only `campaign-declaration.json` in `frozen/`) |
| Re-freeze from scratch | pinned env | token `ed641aea…0bbb`; independent reconstruction byte-identical; preflight ok, exit 0 |
| `git diff --check` | — | clean |

Measured overhead: one identity re-verification (26 file hashes plus the live runtime query) takes about 0.55 ms. At roughly two per evaluation unit, that adds only a few seconds over a full campaign.

## 11. Remaining limitations and deferred items

1. **An interruption costs the whole campaign.** This is by design. Any crash, reboot, signal or limit means `INCOMPLETE`, and recovery requires a new declaration and a new directory. Plan the launch for an uninterrupted 24 h window on a machine that will not reboot or update. The command below wraps the launch in `caffeinate -i` to prevent idle sleep.
2. **System sleep is not charged.** On macOS, Python's `time.monotonic()` does not advance while the machine sleeps. No computation happens during sleep, so the budget measures active owner time, consistent with the earlier "active seconds" semantics. `caffeinate -i` avoids the question.
3. **Durability.** Writes use fsync plus `F_FULLFSYNC` where supported, atomic rename and hard links. A lost write can only produce an earlier non-terminal state, and therefore `INCOMPLETE`, never a false `COMPLETE`.
4. **Identity scope** is unchanged from 4D.3B.1. Versions and build metadata are bound, not every binary. The CPU model is not a unique machine identifier. The bound interpreter lives under `/tmp`, and macOS may clear `/tmp` on reboot. Recreating the venv with the same wheels reproduces the identity fields, and the preflight is the gate.
5. **Unchanged deferred items:**
   - selected-model 512-simulation p95 latency, to be measured after training and before public integration;
   - supplementary searched-root-value diagnostic;
   - the `exhaustive_action_values` terminal guard.

   No search, oracle or metric change was made.
6. **Historical mutation harness.** `research/phase4d3b1-mutation/mutate_launch.py` targets the deleted 4D.3B.1 controller and no longer applies. No new mutation sweep was run in this phase.
7. **No real-scale run.** All lifecycle evidence comes from tiny synthetic campaigns.

## 12. Launch command (do not run without separate authorization)

This command **was not executed**. It requires separate authorization. Commit this work first: Git state is provenance only and does not change the token. Do not edit any execution-closure file afterwards.

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/tmp/board-game-phase4c1-venv/bin/python
D=games/connect4/alphazero_v2/frozen/campaign-declaration.json
T=ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb
$PY -m games.connect4.alphazero_v2.campaign preflight --declaration $D     # must print "ok": true and exit 0
caffeinate -i $PY -m games.connect4.alphazero_v2.campaign launch --declaration $D \
    --campaign-dir experiment-output/phase4d3c-alphazero-v2-official-campaign --authorize $T
```

- The campaign directory **must not exist**. The launcher creates it and refuses any existing directory.
- One invocation runs both seeds, the development selection and the sealed final evaluation.
- Exit codes:
  - **0:** `COMPLETE`; read results only through `official_results`;
  - **4:** `INCOMPLETE`, final;
  - **2:** refused.
- **Never rerun to resume.** A rerun on the same directory only reports `INCOMPLETE`.
- Status at any time is read-only:

  ```sh
  $PY -m games.connect4.alphazero_v2.campaign status --campaign-dir experiment-output/phase4d3c-alphazero-v2-official-campaign
  ```
