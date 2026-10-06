# Phase 4D.3B.3 — Final terminal-commit fix

Completed October 6, 2026 on local branch `phase4d3b-alphazero-v2-preflight`, starting from clean HEAD `8715b38`. **No campaign was launched.**

This phase fixes exactly the two blockers in the [Phase 4D.3B.2 final launch review](phase4d3b2-final-launch-review.md) (NO GO):

- **F1:** a late signal or a deadline crossing during terminal publication could still publish `COMPLETE`.
- **F2:** a publication exception could overwrite a visible `COMPLETE` with `INCOMPLETE`.

Nothing else was redesigned. Architecture, training, search, seeds, packages, solver, evaluation, thresholds, budgets and the scientific protocol are unchanged. Only `campaign.py` and `launch_control.py` changed among the 26 execution-source files. The training source group hash is unchanged.

Every campaign run in this phase was a tiny synthetic run in a temporary directory, driven by fault injection and fake clocks. There was no research training, no package regeneration and no commit.

> **REJECTED / NOT AUTHORIZED (all permanently):**
> `2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8` (Phase 4D.3B),
> `8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39` (Phase 4D.3B.1, NO GO) and
> `ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb` (Phase 4D.3B.2, NO GO).
>
> **New frozen declaration token (format 3):**
> **`34b4d899639d59d1bd4408ecc505b7eb20cb4d9c2fdc12095402c246887ac390`**
> It is valid only with this tree's exact execution sources and runtime. Launching still needs separate authorization.

## 1. Reproduction of both NO-GO findings

Before any edit, three probes ran against unmodified HEAD `8715b38`. Each one copies the review's method and **asserts the violation**, so a passing probe means the defect is present. **All three passed: 3 passed in 19.8 s.**

| Probe | Injection | Observed on `8715b38` |
| --- | --- | --- |
| F1, signal | Real CLI `main` on a tiny declaration. `launch_control.os.replace` is wrapped: when the temporary document holds `COMPLETE`, it confirms the visible state is `RUNNING_FINAL_EVALUATION`, sends `os.kill(getpid(), SIGTERM)`, then runs the real rename. | The CLI's installed handler ran. Exit **0**, `state.json` `COMPLETE`, and `official_results` returned the results. |
| F1, deadline | `campaign_seconds = 10`, fake monotonic clock: **9.9** at `before_campaign_complete`, **10.1** immediately before the terminal rename. | Status `completed`; `COMPLETE` recorded elapsed **9.9**; `official_results` accepted. |
| F2 | One `OSError` at the directory fsync immediately after the terminal rename. Inside that call, the probe reads the state and calls `official_results`. | Inside the call, readers saw `COMPLETE` and accepted results for seed 42. Then the error handler wrote `INCOMPLETE`, and the new history had **no** `COMPLETE` entry. Visible sequence: `RUNNING_FINAL_EVALUATION → COMPLETE → INCOMPLETE`. |

The same three probes on the fixed tree give:

- **F2:** the probe fails (`DID NOT RAISE OSError`). The owner adopts the visible `COMPLETE` and reports `completed`.
- **F1:** both probes still end `COMPLETE`. **This is the specified behaviour, not a missed fix.** In both probes the signal or the late clock reading arrives *after* the completion barrier (§3). The barrier is now the commitment point. A stop handled before it prevents `COMPLETE`; anything after it cannot change the outcome.

  What was wrong at `8715b38` is that the handler **accepted** the stop while the authoritative state was non-terminal (`budget.stop_reason == 'signal 15'`), and `COMPLETE` was certified anyway. There was no defined endpoint, and acceptance never re-checked time. The retained regression tests assert those invariants directly and fail on the old code (§9).

## 2. Exact state-machine fix

```text
CREATED ─► RUNNING_SEED_42 ─► RUNNING_SEED_314159 ─► DEVELOPMENT_SELECTION_COMPLETE ─► RUNNING_FINAL_EVALUATION ─► COMPLETE
   │              │                    │                          │                               │
   └──────────────┴────────────────────┴──────────────────────────┴───────────────────────────────┴──► INCOMPLETE
                                         COMPLETE and INCOMPLETE: no outgoing edge
```

The line is unchanged. The change is in how [`CampaignStateFile`](../games/connect4/alphazero_v2/launch_control.py) writes it:

| API | Rule |
| --- | --- |
| [`transition`](../games/connect4/alphazero_v2/launch_control.py#L305) | Only the next **running** state. A terminal target or an `outcome` argument is refused. |
| [`terminate`](../games/connect4/alphazero_v2/launch_control.py#L318) | **The one terminal-state transition.** `COMPLETE` only from `RUNNING_FINAL_EVALUATION`; `INCOMPLETE` from any non-terminal state. `outcome.status` must name the state. Refused from a terminal state. |
| [`mark_incomplete`](../games/connect4/alphazero_v2/launch_control.py#L330) | **Conditional.** Re-reads the visible file first. If it is already terminal (`COMPLETE` or `INCOMPLETE`), nothing is written and that state is returned. Otherwise `INCOMPLETE` is built from the **visible** document, never from a stale in-memory copy. |
| [`_write`](../games/connect4/alphazero_v2/launch_control.py#L275) | Compare before replace. Refuses (`StateCorrupt`) unless `state.json` still holds exactly the bytes this object last wrote or read. If the atomic replace raises, the object [`refresh`](../games/connect4/alphazero_v2/launch_control.py#L266)es to whatever is now visible (old or new) before re-raising. |
| [`validate`](../games/connect4/alphazero_v2/launch_control.py#L241) | Also requires a terminal `outcome.status` to equal the state. |
| [`atomic_replace`](../games/connect4/alphazero_v2/launch_control.py#L153) | Removes its temporary file if it fails before the rename. A stray temporary `COMPLETE` document is never left behind. |

Every error and interruption path uses `mark_incomplete`:

- [`Campaign._incomplete`](../games/connect4/alphazero_v2/campaign.py#L507), reached from every exception and stop in [`run`](../games/connect4/alphazero_v2/campaign.py#L468);
- [`_refuse_existing`](../games/connect4/alphazero_v2/campaign.py#L438), for a dead owner.

The signal handler writes nothing. Nothing replaces `state.json` with `INCOMPLETE` unconditionally. `run` reports the **visible** terminal state. An exception raised after `COMPLETE` is visible goes into `post_commit_errors`; it is not re-raised, and the status stays `completed`.

F2's mechanism is gone on two counts:

1. after the failed directory fsync, the owner's object holds the visible `COMPLETE`;
2. even a genuinely stale object (tested) re-reads before acting, and its writes are refused.

`STATE_FORMAT` is now `...-state-v2`, because a `COMPLETE` outcome now carries a completion record. `LAUNCH_CONTROL_VERSION` is `...-v3-terminal-commit`.

## 3. Completion-barrier semantics

[`Campaign._complete`](../games/connect4/alphazero_v2/campaign.py#L816) runs one explicit sequence. **Before the barrier**, all of the following must already hold, or the campaign ends `INCOMPLETE`:

| Condition | Where it is established |
| --- | --- |
| Both seed runs complete, with every generation | each seed's `RUNNING_SEED_*` phase; re-checked by `account_problems` (`generations_completed == generations` per seed) |
| Development selection complete | `DEVELOPMENT_SELECTION_COMPLETE`; `_complete` requires the state to be `RUNNING_FINAL_EVALUATION` |
| Sealed/final evaluation complete; no unit started but unfinished | every seed's `final/seed-S.json` published; `account_problems` requires `started == completed` for every unit kind |
| Evidence and result files fully written and durable | `write_once`: temporary file, fsync plus `F_FULLFSYNC`, hard link, directory fsync (the existing artifact contract) |
| Files required for acceptance validated | [`certified_evidence_problems`](../games/connect4/alphazero_v2/campaign.py#L1046): `declaration.json` hash and token not rejected; `final/started.json` hash; every seed result's hash, `seed` and `completeness == "complete"`. This is the same function `official_results` uses. |
| Source and runtime identity match | `verify_identity("campaign complete")` |
| Evaluation-game ceiling, per-seed budgets, overall budget | at the barrier, against its single reading |
| No stop latched; no handled termination signal | at the barrier, after the latch |

**The barrier:** [`Budget.cross_completion_barrier`](../games/connect4/alphazero_v2/launch_control.py#L440).

1. It records **one** monotonic reading (`barrier_time`) **before** it reads the stop flag. That reading is the **authoritative endpoint** for all campaign budget accounting. From then on, `elapsed()` is frozen at it.
2. It then decides eligibility from that reading alone: stop flag, `elapsed`, per-seed training seconds, and games started.
3. It runs once. A refused barrier is never retried.

**After the barrier**, three things happen:

- the `COMPLETE` document is assembled in memory: the counters, plus a `completion` record holding the endpoint, training seconds, games started and the declared limits;
- the atomic `COMPLETE` write runs ([`_commit_complete`](../games/connect4/alphazero_v2/campaign.py#L850));
- two no-op test hooks run (`after_completion_barrier`, `after_complete`).

There is no computation, model selection, evaluation or evidence publication after the barrier.

> **Semantics.** The scientific campaign is completed at the successful completion barrier. The subsequent atomic `COMPLETE` publication certifies already-completed work.

## 4. Signal semantics

[`install_stop_handlers`](../games/connect4/alphazero_v2/campaign.py#L1245) is used by the CLI. It installs SIGINT and SIGTERM handlers that call [`Budget.request_stop`](../games/connect4/alphazero_v2/launch_control.py#L379). The handler **only records**. It never raises, never writes state and never touches `COMPLETE`.

| When the handler runs | Effect |
| --- | --- |
| During work | The existing cooperative stop. The next check raises, and the campaign ends `INCOMPLETE`. |
| Before the barrier's latch, including inside the latch statement itself | `stop_reason` is set. The barrier reads it after latching, so the campaign ends `INCOMPLETE` with reason `signal N`. |
| After the latch (during the terminal commit, or after `COMPLETE`) | Appended to `late_stop_requests`. `stop_reason` stays unset and the outcome is unchanged. The CLI notes it on stderr. |

**Why this mechanism and not signal masking.**

- CPython runs a Python-level handler on the **main thread**, between bytecodes. It does so whichever OS thread actually received the signal.
- A process-directed SIGTERM can be delivered to any thread that has not blocked it, including the torch and OpenMP worker threads.
- `pthread_sigmask` on the main thread alone therefore cannot keep a handler out of the commit window.

The latch-then-read ordering needs no masking. A handler runs atomically with respect to main-thread bytecode, so every handled signal falls on exactly one side of the latch. No signal is masked anywhere, including around training and evaluation.

The commit cannot be interrupted into an ambiguous transition, for three reasons:

- the handler does not raise;
- a non-signal exception during the write is resolved by re-reading the visible state (§6);
- the handler never writes, so it cannot overwrite `COMPLETE`.

## 5. Deadline semantics

Boundary semantics are unchanged and now stated explicitly in the declaration's `launch_control.deadlines`:

| Check | Comparison | Meaning |
| --- | --- | --- |
| Start checks (`check`, `start_game`) | `>=` refuses | No unit, search, move, update or game starts **at** a limit. |
| Acceptance checks (`completion_violation`) | `>` refuses | Work finishing **exactly at** a limit is accepted; past it, never. |
| Completion barrier | `>` refuses | Eligible iff elapsed at the endpoint ≤ `campaign_seconds`, each seed's training ≤ `per_run_training_seconds`, and games started ≤ `evaluation_games_ceiling` (inclusive, consistent with acceptance). |

The **completion-eligibility timestamp is the endpoint**, and there is no work between it and the eligibility decision. Fake-clock results (`campaign_seconds = 10`):

| Clock at the barrier | Outcome |
| --- | --- |
| 9.9 | `COMPLETE`, endpoint 9.9 |
| 10.0 | `COMPLETE`, endpoint 10.0 (inclusive) |
| 10.1 | `INCOMPLETE`: "campaign wall-clock budget exceeded at the completion barrier" |
| 9.9, then 10.05 after the barrier and 10.1 inside the terminal rename | `COMPLETE`; the recorded endpoint stays **9.9** |

`official_results` re-checks the recorded endpoint. A `COMPLETE` record whose endpoint is 10.1 is refused, even when every result file is valid.

## 6. Publication-failure semantics

`terminate(COMPLETE)` is the last operation. The decision after a failure is made from the **re-read visible state**, never from the point where the exception was raised:

| Failure point | Visible after the failure | Outcome |
| --- | --- | --- |
| Building or validating the document; writing or fsyncing the temporary file | old (`RUNNING_FINAL_EVALUATION`) | **A.** Temporary file removed; exception propagates; `mark_incomplete` records `INCOMPLETE`; nonzero exit. |
| `os.replace` raising without renaming | old | **A.** As above. |
| `os.replace` renamed, then an error (OSError or KeyboardInterrupt) | `COMPLETE` | **B.** Owner adopts `COMPLETE`; error recorded in `post_commit_errors`; directory fsync retried; status `completed`, exit 0, warning on stderr. |
| Directory fsync after the rename (F2) | `COMPLETE` | **B.** As above. |
| Any error after `terminate` returned (`after_complete`) | `COMPLETE` | **B.** `run` calls `mark_incomplete`, which is a no-op; status `completed`. |
| `state.json` unreadable when an error path runs | unknown | Nothing is written: the visible bytes differ from the owner's, so `_write` refuses. Never `COMPLETE`; a later invocation refuses it as corrupt. |

The owner and every `official_results` reader therefore see the same outcome. Durability is unchanged from the 4D.3B.2 contract. If the directory fsync failed twice and power is later lost, the rename can only revert to the earlier non-terminal state, which a later invocation records as `INCOMPLETE`. It can never produce a false `COMPLETE`.

## 7. Results acceptance

[`official_results`](../games/connect4/alphazero_v2/campaign.py#L1104) returns the sealed results only if all of the following hold:

- `state.json` loads, validates and is `COMPLETE`;
- the declaration token equals the hash of `declaration.json` and is not a rejected token;
- `final/started.json` matches the hash recorded on entering `RUNNING_FINAL_EVALUATION`, and also the hash in the terminal outcome;
- every declared seed has a certified `final/seed-S.json` whose hash matches, with `seed` and `completeness == "complete"`;
- the counts satisfy the protocol ([`account_problems`](../games/connect4/alphazero_v2/campaign.py#L1068)): every seed completed all declared generations, no evidence unit was started without completing, and games started do not exceed the ceiling;
- the completion record ([`completion_problems`](../games/connect4/alphazero_v2/campaign.py#L1083)) exists, its limits equal the declared budgets, it equals the account it certifies, and its endpoint, per-seed training seconds and games started are within the limits (inclusive).

A stray `COMPLETE` string or file is not sufficient. Twelve forgeries are refused, each with otherwise valid result files. `INCOMPLETE` and non-terminal campaigns remain unusable.

## 8. Old tokens

`launch_control.REJECTED_DECLARATION_TOKENS` now holds all three, and the declaration's `launch_control.rejected_declaration_tokens` lists all three.

In fresh interpreters through the CLI, each old token was tried with:

- the current declaration bytes;
- its original declaration bytes, restored with `git show` at `a663454`, `c302c18` and `ff1ded6` (each hash re-confirmed);
- its original bytes with the **new** token.

**All nine exited 2 with `REJECTED` before any campaign directory was created.** `official_results` also refuses a `COMPLETE` record that names a rejected token.

## 9. Regression tests

All use tiny synthetic campaigns (2 simulations, 2 games, 2 generations), fake clocks, temporary directories, and fault injection inside the real atomic publication. Real `SIGTERM`/`SIGINT` go to the real handler. "Old" means the test run against `8715b38` with the new test files.

| Requirement | Test | On `8715b38` |
| --- | --- | --- |
| Signal immediately before the barrier | `test_signal_immediately_before_the_completion_barrier_is_incomplete` | fails (no barrier) |
| Signal during the terminal commit (temporary fsync, inside rename, directory fsync); the review's F1 probe via CLI `main` | `test_signal_during_terminal_commit_cannot_certify_an_accepted_stop` ×3 | **fails: `assert 'signal 15' is None`**; a stop accepted while non-terminal, then certified `COMPLETE` |
| Signal after `COMPLETE` | `test_signal_after_complete_changes_nothing` | fails |
| Deadline just under, exactly at, just over | `test_completion_barrier_deadline_boundary` ×3; unit `test_completion_barrier_deadline_is_inclusive` ×3 | — |
| Clock advancing after eligibility (the review's F1 deadline probe) | `test_clock_after_eligibility_never_changes_the_recorded_completion_time` | **fails: over-budget endpoint accepted (`DID NOT RAISE`)** |
| Exception before the `COMPLETE` replacement | `test_publication_failure_before_complete_is_visible_is_incomplete` ×2; unit `test_failure_before_complete_is_visible_ends_incomplete` | unchanged behaviour (A) |
| Exception after the replacement; fsync failure after `COMPLETE` is visible (the review's F2 probe); wrapper cannot downgrade | `test_failure_after_complete_is_visible_never_downgrades_it` ×4 (OSError and KeyboardInterrupt after rename, directory fsync, post-commit RuntimeError); unit `test_failure_after_complete_is_visible_is_adopted_and_never_downgraded` ×2 | **fails: the OSError propagates after readers accepted `COMPLETE`**; KeyboardInterrupt aborts the session |
| `COMPLETE` and `INCOMPLETE` immutability, including a stale object | `test_terminal_states_never_reopen` ×2 | fails |
| `INCOMPLETE` built from the visible history; never a blind replace | `test_incomplete_after_a_partial_running_write_keeps_the_visible_history`; `test_state_is_never_blindly_replaced`; `test_terminal_outcome_must_name_its_state` | fails |
| Barrier latch | `test_completion_barrier_latches_one_endpoint`; `test_completion_barrier_latches_before_reading_the_stop_flag`; `test_completion_barrier_refuses_a_stop_or_any_exceeded_limit` ×3 | — |
| Result acceptance validates `COMPLETE` evidence | `test_official_results_validates_the_complete_record` (12 forgeries) | fails |
| Old-token rejection | `test_both_old_tokens_and_their_declarations_never_authorize` ×3 commits; `test_all_three_previous_tokens_are_rejected` | `ff1ded6` case fails |

Changed existing tests:

- the two 4D.3B.2 terminal-deadline tests are replaced by the parametrized boundary test;
- the state-file tests use `terminate`;
- `atomic_replace` now also asserts that the temporary file is removed.

The eleven hard-crash scenarios in fresh interpreters are unchanged and pass, including `before_campaign_complete`.

## 10. Verification

Environment for the torch rows: `/tmp/board-game-phase4c1-venv` (CPython 3.11.17, torch 2.10.0, NumPy 1.26.4), `PYTHONDONTWRITEBYTECODE=1`, all five thread variables `1`.

| Check | Result |
| --- | --- |
| Reproduction probes on `8715b38`, before any edit | **3 passed**: F1 signal, F1 deadline and F2 all reproduced |
| New regression tests on `8715b38` | core F1/F2 tests fail as listed in §9 |
| `test_alphazero_v2_campaign.py` (100) + `test_alphazero_v2_launch_control.py` (31), deselecting `phase4d2f_adapter_is_hash_pinned` and `synthetic_full_scale_replay_is_legal_and_resumable` as in the review | 128 passed before the re-freeze; the one failure was the frozen-declaration test, because the old token was now rejected (expected). Passes after the re-freeze (next row). |
| All backend `tests/` in the torch venv, excluding the five API modules (no `dotenv` there) and deselecting `phase4d2f_adapter_is_hash_pinned` and the two import-isolation gates | **1,230 passed, 3 deselected**, 250 s. Includes all AlphaZero v2, engine, MCTS, neural and DQN suites, the 11 crash points and the full-scale replay check. |
| All backend `tests/` in `.venv` (no torch) | **482 passed, 15 skipped**, 95 s |
| API import isolation (`.venv`) | **2 passed** |
| `packages verify` (frozen packages) | exit 0; all six packages and 2,100 rows verified; `git diff` touches only `campaign-declaration.json` in `frozen/` |
| Re-freeze and independent reconstruction | identical bytes, `34b4d899…c390` (§11) |
| Preflight on the new declaration | `"ok": true`; all five gates pass; 6,272 planned games |
| Old tokens through the CLI | 9 of 9 refused, exit 2, no directory |
| `git diff --check` | clean |

## 11. New frozen declaration

`games/connect4/alphazero_v2/frozen/campaign-declaration.json`: 15,490 bytes. It was regenerated from scratch with `campaign freeze` in the pinned runtime, after the rejected file was removed (its bytes remain in git at `ff1ded6`).

**SHA-256 / authorization token: `34b4d899639d59d1bd4408ecc505b7eb20cb4d9c2fdc12095402c246887ac390`**

It was confirmed three ways:

- the `freeze` output;
- `shasum -a 256` of the file;
- an in-memory reconstruction from `build_declaration`, the configured runtime, the manifest, the frozen records and `FREEZE_NOTES`, which produced byte-identical output.

Against the rejected `ed641aea…` declaration, exactly three of the 22 top-level fields changed:

| Field | Change |
| --- | --- |
| `execution_source` | Combined `c6e474923c532e16acc2999afe38aff065a831f914c8d409aa7910e09eaf393d`, 26 files. **`training` (18 files) `22678c4b1c056f4b2b61a94fdc2eaaa0cd67de884a82c558ab15324406899e35`, unchanged.** `evaluation_and_launch` (8 files) is now `62c42e6fe68862c10369f188dcabd9fc48d1f54c97365e7c82dd5cb011125a90`; only `campaign.py` and `launch_control.py` differ. |
| `launch_control` | `version` `…-v3-terminal-commit`; `state_file` `…-state-v2`; `deadlines` now states the inclusive completion boundary; new `terminal_commit` entry (barrier, latch, post-barrier signals, conditional `INCOMPLETE`); `rejected_declaration_tokens` holds all three. |
| `notes` | Phase 4D.3B.3; the three rejected tokens; the one-way terminal commit. |

These fields are unchanged:

- `format_version` (3);
- `runtime` and all 17 `runtime_identity` fields;
- `config`, `seeds` (42 / 314159), `primary_seed` and `generations` (20);
- `budgets` (28,800 s / 86,400 s / ceiling 6,400 / planned 6,272);
- `evaluation`, `champion`, `final`, `thresholds` and `retention`;
- `packages`, `frozen_records` and `phase4d2f_checkpoint`.

### Launch command (not run; needs separate authorization)

Commit this work first. Do not edit any execution-closure file afterwards.

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/tmp/board-game-phase4c1-venv/bin/python
D=games/connect4/alphazero_v2/frozen/campaign-declaration.json
T=34b4d899639d59d1bd4408ecc505b7eb20cb4d9c2fdc12095402c246887ac390
$PY -m games.connect4.alphazero_v2.campaign preflight --declaration $D     # must print "ok": true and exit 0
caffeinate -i $PY -m games.connect4.alphazero_v2.campaign launch --declaration $D \
    --campaign-dir experiment-output/phase4d3c-alphazero-v2-official-campaign --authorize $T
```

Exit codes are unchanged:

- **0:** `COMPLETE`; read results only through `official_results`. A stderr warning about a post-commit error does not change the certified outcome.
- **4:** `INCOMPLETE`, final.
- **2:** refused.

Never rerun to resume.

## 12. Notes and limitations

1. **Post-barrier signals are absorbed.** A SIGINT or SIGTERM in the few milliseconds between the barrier and the end of `run` does not stop the process early. It is noted on stderr, and the process exits normally after certification. Likewise, a KeyboardInterrupt raised after `COMPLETE` is visible is recorded, not re-raised.
2. **Hard death in the commit window** behaves as before. Before the rename, the state stays `RUNNING_FINAL_EVALUATION` and a later invocation records `INCOMPLETE`. After the rename, it is `COMPLETE`.
3. **Unchanged from 4D.3B.2:**
   - non-resumability;
   - active-time measurement (sleep is not charged);
   - durability primitives;
   - identity scope;
   - the deferred items: learned-checkpoint 512-simulation p95, the searched-root-value diagnostic and the `exhaustive_action_values` terminal guard.
4. **No real-scale run.** All lifecycle evidence comes from tiny synthetic campaigns.
