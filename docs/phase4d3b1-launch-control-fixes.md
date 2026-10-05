# Phase 4D.3B.1 — Launch-control hardening and re-freeze

Completed October 5, 2026 on local branch `phase4d3b-alphazero-v2-preflight` (starting HEAD `d5696cb`, clean). **No campaign was launched.**

This phase produced:

- no research training, learned research checkpoint or strength evaluation;
- no package regeneration, public API/UI integration or deployment;
- no commit, push or merge.

Every campaign in this phase was a tiny synthetic lifecycle run in a temporary directory.

> **REJECTED / NOT AUTHORIZED:** `2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8`
> (the Phase 4D.3B declaration token). The launcher refuses it as an `--authorize` value and refuses any declaration file with that hash, even byte-for-byte copies restored from git.
>
> **New frozen declaration token (format 2):** `8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39`
> It is valid only with this tree's exact execution sources and runtime, and it still needs separate launch authorization.

Authorities: [launch-readiness review](phase4d3b-launch-readiness-review.md) (CONDITIONAL GO; B1–B4, S1–S4), [4D.3B handoff](phase4d3b-alphazero-v2-preflight.md), [4D.3A](phase4d3a-alphazero-v2-core.md), [4D.3 design review](phase4d3-alphazero-v2-review.md).

## Summary

| ID | Review finding | Reproduced on HEAD | Status |
| --- | --- | --- | --- |
| B1 | Declaration does not bind executable source; final evaluation has no source/runtime continuity | Yes | **Fixed** |
| B2 | Generation publication is not a recoverable transaction | Yes | **Fixed** |
| B3 | Final calibration lacks stop checks; completion is not rechecked; the exact game ceiling blocks later work | Yes | **Fixed** |
| B4 | A hard crash erases evaluation-game consumption; time gap; torn ledger; no single-writer rule | Yes | **Fixed** |
| S1 | Partial evidence discarded; `running` status after an exception | Yes | **Fixed** (part of B2–B4) |
| S2 | Row-level evidence and learned-artifact timing measurement | Yes | **Partly fixed**: row-level evidence is durable. The readiness measurement and searched root values are **deferred** (§10). |
| S3 | Oracle action helpers accept already-won histories | Yes (precision note below) | **Deferred**: outside the four authorized areas; frozen labels unaffected (§10) |
| S4 | Preflight is informational; CPU identity can be `null`; 4D.2f checkpoint unchecked | Yes | **Fixed** (part of B1) |
| L1/L2 | Package coverage and interpretation limits | — | Unchanged, nonblocking |

No reproduction disagreed with the review. Two precision notes:

- **B2(c), duplicate inference registration.** It is real: retrying after a crash between the inference and resume manifest entries left two inference records for generation 1. The failure the review mentions is latent. With the archive collision bypassed, the retry still completed, because the in-loop champion check uses the fresh record. It fails only when a later *pending* check calls `_inference_record`.
- **S3.** `exhaustive_action_values` has no root guard, as the review says. On the tested won history it refused incidentally, because a non-winning child reached `exhaustive_value`. `BitboardSolver.action_values` returned a full value map, as reported.

The fix is confined to the four areas. **New** module [`launch_control.py`](../games/connect4/alphazero_v2/launch_control.py) is torch-free and holds the journal, lock, leases, reservations and state fold. **Rewritten** [`campaign.py`](../games/connect4/alphazero_v2/campaign.py) handles declaration v2, binding, staged commits, durable units, preflight, `freeze` and `status`. Three files received minimal hooks that do not change the trajectory:

- [`provenance.py`](../games/connect4/alphazero_v2/provenance.py): in-process CPU probe, source groups, unavailable-field detection;
- [`generation.py`](../games/connect4/alphazero_v2/generation.py) / [`selfplay.py`](../games/connect4/alphazero_v2/selfplay.py): `progress` / `on_game` callbacks;
- [`arena.py`](../games/connect4/alphazero_v2/arena.py) / [`evaluation.py`](../games/connect4/alphazero_v2/evaluation.py): per-game `paired_game` and per-row `search_row`, factored out of existing loops with identical RNG derivation.

Unchanged:

- network, search, replay, training and optimizer code;
- `config.py`, `data.py`, `oracle.py`, `packages.py`, `reference_negamax.py` (so `build-provenance.json`'s builder-source hashes remain current);
- statistics, thresholds, gates, seeds, generations, budgets and every frozen package byte.

## 1. Findings: reproduction, fix and regression tests

All reproductions were run on **unmodified HEAD** before any edit. They use temporary directories and virtual source edits (monkeypatched reads), and no repository file was modified while reproducing. Harness and recorded outputs: [`research/phase4d3b1-mutation/head-reproduction/`](../research/phase4d3b1-mutation/head-reproduction/); it runs only against the pre-fix code.

### B1 — source and runtime binding (BLOCKER)

- **Where:** `campaign.validate_declaration` / `Campaign.__init__` / `preflight` / `final_evaluate` (HEAD `campaign.py:128`, `:590`, `:718`).
- **HEAD behavior:**
  - A virtual one-byte append to `search.py` changed the execution digest from `fae266e4…` to `45176ea0…`.
  - `preflight` still reported token `2741314399…`, and `Campaign(…, old token)` was accepted.
  - The declaration had no source or runtime identity fields.
  - `final_evaluate` never consulted source identity, and loaded inference artifacts carry none.
- **Why it violates the protocol:** the token was supposed to freeze the executed campaign. Instead, a fresh run, a second seed's fresh start, or the sealed evaluation could run different code or a different runtime under the same authorization.
- **Fix:**
  - Declaration **format 2** binds `execution_source` (per-file SHA-256, combined digest and the two group digests), the complete `runtime_identity`, `frozen_records`, and `launch_control` semantics (§6).
  - `Campaign.__init__` configures the deterministic runtime, then refuses unless every identity input matches; it names the differing **group** and files, or the differing runtime fields. It also refuses any unavailable runtime field (for example `cpu_model: null`), a missing or changed 4D.2f checkpoint, and changed packages or frozen records. All of this happens before the campaign directory is touched.
  - At launch it imports the whole closure, so later lazy imports cannot pick up edited code.
  - Identity is re-verified at every generation, commit, champion check, decision, selection, final start and final publication.
  - `final_evaluate` additionally refuses if any journaled earlier attempt recorded a different runtime, execution digest or declaration.
  - `run_seed` checks that the imported `generation` module's import-time digest equals the declaration.
  - Strict resume loads, which compare the boundary's recorded runtime and source, are unchanged.
- **S4 (same area):**
  - `preflight` is now a gate: exit 2 unless the declaration, pinned thread environment, *configured* runtime identity (complete and equal), execution source and 4D.2f checkpoint all pass.
  - The CPU model is now probed in-process with `sysctlbyname` (with the old `sysctl` subprocess as fallback). A sandbox that blocks exec can no longer produce `null`.
- **Regression tests (`test_alphazero_v2_campaign.py`):**
  - `test_changed_execution_source_refuses_launch[...]`: six files across both groups.
  - `test_fresh_interpreter_refuses_a_one_byte_source_edit`: copied tree. Clean launches are accepted, and docs, tests or a non-closure module change nothing. A one-byte edit to `search.py` (training) or `evaluation.py` (evaluation/launch) is refused before any directory is created, and the closure is loaded at launch.
  - `test_source_drift_mid_campaign_stops_before_publication`.
  - `test_every_runtime_identity_field_refuses_launch_and_resume[17 fields]`, with the journal byte-identical after the refusal.
  - `test_unidentified_runtime_is_refused`, `test_cpu_identity_probe_does_not_need_a_subprocess`.
  - `test_runtime_change_between_training_and_final_evaluation_is_refused` (runtime and source).
  - `test_final_evaluation_refuses_an_attempt_recorded_under_another_identity`.
  - `test_preflight_is_a_launch_gate` (passes when pinned; fails with exit 2 when unpinned or declared M4), `test_launch_rejects_changed_packages_frozen_records_and_checkpoint`, `test_source_groups_partition_the_closure`, `test_training_path_loads_only_training_group_modules`.

### B2 — generation publication and recovery (BLOCKER)

- **Where:** HEAD `Campaign.run_seed` (`campaign.py:447–458`): archive, then boundary, then separate manifest appends, then diagnostics, then summary.
- **HEAD behavior:**
  - **Crash after archival:** the retry failed with `FileExistsError: generation-0001.jsonl`. Manual deletion of evidence was needed.
  - **Crash in generation-1 diagnostics:** the retry reported `completed`, with no generation-1 diagnostic call and no generation-1 summary file.
  - **Crash between the inference and resume registrations:** two inference records for generation 1 (latent failure, see above).
- **Why it violates the protocol:** an interruption could block resumption or silently skip required per-generation evidence, and the multi-file generation had no commit point.
- **Fix: journal-first commit.**
  1. Every output is staged under `runs/seed-S/staging/generation-NNNN.attempt-AAA/`: inference and resume via the existing validating `atomic_torch_save`, and archived games via `write_once`. Then the staging directory is fsynced.
  2. Limits, stop requests and identity are rechecked.
  3. One `generation_committed` journal record names every file with its SHA-256, plus the learner weight digest, `state_sha256` and the deterministic summary. **This record is the commit.**
  4. The staging directory is renamed to `generations/generation-NNNN/` and the parent directories are fsynced.

  Recovery (at every attempt start) installs or verifies every committed generation:
  - A committed generation with only its staging directory is renamed into place after a hash check.
  - Any missing or hash-mismatched committed output is refused (`InconsistentCampaign`). Earlier work is never silently redone.
  - Uncommitted staging directories are orphans, never read, and never block a rerun, because each attempt stages into its own directory.

  Post-commit work runs in generation order and is resumable: development diagnostics (`generation_diagnostics` record, then `summary.json` view), pruning (`artifact_pruned` record, then unlink; recovery finishes an interrupted unlink after a hash check), and the scheduled champion check. Resume continues from the latest committed boundary and verifies its file hash, `completed_generations` and `state_sha256` against the commit record. The journal's state fold refuses an out-of-order or duplicate commit, decision or selection.
- **Regression tests:**
  - 30-point hard-crash matrix (§3).
  - `test_inconsistent_published_outputs_are_refused`: a modified committed artifact or a modified view is refused, and no new work happens over it.
  - `test_campaign_directory_stays_bound_to_its_declaration`.
  - Journal transition tests in `test_alphazero_v2_launch_control.py`.

### B3 — deadline enforcement (BLOCKER)

- **Where:** HEAD `_final_seed` calibration loop (`campaign.py:672–678`), `final_evaluate`, `run_seed`, `Budget.check`.
- **HEAD behavior (fake clock):**
  - **Calibration overrun:** with a 1-second cap, the last calibration operation advanced the clock to 2 s and requested a stop. The result was still `completed`, `final/seed-42.json` was written, and calibration had received **no** check callback.
  - **Training overrun:** a training check at 99 of 100 s permitted the last update; the update ran to 101 s and evaluation proceeded.
  - **Exact ceiling:** after the 3rd of 3 allowed games, a non-game evaluation check was refused with `evaluation-game ceiling reached`.
- **Why it violates the protocol:** completion was published past the deadline or after a stop. The training cap was not enforced on the final update. Exactly reaching the game ceiling falsely blocked required non-game evidence.
- **Fix:**
  - Checks are forwarded into calibration self-play and every arena move, search seed and update. Every evaluation row and game is a separately checked unit.
  - Starting work and the game ceiling are separate checks. `check(phase)` limits time and stop requests only. `require_game_slot()` limits games only and is called solely before a game starts.
  - Before any generation commit, diagnostics record, champion decision, selection, final seed result or campaign completion, `_require_completion` rechecks stop requests and every charged limit (`>` the limit refuses; finishing exactly at the limit is within it).
  - A generation whose final update overran the per-run cap is discarded.
  - Reaching a limit records a terminal **INCOMPLETE** outcome (§4).
- **Regression tests:**
  - `test_final_calibration_receives_checks_and_an_overrun_is_incomplete`: the B3 reproduction now yields `incomplete:`, no seed result, INCOMPLETE `outcome.json` and `final/result.json`, and a later invocation does nothing.
  - `test_stop_during_the_last_unit_defers_publication`.
  - `test_last_update_overrunning_the_training_cap_discards_the_generation`.
  - `test_deadline_during_development_evaluation_preserves_evidence`.
  - `test_exact_game_ceiling_completes_and_one_lost_slot_is_incomplete`.
  - Budget tests in `test_alphazero_v2_launch_control.py`.

### B4 — durable budget accounting (BLOCKER)

- **Where:** HEAD `Budget.count_evaluation_game` / throttled `heartbeat` / `consumed` / `JsonLog.records` (`campaign.py:212–290`, `:113`).
- **HEAD behavior:**
  - With a heartbeat at t=0, five counted games and then a hard crash: in-memory count 5, recoverable count **0**.
  - A torn trailing ledger line raised `JSONDecodeError` on every later read.
  - `training_games` counted only successful generations.
  - No lock: two `Campaign` objects on one directory were both accepted.
- **Why it violates the protocol:** each crash could regain up to a heartbeat's worth of games and time, so the "cumulative" 6,400-game ceiling and 24-hour cap were not guaranteed. A torn record needed manual repair.
- **Fix (`launch_control`):**
  - **Games:** an evaluation game is charged by its `unit_begin` record, written and fsynced before the first move. That charge is permanent whether the game completes, is abandoned or the process dies.
  - **Time:** active time is charged through **durable leases**, which are upper bounds written before work outruns them (§4). A crashed attempt is charged through its last lease; an attempt that ends normally is charged its measured time.
  - **Journal:** every record is checksummed and sequence-numbered. A torn **trailing** record is detected, saved beside the journal and discarded with a `torn_tail_discarded` record. Damage before a valid record is refused.
  - **Writers:** `flock` admits one writer per campaign directory, and the read-only `status` command needs no lock.
  - **Counters:** self-play games, plies and optimizer steps are journaled as they happen, separately from accepted work.
- **Regression tests:**
  - `test_game_reservation_is_charged_before_play_and_survives_a_crash`, `test_hard_crash_is_charged_through_its_last_lease_and_never_resets`, torn-tail and corruption tests, `test_second_writer_is_refused_in_process` and `test_lock_held_by_another_process_is_refused_and_released_on_death` (`test_alphazero_v2_launch_control.py`).
  - The crash matrix's charge assertions.
  - `test_concurrent_invocation_is_refused`.

### S1 — partial evidence and status (SHOULD FIX; fixed with B2–B4)

- Every completed game or row is durable evidence the moment it completes.
- A cooperative stop records the in-flight game's partial record in `unit_abandoned`.
- An unexpected exception ends the attempt as `failed: <type>: <message>`, after abandoning open units and discarding any open generation. It no longer ends as `running`.
- Interrupted champion checks and final evaluation resume at unit granularity. The selection is fixed by `final_started` before any sealed inference.
- Tests: `test_cooperative_stops_resume_without_rerunning_completed_units` and the crash matrix.

## 2. Source and runtime binding definition

| Identity | What is bound | Where enforced |
| --- | --- | --- |
| **A. Training source** | Content SHA-256 of the 12 v2 training modules (`__init__`, `artifacts`, `config`, `data`, `diagnostics`, `generation`, `network`, `oracle`, `provenance`, `search`, `selfplay`, `training`) plus the 6 reused engine/v1 modules | Launch, every phase boundary, generation import-time digest, strict resume |
| **B. Evaluation/solver/arena/launch source** | `arena`, `campaign`, `evaluation`, `launch_control`, `packages`, `profiling`, `reference_negamax`, `statistics` | Same; reported as the `evaluation_and_launch` group |
| **C. Frozen package identity** | SHA-256 of the 6 package files; `manifest.json`, `exclusions.json`, `build-provenance.json` (`frozen_records`); 4D.2f checkpoint path and SHA-256 | Declaration validation (every load); checkpoint at launch and preflight |
| **D. Campaign configuration** | Every declaration field (seeds, config, budgets, evaluation, champion schedule and gate, final ladder, thresholds, retention, notes, launch-control semantics) via the file token; gate and thresholds must equal the executed `statistics` constants | Token equality; validation |
| **E. Runtime identity** | All 17 `runtime_identity()` fields: Python implementation/version; torch version, git version and build-config hash; NumPy; platform string (OS version); machine; CPU model; torch CPU capability; intra/inter-op threads; deterministic and warn-only flags; float32 matmul precision; mkldnn; the five library thread variables. No field may be unavailable. | Launch (after configuring), every phase boundary, every journaled attempt before final evaluation, strict resume |

- The closure is the review's 25 files plus the new `launch_control.py`, 26 in total. The groups are a reporting partition: both are always enforced, and a fresh-interpreter test checks that training loads only group A.
- Documentation, tests, research scripts and repository modules outside the closure (for example the legacy `negamax_agent.py`) do not affect executable identity. This is tested.
- Git commit, branch and dirty state are recorded per attempt (`attempt_source` record plus the source snapshot) as provenance only.
- Exactness claimed: deterministic logical continuation under identical bound identity. It is not a hermetic binary hash of installed libraries, and the CPU model is not a unique physical-machine identifier.

## 3. Durable campaign state and crash recovery

`journal.jsonl` is the single authority. The JSON views (`summary.json`, `champion-check-NNNN.json`, `selection.json`, `final/*.json`, `outcome.json`) are derived from it. Each is written once and verified byte-for-byte on every recovery. `status.json` is a replaceable, non-authoritative summary, and `campaign status --campaign-dir C` prints the derived state read-only.

**The journal answers, after any crash:**

- **Seeds and generation:** which seed is active or selected; the current (latest committed) generation; whether a generation was open; which committed artifacts are authoritative and which resume files are pruned.
- **Evaluation:** which diagnostics, champion decisions and final seed results exist; the current champion; every unit's begins, completion evidence and abandonments.
- **Budgets and identity:** consumed budgets; the approved declaration digest (`campaign_created`) and every attempt's identity.

**Explicit transitions:**

| Record | Meaning | Written |
| --- | --- | --- |
| `attempt_start` / `attempt_source` / `lease` | Attempt opened with its identity; provenance; time reservation | Before any work |
| `generation_begin` | Generation attempted | Before collection |
| `selfplay_game`, `optimizer_step` | Attempted work completed inside an open generation | After each game or update |
| `generation_committed` | **Commit point**: staged files hashed and named | After staging + fsync, before install |
| `generation_discarded` | Open generation abandoned (stop, limit, failure, recovered crash) | — |
| `generation_diagnostics` | Development raw-value diagnostics complete | — |
| `artifact_pruned` | Old resume boundary retired | Before unlink |
| `unit_begin` / `unit_complete` / `unit_abandoned` | Evaluation unit attempted (games charged here) / completed with evidence / abandoned | Around each game or row |
| `champion_decision`, `seed_selected` | At most once per scheduled generation / seed; validated against the derived champion | After completion recheck |
| `final_started`, `final_seed_complete`, `campaign_complete` / `campaign_incomplete` | Selections fixed / sealed seed result (SHA-256 of its view) / terminal outcome | — |
| `attempt_end` / `attempt_recovered` | Normal end with measured charges / crash closed by the next invocation with lease charges | — |

**Rules for units that cannot complete atomically:**

- A generation is discarded and rerun from the last committed boundary; the rerun is deterministic and identical.
- A game or row is abandoned and rerun with a new reservation. Its earlier reservation stays charged and is never evidence.
- A completed unit is never rerun; the state fold refuses `unit_begin` on a completed unit.
- A completed unit evaluated after a deadline is kept, flagged `within_budget: false`, and its phase cannot be published.
- Final evaluation resumes at unit granularity. It is never restarted from scratch and never reselected, and each sealed unit completes at most once.

**Crash matrix** (`test_hard_crash_recovers_to_the_uninterrupted_result`):

- **Setup:** a tiny two-seed campaign (2 generations, 2 games each, schedule 1/2, final ladder Random + Negamax-1). It runs in a fresh pinned interpreter and is killed by `os._exit` at each of 30 transitions, then resumed in a new interpreter.
- **Transitions covered:**
  - before generation 2 begins; during collection; after collection before training; during training; after training before publication;
  - staged but uncommitted; before the commit record; after the commit record before install; after install before diagnostics; before and after the diagnostics record; after a prune record before the unlink;
  - during development evaluation (after a game reservation; before a baseline game completes; before a tactics row completes); before and after a champion decision; before and after selection;
  - between seeds (second seed's first commit, and during its training);
  - before and after `final_started`; during the sealed tactical, solved, ladder and calibration units; before and after a final seed record; before campaign completion.
- **Every resumed campaign equals the uninterrupted one** in: committed weights and state digests, deterministic summaries, archived games, diagnostics, champion decisions with their arena and tactics metrics, selections, all unit evidence (timings excluded) and both sealed seed results.
- **Every case also shows:** exactly one recovered attempt; the crashed attempt charged through its lease (charged time ≥ uninterrupted time + 150 s); accepted counters equal; attempted counters ≥ accepted. An interrupted game is charged exactly once more than in the uninterrupted run; other points add no game charge. A generation is discarded exactly when one was open, and orphan staging is present but unused.

## 4. Deadline semantics

- **Limits:** 8 h (28,800 s) collection + optimization per seed; 24 h (86,400 s) combined two-seed campaign including evaluation; 6,400 evaluation games (6,272 planned). All are unchanged.
- **Charged time** is *active invocation time*, as frozen in 4D.3B and accepted by the review.
  - **Downtime between processes is not charged.**
  - An attempt that ends normally is charged its measured elapsed time, and its measured collection + optimization time toward its seed.
  - **A hard-interrupted attempt is charged through its last durable lease.** A lease is journaled at attempt start and renewed whenever fewer than 150 s remain, to elapsed + 300 s; a training lease is renewed likewise while training. A crash therefore over-charges by at most about 300 s per crash and never under-charges, unless a single unchecked operation outlasts the 150 s margin (§10).
  - Charges are summed from the journal, so repeated crashes or restarts never reset any timer (tested).
- **When time is charged:** continuously while an attempt is alive. Training time is the span from `begin_training` to `end_training` around each generation.
- **Before work starts:** every search, arena move and optimizer update is preceded by a check that refuses to *start* once charged time is at the limit (`>=`). A new game also needs a free game slot (charged games below the ceiling). There is no prediction of unit cost: a unit may start with any positive remaining budget, and if the deadline passes it is interrupted at its next check.
- **In-flight work:** an in-flight primitive (one search, move or update) finishes past a cooperative deadline. The next check stops the attempt:
  - an in-flight game is abandoned (charged, partial record kept, not evidence);
  - an in-flight row is abandoned;
  - an in-flight generation is discarded.
- **Completion:** before anything is published, stop requests and limits are rechecked. A limit exceeded (`>`) means nothing is published; finishing exactly at a limit is within it. A generation whose final update overran its seed's cap is discarded.
- **Cooperative versus hard interruption:**
  - *Cooperative:* SIGINT/SIGTERM or a stop request. The attempt ends with `attempt_end` status `stopped: …` and stays resumable (CLI exit 3).
  - *Exhaustion:* exit 4 with status `incomplete: …`.
  - *Hard:* no `attempt_end`; the next invocation writes `attempt_recovered`, abandons open units and discards open generations.
- **Incomplete evidence:** reaching any limit writes a terminal `campaign_incomplete` record and `outcome.json` (and `final/result.json` once final evaluation has started) with status **INCOMPLETE**, the reason and per-seed and final progress. Completed evidence stays in the journal. Every later invocation returns `incomplete: …` without doing work. The launcher never reports PASS: COMPLETE means only that every declared unit finished within the limits, and acceptance is judged separately against the frozen thresholds. No limit is ever extended.

## 5. Durable budget semantics

| Concept | Defined by |
| --- | --- |
| **Attempted** | `unit_begin`, `generation_begin` |
| **Completed** | `unit_complete` (with evidence); per-game `selfplay_game`; per-update `optimizer_step` |
| **Counted toward evidence** | `unit_complete` units of a published phase; games, plies and updates of *committed* generations (`accepted_*`) |
| **Counted toward budget** | Evaluation games at `unit_begin` (whatever happens next); time through leases or measured `attempt_end` |

The durable, monotonic counters (`campaign status`, `state.consumption()`) are:

- **Time:** charged campaign seconds; charged training seconds per seed.
- **Evaluation games:** charged and completed totals; per unit kind (development arena, development baseline, final ladder, final NN-only, calibration games; development tactics, final tactical and final solved rows), the attempted, completed, abandoned and restarted counts.
- **Training, per seed:** generations begun, committed and discarded; self-play games and plies completed (attempted); optimizer steps attempted; accepted games, plies and steps.
- **Attempts:** total, ended, recovered after hard interruption, and open.

Uninterrupted runs show attempted = accepted for every counter (tested). Crashes show attempted > accepted by exactly the discarded work (tested for a mid-collection game and mid-training updates).

## 6. Declaration and launcher

### Before (REJECTED)

`frozen/campaign-declaration.json` format 1, SHA-256 `2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8`.

- **Keys (18):** `budgets, champion, config, evaluation, final, format, format_version, generations, kind, name, notes, packages, phase4d2f_checkpoint, primary_seed, retention, runtime, seeds, thresholds`.
- **Not bound:** no execution source, no concrete runtime identity (`runtime` held only threads=1, deterministic=true and the strict-resume flags), and no hashes for the frozen records.
- **Status:** **REJECTED / NOT AUTHORIZED.** The token is listed in `launch_control.REJECTED_DECLARATION_TOKENS` and inside the new declaration's `launch_control.rejected_declaration_tokens`. The launcher refuses it as `--authorize` and refuses any declaration file with that hash, including the original bytes restored from git commit `a663454` (tested in-process, via preflight, and via the CLI in a fresh interpreter: exit 2, no directory created).

### After (format 2, regenerated from scratch with `campaign freeze`)

**New SHA-256 / authorization token: `8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39`** (checked independently with `shasum -a 256`).

The 18 format-1 keys are byte-equivalent in value except `format_version` (1 → 2) and `notes`. Four keys are new:

| Field | Content |
| --- | --- |
| `execution_source` | Scheme `sha256(json(sorted relative path -> file sha256))-v1`; **26 files**; combined `86a4b8502b1fb708e3c3e97614985dad98bc71ab5cbe46a236f5550874855886`; groups `training` (18 files) `22678c4b1c056f4b2b61a94fdc2eaaa0cd67de884a82c558ab15324406899e35` and `evaluation_and_launch` (8 files) `4c32d8e42049c2ee280552557164deae88c76978f63cc087333b3c6385f77bda`; per-file hashes. |
| `runtime_identity` | CPython 3.11.17; torch 2.10.0 (git `449b1768…`, build-config `ae49aeb8…`); NumPy 1.26.4; `macOS-26.6.2-arm64-arm-64bit`; arm64; **Apple M5**; CPU capability DEFAULT; intra/inter-op threads 1/1; deterministic true, warn-only false; matmul precision highest; mkldnn true; `OMP/MKL/OPENBLAS/VECLIB/NUMEXPR_NUM_THREADS=1`. |
| `frozen_records` | `manifest.json` `d1f0d4df44536346b7f326f88fd0f926af8d34a96f5e03dc9c2e5151ca3f06db`; `exclusions.json` `73c0f5a581a42a508e444c16e465239bad4de5be69d26d8caeec021e5b6de044`; `build-provenance.json` `09e352db84b03080719e9afb0c7160f636ec2b9f7ee99abaf218d59ae7b29a77`. |
| `launch_control` | `connect4-alphazero-v2-launch-control-v1`; journal format; single-writer rule; lease 300 s, renewal below 150 s; time-accounting, evaluation-game, deadline, recovery and final-evaluation semantics (§3–§5); rejected tokens. Validation requires these to equal the launcher's own constants. |

Unchanged bound values:

- **Packages:**
  - openings-development `3990bc23…2f72`, openings-sealed `d30683aa…4eb`;
  - solved-development `41ffd4df…37b9`, solved-sealed `2c5f70dc…394b`;
  - tactical-development `106cc315…6d9d`, tactical-sealed `cb4e2e1e…6f47`.
- **4D.2f checkpoint:** `78276912bdef5f1b1b53ac6b2ec3ede2eed20a8ebb5796a9bdcbb57e4cb28c51`.
- **Budgets:** 28,800 / 86,400 s; 6,400 ceiling, 6,272 planned.
- **Campaign:** seeds 42 / 314159; 20 generations; full `V2Config`; evaluation, champion, final, thresholds and retention.

### What the launcher requires

The token cannot be embedded in launcher source, because the declaration binds that source; the binding is circular in the safe direction. Instead:

- `--authorize` must equal the SHA-256 of the declaration file, and neither may be rejected.
- The declaration must validate, and every bound input must match this process.
- **A one-byte change to any bound input therefore changes or refuses the token:** declaration fields (`test_every_declaration_field_is_bound_by_the_token` mutates each of the 22 keys); sources, runtime, packages, frozen records and checkpoint (§1 tests).
- Only `8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39` authorizes this tree and runtime, and `test_frozen_declaration_is_the_new_format_2_declaration` pins it to this tree's sources.

**Preflight on the new declaration** (pinned environment, this machine): `"ok": true`, exit 0. Every gate passed: declaration, thread environment, configured runtime identity, execution source and 4D.2f checkpoint. Planned games 6,272. Git provenance (recorded, not enforced): HEAD `d5696cb`, tracked tree dirty, `launch_control.py` untracked. Committing changes neither the execution digest nor the token.


## 7. Preserved scientific protocol

Byte-identical to 4D.3B:

- all six package files (hashes in §6), `manifest.json`, `exclusions.json` and `build-provenance.json`; `build-provenance.json`'s builder-source hashes still match `packages.py`/`oracle.py`/`reference_negamax.py`;
- every `V2Config` value, seeds 42 / 314159, 20 generations, 256 games per generation, 256 / 512 simulations;
- champion schedule and gate, acceptance thresholds, budgets and planned games, retention, and the 4D.2f checkpoint hash.

The declaration fields from format 1 carry over unchanged except `format_version` (1 → 2) and `notes`. No blocker made a frozen artifact invalid, so nothing was regenerated.

## 8. Verification

| Check | Environment | Result |
| --- | --- | --- |
| Reproduction of B1–B4, S1–S4 on unmodified HEAD | torch venv | All reproduced (§1) |
| All AlphaZero-v2 suites (`test_alphazero_v2.py` 138, `_evaluation_core.py` 41, `_campaign.py` 89 including the 30-point crash matrix, `_launch_control.py` 20) + engine + every neural, DQN and standalone-MCTS suite | `/tmp/board-game-phase4c1-venv` (CPython 3.11.17, torch 2.10.0, NumPy 1.26.4), pinned thread env, `-k 'not phase4d2f_adapter_is_hash_pinned'` (avoids running the retained learned model, as in the review) | **1,187 passed, 1 deselected, 2 failed.** The 2 failures are the two API-startup isolation gates, which cannot import `dotenv` in this venv (the known 4D.3B environment limitation); they pass in `.venv` (next rows). 252 s. |
| Full backend `tests/` | `.venv` (no torch) | **471 passed, 15 skipped** (torch-dependent modules). This includes the 20 torch-free launch-control tests. |
| API import isolation (`test_neural_mcts_import_isolation.py`, `test_dqn_import_isolation.py`) | `.venv` | **2 passed** |
| Frozen packages, structural verify | torch venv | 2,100 rows verified, exit 0; package, manifest, exclusions and build-provenance bytes unchanged |
| Re-freeze and launch gate | pinned env | New token `8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39`; preflight ok, exit 0; old token CLI exit 2 |
| Mutation sweep (§9) | torch venv | **42 / 42 killed, 0 survived** |
| `git diff --check` and whitespace check of new files | — | Clean |

Lifecycle runs (crash matrix, cooperative stops, deadlines) used tiny synthetic configurations: 2 simulations, 2 games, 2 generations and batch 8, in temporary directories. The crash matrix runs 31 campaigns in fresh interpreters in parallel, in about 50 s.

## 9. Mutation-style testing

[`research/phase4d3b1-mutation/mutate_launch.py`](../research/phase4d3b1-mutation/mutate_launch.py) applies one plausible defect at a time, runs `test_alphazero_v2_launch_control.py` and `test_alphazero_v2_campaign.py` stopping at the first failure, and restores the source. The unmutated baseline passed (107 tests). Every source was verified byte-identical to its backup afterwards. Full results: [`mutation-launch-results.txt`](../research/phase4d3b1-mutation/mutation-launch-results.txt).

| Area | Mutations (all KILLED) |
| --- | --- |
| Source binding | source not compared; group misreported; `oracle` missing from the training group; closure not imported at launch; mid-campaign identity unchecked; declared source digest consistency unchecked |
| Runtime binding | runtime not compared; unavailable fields allowed; preflight reports without configuring; CPU probe subprocess-only; final evaluation skips historical attempt identity |
| Old token / resumed-token verification | rejected list emptied; declaration loader skips rejection; token not compared; campaign directory's declaration unchecked; journal's declaration unchecked |
| Durable transitions | install skips hash check; no staging recovery; completed units rerun; views not verified; duplicate decision allowed; out-of-order commit allowed; open attempts not recovered; torn tail not repaired; unterminated record accepted; damage before a valid record accepted; concurrent writer allowed |
| Deadline persistence | lease not written; lease renewed only after expiry; crash charged zero; clean end charged its lease; start check `>` instead of `>=`; completion not rechecked; completion ignores stop; training completion unchecked; calibration check not forwarded; INCOMPLETE not final |
| Budget persistence | game charged at completion instead of reservation; ceiling not enforced; prior training ignored; self-play progress not journaled; abandoned game not recorded |

These are focused mutants, not a coverage proof. Writing the mutation list first exposed five untested contracts, which got new tests before the sweep:
- directory and journal binding;
- historical attempt identity;
- campaign-level torn tails;
- unterminated records;
- exact attempted = accepted counters.


## 10. Remaining limitations and deferred items

1. **S2 readiness measurement — deferred, needs a decision before launch.** The declared readiness gate (warm whole-move p95 ≤ 500 ms over ≥ 500 varied positions on the approved artifact) is still not executed by `final-evaluate`. Searched root values are not recorded either: `V2SearchResult.root_value` is the raw network value, and adding a searched value would change `search.py`. Both are new measurements or search changes outside the four authorized areas.
   - Row-level evidence (choices, visits and raw root values per seed, NN-only choices, raw values, full game and calibration records) is now durable and included in each sealed result.
   - The NN-only tactical metrics now also carry the prespecified overlap-excluded sensitivity.
   - Before Milestone 3, decide whether readiness is measured inside the campaign (requires a code change and a re-freeze) or as a separately declared post-campaign step.
2. **S3 — deferred.** `BitboardSolver.action_values` and `exhaustive_action_values` still lack an explicit legal-nonterminal guard. Every frozen and evaluation row is nonterminal, so labels are unaffected. Fixing it changes `oracle.py`, which is in the training source group and is recorded in `build-provenance.json`'s builder hashes, so it was left for a separately reviewed change.
3. **Lease bound.** A hard crash is charged through its last lease. The guarantee assumes no single unchecked operation outlasts the 150 s renewal margin. Known unchecked operations are well inside it:
   - boundary save ≈ 10 s, load ≈ 9 s (measured at full scale);
   - one search ≈ 0.25 s, one UCT move ≈ 0.4 s;
   - the source snapshot and journal fold take seconds.

   A longer stall that ends in a hard crash (for example the machine hanging) could be under-charged by the excess. Over-charge is at most one lease (300 s) per hard crash.
4. **Active-time semantics.** Downtime between invocations is not charged, as frozen in 4D.3B. The 24 h cap is cumulative active seconds, not calendar time.
5. **Durability assumptions.** Durability relies on `fsync` and on atomic `rename`/`link` on the local APFS volume. `os.fsync` on macOS does not force the drive's write cache (`F_FULLFSYNC`), so power loss, unlike a process crash, could in principle lose acknowledged records. A torn tail is then handled, but a lost acknowledged reservation would under-charge one game. Process crashes, the tested case, are fully covered.
6. **Identity scope.** Identity binds versions and build metadata, not hashes of every installed binary. The CPU model is not a unique machine identifier. The declaration is bound to the current interpreter (`/tmp/board-game-phase4c1-venv`: CPython 3.11.17, torch 2.10.0, NumPy 1.26.4) on macOS 26.6.2 / Apple M5. An OS update, a different torch build or wheel, or another Mac model refuses the launch, which then needs a re-freeze.
7. **Determinism of resumed evaluation** relies on unchanged per-game, per-row and per-index RNG derivation. Units are independent of execution order, which the crash matrix confirms at tiny scale. Wall-clock timings in evidence differ between runs.
8. **Journal size.** About 30k records per seed (one per self-play game, update and evaluation unit, plus leases) means a few MB per seed. Folding takes about a second.
9. **L1/L2** (package coverage, opening outcomes, untrained-model timings) are unchanged.
10. **No real-scale run.** Lifecycle evidence is from tiny synthetic campaigns. Scale behavior of the new transitions (fsync cadence, journal growth) is estimated, not measured.

## 11. Launch steps (still requires separate authorization)

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/tmp/board-game-phase4c1-venv/bin/python       # the runtime the declaration binds
D=games/connect4/alphazero_v2/frozen/campaign-declaration.json
T=8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39
C=experiment-output/phase4d3c-alphazero-v2-campaign-<date>      # new directory
$PY -m games.connect4.alphazero_v2.campaign preflight --declaration $D          # must print "ok": true, exit 0
$PY -m games.connect4.alphazero_v2.campaign run --declaration $D --campaign-dir $C --seed 42 --authorize $T
$PY -m games.connect4.alphazero_v2.campaign run --declaration $D --campaign-dir $C --seed 314159 --authorize $T
$PY -m games.connect4.alphazero_v2.campaign final-evaluate --declaration $D --campaign-dir $C --authorize $T
$PY -m games.connect4.alphazero_v2.campaign status --campaign-dir $C           # read-only, any time
```

- **Exit codes:** 0 completed, 3 stopped (resume by rerunning the same command), 4 INCOMPLETE (final; never extend), 2 refused.
- **Interruption:** after any interruption, including `kill -9` or a crash, rerun the same command. Recovery is automatic, and no evidence needs manual deletion.
- **Before launching:**
  - commit this work (Git state is provenance only and does not affect the token);
  - resolve the S2 readiness decision (§10.1);
  - do not edit any closure file afterwards, since any edit refuses the launch and requires a re-freeze.
