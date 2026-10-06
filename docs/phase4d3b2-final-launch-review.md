**Phase 4D.3B.2 — Final fail-closed launch review**

**Verdict: NO GO**

> **Status (Phase 4D.3B.3):** F1 and F2 are addressed in [phase4d3b3-terminal-commit-fix.md](phase4d3b3-terminal-commit-fix.md). The token `ed641aea…0bbb` reviewed here remains **REJECTED / NOT AUTHORIZED**; the re-frozen declaration has a new token and still needs its own review and authorization.

The frozen declaration `ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb` is not approved for launch. The non-resumable design removes the previous continuation, journal-repair and lease-accounting failures, but terminal publication still violates the required fail-closed contract. Two blocking defects are reproduced below.

Reviewed October 6, 2026, at initially clean HEAD `ff1ded6431638e8d687d707fd94aa2579af772a1`. Read the previous final review, the Phase 4D.3B.2 handoff, prior superseded notices, both controller modules, rewritten tests, declaration, and relevant generation, provenance, evaluation and package code. This review is the only repository change. No research training, official campaign launch, frozen-package regeneration, retained learned-model evaluation, source edit, commit or deployment occurred. Existing tests and additional fault probes used discarded tiny synthetic fixtures in temporary directories. No inference ran on the actual frozen sealed positions.

**F1 — A signal or deadline crossing before terminal publication can still certify COMPLETE.**

Locations: `campaign.py:794–811` (`_complete`), `campaign.py:1134–1137` (signal handler), `launch_control.py:146–155` (`atomic_replace`).

`_complete` checks identity, takes a counter snapshot, checks the stop flag and limits, then calls `state.transition(COMPLETE)`. That call serializes the state, writes and flushes a temporary file, and only then renames it over `state.json`. No stop or deadline check occurs during that publication interval. Until the rename, the authoritative state is still `RUNNING_FINAL_EVALUATION`.

Independent reproductions exercised this interval rather than the existing `before_campaign_complete` hook:

1. **Actual SIGTERM before rename:** ran the real CLI `main` on a tiny test declaration. Wrapped `launch_control.os.replace` so that, only when its temporary document contained `COMPLETE`, it first verified the existing state was `RUNNING_FINAL_EVALUATION`, delivered `os.kill(os.getpid(), signal.SIGTERM)`, and then called the real rename. The CLI's actual installed handler received the signal. Nevertheless, the command printed `completed`, returned **0**, wrote `COMPLETE`, and `official_results` returned the sealed results. The probe did not alter the handler, completion checks, result files or state bytes.
2. **Deadline crossing before rename:** used a 10-second synthetic limit and a fake monotonic clock. Set time to **9.9** at `before_campaign_complete`, then advanced it to **10.1** immediately before the actual terminal rename. The command returned `completed`; `state.json` certified `COMPLETE` with recorded elapsed time **9.9**, and `official_results` accepted it. This models a stall in serialization, temporary-file flushing or scheduling after the snapshot. No game or evidence was fabricated.

The existing terminal-overrun test advances time before `_complete` starts; it cannot detect this later window. The new single snapshot prevents disagreement with a later lease settlement, but does not eliminate the pre-publication deadline gap. The signal reproduction independently establishes an interruption while the authoritative campaign is nonterminal that is then certified as successful execution. `COMPLETE` is not itself a scientific PASS, but its certification makes these results eligible for acceptance.

Required for launch: a stop or deadline exhaustion before terminal commitment must not publish accepted `COMPLETE` evidence. The terminal decision and its publication need consistent interruption/deadline semantics.

**F2 — A publication exception can overwrite an already visible terminal state and erase its transition.**

Locations: `launch_control.py:154–155`, `launch_control.py:239–242` (`CampaignStateFile._write`), `campaign.py:484–491` (`_incomplete`).

`atomic_replace` makes the new `state.json` visible through `os.replace`, then fsyncs the directory. `_write` updates `self.document` only after both operations return. If directory fsync raises, the file can say `COMPLETE` while the owning object's document still says `RUNNING_FINAL_EVALUATION`. Error handling consults the stale object and writes `INCOMPLETE` over that terminal file.

Independent reproduction: completed a tiny synthetic campaign and injected one `OSError` at the directory-fsync call immediately after the terminal rename. Inside that call, the real state loader observed `COMPLETE`, and the real `official_results` successfully returned the certified results. After the exception propagated into `run`, its ordinary error handler replaced the file with `INCOMPLETE`. The new history contained no `COMPLETE` entry: it was rebuilt from the stale nonterminal document.

Thus the externally observable sequence is **`RUNNING_FINAL_EVALUATION → COMPLETE → INCOMPLETE`**, although the final history conceals the middle transition. This is not malicious state editing or a hypothetical torn JSON record. It is an ordinary write-error boundary in the official path, and readers are explicitly allowed to inspect results without acquiring the owner's lock. The final refusal is conservative, but the earlier success certification and later reversal violate the required terminal-state authority.

Required for launch: once terminal state is exposed as authoritative, error handling must not overwrite it or rewrite its history from stale state. Publication failures must have one consistent outcome for the owner and official-result readers.

**Disposition of the five previous findings**

| Previous finding | Current disposition |
| --- | --- |
| R1: runtime-drifted evidence reused after restart | Structurally eliminated as a continuation failure. Generations and `_unit` verify source/runtime before and after computation; drift prevents acceptance. Evidence stays in memory, and no later invocation imports it. Existing drift tests, including sealed calibration and restoration of the declared runtime before relaunch, passed. |
| R2: interrupted journal repair blocks recovery | Removed. There is no journal or repair-and-continue path. Independent crashes during the replacement refusal transition remained fail-closed on subsequent invocations. |
| R3: terminal deadline/accounting gaps | Not fully eliminated. Post-terminal lease settlement is gone; F1 reproduces the remaining pre-publication gap. F2 also prevents certification of the terminal-state invariant. |
| R4: finite leases undercharge unchecked work | Structurally eliminated. There are no leases, renewals, settlements or cross-process budget reconstruction. A crash cannot authorize additional work in this campaign directory. |
| R5: lost post-work training counters authorize incorrect accounting | Eliminated as a launch blocker by the new contract. Lost snapshots remain lower-bound provenance and never authorize continuation. The existing hard-crash test confirms a physically completed game can be absent from the snapshot, but relaunch performs no work. |

**Lifecycle and evidence audit**

The CLI has exactly one work-executing command, `launch`, which invokes `Campaign.run` once. The removed `run --seed` and `final-evaluate` commands are rejected. Preflight and status perform no campaign training or evaluation.

Independently enumerated all **49 ordered pairs** of the seven requested states through `CampaignStateFile.transition`. Exactly ten edges succeeded: five forward edges along

```text
CREATED → RUNNING_SEED_42 → RUNNING_SEED_314159
        → DEVELOPMENT_SELECTION_COMPLETE → RUNNING_FINAL_EVALUATION → COMPLETE
```

and five edges to `INCOMPLETE`, one from each nonterminal state. All other edges were refused without changing the file. Direct calls also prohibit terminal counter rewrites. F2 shows why correct ordinary transition checks do not suffice when the durable-file operation partially succeeds.

Additionally, for **each of the five nonterminal states**, a fresh subprocess was killed with `os._exit(97)` immediately before or immediately after the rename that records interrupted work as `INCOMPLETE`: **ten new crash cases**. The next refusal produced or retained `INCOMPLETE`; another refusal preserved identical state bytes. These probes invoked only the existing-directory refusal path, with no training/evaluation. The existing eleven campaign crash scenarios also passed, including training, between seeds, sealed evaluation, and the point after both sealed result files exist but before completion.

An existing directory is never adopted for continuation. A live owner excludes another process through `flock`. After owner death, a valid nonterminal state is marked `INCOMPLETE`; terminal states are refused. Missing or unreadable state cannot supply accepted results or a continuation path. Launch identity gates may refuse even before that interrupted-state annotation, which still permits no work. Hard death cannot synchronously write `INCOMPLETE`, so the authoritative nonterminal file remains uncertified until a later invocation records the interruption.

No published generation checkpoint becomes the campaign's runner. There is one precise qualification to the requested literal claim that generic resume utilities are unreachable: `_save_generation → GenerationRunner.save_boundary → save_resume_boundary` invokes `load_resume_boundary` on its own **unpublished temporary file**, with `restore_global_rng=False`, to validate the diagnostic artifact it is writing (`generation.py:326–341`). The returned runner is discarded. The existing instrumented test observed three such temporary-file validations and prohibited all published-boundary loads. This is not a continuation mechanism and does not reintroduce any of R1, R4 or R5; a literal claim of zero calls to the generic loader would be inaccurate.

Both development selections are written before sealed inference and recorded in `RUNNING_FINAL_EVALUATION` together with the hash of `final/started.json`. The final phase uses those selections without reselection. Evaluation units check identity before computation and again before retaining evidence; generation acceptance does the same. Diagnostics, champion decisions, selections and sealed seed publication have identity/limit gates before their evidence is accepted. Detected source/runtime drift leaves no cross-process reuse route. Ordinary exceptions, cooperative stops and sealed interruptions end uncertified or `INCOMPLETE`; F1/F2 are the reproduced terminal-boundary exceptions to the blanket guarantee.

`official_results` requires a valid `COMPLETE` state, matching declaration bytes, matching certified result-file hashes, and a result for every declared seed. Stray files are not enumerated as evidence. Partial sealed files under any nonterminal or `INCOMPLETE` state are refused. Existing tamper and partial-result tests passed; the acceptance exceptions in F1/F2 arise from terminal certification, not accidental discovery of stray files.

The live budget enforces **28,800 seconds of collection plus optimization per seed**, **86,400 monotonic active seconds overall**, and reservations before each evaluation game against the **6,400** ceiling (**6,272** planned). Checks precede searches/moves/updates; acceptance checks reject completed work that overruns. Limits are cooperative, so an in-flight primitive may finish late and then be rejected. An exact-limit finish can be accepted if no further work is needed; starting more work at the limit is refused. Normal training/development/calibration overruns and game-slot exhaustion produced `INCOMPLETE` in the focused tests. The overall terminal-publication exception is F1. No lease or missing-counter recovery can grant extra work.

**Declaration, unchanged science and execution identity**

Independently hashed the **14,236** declaration bytes and reconstructed the exact serialization in memory:

```text
SHA-256: ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb
Serialization: json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n"
```

One reconstruction used current defaults, manifest/package hashes, freshly hashed frozen records, independently assembled runtime fields and `FREEZE_NOTES`. A second started with the original 4D.3B.1 declaration, retained its scientific fields, and applied the inspected format/runtime/launch-control/notes/source changes. Both were byte-identical to the frozen file. Neither called `freeze` or wrote a declaration into the repository.

| Identity | Independently verified SHA-256 |
| --- | --- |
| All 26 execution-source files | `08df122b199eb6eaf6ca5fa8714be89e984b8fa3cb0e2a6c69c0937a84a787ca` |
| 18 training-source files | `22678c4b1c056f4b2b61a94fdc2eaaa0cd67de884a82c558ab15324406899e35` |
| Eight evaluation/launch files | `f6e2646160f27038333ed2b32bbf339a3560e74d6b158ddd2254cd62209571f5` |

Every training-source file is byte-identical to its version at `c302c18`. Only `campaign.py` and `launch_control.py` changed among the 26 execution files. A fresh import of the execution closure loaded 26 repository implementation files, all covered by the independently hashed inventory.

All **17 runtime fields** matched: CPython 3.11.17; torch 2.10.0, git `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`, build hash `ae49aeb898c01a3cd7bdb6311dce7ccea64c05a02c023ba09c707938191c6fea`; NumPy 1.26.4; `macOS-26.6.2-arm64-arm-64bit`; arm64; Apple M5; DEFAULT CPU capability; intra/inter-op threads 1/1; deterministic algorithms enabled with warn-only disabled; highest float32 matmul precision; mkldnn enabled; all five declared thread environment variables `1`. The independent CPU query used in-process `sysctlbyname` after the sandbox refused the external `sysctl` command. This verifies the declared metadata scope, not a unique physical CPU or every installed binary.

Against 4D.3B.1, exactly five top-level fields changed: `execution_source`, `format_version`, `launch_control`, `notes`, `runtime`. The complete scientific config, seeds/primary seed, generation count, budgets, evaluation, champion schedule/gate/baselines, final ladder/calibration, thresholds, retention, packages and retained checkpoint are unchanged. Those scientific fields also match the original 4D.3B declaration. All six package files and three frozen-record files match their bound hashes and their original 4D.3B.1 bytes. Structural package verification passed for all **2,100** rows. The retained checkpoint was hashed only and matched `78276912bdef5f1b1b53ac6b2ec3ede2eed20a8ebb5796a9bdcbb57e4cb28c51`.

Restored the original declaration bytes using `git show` and independently confirmed both superseded digests:

- `a663454`: `2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8`.
- `c302c18`: `8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39`.

For each, three fresh-interpreter CLI checks used (1) current bytes plus old token, (2) original bytes plus old token, and (3) original bytes plus the candidate token. **All six exited 2 with REJECTED before creating a campaign directory**, performing no official work. The focused existing tests also rejected their preflights and in-process loads. The candidate's read-only preflight passed every gate with 6,272 planned games; no candidate campaign was constructed or run.

**Verification record**

| Check | Result |
| --- | --- |
| Existing campaign and launch-control suites | **100 passed, 2 deselected**, 68.10 seconds; includes eleven hard-crash scenarios |
| Independent adversarial harness | **14 probes passed**, 24.43 seconds: exhaustive transition matrix, ten crashes during interruption annotation, and three successful reproductions of F1/F2 |
| Independent declaration/runtime/source reconstruction | Byte-identical declaration; 26 source hashes; 17 runtime fields |
| Old tokens and original bytes | Six additional CLI refusals before directory creation |
| Candidate preflight; frozen package structural verification | Both exited 0 |

Tests used `/tmp/board-game-phase4c1-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, and all five library thread variables pinned to `1`. The two deselected tests were `phase4d2f_adapter_is_hash_pinned` and `synthetic_full_scale_replay_is_legal_and_resumable`, avoiding retained-model inference and an unnecessary full-scale synthetic replay check. Scratch reproductions are `/tmp/phase4d3b2-review/test_adversarial.py`, `identity.py`, and `token_checks.py`. The adversarial harness asserts the observed violations; its passing result does not mean those invariants hold.

The learned-checkpoint 512-simulation p95 measurement, searched-root-value diagnostic, and general-purpose `exhaustive_action_values` terminal guard remain deferred. Unchanged sources/packages and the inspected official callers provide no new reason to make them launch blockers. Approval is withheld solely for F1 and F2.
