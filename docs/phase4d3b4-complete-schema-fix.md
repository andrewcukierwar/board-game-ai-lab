# Phase 4D.3B.4 — Strict COMPLETE schema validation

Completed October 6, 2026 on local branch `phase4d3b-alphazero-v2-preflight`, starting from clean HEAD `bf48303`. **No campaign was launched.**

This phase fixes one launch blocker:

> `official_results` accepted a malformed `COMPLETE` record after seed `"42"` was deleted from both `counters.time.training_seconds` and `outcome.completion.training_seconds`.

The blocker was taken from the Phase 4D.3B.4 task statement. No separate 4D.3B.3 review document is in the repository.

What changed:

- `official_results`;
- the pre-barrier evidence check in `_complete`, which now uses the same schema functions;
- the declaration-bound launch-control text;
- the rejected-token list.

Nothing else changed: training, search, solver, packages, evaluation, thresholds, budgets, seeds, architecture and the campaign lifecycle, including the terminal-commit, signal and deadline semantics reviewed in 4D.3B.3. Only `campaign.py` and `launch_control.py` changed among the 26 execution-source files. The training source group hash is unchanged.

Every campaign run in this phase was a tiny synthetic run in a temporary directory, with forged `state.json` files. There was no research training, no package regeneration and no commit.

> **REJECTED / NOT AUTHORIZED (all permanently):**
> `2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8` (Phase 4D.3B),
> `8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39` (Phase 4D.3B.1, NO GO),
> `ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb` (Phase 4D.3B.2, NO GO) and
> `34b4d899639d59d1bd4408ecc505b7eb20cb4d9c2fdc12095402c246887ac390` (Phase 4D.3B.3, NO GO).
>
> **New frozen declaration token (format 3):**
> **`9f7259827e8abee6db81113147e48c2a3420248296d09df559f2783c5d7d85f4`**
> It is valid only with this tree's exact execution sources and runtime. Launching still needs separate authorization.

## 1. Exact reproduction

This ran on unmodified `bf48303` before any edit. The campaign was a genuine tiny COMPLETE campaign with seeds `[42, 314159]` and `per_run_training_seconds = 28800`; `official_results` accepted the untouched record.

| Forged `state.json` | Result on `bf48303` |
| --- | --- |
| Seed `"42"` = 28,801 in both training-time mappings | refused: `seed 42 training seconds exceed the per-seed budget at completion` |
| Seed `"42"` deleted from both mappings | **accepted**: `completion.training_seconds = {'314159': 0.204…}`, with results for both seeds |

On the fixed tree the same probe is refused:

```text
COMPLETE is not accepted: counters.time.training_seconds must cover exactly seeds ['314159', '42']; has ['314159'];
outcome.completion.training_seconds must cover exactly seeds ['314159', '42']; has ['314159']
```

## 2. Root cause

Acceptance validated only the fields that happened to be present:

```python
for seed, seconds in completion["training_seconds"].items():      # only the keys present
    if not seconds <= budgets["per_run_training_seconds"]: ...
```

The cross-check compared the two training-time copies **with each other**. Nothing tied their key set to the declared seeds. Deleting a seed from both copies kept them equal and removed it from the budget loop.

The same "validate if present" pattern ran through the rest of the record. A mutation sweep on `bf48303` deleted each required field once and broke each per-seed mapping once, giving 156 malformed records:

- **47 refused**;
- **85 accepted**;
- **24 crashed with an uncaught `KeyError`, `TypeError` or `AttributeError` instead of `NotOfficialEvidence`**.

Examples accepted on `bf48303`:

- any single `counters.evaluation.by_kind` entry deleted;
- `development_games`, `sealed_games`, `calibration_games`, `total_games_completed` or `ceiling` deleted;
- any per-seed training counter except `generations_completed` deleted;
- an extra seed added to `counters.training`;
- seed keys deleted, added or renamed (`"042"`) in either history `selections`;
- `phase_seconds` deleted.

The per-seed training-time variants on `bf48303`:

- **accepted:** seed 314159 deleted from both copies, both seeds deleted, an extra seed `"7"`, keys `"042"` or `" 42"`, negative seconds, `true`, `-Infinity`, and a duplicate `"42"` key whose first copy was over budget;
- **crashed:** `null`, strings, and a list in place of the mapping;
- **refused:** a seed deleted from only one copy, disagreeing copies, over the limit, `NaN` and `Infinity` (both caught only by the limit comparison).

## 3. Schema rule

[`official_results`](../games/connect4/alphazero_v2/campaign.py#L1328) accepts only if [`complete_record_problems`](../games/connect4/alphazero_v2/campaign.py#L1303) finds nothing. Every exception during validation becomes `NotOfficialEvidence`. The rule:

1. **Strict JSON.** `state.json`, `declaration.json`, `final/started.json` and every `final/seed-S.json` are parsed with [`strict_json`](../games/connect4/alphazero_v2/campaign.py#L1063). Duplicate object keys and `NaN`/`Infinity`/`-Infinity` are refused. The ordinary `json.loads` would silently keep the last duplicate and parse non-finite numbers.
2. **Required seed set.** [`required_seed_keys`](../games/connect4/alphazero_v2/campaign.py#L1088) derives it from the hash-verified `declaration.json`. The seeds must be a non-empty list of distinct integers (not booleans), and a `research` declaration must declare exactly `[42, 314159]`. For the frozen declaration the set is exactly `{"42", "314159"}`. That is asserted on the frozen file.
3. **Exact coverage.** Every required per-seed mapping must have exactly that key set ([`seed_mapping_problems`](../games/connect4/alphazero_v2/campaign.py#L1107)). Missing seeds, extra seeds and non-canonical keys (`"042"`, `" 42"`) are refused. JSON object keys are always strings, so set equality with the canonical decimal strings rejects every other representation. Duplicate keys are refused by rule 1.
4. **Exact structure.** Required objects must have exactly their required keys: outcome, completion record, counters, `time`, `training[seed]`, `evaluation`, `by_kind`, each kind's `{started, completed}`, and `started.json`.
5. **Domains.**
   - seconds are finite non-negative `int` or `float`, never `bool`;
   - counts are non-negative `int`;
   - digests are 64 lowercase hex characters;
   - text fields (`note`, `endpoint`) are strings.
6. **Limits (inclusive, unchanged).**
   - completion `elapsed_seconds` ≤ `campaign_seconds`;
   - **each declared seed's** training seconds ≤ `per_run_training_seconds` (28,800);
   - games started ≤ `evaluation_games_ceiling`.
7. **Cross-consistency.** Every quantity stored twice must exist in both places, cover the same seeds and be equal (§5).
8. **Optional diagnostics stay optional.** `counters.time.phase_seconds` is required to exist and to map phase names to finite non-negative seconds, but no particular phase name is required. History `utc` and `owner` details are provenance; `CampaignStateFile.validate` already checks their structure.

The barrier in `_complete` runs the same `certified_evidence_problems` and `counter_problems` before `COMPLETE` is written, so the owner cannot certify a record that acceptance would refuse. Every condition added there holds by construction in a genuine run. Every tiny campaign, the two-seed reference campaign and the crash matrix still complete.

## 4. Per-seed mappings audited

| Mapping | Where | Value domain | Before (`bf48303`) | Now |
| --- | --- | --- | --- | --- |
| `counters.time.training_seconds` | `state.json` | finite seconds ≥ 0 | compared only with the completion copy | exact seeds; domain; equal to completion copy |
| `outcome.completion.training_seconds` | `state.json` | finite seconds ≥ 0, ≤ 28,800 | **only keys present budget-checked** | exact seeds; domain; every seed ≤ limit; equal to counters copy |
| `counters.training` | `state.json` | object of six counts | only `generations_completed` read, per declared seed | exact seeds; exact keys; counts; generations attempted = completed = declared; self-play attempted = completed = generations × `games_per_generation` |
| `outcome.final_results` | `state.json` | SHA-256 | exact seeds (sorted list) | exact seeds; digest format; file hash |
| history `DEVELOPMENT_SELECTION_COMPLETE.selections` | `state.json` | selection under this token | **not checked** | exact seeds; each names its own seed and the token; equal to the next entry |
| history `RUNNING_FINAL_EVALUATION.selections` | `state.json` | as above | **not checked** | as above; equal to `started.json` |
| `selections` | `final/started.json` (hash-certified) | selection | hash only | exact seeds; equal to the history selections |
| `descriptions` | `final/started.json` (hash-certified) | object with `agent` | hash only | exact seeds; each has an agent description |
| `seed`, `selection`, `agent` | `final/seed-S.json`, one per seed | int; selection; agent | `seed`, `completeness` | `seed` is an int equal to its key; `completeness == "complete"`; `selection` and `agent` equal `started.json` |
| `phase_seconds` (`seed-S:*` names) | `state.json` | seconds | not checked | diagnostic: typed, names not required |

No other per-seed mapping exists in the `COMPLETE` record. The sweep in §6 finds exactly the six in `state.json` automatically.

## 5. Cross-consistency

| Quantity | Copies | Rule |
| --- | --- | --- |
| Per-seed training seconds | `counters.time.training_seconds`, `outcome.completion.training_seconds` | both present; both exactly `{"42", "314159"}`; equal per seed; finite, ≥ 0, ≤ 28,800 |
| Elapsed seconds at the endpoint | `counters.time.elapsed_seconds`, `outcome.completion.elapsed_seconds` | both present; equal; finite, ≥ 0, ≤ `campaign_seconds` |
| Evaluation games started | `counters.time.evaluation_games_started`, `outcome.completion.evaluation_games_started`, `counters.evaluation.total_games_started` | all equal; equal to the sum of `by_kind[*].started` over game kinds; ≤ ceiling |
| Evaluation aggregates | `development_games`, `sealed_games`, `calibration_games`, `total_games_completed` | each equals the sum of its kinds' `completed` units; `ceiling` equals the declaration |
| Evidence units | `by_kind[kind]` | exactly the 8 unit kinds; `started == completed` for each |
| Completion limits | `outcome.completion.limits` | equal to the declaration's three budgets |
| Sealed-phase start | history `RUNNING_FINAL_EVALUATION.started_sha256`, `outcome.started_sha256`, hash of `final/started.json` | all equal |
| Selections | both history entries, `started.json`, every `final/seed-S.json` | all equal per seed |
| Declaration | `state.json` `declaration_sha256`, hash of `declaration.json`, `started.json`, every selection | all equal; not a rejected token; state seeds equal declared seeds |

No new scientific metric was introduced. Every rule restates a relationship that the existing code already writes.

## 6. Regression tests

The tests are in [`tests/test_alphazero_v2_campaign.py`](../tests/test_alphazero_v2_campaign.py), section "Phase 4D.3B.4 strict COMPLETE schema": 34 tests. One genuine COMPLETE campaign is built per module, with seeds 42 and 314159 and a 28,800 s per-seed limit (tiny synthetic work). Each test forges a copy of it. The "On `bf48303`" column is from running the same tests against the old code.

| Test | Cases | On `bf48303` |
| --- | --- | --- |
| `test_exact_reproduced_exploit_is_refused` | 28,801 s refused; then seed 42 deleted from both copies **must still be refused** | **fails: accepted** |
| `test_per_seed_training_time_requires_both_seeds_in_both_copies` | 15: seed 314159 deleted from both; both deleted; 42 deleted from counters only, and from completion only; 314159 deleted from completion only; extra seed; `"042"`; `" 42"`; copies disagree; negative; string; `true`; `null`; 28,800.5; list instead of mapping | 10 fail (7 accepted, 3 crash); 5 already refused |
| `test_non_finite_training_time_is_refused` | `NaN`, `Infinity`, `-Infinity` in both copies, injected as raw JSON text (the ordinary loader accepts them) | fail (`-Infinity` accepted; the other two refused only by the limit) |
| `test_duplicate_seed_keys_are_refused` | duplicate `"42"` keys: over-budget first copy, valid last copy | **fails: accepted** |
| `test_exactly_the_per_seed_limit_is_accepted` | both seeds at exactly 28,800 → accepted (inclusive) | passes |
| `test_valid_two_seed_record_is_accepted` | untouched record accepted with both seeds | passes |
| `test_started_record_seed_coverage_is_checked_beyond_its_hash` | `started.json` re-hashed consistently without seed 42's description | fails: accepted |
| `test_mutation_sweep_every_required_field_and_seed_key` | **156 mutations**: every required field deleted once, plus delete-42, delete-314159, extra-7 and rename-042 on each of the **6** per-seed mappings. All must be refused; deleting a diagnostic phase name and the untouched record are accepted. | **fails: 85 accepted, 24 crashed** |
| `test_required_seed_keys_are_the_declared_seeds` | 10 declarations: research `[42, 314159]` valid; research `[42]`, `[314159, 42]` or with an extra seed invalid; duplicates, booleans, strings, empty and non-list invalid | n/a (new function) |

Also changed:

- `test_frozen_declaration_is_the_new_format_3_declaration` asserts the frozen required set is `{"42", "314159"}`;
- both old-token tests include the 4D.3B.3 token and its original bytes from `bf48303`;
- the rejected-token comparison is order-insensitive.

## 7. Verification

Environment for the torch rows: `/tmp/board-game-phase4c1-venv` (CPython 3.11.17, torch 2.10.0, NumPy 1.26.4), `PYTHONDONTWRITEBYTECODE=1`, all five thread variables `1`.

| Check | Result |
| --- | --- |
| Reproduction on `bf48303`, before any edit | seed 42 deleted from both copies was **accepted** (§1) |
| New schema tests on `bf48303` | 17 of the 24 non-unit tests fail, as listed in §6 |
| `test_alphazero_v2_campaign.py` (135) + `test_alphazero_v2_launch_control.py` (31), deselecting `phase4d2f_adapter_is_hash_pinned` and `synthetic_full_scale_replay_is_legal_and_resumable` | **164 passed, 2 deselected**, 82 s. Includes the 11 hard-crash points and all 4D.3B.3 terminal-commit tests, unchanged. |
| All backend `tests/` in the torch venv, excluding the five API modules (no `dotenv`) and deselecting `phase4d2f_adapter_is_hash_pinned` and the two import-isolation gates | **1,265 passed, 3 deselected**, 257 s |
| All backend `tests/` in `.venv` (no torch) | **482 passed, 15 skipped** |
| API import isolation (`.venv`) | **2 passed** |
| `packages verify` | exit 0; six packages and 2,100 rows; no package file changed |
| Re-freeze and independent reconstruction | byte-identical, `9f725982…85f4` (§8) |
| Preflight on the new declaration | `"ok": true`, exit 0; all five gates pass; 6,272 planned games |
| Old tokens through the CLI in fresh interpreters | **12 of 12 refused**, exit 2, `REJECTED`, no directory created. Each of the four old tokens was tried with the current bytes, with its original bytes restored from `a663454`, `c302c18`, `ff1ded6` and `bf48303` (each hash re-confirmed), and with its original bytes under the new token. |
| `git diff --check` | clean |

## 8. New frozen declaration

`games/connect4/alphazero_v2/frozen/campaign-declaration.json`: 16,401 bytes. It was regenerated from scratch with `campaign freeze` in the pinned runtime, after the rejected file was removed (its bytes remain in git at `bf48303`).

**SHA-256 / authorization token: `9f7259827e8abee6db81113147e48c2a3420248296d09df559f2783c5d7d85f4`**

It was confirmed three ways:

- the `freeze` output;
- `shasum -a 256` of the file;
- an in-memory reconstruction from `build_declaration`, the configured runtime, the manifest, the frozen records and `FREEZE_NOTES`, which produced byte-identical output.

Against the rejected `34b4d899…` declaration, exactly three of the 22 top-level fields changed:

| Field | Change |
| --- | --- |
| `execution_source` | Combined `8f554b939a894283542ec675304a6f7ba23c88899659869245a9d48f2e162b37`, 26 files. **`training` (18 files) `22678c4b1c056f4b2b61a94fdc2eaaa0cd67de884a82c558ab15324406899e35`, unchanged.** `evaluation_and_launch` (8 files) is now `e4532730d92c37fa8ca516f5b2b56570a56453b954a46cbe35dcfce3785498c8`; only `campaign.py` and `launch_control.py` differ. |
| `launch_control` | New `acceptance` entry describing the strict schema; `rejected_declaration_tokens` holds all four. `version` and `state_file` are unchanged: the lifecycle and the state format did not change. |
| `notes` | Phase 4D.3B.4; four rejected tokens; strict acceptance schema. |

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
T=9f7259827e8abee6db81113147e48c2a3420248296d09df559f2783c5d7d85f4
$PY -m games.connect4.alphazero_v2.campaign preflight --declaration $D     # must print "ok": true and exit 0
caffeinate -i $PY -m games.connect4.alphazero_v2.campaign launch --declaration $D \
    --campaign-dir experiment-output/phase4d3c-alphazero-v2-official-campaign --authorize $T
```

Exit codes are unchanged: 0 `COMPLETE` (read results only through `official_results`), 4 `INCOMPLETE`, 2 refused. Never rerun to resume.

## 9. Notes and limitations

1. **Consistency, not authenticity.** The schema refuses malformed, incomplete and internally inconsistent records. Hashes give integrity against accidental change. Someone with write access who rewrites every file, hash and copy consistently is outside this check, as before. There is no signature.
2. **Planned game count.** Acceptance requires `started == completed` for every unit kind and games started ≤ the ceiling. It still does not require exactly 6,272 games. That was the existing protocol, and the sealed result files themselves carry each arena's planned-game count.
3. **Phase timings stay diagnostic.** `phase_seconds["seed-S:training"]` is wider than the budgeted collection+optimization time (it includes identity checks and bookkeeping), so it is not tied to `training_seconds`.
4. **Unchanged from 4D.3B.3:** terminal-commit, signal and deadline semantics; non-resumability; durability; identity scope; and the deferred items (learned-checkpoint 512-simulation p95, searched-root-value diagnostic, `exhaustive_action_values` terminal guard).
5. **No real-scale run.** All evidence comes from tiny synthetic campaigns and forged records.
