# Phase 4D.3B.5 — Official evidence contract

Completed October 6, 2026 on local branch `phase4d3b-alphazero-v2-preflight`, starting from clean HEAD `85dd988`. **No campaign was launched.** Nothing was committed.

This phase replaces field-by-field acceptance patches with one authoritative contract for every artifact `official_results` relies on.

What changed:

- a new torch-free, read-only module, [`official_evidence.py`](../games/connect4/alphazero_v2/official_evidence.py), which owns every acceptance schema and the single producer of a sealed result;
- [`campaign.py`](../games/connect4/alphazero_v2/campaign.py): the producer, the two in-process call sites and `official_results` now call that module, and the old in-file schema is removed;
- [`launch_control.py`](../games/connect4/alphazero_v2/launch_control.py): the rejected-token list; a total `CampaignStateFile.validate`; and `launch_control_declaration()`, moved here unchanged except for its `acceptance` text;
- a new campaign-directory artifact, `packages/<name>.json`, a hash-verified copy of every declared package;
- the re-frozen declaration.

Unchanged: the network, PUCT/search, training, optimizer and replay, seeds, packages, the exact solver, evaluation algorithms, thresholds, budgets, champion logic and the terminal-commit semantics. The training source group's digest is unchanged (`22678c4b…`). For identical evidence, the sealed results are identical to `85dd988`'s: a tiny two-seed campaign produced the same `final/seed-*.json` and `selection.json`, with timings and declaration hashes stripped.

> **REJECTED / NOT AUTHORIZED (all permanently):**
> `2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8` (Phase 4D.3B),
> `8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39` (Phase 4D.3B.1, NO GO),
> `ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb` (Phase 4D.3B.2, NO GO),
> `34b4d899639d59d1bd4408ecc505b7eb20cb4d9c2fdc12095402c246887ac390` (Phase 4D.3B.3, NO GO) and
> `9f7259827e8abee6db81113147e48c2a3420248296d09df559f2783c5d7d85f4` (Phase 4D.3B.4, NO GO).
>
> **New frozen declaration token (format 3):**
> **`2e39713f6e2791c735406c3bec7def0d336adb6add1cd43d49319e83aac486ab`**
> It is valid only with this tree's exact execution sources and runtime. Launching still needs separate authorization.

## 1. Root cause

The reproduction ran on unmodified `85dd988` against a genuine tiny two-seed COMPLETE campaign. Each forgery deleted fields from `final/seed-42.json` and wrote the file's new SHA-256 into `outcome.final_results`.

| Forgery of `final/seed-42.json` | `85dd988` | Now |
| --- | --- | --- |
| delete `ladder`, `tactical`, `solved`, `evidence` and `calibration` | **accepted** | refused |
| delete any one of `ladder`, `tactical`, `tactical_nn_only`, `solved`, `value`, `evidence`, `calibration`, `nn_only_vs_random`, `overlap_families` | **accepted** (each) | refused (each) |
| delete each of the 1,298 JSON paths, one at a time | **1,273 accepted**, 25 refused | 1,298 refused |

The 25 refused paths on `85dd988` were all under `seed`, `completeness`, `selection` and `agent`.

The producer (`Campaign._final_seed_result`) and the acceptance check (`certified_evidence_problems`) did not share a schema. Acceptance read four fields of a certified result. It trusted the rest because the bytes matched a hash.

A hash shows that a file is the one certified. It does not show that the file is a complete, internally consistent scientific result:

- every summary was trusted because it existed;
- every evidence list was trusted whatever its length;
- nothing tied a result's ladder, rows, games or calibration games to the declaration.

The 4D.3B.4 fix had the same blind spot one level down. It made `state.json` exact, but stopped at the certified files' hashes.

## 2. Design

**One producer.** [`derive_seed_result(context, seed, started, raw)`](../games/connect4/alphazero_v2/official_evidence.py#L705) is the only code that builds a sealed result. `raw` holds the raw evidence:

- the overlap families;
- tactical and solved row evidence;
- calibration records;
- ladder and NN-only game records.

Every other field is derived from `raw`, the declaration, the packages and `started.json`. `Campaign._final_seed_result` now only gathers this process's raw evidence and calls that function ([campaign.py:858](../games/connect4/alphazero_v2/campaign.py#L858)). The arithmetic is unchanged; it was moved, not rewritten.

**One validator.** [`seed_result_problems`](../games/connect4/alphazero_v2/official_evidence.py#L924) runs in two steps:

1. It validates the raw evidence structurally against the declaration and packages ([`raw_evidence`](../games/connect4/alphazero_v2/official_evidence.py#L892)).
2. It requires the **whole stored result** to equal what `derive_seed_result` derives from that stored evidence, type-strictly. Key sets and list lengths must match exactly, `True ≠ 1` and `1 ≠ 1.0` ([`differences`](../games/connect4/alphazero_v2/official_evidence.py#L194)).

So the validator does not keep a second copy of the summary schema to drift from the producer. A summary that is deleted, retyped, extended or inconsistent with its evidence fails step 2. Evidence that is deleted, truncated, retyped or inconsistent with the declaration fails step 1.

**Self-contained directory.** Before this phase the packages lived only beside the original declaration, so acceptance could not derive sealed row IDs from a campaign directory at all. Launch now copies every declared package into `packages/<name>.json` ([campaign.py:352](../games/connect4/alphazero_v2/campaign.py#L352)), verified against the declared hash. Acceptance reads them through [`load_context`](../games/connect4/alphazero_v2/official_evidence.py#L523), which re-verifies each against the token-certified declaration.

**Total validators.** Every validator returns problems for any JSON value and never raises on malformed input. `official_results` still fails closed on an unexpected exception, labelled `internal validation error (refused)`. Every mutation test asserts that this backstop never fires.

## 3. Authoritative artifact schemas

### 3.1 `declaration.json`

Certified by its SHA-256, the authorization token in `state.json`, which is never a rejected token. Strictly parsed: duplicate keys and NaN/Infinity are refused.

[`declaration_problems`](../games/connect4/alphazero_v2/official_evidence.py#L341) is the shared contract: every check that needs neither torch nor files. Launch runs it inside `validate_declaration` ([campaign.py:135](../games/connect4/alphazero_v2/campaign.py#L135)). Acceptance re-runs it on the certified declaration, so the declaration of an accepted campaign is always one that could have launched.

| Field | Rule |
| --- | --- |
| top level | exactly the 22 `DECLARATION_KEYS` |
| `format`, `format_version` | the format-3 constants (integer version) |
| `name`, `kind` | strings; `research` requires seeds exactly `[42, 314159]` and exactly the three frozen records |
| `seeds`, `primary_seed` | non-empty list of distinct integers (not booleans); primary among them |
| `generations`, `config` | positive integer; `config` is a valid `V2Config` without `seed` whose `max_generations` equals `generations` |
| `budgets` | exactly 4 keys; positive finite seconds; positive integer ceiling; `planned_evaluation_games` equals the planned count and is ≤ the ceiling |
| `champion` | exactly 6 keys; strictly increasing schedule within `1..generations`; positive `arena_openings`; distinct known baseline opponents; distinct baseline rows; gate equal to the executed gate; a declared `tactical_package` |
| `final` | exactly 5 keys; distinct ladder of known opponents; positive `ladder_openings` and `calibration_games`; `calibration_constant` null or finite; `one_time` true |
| `evaluation` | exactly 6 keys; positive simulations; distinct integer tactical seeds; noise off, guard off, temperature 0, seeded ties |
| `packages` | the six required packages; each a plain name bound to exactly `{path, sha256}` |
| `thresholds` | equal to the executed `ACCEPTANCE_THRESHOLDS` |
| `phase4d2f_checkpoint` | exactly `{path, sha256}` |
| `runtime`, `runtime_identity` | exactly `{threads, deterministic_algorithms}`; identity fully available and consistent with it (threads, deterministic flag, thread environment) |
| `execution_source` | exactly `{scheme, sha256, files, groups}`; digest of its files; groups partition the files, each a digest of its members |
| `retention`, `frozen_records`, `notes`, `launch_control` | exact retention keys; records exactly `{path, sha256}` each; notes a list of strings; launch-control text equal to this launcher's |

Launch-only (needs torch or files): exact runtime field names, the training/evaluation source partition, frozen-record file hashes and package loading.

### 3.2 `packages/<name>.json` (new)

[`load_context`](../games/connect4/alphazero_v2/official_evidence.py#L523) and [`package_problems`](../games/connect4/alphazero_v2/official_evidence.py#L470) require, for every declared package:

- SHA-256 equal to the declaration's;
- a valid package whose rows match their content hash;
- `kind-split` equal to its name;
- distinct string row IDs.

Across packages:

- the sealed openings hold at least `ladder_openings` rows;
- the development openings hold at least `arena_openings` rows;
- every baseline row exists in the development openings.

### 3.3 `state.json` COMPLETE record

| Part | Rule |
| --- | --- |
| envelope | exactly 9 keys; this launcher's `format` and `launch_control`; `state` COMPLETE; `seeds` equal to the declaration's; `CampaignStateFile.validate` now refuses a non-list history or non-object entry/outcome instead of crashing |
| `owner` | exactly `{pid: int, host: str, started_utc: str}` |
| `history` | exactly CREATED, RUNNING_SEED_<s> for each declared seed, DEVELOPMENT_SELECTION_COMPLETE, RUNNING_FINAL_EVALUATION, COMPLETE; each entry exactly `{state, utc}`, plus `selections` (selection complete) or `selections, started_sha256` (sealed start); `utc` a string |
| `outcome` | exactly `{status, final_results, started_sha256, completion, note}` |
| `outcome.final_results` | exactly the declared seeds → SHA-256 of `final/seed-S.json` |
| `outcome.completion` | as in 4D.3B.4: exact keys; limits equal the declaration; endpoint, each seed's training seconds and games started within the (inclusive) limits; each equal to its counter copy |
| `counters` | as in 4D.3B.4 (exact keys, seed coverage, domains, aggregates), plus a string `note`, plus **every unit kind's `started = completed =` the declaration-derived count** (§4) |

### 3.4 Selection (four copies)

[`selection_problems`](../games/connect4/alphazero_v2/official_evidence.py#L573) applies to every copy: both history entries, `started.json` and every seed result. All copies must be identical.

The selection has exactly `{seed, generation, inference, decisions, rule, declaration_sha256}`:

- `seed`: the integer of its key;
- `declaration_sha256`: the token;
- `rule`: the development-only selection rule;
- `decisions`: exactly the scheduled generations, each mapped to a boolean;
- `generation`: derived from `decisions`, the latest promoted scheduled generation, or 0 if none;
- `inference`: exactly `{path, sha256, bytes}`, with the path inside the selected generation's directory.

### 3.5 `final/started.json`

[`started_problems`](../games/connect4/alphazero_v2/official_evidence.py#L671): exactly `{selections, descriptions, runtime, execution_sha256, declaration_sha256}`.

- `declaration_sha256` is the token, and `execution_sha256` is the declared execution digest.
- `runtime` equals the declared runtime identity.
- `selections` are valid and equal to `state.json`'s.
- `descriptions` covers exactly the declared seeds, each exactly `{agent, opponents}`.

`agent`:

- exactly the 10 description keys;
- named `seed<S>-champion`;
- path and SHA-256 equal to the selection's inference;
- simulations and deployment conventions from the declaration.

`opponents`: exactly the declared ladder. [`opponent_problems`](../games/connect4/alphazero_v2/official_evidence.py#L621) gives each opponent kind its exact keys, its own name and its declaration-derived values:

- Negamax depth (and seeded ties);
- UCT simulations;
- v2 simulations and conventions;
- the declared 4D.2f checkpoint path and hash.

### 3.6 `final/seed-S.json` (`FinalSeedResult`)

Exactly 13 keys: `seed, completeness, selection, agent, overlap_families, tactical, tactical_nn_only, solved, value, ladder, nn_only_vs_random, calibration, evidence`. **No field is optional.**

Raw evidence, validated against the declaration and packages ([`raw_evidence`](../games/connect4/alphazero_v2/official_evidence.py#L892)):

| Path | Rule |
| --- | --- |
| `evidence` | exactly `{tactical, solved, calibration}` |
| `evidence.tactical` | exactly the sealed tactical row IDs; each exactly `{choices, visits, root_values, nn_only_choice}` |
| `evidence.solved` | exactly the sealed solved row IDs; each exactly `{choices, visits, root_values, raw_value}` |
| search evidence | one entry per declared tactical seed; choices legal in the row's position; 7 visit counts summing to the declared simulations, zero on illegal columns, maximal at the choice; root values finite in [−1, 1] and identical across seeds; NN-only choice legal; solved `raw_value` equal to the root value |
| `evidence.calibration` | records exactly `{game, ply, actor, value, outcome}`; games exactly `0..calibration_games−1`, each one contiguous block in order; plies `0..n−1`; actor = ply parity; value finite in [−1, 1]; outcome in {−1, 0, 1}, alternating sign; a draw has 42 plies; a decisive game has 7–42 plies and its last mover won |
| `overlap_families` | sorted, distinct strings, each the family of a sealed tactical or solved row |
| `ladder` | exactly the declared ladder opponents; each exactly `{summary, records, gate, opponent}` |
| `nn_only_vs_random` | exactly `{summary, records, gate}` |
| arena `records` | exactly the `2 × ladder_openings` paired games of the declared sealed openings, in order; each exactly 14 keys; see below |

Each arena record:

- takes its opening, family, stratum, prefix length, game index and colour from the package and the pairing;
- names the agent `seed<S>-champion` (or `-nn-only`) and the declared opponent;
- is not abandoned;
- has moves that extend the opening and [replay](../games/connect4/alphazero_v2/official_evidence.py#L825) legally to a finished game, under the oracle's bitboard rules (independent of the engine that played it);
- has `winner` and `result` equal to the replayed outcome;
- has one `{color, move, seconds}` decision per played move, with the mover's colour and that move;
- has finite non-negative seconds.

Two of these checks rest on invariants verified before they were enforced, on 468 real sealed rows with 4 search seeds:

- **Root value.** `root_values` are identical across search seeds, and the solved `raw_value` equals them bitwise. Both are the raw network value at the root, which does not depend on the search seed.
- **Visits.** Each seed's visits sum to the simulations and peak at the choice.

## 4. Declaration-derived completeness

| Set or count | Source |
| --- | --- |
| Seed keys of every per-seed mapping and `final_results` | `declaration.seeds` (canonical decimal strings; research: exactly 42 and 314159) |
| Seed of each result | its key, and its agent name `seed<S>-champion` |
| Ladder opponents (result `ladder`, `started.json` opponents) | `declaration.final.ladder` |
| Tactical / solved evidence row IDs | the `tactical-sealed` / `solved-sealed` package copies |
| Arena records per opponent and NN-only | `2 ×` the first `ladder_openings` rows of `openings-sealed`, in pairing order |
| Calibration games | `0..final.calibration_games − 1` |
| Search entries per row | `len(evaluation.tactical_seeds)`, visits summing to `evaluation.simulations` |
| Selection decisions | `champion.schedule` |

[`expected_units`](../games/connect4/alphazero_v2/official_evidence.py#L553) gives the evidence units per declared seed. Totals multiply by the number of seeds; the game kinds sum to `planned_evaluation_games` (6,272 for the research declaration).

| Unit kind | Count per seed |
| --- | --- |
| `development_arena_game` | \|schedule\| × 2 × `arena_openings` |
| `development_baseline_game` | \|schedule\| × 2 × \|baseline rows\| × \|baseline opponents\| |
| `development_tactics_row` | (\|schedule\| + 1) × \|champion tactical package rows\|: each candidate, plus generation 0 as the first check's incumbent (later incumbents are earlier candidates, whose rows the owner reuses) |
| `final_tactical_row`, `final_solved_row` | \|sealed tactical rows\|, \|sealed solved rows\| |
| `final_ladder_game` | \|ladder\| × 2 × `ladder_openings` |
| `final_nn_only_game` | 2 × `ladder_openings` |
| `calibration_game` | `calibration_games` |

Before this phase, acceptance required only `started == completed` per kind. A record consistently reduced by one unit (aggregates and every games-started copy rewritten) was accepted. It is now refused, for each of the 8 kinds.

## 5. Summary and evidence cross-checks

Every summary is re-derived, not trusted:

| Derived field | From |
| --- | --- |
| `seed`, `completeness` | the result's key; `"complete"` |
| `selection`, `agent`, `ladder[*].opponent` | `started.json` |
| `tactical` (complete set, overlap-excluded, excluded-row count) | tactical choices and visits over the sealed tactical rows |
| `tactical_nn_only` | NN-only choices over the sealed tactical rows |
| `solved` | solved choices over the sealed solved rows |
| `value` | solved raw values over the sealed solved rows |
| `ladder[*].summary` / `gate` | that opponent's records / the declared threshold (null for reported-only opponents) |
| `nn_only_vs_random.summary` / `gate` | its records / the declared NN-only threshold |
| `calibration` | the calibration records and the declared constant |

Raw evidence is cross-checked against itself:

- arena moves (replayed), `winner`, `result` and `decisions`;
- search choices against visits;
- root values across seeds, and against the solved raw value;
- calibration outcomes across a game's plies.

Quantities stored more than once must agree:

| Quantity | Copies |
| --- | --- |
| Declaration | token in `state.json`; SHA-256 of `declaration.json`; `started.json`; every selection |
| Selections | both history entries; `started.json`; every seed result |
| Agent and opponent descriptions | `started.json`; each seed result's `agent` and `ladder[*].opponent` |
| Selected checkpoint | selection `inference`; agent description `path`/`sha256` |
| Sealed-phase start | history `started_sha256`; `outcome.started_sha256`; SHA-256 of `started.json` |
| Seed results | `outcome.final_results`; SHA-256 of each file |
| Endpoint, training seconds, games started | `counters.time`; `outcome.completion`; `counters.evaluation.total_games_started` |
| Evidence units | `by_kind`; the declaration-derived counts; aggregates |
| Packages | declaration hashes; `packages/` copies |
| Runtime and execution identity | declaration; `started.json` |

Real-scale cost: re-deriving one seed's summaries takes about 6.5 s, almost all of it the 10,000-resample calibration bootstrap over 256 games. Replay and structural checks add a few seconds. The owner validates each result twice (A and B), adding about 30 s to a 24-hour campaign.

## 6. What COMPLETE means

The invariant chain `official_results` enforces ([campaign.py:946](../games/connect4/alphazero_v2/campaign.py#L946) → [`complete_record_problems`](../games/connect4/alphazero_v2/official_evidence.py#L1130)):

1. **Frozen declaration.** `declaration.json` hashes to the recorded token. The token is not rejected. The declaration satisfies the full declaration contract.
2. **Exact package set.** Every declared package's copy in `packages/` hashes to the declared value and is consistent with the declaration.
3. **Exact required seed set.** Every per-seed mapping, the history sequence and the certified result set cover exactly the declared seeds.
4. **Complete counters and budgets.** Every evidence unit started equals completed equals the declaration-derived count. Training finished every generation and self-play game. The completion endpoint, each seed's training time and the games started are within the declared limits, and equal in every copy.
5. **Complete started record.** `started.json` hashes to the recorded value, satisfies its schema and agrees with the recorded selections and the declaration.
6. **Complete, semantically valid result for each seed.** Each `final/seed-S.json` hashes to its certified value and covers exactly the declared opponents, rows, games and calibration workload. Every game replays. Every summary equals its re-derivation.
7. **Certified hashes.** All of the above are bound by the COMPLETE record.

Together these are **official evidence**. If any component is absent, malformed or inconsistent, `official_results` raises `NotOfficialEvidence` (or `StateCorrupt` for an unreadable `state.json`), and it returns exactly the documents it validated.

## 7. Exact shared-validator call sites

| Site | Where | Validators |
| --- | --- | --- |
| **A** before `final/started.json` is published | `Campaign._final` ([campaign.py:730](../games/connect4/alphazero_v2/campaign.py#L730)) | `load_context` (from the campaign directory), `started_problems` |
| **A** before each `final/seed-S.json` is published | `Campaign._final` ([campaign.py:739](../games/connect4/alphazero_v2/campaign.py#L739)) | `seed_result_problems` on the exact normalized object then written by `publish_view` |
| **B** before the completion barrier | `Campaign._complete` ([campaign.py:773](../games/connect4/alphazero_v2/campaign.py#L773)) | `load_context`, `certified_evidence_problems` (strict parse from disk → `selections_problems`, `started_problems`, `seed_result_problems`), `counter_problems` |
| **C** acceptance | `official_results` ([campaign.py:958](../games/connect4/alphazero_v2/campaign.py#L958)) | `complete_record_problems`: `envelope_problems`, `load_context` (→ `declaration_problems`, `package_problems`), `history_problems`, `certified_evidence_problems`, `counter_problems`, `completion_problems` |
| launch | `validate_declaration` ([campaign.py:135](../games/connect4/alphazero_v2/campaign.py#L135)) | `declaration_problems`, `package_problems` |

A violation at A or B raises before anything is certified: the campaign ends INCOMPLETE and nothing is published. The round-trip test records every call and its site. Each seed result is validated at A, B and C; `started.json` at A, B and C. Every call returns no problems.

## 8. Mutation coverage

The tests are in [`tests/test_alphazero_v2_campaign.py`](../tests/test_alphazero_v2_campaign.py), section "Phase 4D.3B.5 official evidence contract": 20 tests.

The section uses one genuine two-seed COMPLETE campaign (seeds 42 and 314159) with a richer tiny protocol than earlier fixtures:

- four ladder opponents: `random`, `negamax1`, reported-only `negamax4` (null gate) and the v2 search agent `initial_v2_512`;
- three sealed openings, including a prefix base and its mirror;
- two calibration games.

**Generator.** Mutations are generated from the produced artifacts, not from a hand list. For every schema path (list indices collapsed, each mutated at its first and last occurrence; dict keys such as seeds, row IDs and opponents never collapsed), the generator:

- deletes it;
- replaces it with another JSON type (object↔list, number→string, string→number, boolean→integer, null→string), and turns every integer into a boolean;
- adds an unexpected member to every object.

The only expected acceptances are the declared optional diagnostics ([`OPTIONAL_PATHS`](../games/connect4/alphazero_v2/official_evidence.py#L115): phase timings) and the declared free declaration items ([`DECLARATION_FREE_ITEMS`](../games/connect4/alphazero_v2/official_evidence.py#L48): notes, and non-research frozen records).

| Artifact | Schema paths (of concrete) | Mutations | Wrongly accepted |
| --- | --- | --- | --- |
| `final/seed-42.json` | 791 (3,079) | 1,951: 791 delete, 1,033 retype, 127 add | 0 |
| `final/seed-314159.json` | 788 (3,336) | 1,945 | 0 |
| `state.json` (through `official_results`) | 151 (163) | 395: 8 optional deletions and 1 optional addition, which are accepted as required | 0 |
| `final/started.json` (re-hashed into the record) | 121 | 286 | 0 |
| `declaration.json` (coherently re-tokened: every token copy and certified hash rewritten) | 235 (263) | 544, of which 2 free deletions (frozen records of a `test` declaration) are accepted as required | 0 |
| **Total** | | **5,121** | **0** |

The seed-result sweeps call `seed_result_problems` directly; the others go through `official_results` on disk. A one-off run outside the suite mutated **every concrete path** of both seed results (17,212 mutations). It accepted 0.

Targeted tests:

| Test | Shows |
| --- | --- |
| `test_reproduced_exploit_certified_result_without_scientific_fields_is_refused` | the exact blocker, and each of the 13 top-level keys deleted, are refused through `official_results` |
| `test_round_trip_produce_validate_serialize_parse_validate_complete_accept` | produce → validate (A) → serialize (the validated object is the published bytes) → strict parse → validate (B) → COMPLETE → `official_results` (C) returns the identical documents |
| `test_produced_artifacts_have_exactly_the_contract_shape` | genuine artifacts carry exactly the declared key sets, and counters equal `expected_units` |
| `test_new_producer_field_without_acceptance_semantics_stops_publication` (×3) | an undeclared field added to search rows, arena records or calibration records fails at A; the result is never published and the campaign is INCOMPLETE |
| `test_declaration_derived_seed_sets` | seed removed, added, renamed `042` or results swapped; seed 314159's genuine result certified as seed 42's |
| `test_declaration_derived_evidence_sets` | ladder opponent removed, added or renamed; tactical row removed; solved row removed; calibration game removed; calibration truncated; arena or NN-only game removed: **each refused even after every summary is re-derived from the forged evidence** |
| `test_summaries_are_cross_checked_against_their_evidence` | 14 summary-only forgeries (scores, gates, a gate on a reported-only opponent, metrics, calibration, sensitivity, selection, agent, opponent); 6 self-inconsistent evidence forgeries (flipped result, changed winner, truncated moves, changed raw value, non-maximal choice, changed outcome) refused by the structural checks before any summary comparison |
| `test_coherently_reduced_unit_counts_are_refused` | each of the 8 unit kinds |
| `test_optional_diagnostics_may_be_removed_but_stay_typed` | every phase timing may be removed (and all at once); a retyped one is refused; `OPTIONAL_PATHS` is exactly that one pattern |
| `test_research_ladder_descriptions_satisfy_the_contract` | the full research ladder (all 7 opponents, the real 4D.2f checkpoint) built by the producer's `ladder_opponents` satisfies the opponent schema; each field's removal and a wrong checkpoint hash are refused |
| `test_campaign_directory_holds_the_declared_packages` | a changed or missing package copy is refused |
| `test_validators_are_total_on_arbitrary_json` | every validator returns problems, never raises, on 10 arbitrary JSON values |

**Adding a field later.** Raw evidence and every non-derived object are exact-key, and validation runs at A. So a new producer field without declared acceptance semantics makes the genuine campaign fail before publication; the test above shows this for all three raw evidence kinds. A new *derived* field is defined by the single producer, so its acceptance semantics are "equal to its derivation" automatically.

## 9. Verification

Environment for torch rows: `/tmp/board-game-phase4c1-venv` (CPython 3.11.17, torch 2.10.0, NumPy 1.26.4), `PYTHONDONTWRITEBYTECODE=1`, all five thread variables set to `1`.

| Check | Result |
| --- | --- |
| Reproduction on `85dd988`, before any edit | the blocker and 1,273 of 1,298 single-path deletions **accepted** (§1) |
| Same probes on the new tree | all refused |
| New contract tests (20) | **20 passed** |
| `test_alphazero_v2_campaign.py` (155) + `test_alphazero_v2_launch_control.py` (31), **nothing deselected** | **186 passed**, 0 skipped, 130 s. Includes the 4D.3B.4 schema tests, the crash matrix, the 4D.3B.3 terminal-commit tests and `phase4d2f_adapter_is_hash_pinned` (deselected in 4D.3B.4). |
| All other backend `tests/` in the torch venv (AlphaZero v2, evaluation core, solver/oracle, packages, DQN, MCTS, neural), excluding only the five API modules (no `dotenv`) and the two import-isolation gates | **1,100 passed**, 0 skipped, 196 s |
| Torch-free import gate | `official_evidence` and `launch_control` import with torch blocked (added to `test_evaluation_core_imports_without_torch`) |
| All backend `tests/` in `.venv` (no torch) | **482 passed, 15 skipped** (unchanged from 4D.3B.4) |
| API import isolation (`.venv`) | **2 passed** |
| `packages verify` | exit 0; six packages, 2,100 rows; no package file changed |
| Scientific output unchanged | tiny two-seed campaign: `final/seed-42.json`, `final/seed-314159.json` and `selection.json` identical to `85dd988`'s, with timings and declaration hashes stripped |
| Re-freeze and independent reconstruction | byte-identical, `2e39713f…86ab` (§10) |
| Preflight on the new declaration | `"ok": true`, exit 0; all five gates pass; 6,272 planned games |
| Old tokens | the 4D.3B.4 token `9f725982…` refused like the four before it: through `Campaign`, `load_declaration`, preflight and the CLI in a fresh interpreter (exit 2, `REJECTED`, no directory), with the current bytes and with its original bytes from `85dd988` (`test_both_old_tokens_and_their_declarations_never_authorize[85dd988]`, plus the four earlier commits) |
| `git diff --check` | clean (including the two new files) |

## 10. New frozen declaration

`games/connect4/alphazero_v2/frozen/campaign-declaration.json`: 17,440 bytes. Regenerated from scratch with `campaign freeze` in the pinned runtime, after the rejected file was removed (its bytes remain in git at `85dd988`).

**SHA-256 / authorization token: `2e39713f6e2791c735406c3bec7def0d336adb6add1cd43d49319e83aac486ab`**

Confirmed four ways:

- the `freeze` output;
- `shasum -a 256` of the file;
- an in-memory reconstruction in a fresh pinned interpreter, from `build_declaration`, the configured runtime, the manifest, the frozen records and `FREEZE_NOTES`, which was byte-identical;
- the execution-source digest, recomputed from `shasum` of each of the 27 files without the provenance code, matching both the per-file map and the combined digest.

Against the rejected `9f725982…` declaration, exactly 3 of the 22 top-level fields changed:

| Field | Change |
| --- | --- |
| `execution_source` | Combined `bbc74893998e1971dcb245611b5ac94d6266fcf9b339f3750aa3c545fc03fa29`, 27 files (`official_evidence.py` added). **`training` (18 files) `22678c4b1c056f4b2b61a94fdc2eaaa0cd67de884a82c558ab15324406899e35`, unchanged.** `evaluation_and_launch` (9 files) is now `f6444ff21dfa4e6215766038c8177eecc95bf96fbafdebe47668a8f092708a70`. Only `campaign.py`, `launch_control.py` and the new `official_evidence.py` differ. |
| `launch_control` | `acceptance` now describes the evidence contract; `rejected_declaration_tokens` holds all five. `version` and `state_file` are unchanged: the lifecycle and the state format did not change. |
| `notes` | Phase 4D.3B.5; five rejected tokens; the evidence contract. |

Unchanged:

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
T=2e39713f6e2791c735406c3bec7def0d336adb6add1cd43d49319e83aac486ab
$PY -m games.connect4.alphazero_v2.campaign preflight --declaration $D     # must print "ok": true and exit 0
caffeinate -i $PY -m games.connect4.alphazero_v2.campaign launch --declaration $D \
    --campaign-dir experiment-output/phase4d3c-alphazero-v2-official-campaign --authorize $T
```

Exit codes are unchanged: 0 `COMPLETE` (read results only through `official_results`), 4 `INCOMPLETE`, 2 refused. Never rerun to resume.

## 11. Notes and limitations

1. **Consistency, not authenticity.** Someone with write access who rewrites every file, hash and copy consistently is outside this check; there is no signature. Within that scope one narrow coherent forgery remains:
   - **Calibration truncation.** Calibration records carry no moves. Truncating a decisive game by an even number of plies keeps its parity and last-mover outcome. If every calibration summary is also re-derived from the truncated records, the result validates. With the summary left as published (the "just update the hash" case), it is refused (tested). Closing this would require `evaluation.calibration_games` to record the game's moves, which is out of this phase's scope.
2. **Overlap families are evidence, not recomputed.** `overlap_families` comes from the archived training games, which the COMPLETE record does not certify. Acceptance checks that the list is sorted, distinct and drawn from the sealed rows' families, and re-derives every overlap-excluded summary from it. It does not re-scan the training archive.
3. **Acceptance is version-bound.** The certified declaration must carry this launcher's launch-control text and executed thresholds. A campaign is therefore accepted by the code it was declared with. Any code change already requires a new declaration.
4. **Planned game count.** It is now enforced exactly, through the per-kind unit counts. 4D.3B.4 listed it as not required.
5. **Unchanged from 4D.3B.4:** terminal-commit, signal and deadline semantics; non-resumability; durability; identity scope; and the deferred items (learned-checkpoint 512-simulation p95, searched-root-value diagnostic, `exhaustive_action_values` terminal guard).
6. **No real-scale run.** All evidence comes from tiny synthetic campaigns and forged records. Real-scale validator cost is estimated from benchmarks (§5).
