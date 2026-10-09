# Evaluation provenance and integrity

Phase 5E exports `connect4-season-evaluation`, `schema_version: 1`. A second
person can validate schedules, replay every board and recompute analytics without
the React UI. Browser exports and the Node verifier reuse the Season validator
and analytics; the benchmark runner manually steps the existing Season controller.

## Implementation registry

These integers identify explicit implementations, **not semantic-version guarantees**.
Backend constants live in `api/connect4/provenance.py`; the matching browser/Node
registry is `ui/src/evaluation/provenance.js`. Keep them aligned.

| Identifier | Value | Meaning |
| --- | ---: | --- |
| Engine | 1 | Current public 6×7 gravity engine/replay rules: Red first, four in any direction, full-board draw |
| Random | 2 | Uniform legal-column choice with Phase 5B injectable local RNG; previous implementation relied on global RNG |
| Negamax | 2 | `1d08741` public correction: dominant terminal scores, zero draws, bound-typed transposition entries, center-first exact ties and bitboards; unsafe-cache predecessor is version 1 |
| MCTS | 2 | `c213ddf` correction: alternating-mover UCT rewards, half-point draws, robust visit-count selection, immediate-win/safe-reply root guards and injectable RNG; legacy MCTS is version 1 |
| API contract | 2 | Revision-bound single-ply API/history, Phase 5B uint32/per-ply seed contract and additive safe manifest |
| API provenance | 1 | First safe public implementation-manifest contract |
| Stochastic seed | 1 | Seed XOR revision/player salts followed by uint32 avalanche (`ply_seed`) |
| Season schedule | 1 | Phase 5D seeded circle rotation, mirrored halves, domain-separated fixture/game seeds |
| Evaluation methodology | 1 | Phase 5D scoring, standings, sequential Elo and observed-score bootstrap |
| Export schema | 1 | First evaluation envelope/canonicalization/integrity contract |

Numerical evaluation constants come directly from `season/analytics.js`, and the
schedule identifier from `season/model.js`. No agent algorithm changes were made.

`GET /v1/connect4/provenance` is read-only, uses existing `Cache-Control: no-store`
and exact-origin CORS/Vary policy, and exposes only these safe identifiers and an
optional sanitized commit. History responses include the same manifest. Season
`compactHistory` retains it in each completed game's optional `backendProvenance`.
It exposes no search traces, private model settings, environment, paths or secrets.

## Historical provenance and source commits

Exports carry each game's `backend_provenance`, including its backend commit.
Legacy saved Seasons remain valid; games without captured metadata export null,
and `execution_capture` counts recorded/unrecorded games. The top-level registry
identifies the supported reference implementations, **not proof of a legacy
game's backend identity**. Such artifacts produce an explicit verifier notice.
The exporter never substitutes a new endpoint response for old game evidence.

The runner requires a supported preflight manifest, checks it before each game,
and checks that completed history matches it. Changed/unknown implementations stop
the run. Future implementation support requires an intentional verifier update.

Optionally configure backend `EVALUATION_SOURCE_COMMIT` with a real full lowercase
40- or 64-character Git object ID (test/config override: `SOURCE_COMMIT`). Missing
or malformed values become null. Optionally configure frontend
`VITE_EVALUATION_SOURCE_COMMIT`; this supplies the top-level artifact generator's
`source_commit`. No browser invokes Git. CLI top-level source commit defaults to
null, while each game's backend commit is retained. Do not fabricate a SHA.

Commits are declarations, not attestations. Dirty trees, wrong injection, changed
dependencies/Python RNG or mixed builds can prevent exact reproduction. Record the
actual reviewed revision and runtime environment with a real benchmark. No neural
checkpoint/model is involved; frozen research campaign artifacts remain untouched.

## Artifact contract and digest coverage

All export fields use snake_case:

| Field | Contents |
| --- | --- |
| `format`, `schema_version` | Versioned artifact identity |
| `exported_at` | ISO-8601 UTC export metadata |
| `provenance` | Reference versions, optional generator commit, execution-capture counts |
| `methodology` | Numeric scoring/Elo/bootstrap constants, algorithms, domains, sort and balance identifiers |
| `evidence` | Seed, field, length, entrant IDs/configs, exact ordered schedule, compact completed games, partial/complete counts |
| `derived` | Recomputed standings, sides, pairwise scores, full precision Elo/change/peak/low/history, bootstrap intervals or null |
| `integrity` | SHA-256, canonicalization/coverage identifiers, lowercase evidence digest |

Each completed game contains fixture ID, zero-based completion index, Red/Yellow
entrant IDs, uint32 seed, ordered API player configs, zero-based move columns 0–6,
count, terminal status/winner index and backend manifest or null. Schedule rows
include round/cycle. There are no duplicate boards. The entire future schedule
remains present for a partial evaluation.

`canonicalPayload` hashes **exactly**:

```text
{format, schema_version, provenance, methodology, evidence}
```

This includes implementation declarations/commits, identities/configs, schedule
and completion order, colors/seeds/moves/results, counts and material analytics
constants. It excludes `exported_at`, `derived`, `integrity`, UI/browser state,
cosmetic labels and machine-specific runtime metadata. Timestamp-only changes
preserve the digest. Boards and analytics are reconstructed rather than trusted.
Unfinished active moves and server session IDs are omitted; this evaluates completed
games and cannot import/resume browser state.

## Canonical JSON and verification

`sorted-json-ecmascript-v1` recursively sorts object keys by JavaScript UTF-16
lexicographic order, preserves array order, uses ECMAScript JSON string escaping
and shortest round-tripping finite numeric formatting, and inserts no whitespace.
Negative zero becomes zero. Unicode is not normalized. It rejects undefined,
nonfinite numbers, BigInt, functions, symbols, class instances, cycles, sparse or
decorated arrays, accessors and non-enumerable object fields. Plain and
null-prototype objects are supported. This is not a claim of RFC 8785 compliance.

SHA-256 hashes UTF-8 canonical bytes using browser `crypto.subtle.digest` or Node
`createHash('sha256')`. Web Crypto/validation failures block downloads with an
actionable error. Native Blob/ObjectURL downloads release URLs. CSV uses commas,
CRLF records, quotes/doubled quotes, and apostrophe protection for formula-like
text (also whitespace/control prefixes). Numeric negative Elo stays numeric.
Summary CSV repeats evaluation status and completed/scheduled counts per row;
for partial evaluations, `final_elo` means the latest rating for that exported
prefix, rather than a completed-season result.

```sh
node ui/scripts/verify-evaluation-export.mjs path/to/evaluation.json
```

The structured pure verifier checks envelope/schema/methodology, reconstructs and
validates the Season, checks exact schedule and pairwise color balance, validates
game provenance/order/seeds/configs, replays through terminal outcomes, checks
SHA-256 and compares all recomputed analytics. Errors identify a stage and affected
analytics path/game when possible; CLI failures return nonzero. Elo/history allows
absolute differences ≤1e-9 rating points for cross-engine exponent rounding; all
other analytics require exact agreement. Canonical evidence contains no rounded Elo.

The digest is a checksum, **not a signature**. Valid evidence edited together with
a recalculated digest is another internally valid artifact; compare with a
separately trusted published digest to establish unchanged evidence. Legal replay
does not prove the named agent selected the moves; execution reproduction requires
the declared implementation, seeds and runtime. Independent exact win/draw fixtures
and direct standings/Elo calculations supplement shared exporter/verifier tests.

## Bump rules

- Engine: rule, legality or terminal-outcome changes.
- Agents: search/scoring, meaningful move ordering/ties, stochastic/rollout policy
  or tactical guard changes.
- API/stochastic seed: material execution or seed derivation changes; API provenance
  for incompatible manifest contracts.
- Schedule: pairing generation/order, color assignment or seed derivation changes.
- Methodology: scoring/standings, Elo constants/formula, bootstrap sampling/PRNG,
  percentiles or minimum-game threshold changes.
- Export schema: incompatible machine-readable contract changes. Preserve old
  version verifiers or clearly reject unsupported versions.
- Cosmetic UI, labels, comments and export timestamp do not require algorithm bumps.

See [benchmark methodology](../benchmarks/connect4/canonical-season-v1.md),
[quickstart](quickstart.md) and [Phase 5E report](phase5e-evaluation-provenance-export.md).

## Victor competition extension (October 9, 2026)

Ordinary seasons still emit the frozen schema 1 / methodology 1 envelope. Victor
seasons emit schema 2 / methodology 2, retaining the same schedule, scoring,
analytics, canonicalization and integrity coverage. Neither old canonical results
nor their hashes are changed. Persistence remains Season v1 / Tournament v1.

The v2 top-level reference registry adds `victor_research: 1`: the bounded public
hybrid policy released at `31da353143c69c7e5ab4efd0fb6788a62e22e0ff`, merged at
`8a3c3e0e1e0d612ae121819d5fde4341aee8e3ab`. This identifies a reference policy,
not an exact replay promise. Methodology 2 adds explicit wall-clock budget,
seed-scope, historical replay and captured-provenance limitations. Engine 1
continues to identify board/replay semantics; it does not identify Victor.

The current API manifest in `api/connect4/provenance.py` identifies the core
engine and classical agents only. It is preserved verbatim per game and is not
silently upgraded to include Victor. Both the verifier and export panel explain
that the declared Victor reference version does not authenticate which algorithm
executed historical moves. Capture a backend source commit and runtime details
when conducting a real campaign. Do not infer missing historical identity.

The verifier accepts exactly Victor's `{type: 'victor_research'}` configuration,
rejects unknown agents and extra fields, and requires v2 for Victor evidence.
It independently reconstructs the schedule and boards and recomputes analytics.
The same CLI verifies both versions, including completed Victor seasons while
the UI research flag is disabled. See [integration report](victor-labs-integration.md).
