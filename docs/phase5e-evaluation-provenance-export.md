# Phase 5E — Evaluation Provenance & Export

Implemented October 7, 2026 (America/New_York), on
`phase5e-evaluation-provenance-export`. **GO for review and subsequent merge/deploy;
canonical benchmark execution requires separate authorization.** No commit, push,
merge, rebase, cherry-pick, deployment or canonical benchmark execution occurred.
The AlphaZero preflight branch and frozen artifacts were not modified.

## Repository gate

Fetched origin first; both local `main` and `origin/main` contained exactly
`58a88ad2f397ea388b1f98032241ed26eafef197`. GitHub had no associated pull request;
the user explicitly confirmed Phase 5D was reviewed and approved. The tree was
clean and the requested Phase 5E branch did not exist before creation.

## 1. Architecture

`ui/src/evaluation/` separates canonical serialization, version registry, export
projection/methodology, pure verification, CSV encoding, native download behavior
and the React export panel. The existing Season validator/replay and pure analytics
remain authoritative. Backend provenance is a small safe read-only manifest,
also included in history responses and retained alongside compact completed games.

The CLI verifier uses the same pure verifier with Node crypto. The benchmark
runner reuses `SeasonController` and manual sequential stepping, with isolated
in-memory storage, validated checkpoints and no second gameplay/evaluation engine.
Existing saved Seasons remain compatible through an optional per-game metadata field.

## 2. Export schema

`connect4-season-evaluation`, schema 1, snake_case fields:

- `evidence`: seed/field/length, entrant identities/configs, full deterministic
  schedule, compact completed games and explicit complete/partial counts.
- `methodology`: numeric scoring, Elo and bootstrap settings; formula, sampling,
  sort, seed derivation/domain and side-balance identifiers.
- `derived`: fresh standings, Red/Yellow splits, pairwise scores, full-precision
  Elo/change/peak/low/history and bootstrap intervals or null.
- `provenance`: explicit reference implementations, optional generator commit and
  recorded/unrecorded execution-provenance counts; game evidence holds backend manifests.
- `integrity`: SHA-256, canonicalization/coverage identifiers and lowercase digest.
- `exported_at`: ISO UTC metadata, outside the digest.

Completed games include fixture/completion index, Red/Yellow IDs, seed, ordered
API configs, columns/count and replayed terminal result. No board snapshots or
live-session IDs are exported. Partial analytics are explicitly provisional.

## 3. Canonicalization

Recursive UTF-16 lexicographic object-key sorting, preserved array order,
ECMAScript JSON primitive formatting and no whitespace. Finite numbers retain
round-trip precision; no Elo rounding is introduced. Unsupported/non-JSON values,
undefined, cycles, sparse/decorated arrays and accessors are rejected.

Hash coverage is exactly `{format, schema_version, provenance, methodology,
evidence}` over UTF-8. Timestamp, derived analytics, UI/browser/cosmetic state
and separate runtime metadata are excluded. Browser hashing uses Web Crypto;
Node uses `createHash('sha256')`. This project canonicalization is not advertised
as RFC 8785 compliance. See [the complete contract](evaluation-provenance.md).

## 4. Provenance

Engine 1; Random 2; corrected Negamax 2; corrected MCTS 2; API contract 2;
API provenance 1; stochastic seed 1; Season schedule 1; methodology 1; schema 1.
These integer implementation identifiers do not promise semantic compatibility.
The provenance guide documents implementation/history meanings and bump policy.

Backend `EVALUATION_SOURCE_COMMIT` and optional frontend
`VITE_EVALUATION_SOURCE_COMMIT` accept full lowercase Git object IDs. Missing or
invalid values are null; no SHA is fabricated. This local uncommitted build used
null source commits. Commit declarations do not attest clean builds or dependencies.
Legacy game manifests remain null and trigger an explicit notice, rather than a
retrospective claim of known backend identity.

## 5. Browser exports

Season Lab now contains a compact **Evaluation export** panel beneath execution
controls. It displays counts/status, digest, schema/methodology/schedule versions,
optional commit and expandable provenance/integrity explanation. Actions:

- Export evaluation JSON.
- Export games CSV (completed games only; `|`-separated zero-based columns).
- Export summary CSV (one row per entrant; missing intervals blank).
- Copy evidence digest, with live successful-copy feedback.

Every download constructs and verifies fresh evidence/analytics. Crypto and
validation failures block downloads and display an actionable error. Native
Blob/ObjectURL downloads release URLs; no file-export dependency was added.
CSV quotes/doubles quotes and protects textual formula prefixes, including
leading whitespace/control bypasses. Numeric negative Elo remains numeric.
A final fix prevents an older in-flight action from replacing a newer preview.
Summary CSV includes explicit status/completed/scheduled columns so provisional
ratings remain identifiable when opened independently of the UI/JSON.

## 6. Verifier and tampering

Pure structured results identify failure stages. Verification checks envelope,
schema, methodology, provenance, canonical shape, exact reconstructed schedule,
per-pair Red/Yellow balance, seeds/configs/colors/completion order, all boards and
terminal outcomes, digest and recomputed standings/Elo/history/sides/pairwise/bootstrap.
The CLI prints PASS, schema, hash, field, counts/status and versions; failures exit
nonzero. Elo-only tolerance is 1e-9 points for cross-engine exponent rounding.

Tests detect changed moves, game seeds, agent settings, winners, schedule/completion
order, Elo/history, standings, sides, pairwise results, intervals, digest,
methodology, provenance, counts, schema and unknown evidence fields. Timestamp-only
changes retain the digest and verify. Golden partial/complete JSON fixtures and
synthetic draw/stochastic/duplicate/maximum fixtures use stable seeds. Known exact
one-win/draw and repeated-opponent fixtures provide independent expectations;
the real smoke outputs receive direct standings/Elo/CSV checks.

## 7. Canonical benchmark

[JSON definition](../benchmarks/connect4/canonical-season-v1.json) and
[methodology](../benchmarks/connect4/canonical-season-v1.md): Random; Negamax
depths 2/4/6/8; MCTS 100/400/800. Eight entrants, four games per pair, 112 total,
28 per entrant, season seed **20261008**. Every pair has two games per color;
every entrant has 14 games per color. This is a modest portfolio sample, not
proof of solved strength. **This canonical benchmark was not executed.**

## 8. Safe runner

Default dry-run prints configurations, schedule count, stochastic work upper
bounds, API and output targets, with no HTTP/files/gameplay. `--execute` is required
for actual execution. Default API is `http://127.0.0.1:8000`; nonlocal origins
require `--allow-nonlocal`, and redirect/credential/path/query/fragment targets
are refused. No explanation/provider endpoint is called.

Each real run creates a new exclusive timestamp/UUID directory containing
`evaluation.json`, `games.csv`, `summary.csv`, `run-metadata.json`. JSON checkpoints
atomically after each completed game; CSVs finalize on completion/caught failure.
Backend manifest checks before each fixture and on completed history reject drift.
No automatic resume/import is implemented. Hard kills retain the last completed
checkpoint but may lack final CSVs/metadata; interrupted mutations are not blindly retried.

## 9. Actual cheap integration benchmark

Only the authorized smoke configuration ran against an isolated local backend:
Random / Negamax 1 / Negamax 2 / Random; seed 1234; two games per pair.

| Measurement | Result |
| --- | ---: |
| Games / plies | 12 / 210 |
| Maximum concurrent POSTs | 1 |
| POST attempts | 222 (12 starts + 210 plies) |
| Retired prior sessions | 11 |
| Descriptive total wall time (final run) | 339.63 ms |
| Per-game wall time (final run) | 8.24–54.01 ms |
| JSON verifier | PASS |
| Games / summary CSV rows | 12 / 4 |
| Independent standings, splits and Elo | PASS |
| Captured backend manifests | 12; none unrecorded |

| Entrant | Played | W-D-L | Points | Final Elo (display rounded here only) |
| --- | ---: | --- | ---: | ---: |
| Random #1 | 6 | 2-0-4 | 2 | 1477.591746 |
| Negamax 1 #2 | 6 | 4-0-2 | 4 | 1522.408254 |
| Negamax 2 #3 | 6 | 6-0-0 | 6 | 1564.251032 |
| Random #4 | 6 | 0-0-6 | 0 | 1435.748968 |

Artifacts are in ignored local directory
`benchmarks/connect4/runs/smoke-season-v1-2026-10-08T02-13-24-956Z-e02767f8/`.
Their evidence SHA-256 is
`a3ef914e386fee36355f9d7182d6ef6866c346beaaf8f05fd311c94023885fdc`.
The UTC directory date is October 8; execution was October 7 in the client timezone.
No expensive canonical run directory was created. An earlier 12-game smoke run
took 357.17 ms; the final run after adding CSV status columns took 339.63 ms.
Both produced the identical evidence digest despite different export/runtime timestamps.

## 10. Maximum synthetic performance

Apple M5, macOS/arm64, Node v25.9.0; legal all-draw synthetic 12-entrant,
8-game/pair Season: **528 games / 22,176 plies**. No actual 528-game campaign.
Representative final measurements from `node ui/scripts/evaluation-benchmark.mjs`:

| Operation / size | Result |
| --- | ---: |
| Export construction including full validation | 249.26 ms |
| Canonical serialization | 5.34 ms |
| Node SHA-256 | 0.13 ms |
| Web Crypto SHA-256 | 0.27 ms |
| Games CSV construction with reconstruction | 130.47 ms |
| Summary CSV construction with recomputation | 131.91 ms |
| Parsed-JSON verifier | 155.91 ms |
| Pretty JSON size | 1,014,263 bytes |
| Canonical hashed payload size | 381,189 bytes |
| Games CSV size | 107,668 bytes |
| Summary CSV size | 1,810 bytes |

Construction/CSV/verification average five repetitions; serialization/hash average
20. CSV timings intentionally include evidence validation; these are local
infrastructure timings, small relative to expensive gameplay and not strength metrics.

## 11. Files changed

| Group | Files |
| --- | --- |
| Backend | Added `api/connect4/provenance.py`; updated `api/app.py`, `api/connect4/__init__.py` |
| Evaluation modules | Added `ui/src/evaluation/canonical.js`, `provenance.js`, `export.js`, `verify.js`, `csv.js`, `download.js`, `ExportPanel.jsx` |
| Season integration | Updated `ui/src/season/model.js`, `ui/src/pages/SeasonLab.jsx`, `season-lab.css` |
| CLI / measurements | Added `ui/scripts/evaluation-node.mjs`, `verify-evaluation-export.mjs`, `run-connect4-benchmark.mjs`, `verify-benchmark-smoke.mjs`, `evaluation-benchmark.mjs` |
| Definitions | Added `benchmarks/connect4/canonical-season-v1.json`, `canonical-season-v1.md`, `smoke-season-v1.json`; updated `.gitignore` for local runs |
| Backend tests | Added `tests/test_connect4_provenance.py`; updated `test_connect4_history_api.py` to validate additive manifest |
| Frontend tests | Added `ui/tests/evaluation.test.js`, `benchmark.test.js`, `fixtures/evaluation.js`, `fixtures/evaluation-partial.json`, `fixtures/evaluation-complete.json` |
| Browser | Added `ui/e2e/evaluation.smoke.spec.js`; updated `ui/playwright.config.js` preserving prior smoke coverage |
| Documentation | Updated `README.md`, `docs/quickstart.md`; added `docs/evaluation-provenance.md`, this report |

No dependencies, existing AI algorithms, neural models or frozen artifacts changed.

## 12. Regression review

Play retains all public agents, both turn orders and grounded Analysis. Match Lab
retains Human/Human, Human/AI, AI/AI and replay. Tournament retains seeded AI brackets,
Human participation and recovery. Season retains balanced schedules, execution,
analytics, persistence, paused reload/recovery and local replay. All prior product
tests remain; the sole internal evidence extension is optional per-game provenance.

## 13. Verification

| Check | Result |
| --- | --- |
| Full backend suite | 501 passed; 14 existing optional PyTorch skips |
| Full frontend suite | 398 passed; no skips |
| Full Chromium regression | 122 passed |
| Firefox configured product/export smoke | 33 passed |
| WebKit configured product/export smoke | 33 passed |
| Browser total | 188 passed, 2.6 minutes |
| Focused rebuilt export tests | 12 passed across three engines |
| Final captured-provenance screenshot/download tests | 12 passed across three engines |
| General production build | Passed |
| Render production build with explicit HTTPS API base | Passed; no deployment/contact |
| Separate-origin local browser build | Passed |
| Actual 12-game local runner + CLI/independent verification | Passed |
| Maximum synthetic export benchmark | Passed |
| `git diff --check` | Passed |

The first backend pass found five exact-key history assertions; they were updated
to require/validate the additive provenance field, with all original replay checks
retained. Final backend tests are green. The 14 skips require unavailable PyTorch;
no mandatory product/API tests were skipped. Providers were disabled and backend
tests prohibit live HTTPS. All gameplay requests targeted isolated localhost;
no production requests, deployments or 112-game benchmark execution occurred.

Reproduction (root unless noted):

```sh
PYTHONDONTWRITEBYTECODE=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests -q -rs
node ui/scripts/evaluation-benchmark.mjs
node ui/scripts/run-connect4-benchmark.mjs --config benchmarks/connect4/canonical-season-v1.json
# ui/:
npm test
npm run build -- --outDir dist/general
VITE_API_BASE=https://board-game-ai-lab.onrender.com npm run build:render -- --outDir dist/render
VITE_API_BASE=http://localhost:8009 npm run build
# Isolated backend root command, providers disabled:
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' CORS_ALLOWED_ORIGINS=http://localhost:4176 .venv/bin/gunicorn api.app:app --bind 127.0.0.1:8009 --workers 1 --threads 4
# ui/, separate terminal:
npm run preview -- --host localhost --port 4176 --strictPort
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/phase5a-browsers PLAYWRIGHT_BASE_URL=http://localhost:4176 PLAYWRIGHT_API_URL=http://localhost:8009 npx playwright test
# Root: inexpensive smoke only; never substitute canonical config during implementation:
node ui/scripts/run-connect4-benchmark.mjs --config benchmarks/connect4/smoke-season-v1.json --api http://127.0.0.1:8009 --execute
node ui/scripts/verify-evaluation-export.mjs path/to/smoke-run/evaluation.json
node ui/scripts/verify-benchmark-smoke.mjs path/to/smoke-run
git diff --check
```

## 14. Responsive and accessibility

375px and 320px checks pass without page overflow. SHA-256 wraps as selectable
text, with copy access; controls use clear native button names and visible focus.
Keyboard activation/tab return and perceivable live-copy feedback are tested in
all engines. Status is conveyed in text, not color alone. Panel metadata wraps;
mobile controls fit in two columns. Details disclose legacy provenance limitations.
Visual review supplements role/focus/layout tests; no formal screen-reader
certification is claimed.

## 15. Screenshots

Ignored local `ui/playwright-report/phase5e/` contains eight reviewed PNGs:

- `partial-1440.png`, `partial-375.png`, `partial-320.png`.
- `complete-1440.png`, `complete-375.png`, `complete-320.png`.
- `provenance-integrity.png` (expanded details).
- `desktop-complete.png` (full completed Season dashboard/export controls).

These use **synthetic fixtures, seed 1234**, not canonical strength measurements.
Final fixtures show captured per-game manifests. Screenshots contain no secrets.
Dry-run output was recorded locally as text; a terminal screenshot was unnecessary.

## 16. Known limitations

Checksums are not signatures or proof that an agent chose a move. Publish a trusted
digest separately and retain reviewed source/runtime details. Legacy games cannot
recover missing historical backend provenance. Optional commit injection cannot
attest dirty builds. Unknown future implementations/schemas require verifier support.
There is no browser import, archive service, distributed or automatic resume,
cross-tab ownership, account or cloud storage. Hard termination can leave CSVs
unfinished; completed JSON checkpoints remain. Backend Python/dependency/runtime
metadata must be supplied alongside a real run. Pool-relative Elo and descriptive
IID bootstrap intervals retain all Phase 5D sampling/dependence limitations.

## 17. Future canonical launch recommendation

On similar Apple M5 hardware, existing measured MCTS latency and a conditional
20–35-ply/game workload suggest **3–12 minutes**, with **15–20 minutes reserved**.
This is a planning extrapolation, not measured canonical runtime or a strict bound.
The 357-ms cheap integration run cannot estimate expensive search strength/runtime.
See [the benchmark guide](../benchmarks/connect4/canonical-season-v1.md) for exact
latencies, hardware/runtime recording and limitations.

After review/merge and **separate canonical-run authorization**, with a provider-
disabled backend already listening locally at 8000, the exact launch command is:

```sh
node ui/scripts/run-connect4-benchmark.mjs \
  --config benchmarks/connect4/canonical-season-v1.json \
  --api http://127.0.0.1:8000 --execute
```

**This command was not executed.**

## 18. GO / NO-GO

**GO for Phase 5E review and subsequent merge/deploy.** Keep production rollout
subject to normal local/public smoke review; no deployment has occurred here.
**NO-GO for launching the full canonical benchmark until separately authorized**
after this infrastructure is reviewed/frozen. Portfolio results/README/resume
completion follow that real reviewed run.
