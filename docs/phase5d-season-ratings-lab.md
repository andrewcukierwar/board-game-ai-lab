# Phase 5D — Season & Ratings Lab

Season Lab at `/connect4/season` is the primary comparative agent-evaluation UI.
It runs repeated AI-only fixtures with exact Red/Yellow balance, local evidence,
standings, pairwise results, pool-relative Elo and observed score-rate uncertainty.
Tournament Lab remains the separate single-elimination competition, with optional
Human participation. Play, Match Lab, Agents and Research remain available.

## Prerequisites and scope

Remote state was fetched before editing. Both `main` and `origin/main` contained
Phase 5C commit `ba2ac9b70f9cdb049adbba7e9cb19c8d2b3376ae`; the working tree was
clean, and `phase5d-season-ratings-lab` did not exist. GitHub had no associated
pull request; the user explicitly confirmed Phase 5C was reviewed and approved
outside GitHub. Work then began on the new branch. No commits, pushes, deployment,
AlphaZero branch integration, or frozen research artifact changes are part of this phase.

This phase supports only the public Random, Negamax 1/2/4/6/8 and MCTS
100/400/800 configurations. Configurations may repeat; identities do not.
Each slot has a stable `entrant-N` identity and seed number. Human seasons,
learned agents, Mancala, remote multiplayer, server persistence, workers,
streaming, universal ratings, matchmaking, Glicko and TrueSkill are out of scope.

## Architecture

| Module | Responsibility |
| --- | --- |
| `ui/src/season/model.js` | Pure schedule, fixture plans, compact records, replay validation and canonical hydration |
| `ui/src/season/analytics.js` | Pure standings, side splits, pairwise results, Elo/history and bootstrap |
| `ui/src/season/storage.js` | Versioned localStorage adapter, safe load/save errors |
| `ui/src/season/controller.js` | Sole HTTP mutation owner, timers, sequential scopes, authoritative reconciliation |
| `ui/src/season/useSeason.js` | React subscription and retained controller identity across SPA remounts |
| `ui/src/season/Schedule.jsx` | Round browser, workload/progress, preview/active/replay selection |
| `ui/src/season/MatchViewer.jsx` | Shared board, playback controls and local timeline |
| `ui/src/season/AnalyticsViews.jsx` | Standings, ratings and responsive pairwise views |
| `ui/src/season/RatingHistory.jsx` | SVG comparison chart and textual history |
| `ui/src/pages/SeasonLab.jsx` | Setup, dashboard hierarchy, execution, completion and reproducibility |
| `ui/src/pages/season-lab.css` | Season-specific layout; established lab styling is reused |

The model and analytics have no React or HTTP dependencies. Reused Connect 4
components include GameBoard, MatchControls, MatchTimeline, replaySnapshot,
validateMatchHistory, competitorLabel/toPlayerPayload and requestMatchPly.
Tournament utilities supply public preset validation, deterministic seed derivation,
seeded Fisher–Yates permutation, default configurations and replayColumns.
There is no new board implementation, chart dependency, backend endpoint or RNG API.

## Field and season length

Supported field sizes are 4, 6, 8, 10 and 12; unsupported and odd sizes are rejected.
The default eight entrants are Random, Negamax 2/4/6/8 and MCTS 100/400/800.
Supported games per pairing are 2, 4 and 8, defaulting to 2.

`total games = N × (N − 1) / 2 × games per pairing`.

Examples: 8 × 2 yields 56 games, 8 × 4 yields 112, and 12 × 8 yields 528.
Setup displays the pairing multiplication, exact game total and a reminder that
higher-cost entrants may take several minutes or longer. No exact runtime is promised.
Reset default field restores the representative presets. Create allocates no API sessions.

## Schedule and reproducibility

Schedule version 1 uses the circle method. Seeded Fisher–Yates first permutes
slot identities; one identity stays fixed and the remaining ring rotates.
A single half has `N − 1` conceptual rounds, each containing `N / 2` fixtures.
Every entrant appears exactly once per round; every unordered pair appears once
per half. Fixtures in a round still execute sequentially.

Each balanced cycle contains the first half followed by the same round/pair order
with colors reversed. Two, four and eight games per pairing produce one, two and
four balanced cycles. Every pair has exactly half its games with each entrant
as Red, and every entrant finishes with exactly `(N − 1) × games per pairing / 2`
games per side. A draw is a completed fixture, with no rematch.

The unsigned 32-bit season seed accepts 0–4294967295. The UI initializes it with
`crypto.getRandomValues`, and lets the user edit it. The existing domain-separated
FNV-1a/32-bit avalanche derives:

- Initial permutation seed: `season:schedule:v1`.
- First-half color bit: `season:color:<cycle>:<round-index>:<pair-index>`;
  the return fixture reverses that bit.
- Game seed: `season:game:v1:<fixtureId>`.
- Bootstrap seed: `season:bootstrap:<entrantId>`.

Fixture IDs are `c<cycle>-h<half>-r<round-within-half>-f<pair-index>`.
The exact schedule is persisted, and hydration verifies it against regeneration.
The same configurations, slot order, seed and length produce the same schedule,
colors and game seeds. Slot order is meaningful even for duplicate configurations.
No timestamps, runtime hashes or global RNG state participate in scheduling.
The existing backend `rng_seed` controls stochastic play. Reproducibility assumes
unchanged engine behavior; the season schema is not an engine version archive.

## Canonical evidence and persistence

Storage uses the separate key `board-game-ai-lab:season:v1`.
Canonical state includes schema/schedule versions, season ID/seed, field size,
games per pairing, entrants, exact schedule, next game index, season status,
completed games, active fixture columns/session identity and retained replacement ID.
Runtime execution mode and playback timers are not persisted.

Each completed record holds fixture ID, game seed, color/entrant identities,
API player configurations, move columns, terminal result, move count and completion
index. Boards and replay snapshots are reconstructed in memory. Standings, ratings,
history, intervals and pairwise aggregates are derived rather than stored.
Fixture status/result are checked against the compact evidence during hydration.

Validation rebuilds the canonical season in schedule order and rejects malformed
JSON, unsupported versions, Human/invalid presets, bad identities, altered schedules,
duplicate fixture IDs, wrong seeds/colors/configs, gaps, invalid terminal results,
illegal columns, moves after terminal play, impossible indices/statuses and session
identity mismatches. Persisted rating/standings/history aggregates are rejected.
Unknown top-level fields are discarded. Active terminal columns are permitted so a
crash between evidence persistence and result compaction can finish by history GET.
Unconfirmed starts without a received game ID cannot contain moves.

The active season is local to this browser/origin. Storage errors display an
in-memory-only notice. New season replaces the saved season after explicit creation;
it carries the known prior API session ID forward for sequential retirement.
There is no import/export, season archive, cross-tab locking or server-side season store.

## Execution and recovery

The controller owns one mutation lock and one timer. A WeakMap retains that owner
for the same HTTP/storage pair across SPA route remounts, including in-flight requests.
Execution never schedules concurrent games or preallocates the full field on the API.
Only the current fixture has a live API game; each following start supplies the known
prior `replace_game_id`. Result evidence persists before the next start.

| Control | Behavior |
| --- | --- |
| Start current fixture | Start paused at move 0, then verify authoritative history |
| Next move | One revision-bound AI ply, followed by history reconciliation |
| Autoplay game | Finish only the current scheduled game |
| Run round | Finish the remaining fixtures of the current conceptual round |
| Run season | Continue through all remaining fixtures sequentially |
| Pause | Cancel future scheduling; allow any in-flight mutation/read to settle and persist |
| Playback speed | Slow/normal/fast delay between requests; never an AI search-time promise |

The UI distinguishes Paused, Running current game, Running current round,
Running season, Recovering, Recovery required and Season Complete.
Selecting a fixture pauses execution. Rewinding disables live moves until returning
to live. Completed fixtures always use local replay, including after replacement,
backend restart or browser reload.

Reload never resumes autoplay. A saved known live session is reconciled by history
GET and remains paused. Without a live game, local results immediately reconstruct
all analytics. Interrupted starts require explicit restart.

- Lost move response: GET authoritative history, validate the seed/players/revision
  and already accepted move prefix; never blindly repeat the POST. If the lost
  response was terminal, the compact result is recorded; an explicit Continue
  after confirmed result clears the recovery notice before the next fixture.
- `agent_busy`: pause, reconcile and require explicit Refresh history / Continue.
- Expired session: preserve completed games and derived ratings; mark only the
  active fixture interrupted. Explicit restart uses its same seed, colors and configs.
- Lost start response: persist the interrupted state, never repeat automatically.
  An unknown allocated session cannot be retired by ID and expires normally.
- Invalid/unavailable history: block moves until explicit successful reconciliation.

An interrupted or unfinished game contributes nothing to standings or Elo.
Restart starts only that fixture at move 0; earlier results are retained.

## Standings, sides and pairwise results

A win scores 1, a draw 0.5 and a loss 0. Score rate is points / played, displayed
as a percentage. Standings order is points, then score rate, then entrant seed number.
Played, W-D-L, points and score rate are shown alongside separate Red and Yellow
W-D-L and score percentages. Zero-game rates show a dash.

Pairwise results derive both perspectives of every completed game exactly once.
Desktop uses an opponent-seed matrix with W-D-L and score percentage; accessible
cell text includes the entrant, opponent, played count and perspective. Mobile uses
a selected-entrant dropdown and opponent list with played, W-D-L and score rate.
Standings and Elo leaders are calculated separately and can differ.

## Elo and history

Every entrant begins at 1500; K is fixed at 24.

```text
EA = 1 / (1 + 10 ^ ((RB − RA) / 400))
delta = 24 × (SA − EA)
RA' = RA + delta
RB' = RB − delta
```

Observed score is 1/0.5/0. Ratings update in canonical scheduled completion order.
Both changes sum to zero (within floating tolerance). Full precision is preserved
between updates; UI ratings/change/peak/low display rounded integers.
Ratings sort by descending full-precision Elo, then entrant seed number.

History derives every entrant's rating after every completed season game, including
unchanged values when that entrant did not play. The SVG compares user-selected
entrants, labels progress/Elo axes, includes a 1500 reference line, and uses labeled
legend controls and line patterns as well as colors. A summary table shows start,
current, peak and low; expandable game-by-game values provide a text equivalent.

Elo is descriptive within this field, pool-relative, and schedule/order dependent.
It is not a universal Connect 4 rating, and its relative position is not a statistical
significance claim. Balanced colors reduce first-player bias but do not eliminate
sampling variation. The same seeded result evidence reproduces the ratings.

## Observed score-rate uncertainty

For an entrant with at least **8 completed games**, collect per-game observations
1, 0.5 or 0. Independently resample that observed vector with replacement, using
**1000** samples of the same length and a local seeded Mulberry32 PRNG. Linear
interpolation at empirical bootstrap percentiles 2.5% and 97.5% yields the displayed
**observed score-rate bootstrap 95% interval**. This is not an Elo confidence interval.
The PRNG seed is derived from season seed, entrant identity and the fixed bootstrap
domain; it neither consumes nor mutates game/global randomness.

Below eight games, the UI states “Too few games for a useful interval.”
These are descriptive IID per-game resampling intervals. Repeated opponents and
shared deterministic search behavior can introduce dependence that they do not model.
They do not cover engine changes, opponent-pool choice or unseen outcomes. Homogeneous
all-win/all-loss/all-draw samples yield collapsed intervals; those do not establish
certainty about true playing strength. No rating uncertainty model is introduced.

## Scale and performance

`node ui/scripts/season-benchmark.mjs` builds a synthetic completed 12-player season
with 8 games per pairing, **528 games**, and legal **42-ply** compact draw histories
(**22,176 columns**). No high-cost actual 528-game season runs.

Representative local timings (Node, October 7, 2026):

| Operation | Milliseconds |
| --- | ---: |
| Create schedule | 0.20 |
| Serialize | 0.25 |
| Parse and fully validate/hydrate | 71.80 |
| Standings and side splits | 0.10 |
| Elo and complete rating history | 0.14 |
| Bootstrap all 12 entrants, 1000 samples each | 6.70 |
| Pairwise results | 0.09 |
| Reconstruct one 42-ply replay | 0.10 |

Serialized evidence is **295,236 UTF-8 bytes** (about 288 KiB; roughly 590 KB if
counted as UTF-16 storage). This is comfortably below common multi-megabyte limits,
though available quota depends on the browser/origin. Creation averages 200 runs;
hydration/bootstrap average 5, other analytics/serialization 20, replay 200.
These figures measure local derivation, not AI runtime or browser-render latency.
Active-game updates preserve completed-evidence references so React does not rerun
bootstrap calculations on each ply. Schedule browsing renders one round at a time.
The full rating-value table remains collapsed until requested.

## Real local integration season

`ui/scripts/verify-season.mjs` used the actual isolated Flask API with provider
calls disabled, seed **1234**, field Random / Negamax 1 / Negamax 2 / Random,
2 games per pairing. It completed **12 games, 210 plies**, maximum simultaneous
POSTs **1**, and **11 retired sessions**; evidence serialized to **6,360 bytes**.
Observed zero-delay-controller wall time was 687ms; this is a local sanity measurement,
not a user-facing runtime estimate. Every game had an actual backend terminal result.

| Entrant | Played | W-D-L | Points | Final Elo (full precision) |
| --- | ---: | --- | ---: | ---: |
| #3 Negamax 2 | 6 | 6–0–0 | 6 | 1564.251032325214 |
| #2 Negamax 1 | 6 | 4–0–2 | 4 | 1522.408253539209 |
| #1 Random | 6 | 2–0–4 | 2 | 1477.591746460791 |
| #4 Random | 6 | 0–0–6 | 0 | 1435.748967674786 |

The script independently recomputed standings and the direct Elo formula from raw
outcomes, verified all six pairs play twice with swapped colors, checked replaced
sessions return 404, locally replayed all terminal games, and hydrated a new paused
controller from browser-compatible persisted evidence. Duplicate Random identities
remained distinct. These results do not establish general agent strength.

## Accessibility and responsive review

Setup fields have explicit labels; seed help, entrant slot labels, workload and
completion announcements are exposed to assistive technology. Schedule fixtures
are buttons with text Red/Yellow assignments and selected state. Run/Pause buttons
reflect execution/recovery locks, and turn status is textual. Charts have title and
description, labeled selection controls, summary and full value tables. Tables use
captions and scoped row/column headers. Pairwise cell labels explain perspective.

375px and 320px layouts use stacked setup/viewer cards and a vertical pairwise list.
Standings/ratings scroll within named, keyboard-focusable regions, with mobile
scroll hints. Page-level overflow is checked. Visible focus and keyboard fixture
start/move/replay flow are tested, including WebKit's macOS Option-Tab behavior.
The shared navigation now wraps cleanly at tablet widths and uses three mobile rows.
Automated roles/focus/copy checks and visual screenshot review are performed;
there is no claim of a formal screen-reader certification.

## Tests and verification

Pure tests cover all 15 field/length combinations, mirrored halves, exact pair and
side counts, one appearance per round, deterministic seeds/IDs, duplicate configs,
invalid options, compact replay, canonical reconstruction and malformed persistence.
They also cover standings/scoring/splits/pairwise ties, Elo invariants/history and
bootstrap threshold/determinism/bounds/extreme samples/global RNG independence.

Controller tests cover creation without sessions, one-session start, one-ply step,
mutation lock, pause/unmount settlement, all execution scope boundaries, mid-round
continuation, no draw rematches, result-before-next-start persistence, expiry/restart,
lost move/start responses, `agent_busy`, wrong history seed, paused reload and terminal
crash recovery. A maximum-scale mock executes all 528 games without parallel POSTs.

Browser tests cover discovery/setup/options/workload, schedule preview and active
selection, Next move, review, Run round/season, Pause, compact persistence, reload,
completed replay, ratings/history/pairwise, distinct completion leaders, recovery,
route remount locking, 528-game synthetic rendering and narrow mobile layouts.
The independent browser Elo calculation also checks displayed final ratings.
Existing Play, Analysis, Match Lab, AI Tournament and Human Tournament suites remain.
The navigation keyboard test was extended for the new Season link.

Final verification on October 7, 2026:

| Check | Result |
| --- | --- |
| Full backend suite | 490 passed; 14 existing optional PyTorch skips |
| Frontend unit suite | 359 passed; no skips |
| Full Chromium regression | 118 passed; no skips |
| Firefox configured smoke suite | 29 passed; no skips |
| WebKit configured smoke suite | 29 passed; no skips |
| Final browser total | 176 passed, about 2.4 minutes |
| General production build | Passed |
| Render production build with explicit HTTPS API origin | Passed |
| Separate-origin local production build | Passed |
| Actual 12-game local integration season | Passed; 210 plies; maximum concurrent POSTs 1 |
| Maximum synthetic benchmark | Passed; 528 games / 22,176 plies / 295,236 bytes |
| `git diff --check` | Passed |

The 14 optional research skips require unavailable PyTorch; no mandatory API or
product test was skipped. All 12 screenshot artifacts were visually reviewed.
The chart axis-label fill and a lost-terminal-response continuation corner case
were corrected before the final passing run. The final checks used a fresh local
API session store to avoid contaminating results with earlier test sessions.
No live provider calls or production requests were made, and no production service
was deployed. Frozen AlphaZero artifacts and campaign branches were untouched.

Reproduction (repository root unless noted):

```sh
PYTHONDONTWRITEBYTECODE=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests -q -rs
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' CORS_ALLOWED_ORIGINS=http://localhost:4175 .venv/bin/gunicorn api.app:app --bind 127.0.0.1:8008 --workers 1 --threads 4
PLAYWRIGHT_API_URL=http://localhost:8008 node ui/scripts/verify-season.mjs
node ui/scripts/season-benchmark.mjs
# ui/:
npm test
npm run build -- --outDir dist/general
VITE_API_BASE=https://board-game-ai-lab.onrender.com npm run build:render -- --outDir dist/render
VITE_API_BASE=http://localhost:8008 npm run build
npm run preview -- --host localhost --port 4175 --strictPort
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/phase5a-browsers PLAYWRIGHT_BASE_URL=http://localhost:4175 PLAYWRIGHT_API_URL=http://localhost:8008 npx playwright test --output=test-results/phase5d-final
# root:
git diff --check
```

Use a fresh isolated API for full browser runs. Local process/browser launch may
need sandbox permission. Tests must remain pointed at localhost; the Render
command builds a bundle without contacting or deploying to production.

**GO** for commit and deployment following review. No commit, push or deployment
has been performed, and no further phase has begun.

## Files changed

19 files added and 9 existing files updated:

| Group | Files |
| --- | --- |
| Season domain/controller | `ui/src/season/model.js`, `analytics.js`, `storage.js`, `controller.js`, `useSeason.js` |
| Season view components | `ui/src/season/Schedule.jsx`, `MatchViewer.jsx`, `AnalyticsViews.jsx`, `RatingHistory.jsx` |
| Page/styles | `ui/src/pages/SeasonLab.jsx`, `season-lab.css`; updated `ui/src/styles/global.css` |
| Navigation/discovery | Updated `ui/src/App.jsx`, `ui/src/components/AppShell.jsx`, `ui/src/pages/Home.jsx`, `ui/src/pages/TournamentLab.jsx` (method copy only) |
| Unit verification | `ui/tests/seasonModel.test.js`, `seasonExecution.test.js` |
| Browser verification | `ui/e2e/fixtures/season.js`, `ui/e2e/season.spec.js`, `season.smoke.spec.js`; updated `ui/e2e/polish.spec.js`, `ui/playwright.config.js` |
| Integration/scale scripts | `ui/scripts/verify-season.mjs`, `season-benchmark.mjs` |
| Documentation | `docs/phase5d-season-ratings-lab.md`; updated `README.md`, `docs/quickstart.md` |

Backend routes, existing agent implementations, frozen campaigns and dependencies
are unchanged. Existing gameplay/research tests were retained.

## Screenshots

Local review artifacts are in ignored `ui/playwright-report/phase5d/`:

1. `01-setup-8x4.png` — representative seeded setup and workload.
2. `02-active-schedule.png` — desktop round and active fixture.
3. `03-live-viewer.png` — shared board, one-ply controls and timeline.
4. `04-standings.png` — partial-season results and both side splits.
5. `05-ratings-history.png` — ratings plus selectable Elo lines and summary.
6. `06-pairwise-sides.png` — pairwise desktop comparison; side splits appear in 04.
7. `07-complete-leaders.png` — full completed dashboard with separate leaders.
8. `08-large-12x8.png` — maximum-scale synthetic 528-game desktop state.
9. `09-mobile-standings.png` — mobile table contained within its scroll region.
10. `10-mobile-ratings-viewer.png` — mobile full dashboard.
11. `11-mobile-current-fixture.png` — readable mobile replay/viewer detail.
12. `12-mobile-ratings.png` — mobile ratings detail.

Screenshots are deterministic fixtures, not agent-strength measurements.
The all-draw maximum state intentionally exercises worst legal compact storage.

## Limitations and next recommendation

Runs require the page to remain open; there is no background execution or server
season state. A season exists per browser origin with no cross-tab ownership protocol.
There is no archived engine/version manifest, import/export, learned agent or Human
season. Ratings depend on pool/order; bootstrap does not model opponent dependence.
Long searches still take engine time and Pause waits for the current mutation to settle.
Browser storage can be cleared or unavailable; completed local evidence has no remote backup.

Next recommended work is a separate review of evaluation methodology and engine-version
provenance/export requirements before any broader ratings claims. It is not implemented
here, and no next phase starts as part of this work.
