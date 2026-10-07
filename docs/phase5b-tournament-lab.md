# Phase 5B — AI Tournament Lab

Phase 5B adds `/connect4/tournament` beside Play and Match Lab. It supports exactly
8, 16, 32 and 64 AI entrants in a seeded single-elimination competition. It does
not add Human participation, seasons, ratings, server archives or learned agents.

Work began from clean local and fetched remote `main` at
`ed8cd20533fa6ecbad550a960365550b08f29250`, on the new
`phase5b-tournament-lab` branch. No commits, pushes, deployments, frozen campaign
changes or changes to `phase4d3b-alphazero-v2-preflight` are part of this work.

## Architecture and reuse

- `ui/src/tournament/model.js`: pure bracket creation, provenance, game plans,
  legal column replay, compaction, progression and storage validation.
- `controller.js`: sole sequential mutation/scheduling owner. HTTP and storage,
  plus timer scheduling/cancellation, are injectable. React subscribes to snapshots; a per-transport/storage controller survives SPA
  route remounts so an in-flight mutation retains its single owner and lock.
- `storage.js`: versioned browser storage, safe load/save and memory fallback.
- `useTournament.js`: small controller subscription/lifetime hook.
- `Bracket.jsx`: one shared data source for desktop rounds and mobile round list.
- `MatchViewer.jsx`: active seeded game or completed local multi-game replay.
- `pages/TournamentLab.jsx` and `tournament-lab.css`: setup, summary, run controls,
  champion, reproducibility and page layout.

Phase 5A provides the public competitor presets/labels/API translation,
`requestMatchPly`, `validateMatchHistory`, `replaySnapshot`, `GameBoard`,
`MatchControls`, `MatchTimeline` and playback speeds. No second board or copied
Match Lab hook was introduced. Timeline gained an optional end-label so completed
local games say “Return to end.” Play/Match Lab retain their default labels.

## Schema and progression

The `version: 1` record contains `tournamentId`, `size`, `tournamentSeed`,
`status` (`paused` or `complete`), `entrants`, `bracketOrder`, `rounds`, `active`,
`retainedGameId` and `championEntrantId`.

Each entrant is `{ entrantId: 'entrant-N', seedNumber: N, config }`.
Identity is independent of configuration; duplicates are valid. Configurations
come from Phase 5A public presets: Random, Negamax depths 1/2/4/6/8 and MCTS
100/400/800 simulations. Default eight slots are Random, Negamax 2/4/6/8 and
MCTS 100/400/800; larger fields cycle that pool. Slots can be edited independently.
The field and seed are frozen after creation until an explicit New tournament.

All N−1 matchups and log2(N) rounds are prebuilt. A matchup has `matchupId`
(`rR-mM`, one-based identity), zero-based `round`/`index`, `entrantAId`,
`entrantBId`, `status` (`pending`, `active`, `complete`), `games`,
`winnerEntrantId` and `resolution`. Only first-round entrants are initially known.
Winner of index i advances to index floor(i/2) in the next round, slot A for even
indices and B for odd. Quarterfinals/Semifinals/Final labels follow remaining field
size. Champion is emitted only after the final. Completion must follow bracket
order; malformed provenance and impossible advancement throw before mutation.

## Deterministic seeds

Both seed inputs are bounded unsigned 32-bit integers. The setup text field
explicitly parses digits only; model/API reject booleans, floats, negative values,
non-parsed strings and overflow. A randomly generated initial form seed is editable.
These algorithms provide reproducibility, not cryptographic randomness.

Frontend `deriveSeed(seed, domain)` starts with `(2166136261 XOR seed) >>> 0`.
For every ASCII domain character, multiply `(h XOR charCode)` by 16777619 with
`Math.imul`, retaining 32 bits. Finish with these avalanche operations:

```text
h = imul(h XOR (h >>> 16), 0x7feb352d)
h = imul(h XOR (h >>> 15), 0x846ca68b)
return (h XOR (h >>> 16)) >>> 0
```

Known vector: `deriveSeed(1234, 'r1-m1:game:1') = 1845687796`.
Fisher–Yates uses `shuffle:COUNTER` domains and rejection sampling for unbiased
bounded picks. Entrant ordering before shuffling, configurations and seed fully
define initial bracket/provenance. No UUID, timestamp or runtime hash is used.

Game N seed is `deriveSeed(tournamentSeed, 'MATCHUP:game:N')`. Game 1 colors use
`MATCHUP:color:1`; an odd low bit reverses A/B. Game 2 swaps Game 1 colors. Game 3
uses separate `MATCHUP:color:3`. Three drawn games use `MATCHUP:draw-tiebreak`;
low bit zero advances A, one advances B. This is a deterministic seeded 50/50 pick
under the seed generator, not an additional game win.

Backend `start_game` optionally accepts `rng_seed`. The session stores it; seeded
history exposes it at the top level, while unseeded history keeps the old shape.
No search state, tree/UCB values or reasoning is exposed. History validation
requires the exact expected tournament game seed.

Backend `ply_seed` computes:

```text
x = (gameSeed XOR ((revision+1)*0x9e3779b9)
              XOR ((currentPlayer+1)*0x85ebca6b)) & 0xffffffff
x = ((x XOR (x >> 16))*0x7feb352d) & 0xffffffff
x = ((x XOR (x >> 15))*0x846ca68b) & 0xffffffff
return (x XOR (x >> 16)) & 0xffffffff
```

Each AI request builds `random.Random(ply_seed(...))` and injects it into MCTS or
RandomAgent. Negamax remains deterministic. There is no global `random.seed`.
Unseeded construction/behavior remains unchanged. Complete Random/Random,
MCTS/Random and MCTS/MCTS games reproduce their entire histories after replacement
or interruption/restart, without changing Python global RNG state. Reproducibility
assumes the same agent implementation/runtime; this is not a permanent cross-version
algorithm guarantee.

## Matchup format and scientific interpretation

Normally one Connect 4 game decides a matchup. Red/Yellow assignment is recorded
separately from bracket order. A first draw causes a color-swapped rematch; a
second draw causes Game 3; a third draw advances an entrant by the explicit
`seeded_draw_tiebreak` resolution. All games retain their actual draw/win results.
The viewer says “Advanced by seeded tiebreak after three draws,” and exposes all
three replays and why each later game occurred.

Single elimination depends on path and first-move/color assignment. One-game
matchups are practical entertainment/comparison, not rigorous strength estimates.
Future seasons/ratings will need repeated balanced comparisons. The champion is
an entrant in this tournament, not proof of general algorithm superiority.

## Execution and safe pause

Tournament creation allocates no server session. Start / Watch starts only the
next matchup at revision zero, paused. Next move sends exactly one revision-bound
POST via existing `make_move`; no full-game/tournament endpoints exist.

Autoplay runs the current game. Run matchup includes required draw rematches;
Run round completes unresolved matchups in the current round; Run tournament
continues through the final. Snapshot run modes distinguish those scopes, paused,
waiting/recovery, and completed tournament. Run modes are deliberately not restored
as automatic execution after reload.

A synchronous controller lock and one timer boundary serialize requests. Pause
clears the timer but never aborts an in-flight POST. That mutation is accepted or
reconciled and persisted before another ply could be scheduled. Detaching the page
also stops scheduling while letting mutation reconciliation/persistence finish.
Browser native timers are invoked through wrappers to preserve their required
receiver and allow injectable clocks in tests.

For terminal games, authoritative history is fetched and validated, columns are
persisted, the compact result is persisted, winner/next slot are recorded, and
only then another game may start. `replace_game_id` retires the last known live
session on each next start, including draw rematches. Completed replays remain
independent of the retired server sessions.

## Persistence, replay and validation

Local key: `board-game-ai-lab:tournament:v1`. No timers, functions, React values,
AbortControllers or derived boards are serialized. The active record holds only
matchup ID, game number, current server ID, starting/running/interrupted status and
validated columns. The last known ID is retained for session replacement.

A completed game has `gameNumber`, `gameSeed`, `playerEntrantIds`, `playerConfigs`,
`columns`, `result` (`status`, `winnerIndex`) and `moveCount`. Player-index order
records the actual color assignment. Boards and timeline records are reconstructed
in memory from legal move columns; moves after a win/draw or into full columns
are rejected. No `board_before`/`board_after` arrays are stored.

Hydration parses safely and reconstructs the bracket from seed/field, replaying
all completed games and checking exact IDs, topology, colors, seeds, winners,
resolution and champion. It rejects unsupported versions, gaps/impossible
advancement, illegal move histories and malformed active games. Unknown derived
fields are discarded by canonical reconstruction. Invalid storage never partially
hydrates and presents Create tournament as a reset path. Security/quota errors
fall back to working memory state with a visible persistence notice. Storage is
injectable for unit tests.

## Recovery

- Lost/rejected move response: pause; GET history; validate players, seed, revision
  and already-confirmed column prefix. Never automatically repeat the POST.
- `agent_busy`: pause and reconcile, retain the error; explicit Refresh history
  clears it, then Next move/run is a separate user action. No retry hammering.
- Failed/invalid history: retain confirmed columns and lock execution until an
  explicit successful refresh. Seed/config/history mismatches cannot unlock play.
- Expiry/backend restart: keep prior results, bracket and columns; mark active
  game interrupted. Explicit Restart interrupted game uses exactly its original
  configs/seed and clears only that game's interrupted columns.
- Lost start ID: persist interrupted pairing; do not replay start automatically.
  Explicit restart is required. The unknown orphan session may expire naturally.
- Reload: stays paused; known running IDs reconcile with the server. A persisted
  in-progress start becomes interrupted rather than silently restarted. Completed
  replay requires no server, including after reload.

## UX, responsiveness and accessibility

Navy/dark surfaces, restrained cyan and Red/Yellow reuse the existing lab design.
Entrant setup uses compact preset selects in a bounded list, including 64 slots.
Desktop displays round columns, paired progression connectors, winner marks,
active/selected outlines and a champion destination. Local bracket scrolling is
bounded so the active viewer stays within reach. Below 760px, a labeled round
select and vertical matchup list replace the horizontal bracket. Both views use
the same record and selection. New tournament replaces configuration explicitly.

Native labeled size/seed/entrant/speed controls and buttons support keyboard input.
Each matchup has a textual accessible round, match number, entrants, status and
winner label; connectors are supplementary. Selection uses `aria-pressed`, replay
uses `aria-current`, execution reports `aria-busy`, and the polite status announces
run/recovery/completion changes rather than each ply. Focus rings, disabled states,
44px targets and existing reduced-motion rules apply. Completed games use “Return
to end”; active games retain “Return to live.” Reproducibility details show field
in bracket order, slot identities, size and seed.

Responsive/focus/overflow checks cover 1440, 1024, 820, 768, 375 and 320px, active
8-player views, completed 16-player replay/champion and 64-player round navigation.
Desktop scrolling stays within the bracket; no mobile page overflow is allowed.

## Storage and performance

Local Node benchmark (`node ui/scripts/tournament-benchmark.mjs`), October 7, 2026:
64 entrants, 63 matchups, maximum bounded 189 full 42-ply draw games (7,938 columns):

| Measurement | Observed |
| --- | ---: |
| Serialized compact record | 79,776 bytes (about 78 KiB) |
| 64-player creation, average of 200 | 0.042 ms |
| Immutable model update, average / maximum | 0.717 / 2.148 ms |
| Serialization | 0.063 ms |
| Validation + JSON rehydration | 117.0 ms |
| One full-game local replay construction | 0.131 ms |

This is comfortably below common multi-megabyte localStorage budgets. Immutable
updates clone the bounded competition record; validation replays and reconstructs
the complete record and is the most expensive measured operation. No duplicate
boards are stored. These are local sanity measurements, not universal browser
latency guarantees. No expensive 63-game MCTS campaign was run.

## Verification

Automated tests prohibit real provider calls. The separate local API runs with
explanations disabled and an empty provider key; the frontend verification bundle
uses `http://localhost:8006`. Browser configuration rejects non-local targets.

Pure model/storage tests cover sizes, duplicates, stable IDs, topology, shuffled
ordering, seed vectors, advancement, champion/reset, all draw branches, legal local
replay and hostile persistence. Controller tests cover serialized one-ply requests,
run scopes, safe pause/detach, lost starts/moves, busy/expired sessions, explicit
seeded restart, reload, persistence-before-next-start, replacement and a full cheap
64-player synthetic tournament. Backend tests exercise optional seed validation,
provenance, complete stochastic replay and global-RNG isolation.

Browser functional tests cover creation, seed validation, duplicate selections,
run scopes, terminal advancement, local replay, persistence/reload, explicit
restart, three-draw resolution and screenshots. Cross-browser smoke retains
existing Play/Analysis and Match Lab tests, adding Tournament Lab setup, bracket,
viewer, replay, focus, mobile navigation and overflow.

Actual local integration (`PLAYWRIGHT_API_URL=http://localhost:8006 node
ui/scripts/verify-tournament.mjs` from root) completed an eight-player Random/Negamax 1/2
field: 7 matchups, 7 games, 162 plies, champion `entrant-2`, maximum concurrent
POSTs 1, and 6 retired sessions confirmed 404. Every compact game replayed locally.
Final observed integration time was 573 ms with injected zero-delay scheduling.

Final verification on October 7, 2026:

| Check | Result |
| --- | --- |
| Full backend `pytest tests -q -rs` | 486 passed; 14 optional skips |
| Frontend `npm test` | 250 passed; no skips |
| Full Chromium regression | 94 passed; no skips |
| Firefox focused supported project | 22 passed; no skips |
| WebKit focused supported project | 22 passed; no skips |
| Total final browser run | 138 passed |
| General Vite production build | Passed |
| Render build, `https://board-game-ai-lab.onrender.com` | Passed |
| Separate-origin local production bundle | Passed |
| Root `git diff --check` | Passed |

The 14 backend skips are optional PyTorch research modules/tests in the current
lightweight environment, the same optional skip set as Phase 5A. No mandatory
API/product check was skipped. The final browser run took about 1.5 minutes and
includes existing Play, both turn orders, Random/Negamax/MCTS, grounded mocked
Analysis, recovery and Match Lab keyboard/manual/autoplay/replay checks. All
requested screenshot states were visually inspected. No live provider call or
production request was made.

Reproduction commands (from root unless stated):

```sh
PYTHONDONTWRITEBYTECODE=1 EXPLANATIONS_ENABLED=false .venv/bin/python -m pytest tests -q -rs
node ui/scripts/tournament-benchmark.mjs
PLAYWRIGHT_API_URL=http://localhost:8006 node ui/scripts/verify-tournament.mjs
# From ui/:
npm test
npm run build
VITE_API_BASE=https://board-game-ai-lab.onrender.com npm run build:render
# For browser verification, build with a separate-origin LOCAL backend, not Render:
VITE_API_BASE=http://localhost:8006 npm run build
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/phase5a-browsers \
PLAYWRIGHT_BASE_URL=http://localhost:4173 PLAYWRIGHT_API_URL=http://localhost:8006 \
npx playwright test
# From root:
git diff --check
```

The general and Render final builds additionally used `--outDir dist/general`
and `--outDir dist/render` so the running local-verification bundle stayed intact.
The local Flask API used port 8006, `EXPLANATIONS_ENABLED=false`,
`OPENAI_API_KEY=''`, and `CORS_ALLOWED_ORIGINS=http://localhost:4173`.

GO for committing after local review. GO for public review after an explicitly
authorized API/frontend deployment including the seed contract and tournament
SPA route. This work does not publish changes. Human participation is deferred.

## Screenshots

Ignored local review artifacts under `ui/playwright-report/phase5b/`:

1. `01-desktop-setup.png`
2. `02-desktop-active-bracket.png`
3. `03-desktop-live-viewer.png`
4. `04-desktop-champion.png`
5. `05-desktop-completed-replay.png`
6. `06-desktop-64-bracket.png`
7. `07-mobile-active.png`
8. `08-mobile-champion.png`

The captures use legal controlled game evidence and real preset labels; their
outcomes illustrate UI states, not algorithm strength. Actual agent execution is
verified separately. No screenshots imply public deployment.

## Limits and Phase 5C handoff

Storage is browser-local to this origin/profile and one active tournament; clearing
site data clears the record. There is no cloud sync/export/import/account archive.
Use one active Tournament Lab tab for execution; multi-tab leadership is not
implemented. Game sessions remain bounded/process-local, expire after idle time,
and disappear on backend restart. Unknown lost-start orphans cannot be retired
without their IDs. Search remains synchronous; pacing adds delay between searches.
Persistence-unavailable behavior works in memory but cannot survive closing the page.
The completed-game provenance is auditable/replayable, not tamper-proof attestation.

Exact Phase 5C extension points (not implemented):

1. Permit `{ type: 'human' }` in the field/schema's entrant-config gate; preserve
   entrant IDs, seed numbers, bracket graph, provenance and winner progression.
2. Add a Human setup option through shared competitor utilities.
3. Add a controller human-column mutation method using `requestMatchPly`; determine
   whether the active player is human and suspend automatic scheduling at that turn.
4. Enable the existing `GameBoard` interaction only for the live, validated Human
   turn; hand off/resume the selected run scope after a legal human ply.
5. Add Human waiting/accessibility copy, interaction/recovery tests and document
   that Human decisions cannot be reproduced solely from a stochastic seed.

The bracket, draw policy, compact columns and completed replay do not need replacement.
Phase 5C, seasons/ratings and learned-agent deployments remain out of scope.


## Changed-file inventory

Added:

- `docs/phase5b-tournament-lab.md`
- `tests/test_connect4_seeded_api.py`
- `ui/src/tournament/model.js`
- `ui/src/tournament/controller.js`
- `ui/src/tournament/storage.js`
- `ui/src/tournament/useTournament.js`
- `ui/src/tournament/Bracket.jsx`
- `ui/src/tournament/MatchViewer.jsx`
- `ui/src/pages/TournamentLab.jsx`
- `ui/src/pages/tournament-lab.css`
- `ui/tests/tournamentModel.test.js`
- `ui/tests/tournamentExecution.test.js`
- `ui/e2e/fixtures/tournament.js`
- `ui/e2e/tournament.spec.js`
- `ui/e2e/tournament.smoke.spec.js`
- `ui/scripts/tournament-benchmark.mjs`
- `ui/scripts/verify-tournament.mjs`

Modified:

- `README.md`
- `docs/quickstart.md`
- `api/connect4/__init__.py`
- `api/connect4/state.py`
- `games/connect4/agents/random_agent.py`
- `ui/src/App.jsx`
- `ui/src/components/AppShell.jsx`
- `ui/src/pages/Home.jsx`
- `ui/src/styles/global.css`
- `ui/src/connect4/matchRecord.js`
- `ui/src/connect4/MatchTimeline.jsx`
- `ui/e2e/polish.spec.js`
- `ui/playwright.config.js`

Removed: none. Build files, reports and screenshots are ignored local review
artifacts; no research campaign artifact is modified.
