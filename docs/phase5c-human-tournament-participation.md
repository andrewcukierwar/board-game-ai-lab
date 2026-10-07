# Phase 5C — Human Tournament Participation

Phase 5C extends the existing browser-local Tournament Lab at
`/connect4/tournament`. Fields of 8, 16, 32 or 64 entrants can contain zero or one
local Human alongside Random, public Negamax depths and public MCTS budgets.
Human participation is opt-in; defaults remain AI-only. This phase adds no accounts,
network multiplayer, backend bracket state, seasons, ratings or research-agent deployment.

Work began from fetched, clean `main` at
`8807f40eb17bd73fc4219635cc393b4ecbac044c`, on the new
`phase5c-human-tournament-participation` branch. No commit, push or deployment is
part of this work. Frozen AlphaZero artifacts and the preflight branch are untouched.

## Architecture and entrant model

`model.js` accepts the exact `{ type: 'human' }` configuration and rejects a second
Human, unsupported presets and extra configuration fields. Existing duplicate AI
configurations remain valid. Identity is still `entrant-N` plus the permanent
slot number; configuration never becomes identity. `humanEntrant`,
`matchupHasHuman` and `humanTournamentStatus` derive the participant's path,
readiness, activity, advancement, elimination and champion state.

The record stays **version 1** and retains the existing storage key. Its topology,
seed derivation, seeded shuffle, color assignment, draw policy, result validation,
compact columns, canonical hydration and session replacement are unchanged.
Existing Phase 5B AI records still hydrate. No redundant Human status or board
snapshots are saved. Generic competitor labels remain unchanged for Match Lab;
tournament labels use `#N · You`.

The setup appends one `You · Human` option to each existing selector and disables
it in other slots once chosen. Model validation also enforces the rule. The
existing bounded field list keeps 64-slot setup compact. Bracket cards preserve
match number/status/winner information and add a restrained YOU badge and path
accent, including completed matches. Mobile round navigation identifies rounds
containing your path and follows the selected matchup.

## Execution boundary

The existing controller remains the sole mutation owner with its synchronous
lock and one scheduled timer. Run matchup, Run round and Run tournament keep
bracket order. At the next Human pairing, they finish and persist prior AI games,
pause, set the ephemeral waiting-for-Human state and show **Play your match**.
They do not create a Human server session or skip it to run later matchups.

An explicit Play your match starts exactly one revision-zero game through
`POST /v1/connect4/start_game`, using the existing game plan, AI seed and seeded
Red/Yellow assignment. Rapid repeated starts share the existing lock. Starting
from either the progress banner or viewer retains the selected Human matchup so
its result remains visible afterward.

During an active Human matchup, run/autoplay can progress AI turns. Every Human
game completion pauses broader execution, even if a round or whole-tournament
scope was selected. The user explicitly resumes the bracket after inspecting the
result. After elimination, the remaining field runs normally.

## Human move lifecycle and playback

The shared `GameBoard` displays seeded Red/Yellow identities and first-move copy.
Human can be Red or Yellow; the seed algorithm is unchanged. Human-first starts
interactive at revision zero. AI-first starts paused; Next move or Autoplay plays
the AI opener, then yields an interactive Human turn.

Human turns enable only legal columns in the live, validated position. Both board
cells and optional accessible column controls invoke `humanMove(column)` through
`requestMatchPly`, sending exactly one POST with current revision and `column`.
AI requests omit `column`. The controller blocks invalid/full columns, duplicate
clicks, concurrent mutations, recovery state and moves while reviewing history.

Autoplay remains armed while waiting for the person, schedules no Human request,
and resumes sequential AI turns after a Human move. Pause clears the next timer
and lets in-flight acceptance/reconciliation finish. A paused Human can still
choose a legal column; the AI then waits for Next move or Autoplay. Next move is
disabled on Human turns. Replay pauses execution and disables interaction; Return
to live restores the authoritative position. Explicit broader run controls return
the viewer to live rather than running behind a historical board.

No special endpoint, backend change or production Human policy was introduced.
The deterministic legal Human policies in tests/integration scripts exist solely
for verification and never select moves in the product.

## Outcomes, draws and completed replay

A decisive Human win advances the same entrant into the existing next slot;
a loss preserves the highlighted completed path and derives elimination status.
The Human champion presentation says **You are the Tournament Champion** and
includes entrant number, size and seed. AI champions retain their presentation.

Draw Game 1 is saved permanently, then **Play rematch** explicitly starts Game 2
with swapped colors. Draw Game 2 pauses again before separately seeded Game 3.
The viewer previews the planned new colors before either start. After three draws,
the existing seeded tiebreak advances an entrant; all three games stay draws.
Copy says **Advanced by seeded tiebreak after three draws** and identifies who
advances, including when the Human is eliminated. It never invents a Game 3 win
or loss.

Terminal authoritative history is validated before compaction and advancement.
Completed Human games have the same column-only records, color-ordered entrant
IDs/configurations, game seed and actual result as AI games. Replays reconstruct
entirely locally, after subsequent session replacement and browser reload.

## Persistence and recovery

Browser storage preserves the Human config, bracket progress, completed game
columns and validated active columns/session identity. Corrupt Human configs,
multiple Humans, illegal columns and invalid provenance reject the whole record.
Storage failure retains the existing visible in-memory fallback.

Reload hydrates and reconciles a known live session with history GET, restores
the correct Human/AI turn and remains paused. Hydrated interrupted Human games
retain an explicit restart-from-beginning explanation.

A lost Human or AI POST response pauses and reconciles authoritative history. If
the Human move committed, the new revision/column is accepted; the controller
never repeats the POST automatically. Invalid/unavailable history leaves mutation
blocked. Refresh and continuation are explicit actions.

If the session expires or the backend restarts, prior results and completed draw
games are retained. **Restart this game** resets only the interrupted current game
from move zero with the same game number, seed and colors. Previous Human columns
are never silently replayed. The UI explains: “The server session expired. Prior
tournament results are safe, but this game must restart from the beginning.”
A lost start identity also requires explicit restart, under the original recovery
architecture; an unknown orphan can expire naturally.

## Reproducibility and interpretation

Seeded AI behavior is reproducible for a fixed Human move sequence; Human
decisions themselves are not determined by the tournament seed. Columns record
those decisions explicitly. Bracket ordering, identities, game seeds, colors and
three-draw advancement remain deterministic. AI implementation/runtime changes
can affect future reproduction.

Single elimination remains sensitive to path and color assignment; these
one-game matchups are **not rigorous strength estimates**. Champion status describes
this tournament, not general algorithm superiority.

## Responsive and accessibility review

Native labeled selectors prevent a second Human and keep keyboard setup simple.
Progress and turn changes use polite atomic status announcements. AI-thinking
copy remains stable across AI plies during autoplay, avoiding a revision-by-revision
announcement. Human turn arrival, advancement, elimination and rematches are textual.
Explicit Human start/restart focuses the viewer and brings it into view.

The shared board retains keyboard cells and adds optional labeled column controls
for this flow. Column controls are at least 44px tall, including at 320px; all use
the same legal/disabled/busy/replay guard. The seven-column layout remains within
the viewport. Disabled Next move cannot imply an automated Human move. Existing
focus styling and reduced-motion rules apply. No new animation was added.

Smoke coverage exercises 375px and 320px setup, readiness, CTA, turn interaction,
AI-first state, replay, 64-player path navigation and champion/elimination layout without page
overflow. Controlled win/loss/draw fixtures cover outcome states on desktop;
screenshots include mobile Human turns and champion state. Keyboard/focus checks
are automated; announcements are verified through DOM roles/copy, without a
physical screen-reader session.

## Verification

Tests use deterministic local mocks or a separate local API with explanations
disabled and an empty provider key. No live OpenAI/provider calls or production
requests are made. The Render bundle is built, not deployed.

Pure tests cover zero/one/two Humans, every slot in every supported field size,
unchanged seed/colors, both colors, version-1 hydration, malformed storage,
Human status, all six rounds of a 64-player Human championship, elimination
followed by an AI champion, compact local replay and both seeded-tiebreak outcomes.
Controller tests cover sequential scope boundaries, explicit single starts,
Human/AI mutation paths, waiting autoplay, pause, duplicate clicks, full columns,
replay, outcome pauses, three explicit draw starts, lost Human/AI responses,
reload at both turns and interrupted Games 1/2/3 with preserved seeds/colors/draws.

Browser tests cover setup and the Human path, both colors, AI opener, keyboard
Human moves, waiting autoplay, sequential AI matchups before Human, win/loss,
three draws/rematch previews, lost response/double click, replay, reload, expiry,
local champion and mobile flow. Existing Play/Analysis, Match Lab and AI-only
Tournament coverage remains enabled. Backend tests additionally verify fixed recorded Human columns reproduce Random/MCTS
AI behavior in both colors without changing Python global RNG. A real local API browser test also verifies
Human and AI POST paths, completed compaction, session retirement and local replay.

A complete actual-backend eight-player Human/Random/Negamax 1/2 tournament completed
7 games and 7 matchups: **10 Human plies, 160 AI plies**, maximum concurrent POSTs
**1**, **6** retired sessions and **4,081** serialized bytes. Human was eliminated
and the AI field completed; `entrant-2` became champion. All completed games
replayed locally. Observed integration time was 605ms with zero-delay test scheduling.
The retained AI-only integration also completed 7 games/162 plies, maximum concurrent
POSTs 1, and 6 retired sessions. These timings are local sanity checks, not browser
latency or algorithm-strength claims. No expensive 64-player live MCTS event ran.

Final verification on October 7, 2026:

| Check | Result |
| --- | --- |
| Full backend suite | 490 passed; 14 optional PyTorch skips |
| Frontend unit suite | 278 passed; no skips |
| Full Chromium regression | 106 passed; no skips |
| Firefox configured smoke suite | 26 passed; no skips |
| WebKit configured smoke suite | 26 passed; no skips |
| Final browser total | 158 passed (about 1.8 minutes) |
| General production build | Passed |
| Render build with `https://board-game-ai-lab.onrender.com` | Passed |
| Separate-origin local production build | Passed |
| `git diff --check` | Passed |

The optional skips are the existing research tests requiring unavailable PyTorch;
no mandatory API/product test was skipped. All nine screenshots were visually
reviewed. Final verification used a fresh API and a dedicated artifact directory
following resolved session-capacity and overlapping-run artifact issues.

**GO** for commit review and public deployment review. Deployment itself was not
performed. Phase 5D remains a recommendation only.

Reproduction (root unless noted):

```sh
PYTHONDONTWRITEBYTECODE=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests -q -rs
PLAYWRIGHT_API_URL=http://localhost:8007 node ui/scripts/verify-human-tournament.mjs
PLAYWRIGHT_API_URL=http://localhost:8007 node ui/scripts/verify-tournament.mjs
# ui/:
npm test
npm run build -- --outDir dist/general
VITE_API_BASE=https://board-game-ai-lab.onrender.com npm run build:render -- --outDir dist/render
VITE_API_BASE=http://localhost:8007 npm run build
npm run preview -- --host localhost --port 4174 --strictPort
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/phase5a-browsers \
PLAYWRIGHT_BASE_URL=http://localhost:4174 PLAYWRIGHT_API_URL=http://localhost:8007 npx playwright test --output=test-results/phase5c-final
# root:
git diff --check
```

The isolated API uses `CORS_ALLOWED_ORIGINS=http://localhost:4174`,
`EXPLANATIONS_ENABLED=false` and `OPENAI_API_KEY=''`. Ports are local verification
choices. Browser launch/listening may require sandbox permission. Start with a fresh isolated
API for each full regression run; repeated suites against the same process can fill
the existing 128-session store (the API correctly returns `session_limit`).

## Screenshots

Ignored local review artifacts in `ui/playwright-report/phase5c/`:

1. `01-setup-human.png`
2. `02-desktop-human-path.png`
3. `03-your-match-ready.png`
4. `04-human-red.png`
5. `05-human-yellow-ai-opener.png`
6. `06-human-advanced.png`
7. `07-human-champion.png`
8. `08-mobile-human-turn.png` (375px)
9. `09-mobile-human-champion.png` (320px)

These controlled legal fixtures illustrate interface states, not measured Human
or agent strength. Actual backend behavior is verified separately.

## Known limitations and Phase 5D recommendation

There is one local user and one tournament per browser origin/profile. Use one
active execution tab; cross-tab leadership, sharing, remote multiplayer and cloud
archives remain absent. Storage can be cleared by the browser. Active sessions
remain process-local and expire; interrupted Human games restart from zero.
Completed columns are replayable evidence, not tamper-proof attestation.

Phase 5D should introduce a separate pure Season model with immutable participant
configuration identities and repeated balanced pairings, both colors, domain-separated
seeds and per-game provenance. Reuse the sequential controller and compact records.
Separate comparison statistics (wins/draws, sample counts and uncertainty) from
bracket entertainment. Add a versioned rating calculation over completed balanced
comparison events, with deterministic update order, an explicit initialization
policy and transparent assumptions. Keep local persistence first. Evaluate rating
choice and sufficient sample size before presenting ranks; keep Human results
separate from deterministic AI benchmarks. This is a recommendation only; no
Season, rating or Phase 5D implementation is included.

## Changed-file inventory

Production changes:

- `ui/src/tournament/model.js`
- `ui/src/tournament/controller.js`
- `ui/src/tournament/MatchViewer.jsx`
- `ui/src/tournament/Bracket.jsx`
- `ui/src/pages/TournamentLab.jsx`
- `ui/src/pages/tournament-lab.css`
- `ui/src/connect4/GameBoard.jsx` (optional column controls; existing consumers keep defaults)

Verification changes:

- `ui/tests/humanTournamentModel.test.js` (new)
- `ui/tests/tournamentExecution.test.js`
- `ui/tests/tournamentModel.test.js`
- `ui/e2e/fixtures/human-tournament.js` (new)
- `ui/e2e/fixtures/tournament.js`
- `ui/e2e/human-tournament.spec.js` (new)
- `ui/e2e/human-tournament.smoke.spec.js` (new)
- `ui/e2e/tournament.spec.js`
- `ui/playwright.config.js`
- `ui/scripts/verify-human-tournament.mjs` (new)

Documentation: this file (new), `README.md` and `docs/quickstart.md`.
Backend test coverage additionally extends `tests/test_connect4_seeded_api.py`; no
backend implementation file, frozen research artifact or existing Match Lab implementation changed.
