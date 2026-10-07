# Phase 5A — Match Lab

## Purpose and scope

Match Lab at `/connect4/match-lab` compares any two supported competitors. Red
(Player 1) moves first and Yellow (Player 2) moves second. Human, Random,
Negamax depth 1/2/4/6/8 and MCTS 100/400/800 simulations are independently
configurable on either side. The homepage and main navigation make it discoverable
while Play at `/connect4` remains the simplest Human-vs-AI entry.

This is a local, uncommitted Phase 5A change from verified main
`1d0874102c8a78935ab68d8cd470f24f66431e9f` on `phase5a-match-lab`.
No public deployment, commit, push, tournament, season, rating, learned-agent
integration, research training, or persistent competition storage is included.
The frozen AlphaZero branch, artifacts and campaign state are untouched.

## Architecture

`MatchLabPage` composes `CompetitorSelector`, `MatchStatus`, the existing
`GameBoard`, `MatchControls`, and `MatchTimeline`. Its independent
`useConnect4Match` hook owns configuration, authoritative game, immutable history,
viewed revision, request lock, recovery, playback state, and one scheduled timer.
It uses a synchronous state mirror to guard duplicate events before React renders.
There is no global state library or second board implementation.

Shared pure modules:

- `competitorConfig.js`: public presets, API translation, display labels, colors,
  and `matchStartPayload` with optional `replace_game_id`.
- `gameSnapshot.js`: the existing Play snapshot validator, moved without changing
  its contract or behavior.
- `matchTransport.js`: `requestMatchPly` sends exactly one revision-bound POST and
  validates its one-revision response and unchanged competitors. It never retries
  or chains moves. Serialization belongs to the owning controller.
- `matchRecord.js`: verifies ordered history by replaying legal columns from an
  empty board, checking every before/after board, agent identity, revision,
  move number, gravity, terminal outcome, and final snapshot. Also provides
  `replaySnapshot`, `timelineLabel`, `matchResult`, and `resultLabel`.

Play retains its own hook, automatic AI response/opener, explanation hook, setup,
status copy, highlights and recovery. Its only gameplay changes are importing the
unchanged snapshot validator and the pure competitor payload helper. `GameBoard`
adds optional `interactive` and `labels` props; omitted props retain Play's prior
behavior and appearance. Match Lab passes its viewed snapshot as `game`.

## Competitor model

UI configs have `type`, a retained `depth` setting and a retained `simulations`
setting. Translation emits only the active competitor's supported fields:

```json
{"type":"human"}
{"type":"random"}
{"type":"negamax","depth":6}
{"type":"mcts","simulation_limit":400}
```

Labels are centralized: Human, Random, Negamax · depth 6, and MCTS · 400
simulations. Live board, history and results use the server's accepted player
configuration, so changing pending settings does not relabel an existing match.
Backend validation/limits remain unchanged, including compatible API-only
Negamax depths 3/5/7 and MCTS budgets 50/250. They are not added to UI presets.

## Execution, autoplay and pause

Start is one `start_game` POST. It validates revision 0 and both requested
competitors, resets replay/history, and remains paused. There is no AI opener in
Match Lab. A new start stops playback and replaces the known old game when possible.
Controls cannot start a replacement during an active request.

On live Human turns, a legal board click sends one POST including `column`.
On AI turns, board cells are disabled and Next move sends one POST without
`column`. The synchronous busy lock spans the move and its history read. Manual
stepping does not automatically reply on behalf of the other competitor.

Autoplay schedules one AI ply, waits for its authoritative response and validated
history, then schedules another timer. Slow/Normal/Fast are 1600/850/300 ms viewer
delays, including the first autoplay step. These delays do not measure or constrain
agent search time. There are never overlapping POSTs, an entire-match request,
or an unbounded recursive move chain.

Autoplay waits at a Human turn and retains its enabled state. After the human
moves, the next AI turn resumes through the same timer boundary. Human/Human has
no enabled autoplay control. Terminal state stops playback.

Pause clears scheduled work synchronously. It does not abort a mutation already
in flight: its result/history are accepted or reconciled, then no next ply is
scheduled. Rewinding also pauses; it may inspect already recorded positions while
an in-flight ply resolves. Returning live does not automatically resume playback.

Unmount invalidates the request lifetime, aborts pending browser requests and
clears timers. The server may still commit an in-flight mutation, but late results
cannot render or schedule moves after navigation. Such abandoned sessions expire.

## Read-only history API and replay contract

`GET /v1/connect4/games/<game_id>/history` returns:

```json
{
  "game_id": "session-id",
  "revision": 0,
  "players": [{"type":"negamax","depth":6}, {"type":"mcts","simulation_limit":400}],
  "state": {
    "game_id":"session-id", "revision":0,
    "board":[
      [" "," "," "," "," "," "," "],
      [" "," "," "," "," "," "," "],
      [" "," "," "," "," "," "," "],
      [" "," "," "," "," "," "," "],
      [" "," "," "," "," "," "," "],
      [" "," "," "," "," "," "," "]
    ],
    "players":[{"type":"negamax","depth":6}, {"type":"mcts","simulation_limit":400}],
    "currentPlayer":0, "gameOver":false,
    "legalMoves":[3,2,4,1,5,0,6], "winner":null
  },
  "moves": []
}
```

This is the paused revision-0 response. As moves complete, `moves` contains the
actual `MoveRecord.to_dict()` records. Each includes `move_number`,
`revision_before`, `revision`, `player`, safe `agent` identity, zero-based
`column`, `board_before`, `board_after`, and `{status,winner}` outcome.
The existing record has no MCTS budget field; labels obtain it from `players`,
which is fixed for the session. No search tree, UCB, private reasoning, or model
state is serialized. This change does not modify `MoveRecord` or `record_move`.

The endpoint captures snapshot and records under the existing nonblocking session
lock, preventing separate-read revision disagreement. It returns detached JSON,
uses `Cache-Control: no-store` and existing CORS/error policy, and accepts no
mutations. Normal session reads refresh idle TTL as before; board/revision/history
are unchanged. Missing/expired sessions return 404; an active move returns
`game_busy`/409. Capacity, per-game locking, one-worker/one-instance deployment and
MCTS's process-wide nonblocking reservation are preserved.

One history GET follows each accepted move. At most 42 records and boards make
full-history replacement simple and bounded; no cache/polling system is necessary.
Recovery uses this same endpoint's atomic authoritative snapshot rather than two
potentially inconsistent game/history reads. There is no background polling.

Live state is the authoritative snapshot. `viewedRevision=null` follows Live;
otherwise the board shows revision 0's empty board or a recorded `board_after`.
Previous, Next and clickable move entries change only the viewed revision. They
send no network calls, and all move/autoplay controls are blocked while reviewing.
Return to live restores the latest authoritative board, including terminal boards.
Timeline labels identify the actual color, competitor, move number and one-based
column. The move list follows new live records within its bounded scroll area.

## Recovery guarantees

An uncertain/rejected/invalid POST response stops autoplay and triggers one
read-only authoritative reconciliation. The client never replays the POST.
A lost committed AI or Human response is recovered with the new revision and full
history. A history read failure after an accepted move retains the accepted
revision, locks further execution and reconciles through GET.

`agent_busy` stops immediately and never triggers repeated AI attempts. The error
remains visible after successful reconciliation. Refresh match is an explicit,
read-only action; then the user separately presses Next move or Autoplay to
continue. If GET fails or history disagrees, execution stays locked behind Refresh.
Expired sessions clear active state and offer Start match. A lost start/replacement
response cannot recover an unknown new game ID; there is no automatic start replay.

## Accessibility and layout

Native fieldsets/legends group each player; radio groups are independent and
keyboard-operable. Search budget and playback speed selects have explicit labels.
Board cells retain accessible column/row/piece names and native disabled semantics.
Timeline controls are buttons, selected history uses `aria-current="step"`, and
Autoplay/Pause uses `aria-pressed`. Shared focus rings remain visible.

The polite atomic status announces playback, recovery, human handoff, replay and
terminal changes. AI autoplay's status title stays stable between AI plies to
avoid announcing every board update. Human-turn status identifies the color.
No animation communicates progression; global reduced-motion rules apply.

Desktop places the board beside controls/history. Mobile stacks status, both
competitors, board, controls and timeline. Replay targets are at least 44 px high.
The page and history stay within 375 px and 320 px; the move list has bounded
vertical scrolling, not horizontal scrolling. Tablet is additionally checked.

## Verification

Final verification on October 7, 2026:

| Check | Result |
| --- | --- |
| Full backend `pytest tests -q -rs` | 468 passed; 14 optional skips |
| Frontend `npm test` | 193 passed; no skips |
| Full Chromium regression | 71 passed; no skips |
| Firefox supported smoke project | 10 passed; no skips |
| WebKit supported smoke project | 10 passed; no skips |
| Total browser run | 91 passed |
| General Vite build | Passed |
| Render build with `https://board-game-ai-lab.onrender.com` | Passed |
| Local separate-origin production build | Passed |
| `git diff --check` | Passed |

Thirteen backend tests were added for the new read-only contract. The 14 optional
backend skips are 13 research modules requiring unavailable PyTorch and one
PyTorch-dependent gradient test; all non-optional backend tests ran. No learned
training campaign or paid provider request was launched. Play regression includes
real full games with Random, Negamax and MCTS, both turn orders, AI-first openers,
revision recovery and mocked Analysis. Match Lab keyboard/focus/layout smoke ran
at 1440/820/375/320 px in all three engines, with reduced motion enabled. Existing
Play/Analysis polish also ran at 1024 and 768 px. Final five screenshots were
visually inspected after capture; no horizontal overflow was found. Automated
explanations are mocked; backend tests prohibit live HTTPS provider calls. Browser
tests use a dedicated local API with `EXPLANATIONS_ENABLED=false`, empty provider
credentials and a separate-origin local production bundle.

Coverage includes every public config payload and 100 independent pairings,
Human/Human, both Human/AI orders, distinct budgets, paused revision-0 AI starts,
one-ply manual moves, duplicate-click locking, sequential timers, in-flight pause,
Human waiting/resume, terminal stop, busy/expired/lost-response recovery, invalid
history, unmount/late results, local rewind and full 42-move terminal replay.
Existing Play/Analysis coverage remains, including AI-first openers for all agents.

Commands:

```sh
PYTHONDONTWRITEBYTECODE=1 EXPLANATIONS_ENABLED=false .venv/bin/python -m pytest tests -q -rs
cd ui
npm test
npm run build
VITE_API_BASE=https://board-game-ai-lab.onrender.com npm run build:render
# Build a separate local verification bundle and serve on localhost:4173.
VITE_API_BASE=http://localhost:8005 npm run build
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/phase5a-browsers \
PLAYWRIGHT_BASE_URL=http://localhost:4173 PLAYWRIGHT_API_URL=http://localhost:8005 \
npx playwright test --project=chromium
# Firefox/WebKit smoke projects include Play polish and Match Lab responsive checks.
# From repository root:
git diff --check
```

Representative screenshots use a deterministic network-controlled legal match
with Negamax depth 6 vs MCTS 400 labels; they illustrate UI states, not strength
measurements. Real-backend representative tests separately exercise agent dispatch.
The screenshot test saves five captures to `ui/playwright-report/phase5a/`:

1. `01-desktop-configured.png`
2. `02-desktop-autoplay.png`
3. `03-desktop-replay.png`
4. `04-mobile-active.png`
5. `05-mobile-replay.png`

These are ignored local review artifacts. A controlled complete draw verifies 43
positions, bounded rendering, one GET per move, no network calls during replay,
and no overlapping search. No tournament benchmark was run.

## Limitations and exact Phase 5B handoff

Sessions/history remain bounded and in memory: 128 sessions, 30-minute idle
expiry, one process/worker/instance, and loss on backend restart/deploy. The browser
owns playback; leaving/reloading does not resume a campaign. Start responses lost
before the ID arrives can leave an abandoned session. Agent search remains
synchronous and its duration varies independently of viewer pacing. No durable
match archive/export, multiuser transport, campaign, tournament engine, brackets,
season, ratings or background jobs exist. Match Lab intentionally omits AI Analysis;
historical-position analysis needs a separate revision-safe contract in a later phase.

Phase 5B can consume `competitorConfig` and `matchStartPayload`, create matches,
use `requestMatchPly` with an explicit sequential owner, collect validated
`matchRecord` history/results, and display `GameBoard`/`MatchTimeline` replay.
Introduce a small tournament orchestration module separate from the match viewer:
entrant config -> pairing -> match execution -> result -> bracket advancement.
The bracket should consume results and match IDs without owning board revisions or
agent configuration translation. Keep scheduling/manual Human participation and
recovery at the match boundary. Define persistence/lifecycle and tournament format
when Phase 5B is authorized; no unused bracket abstractions are introduced here.

## Changed-file inventory

Added:

- `docs/phase5a-match-lab.md`
- `tests/test_connect4_history_api.py`
- `ui/e2e/fixtures/match.js`
- `ui/e2e/match-lab.smoke.spec.js`
- `ui/e2e/match-lab.spec.js`
- `ui/src/connect4/CompetitorSelector.jsx`
- `ui/src/connect4/MatchControls.jsx`
- `ui/src/connect4/MatchStatus.jsx`
- `ui/src/connect4/MatchTimeline.jsx`
- `ui/src/connect4/competitorConfig.js`
- `ui/src/connect4/gameSnapshot.js`
- `ui/src/connect4/matchRecord.js`
- `ui/src/connect4/matchTransport.js`
- `ui/src/connect4/useConnect4Match.js`
- `ui/src/pages/MatchLab.jsx`
- `ui/src/pages/match-lab.css`
- `ui/tests/matchConfig.test.js`
- `ui/tests/matchLab.test.js`

Modified:

- `README.md`
- `api/connect4/__init__.py`
- `docs/quickstart.md`
- `ui/e2e/polish.spec.js`
- `ui/playwright.config.js`
- `ui/src/App.jsx`
- `ui/src/components/AppShell.jsx`
- `ui/src/connect4/GameBoard.jsx`
- `ui/src/connect4/useConnect4Game.js`
- `ui/src/pages/Home.jsx`
- `ui/src/pages/home.css`
- `ui/src/styles/global.css`

Removed: none. Screenshots and test reports are ignored local artifacts, not source
removals or research changes.

## Review recommendation

GO for committing after local product review: the completion gate and automated
regressions pass. Proceed to public review only after an explicitly authorized
API/frontend deployment that includes the new history route and correct SPA
fallback. Current public production was not changed. Phase 5B should be a separate
branch/task using the handoff above.
