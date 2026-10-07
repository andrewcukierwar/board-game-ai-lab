# UI-2: React gameplay and Connect 4 workspace

## Safety and scope

Started on `portfolio-ui-redesign` with a clean working tree and verified the
UI-1 shell, homepage, public-agent content, and global design tokens. No branch
switch, commit, push, backend change, provider configuration change, or research
campaign change. UI-3 has not begun.

## Architecture

`Connect4Page` composes `useConnect4Game`, `GameBoard`, `AgentSelector`,
`GameStatus`, `GameControls`, and the temporary `ExplanationPanel` compatibility
wrapper. React owns gameplay state, board rendering, settings, status, and
recovery controls. No state-management or testing dependency was added.

The hook's reducer represents `game`, `phase`, `uncertain`, `retryAI`, `notice`,
`message`, and `selection`. Phases are idle, starting, human-move, ai-move, and
refreshing; busy is derived from phase. Game snapshots determine human turns and
terminal outcomes. Selected settings are separate from the running game's
players. A synchronous mirror of reducer transitions protects the busy lock and
latest revision before React commits; it does not render or own separate game
state. Each effect lifetime has its own AbortController and active guard.

Invariants:

- Only one gameplay request chain owns the busy lock. Controls and all cells are
  disabled while it runs. Illegal columns, terminal positions, AI turns, and
  uncertain positions cannot accept human input.
- Human POST, snapshot acceptance, AI POST, and AI snapshot acceptance remain
  separate sequential steps. No automatic uncertain POST replay.
- Failed requests with an existing game trigger an authoritative GET. Only an
  explicit recovery action can initiate an AI POST after reconciliation.
- Every move uses the authoritative game ID and revision. Restart sends
  `replace_game_id`; selection changes only affect a subsequent start/restart.
- Cleanup aborts the effect lifetime. Late start, human, AI, and refresh results
  cannot dispatch to an unmounted or newer lifetime.

## Explanation compatibility

`ExplanationPanel` mounts the unchanged `ui/legacy/explanations.js` controller in
a layout effect and holds its controller in a stable ref. Another layout effect
calls `update(game, busy || uncertain)` after React commits the board, ensuring
analysis is invalidated as gameplay becomes unavailable. Cleanup cancels analysis
and removes its listeners. The gameplay hook never calls the analysis controller.

All analysis IDs, `.cell`, `data-column`, `data-square`, `.explanation-square`, and
`.square-label` remain. React supplies static analysis scaffolding; the existing
controller retains requests, generation/cancellation, validation, safe result
rendering, catalog sources, and highlights. Settings rerenders preserve external
highlights; moves/restart clear them. The wrapper makes a newly populated
hypothetical-column select show its first legal option if the legacy controller
preserved an empty pre-game value. No explanation validation or lifecycle code
was changed.

## Files

Added:

- `ui/src/connect4/useConnect4Game.js`
- `ui/src/connect4/GameBoard.jsx`
- `ui/src/connect4/AgentSelector.jsx`
- `ui/src/connect4/GameStatus.jsx`
- `ui/src/connect4/GameControls.jsx`
- `ui/src/connect4/ExplanationPanel.jsx`
- `ui/tests/register-jsx.js`
- `ui/tests/jsx-loader.js`
- `ui/e2e/workspace.spec.js`
- `docs/ui-2-gameplay-migration.md`

Modified:

- `ui/src/pages/Connect4.jsx`
- `ui/connect4/connect4.css`
- `ui/src/styles/global.css` (removed UI-1's temporary light gameplay override)
- `ui/package.json` (JSX loader for the existing Node test runner)
- `ui/tests/connect4.test.js`
- `ui/e2e/connect4.spec.js`
- `ui/e2e/explanations.spec.js`
- `ui/e2e/home.spec.js`

Removed: `ui/legacy/connect4.js`. No active code or tests reference it.

`ui/legacy/explanations.js` remains unchanged for UI-3. Unused standalone
`ui/connect4/connect4.js`, `connect4.html`, `old/connect4_old.html`, and duplicate
`ui/src/pages/connect4.css` remain for later audited cleanup.

## Behavioral parity and evidence

| Previous behavior | React implementation / regression coverage |
| --- | --- |
| Random, Negamax, MCTS | Exact opponent payloads and defaults; complete browser games with each public agent |
| Restart and settings | Separate selection state; replacement ID; old session becomes 404; no leaked agent settings |
| Independent browsers | Per-mounted hook/session state; browser regression checks separate IDs and unchanged second board |
| Failed start | No accepted snapshot; unlocked explicit Start; successful second attempt |
| Slow cold start / HTML success | Busy lock persists until completion; HTML rejected by snapshot validation; explicit fresh start |
| Human move failure / lost response | GET reconciles before human input or explicit AI retry; no human POST replay |
| AI failure / MCTS busy | Authoritative AI-turn snapshot enables explicit Retry AI move; refreshed revision used |
| Lost AI response after commit | GET returns advanced human-turn snapshot; no retry offered and no duplicate AI POST |
| Failed reconciliation | Uncertain flag disables board and analysis; Refresh game is explicit; restart remains available |
| Expired/restarted server | GET 404 clears local game and returns usable Start screen |
| Failed restart | Previous snapshot retained until a valid response; recovery GET and another explicit attempt |
| Win, loss, draw | Terminal snapshot disables cells and stops AI chain; new-game control remains available |
| Revision binding / concurrency | Synchronous busy gate prevents same-tick duplicates; separate POSTs use successive revisions |
| Keyboard / illegal columns | Native buttons, coordinate labels, focus-visible outline, Enter and Space; revision-36 full-column checks |
| Navigation / late results | Abort plus lifetime guard; unit coverage at start, human, AI, refresh stages; browser navigation regression |
| Analysis modes and highlights | Existing browser cases pass against React cells, including highlights surviving settings rerenders |
| Analysis cancellation | Move, restart, and navigation abort pending analysis and prevent stale rendering |
| No paid provider requests | Unit HTTP mocks; browser explanation fixtures; local backend disabled with empty API key |

## Verification

Commands from `ui/`:

```sh
npm test
npm run build
VITE_API_BASE=http://127.0.0.1:8001 npm run build
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/board-game-ai-lab-playwright PLAYWRIGHT_BASE_URL=http://127.0.0.1:4173 PLAYWRIGHT_API_URL=http://127.0.0.1:8001 npm run test:e2e
```

Local backend command from repository root:

```sh
EXPLANATIONS_ENABLED=false OPENAI_API_KEY= CORS_ALLOWED_ORIGINS=http://127.0.0.1:4173 .venv/bin/gunicorn api.app:app --bind 127.0.0.1:8001 --workers 1 --threads 4
```

Final results: 42 unit tests passed; 30 Chromium browser tests passed; both build
variants passed; `git diff --check` passed. No skipped unit/browser tests. The
original 35 unit cases were retained and migrated where necessary; seven cases
were added for slow-start locking, in-flight cleanup, and terminal outcomes.
All original 24 browser cases remain; six workspace/recovery cases were added.
The existing Node/JSDOM runner uses Vite's JSX transform and React's `act` and
`createRoot`, with no new dependency or lockfile change.

Sandbox restrictions initially prevented local port binding; approved execution
allowed the disposable backend, preview, and Chromium suite. Backend/Python
regressions were not run because backend contracts and implementation are
unchanged. Temporary servers were stopped and the standard build restored.

## Responsive and visual review

- 1440px: dominant navy board on the left, opponent cards and game controls on
  the right, status below the board, analysis below the workspace.
- 820px: two columns with narrower setup panel and proportionate board.
- 375px and 320px: status, board, opponent configuration/controls, then analysis
  stack vertically. Board preserves its 7:6 proportions and fits without overflow.
- Keyboard focus, active-game controls, AI-thinking lock, explanation highlights,
  and long uncertain-state errors are verified. Red/yellow pieces remain legible
  when input is disabled. No interaction-delaying animation was added.

Full-page active-game screenshots (ignored test artifacts):

- Desktop: `ui/test-results/workspace-active-workspace-cc230-nalysis-placement-at-1440px/connect4-desktop-active.png`
- Tablet: `ui/test-results/workspace-active-workspace-a0de3-analysis-placement-at-820px/connect4-tablet-active.png`
- Mobile: `ui/test-results/workspace-active-workspace-fd005-analysis-placement-at-375px/connect4-mobile-active.png`
- Narrow mobile: `ui/test-results/workspace-active-workspace-ebb4a-analysis-placement-at-320px/connect4-narrow-mobile-active.png`

Each viewport also has a focus screenshot. Additional screenshots show the
AI-thinking and long-error uncertain states at 320px.

## Deferred work and UI-3 recommendation

The analysis DOM island is intentional technical debt. UI-3 should introduce an
independent explanation hook/reducer, retain the abort/generation guards and
revision/mode/column binding, and carry forward the exact validation contract.
Extract the validator without changing it, then render sections, disclosures,
and catalog source links as React components with escaped text. Feed validated
relevant squares to `GameBoard` as props so highlights and labels become React
owned. Preserve the independent loading state that never locks gameplay and
keep provider calls mocked/disabled in the same regressions. Remove the legacy
analysis controller only after parity is established. Standalone duplicates can
be audited separately; no such cleanup or UI-3 implementation is included here.
