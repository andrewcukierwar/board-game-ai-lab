# UI-1: shell, visual foundation, and homepage

## Branch and scope

Started with a clean working tree. The initial checkout was the separate research
branch; switched away without inspecting or changing its work. Local `main` was
behind, so `portfolio-ui-redesign` was created from `origin/main`, exactly
`a58abccb2f62c4308b074ac123cd6f3ad798c2c0`. The existing ignored experiment output
was left alone. No commit, push, research-code change, or UI-2 migration is included.

## Frontend audit

| File / area | Active role and migration boundary |
| --- | --- |
| `ui/index.html` | Vite HTML entry, `#root`, module entry, page metadata. |
| `ui/src/main.jsx` | React root and BrowserRouter; now delegates route definitions to `App.jsx`. Previously contained the inline homepage. |
| `ui/src/pages/Connect4.jsx` | Active React route. Declares the controller's fixed DOM scaffold and creates its Axios client with a 90-second timeout. Effect returns the mount cleanup. Unchanged. |
| `ui/legacy/connect4.js` | **Active despite its directory name.** Owns game state, board rendering, requests, revision binding, busy/recovery flags, and cleanup. Unchanged. |
| `ui/legacy/explanations.js` | **Active.** Owns explanation state, requests, validation, disclosures, safe text rendering, and square highlights. Unchanged. |
| `ui/connect4/connect4.css` | **Active**, imported by `Connect4.jsx`. Includes responsive board sizing, native button cells, `[hidden]`, explanations, highlights, and focus rules. Unchanged. |
| `ui/src/api-config.js`, `ui/vite.config.js` | Active API-origin normalization, development `/v1` proxy, and Render-mode validation. Unchanged. |
| `ui/tests/` | Node/JSDOM tests for API config and the two controllers; mocked HTTP, no live provider. All unchanged. |
| `ui/e2e/connect4.spec.js`, `explanations.spec.js` | Active Chromium regressions against local servers. Explanation requests are intercepted with fixtures. Unchanged. |
| `ui/playwright.config.js` | Localhost-only URL validation, single worker, Chromium, retained failure traces. Unchanged. |
| `ui/package.json`, `package-lock.json` | Active scripts/dependencies. Only description, author, and keywords changed; no dependency or lockfile changes. |
| `ui/nginx.conf`, `docker/ui.Dockerfile` | Production static bundle / SPA fallback. Unchanged. |

Likely obsolete or duplicate **within the active Vite application**:

- `ui/src/pages/connect4.css`: no import; older fixed-size board and `.cell.disabled`
  styling, unlike the active native-button disabled rules.
- `ui/connect4/connect4.html` and `connect4.js`: standalone earlier app with CDN
  Axios, browser globals, two-player selectors, and older API behavior. They are
  not the React entry or controller. Experimental options here are not evidence
  that those agents are playable publicly.
- `ui/connect4/old/connect4_old.html`: older self-contained HTML/JS/CSS implementation.

Retained all these files. Vite builds from `ui/index.html`; no active React import,
test, or deployment entry points to the standalone HTML/JS or duplicate stylesheet.
Before deleting them later, check any external/manual consumers as well.

### Important DOM and behavior contracts

- Settings IDs: `opponent-type`, `opponent-depth`, `opponent-simulations`,
  `negamax-options`, `mcts-options`. Defaults remain Negamax depth 2 and MCTS 100;
  presets remain 50/100/250 and settings apply on start/restart.
- Lifecycle IDs: `start-button`, `restart-button`, `retry-button`, `message`,
  `loading`, `game-board`. Keep exact Start/Start new game/Retry labels and the
  unique `role="status"` live region. Hidden state must take precedence over layout.
- Board: 42 native button `.cell` elements, `data-column`, `data-square`,
  `.circle.x`, `.circle.o`, `.circle.empty`, disabled state, and accessible labels.
- Explanation IDs: `explanation-panel`, `explanation-question`, `explain-last`,
  `analyze-position`, `what-if-column`, `what-if`, `explanation-status`,
  `explanation-result`. Preserve the 500-character question limit, legal-column
  options, `aria-busy`, and live status feedback.
- Explanation rendering: `.primary-explanation`, top-level evidence sections,
  native `details`/`summary`, catalog-only source links, `.explanation-square`,
  `.square-label`, and exact board coordinates. Provider text is inserted as text.
- Routing tests depend on one homepage link named `Play Connect 4` and one game
  link exactly named `Home`; maintain these without ambiguous duplicate matches.
- A single human-plus-AI request chain owns the busy lock. Never replay an
  uncertain POST: fetch a snapshot first. Preserve revision checks, explicit
  retries, expired-session replacement, cold-start HTML handling, and MCTS busy
  recovery.
- Unmount aborts requests and removes listeners; late responses cannot mutate
  detached DOM. Explanations have their own abort/generation lifecycle and never
  lock gameplay. Moves, restart, and navigation invalidate stale explanations.
- Explanation responses must match game/revision/mode/hypothetical column and
  validate grounded facts, references, and square coordinates before rendering.

## Design and implementation

Added `App.jsx` for nested routes, `components/AppShell.jsx` for header/footer,
skip link and route/anchor focus, `pages/Home.jsx`, `components/primitives.jsx`
for action links/badges/section headings, and `components/BoardIllustration.jsx`
for a static hero independent of the gameplay IDs and classes.

`styles/global.css` supplies charcoal/navy surface tokens, restrained cyan,
system typography, borders, spacing, focus states, and shell/primitives.
`pages/home.css` owns the landing-page layout. The existing gameplay page sits
on a light surface inside the dark shell, preserving its original styling and
native controls; a board redesign belongs to UI-2.

Homepage sections cover the three playable agents, grounded post-hoc analysis,
the experimental research progression, and final game/source links. No strength
rankings, search trace claims, or public learned-agent availability claims.
The repository URL is centralized. Metadata now includes the consistent product
title, viewport, description, and dark theme color. No new dependencies, external
fonts, images, animations, or remote resources are required.

Desktop uses a two-column hero, three agent cards, split explainability section,
and five research columns. Tablet research wraps to two rows. Below 760px the
header stacks with always-visible navigation, sections become single-column,
and cards/illustration fit the viewport; narrow-screen CTAs stack. No menu state
or animation is needed. `e2e/home.spec.js` covers navigation, repeated section
links, metadata, focus, console/page errors, overflow, direct route refresh,
and keyboard gameplay at desktop/tablet/mobile widths.

## Verification commands

Run from `ui/` unless marked otherwise:

```sh
npm test
npm run build
VITE_API_BASE=http://127.0.0.1:8001 npm run build
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/board-game-ai-lab-playwright npx playwright install chromium --only-shell
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/board-game-ai-lab-playwright PLAYWRIGHT_BASE_URL=http://127.0.0.1:4173 PLAYWRIGHT_API_URL=http://127.0.0.1:8001 npm run test:e2e
```

Separate local API command, from repository root:

```sh
EXPLANATIONS_ENABLED=false OPENAI_API_KEY= CORS_ALLOWED_ORIGINS=http://127.0.0.1:4173 .venv/bin/gunicorn api.app:app --bind 127.0.0.1:8001 --workers 1 --threads 4
git diff --check
```

The API uses explicit environment overrides, leaving private `.env` untouched.
Existing port 8000 processes were left alone. The localhost-origin build and
preview verify the production bundle and direct SPA refresh. Build outputs and
browser screenshots are ignored artifacts, not deployment changes.

Initial runs could not launch because the cached browser was missing and the
sandbox denied Chromium startup. Installing the headless browser and using
approved local server/browser execution resolved that. An initial browser run
was interrupted to add the exact preview origin to the disposable API's CORS
settings. No gameplay or provider configuration in source changed.

Final results: **35 unit tests passed, 0 failed; 24 Chromium tests passed,
0 failed** (16 existing regressions and 8 new shell/responsive/keyboard tests).
Both the standard production build and the localhost API-origin build passed.
`git diff --check` passed. No tests were skipped in the final run. No live OpenAI
request was made. No console or page errors were observed in the new navigation
and responsive gameplay checks. The protected game/controller/API/research
files are unchanged relative to the baseline.

Screenshots under ignored `ui/test-results/` include full homepage and hero
views at 1440, 820, 375, and 320px, plus initialized game views at 820, 375,
and 320px. Desktop and mobile screenshots were visually inspected. Temporary
test servers were stopped after verification, and the standard build restored.

## UI-2 recommendations

1. Port controller state into a React hook/reducer with explicit idle, busy,
   uncertain, retry, and terminal transitions. Keep the current network contract
   and cleanup behavior; retain the existing regression suite as acceptance tests.
2. Render the board in React while preserving native keyboard-operable buttons,
   column/square identities, legal-move disabling, and live status messages.
   Keep board state and requests out of the shared application shell.
3. Migrate explanations separately, preserving independent cancellation,
   revision/column validation, safe text rendering, and post-hoc wording.
   Keep all automated explanation requests mocked or disabled.
4. Replace the light game surface with token-based game styling once React owns
   the board. Verify mobile layout and focus alongside recovery regressions.
5. Remove duplicates only after identifying any standalone/manual consumers;
   the two `legacy/` controllers are active until their migrations finish.
