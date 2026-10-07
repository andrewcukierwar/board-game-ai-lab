# UI-4: final frontend polish and production review

## Starting audit (before application edits)

On October 7, 2026, verified `portfolio-ui-redesign`, a clean working tree,
committed UI-3 (`c970b46`) and its What If follow-up (`533adac`). Both removed
legacy controllers are absent; reference searches find only historical docs.
No other branch, backend, research or campaign files are part of this pass.

| Active area | Files and responsibilities |
| --- | --- |
| Entry / routes | `ui/index.html` → `src/main.jsx` → `src/App.jsx`; BrowserRouter routes `/` and `/connect4` |
| Shared shell | `components/AppShell.jsx` (header, navigation, skip link, route/anchor focus, footer); `primitives.jsx` (links, badges, headings, repository URL) |
| Homepage | `pages/Home.jsx`, `components/BoardIllustration.jsx`; static decorative SVG/board, playable agent and experimental research distinctions |
| Game | `pages/Connect4.jsx`; `connect4/useConnect4Game.js`, `GameBoard.jsx`, `AgentSelector.jsx`, `GameStatus.jsx`, `GameControls.jsx` |
| Analysis | `connect4/useConnect4Analysis.js`, `analysisValidation.js`, `AnalysisPanel.jsx`, `AnalysisControls.jsx`, `AnalysisResult.jsx`; independent cancellation, complete response validation, declarative highlights |
| CSS | `src/styles/global.css`, `src/pages/home.css`, **active** `connect4/connect4.css` imported by `Connect4.jsx` |
| Unit tests | `tests/api-config.test.js`, `connect4.test.js`, `explanations.test.js`, `analysisValidation.test.js`; Node/JSDOM via `register-jsx.js` / `jsx-loader.js` |
| Browser tests | `e2e/home.spec.js`, `connect4.spec.js`, `workspace.spec.js`, `explanations.spec.js`; localhost-only `playwright.config.js` |
| Build / hosting | `vite.config.js`, `src/api-config.js`, `.env.production`, `package*.json`, `nginx.conf`, `docker/ui.Dockerfile`, Compose, deployment/quickstart docs |
| Standalone candidates | `src/pages/connect4.css`, `connect4/connect4.html`, `connect4/connect4.js`, `connect4/old/connect4_old.html` |

Read the complete active components, both lifecycle hooks, validator, CSS,
test structure, and build/hosting inputs. Repository-wide searches include
hidden files, tests, Docker/nginx, CI and docs. The standalone HTML references
its own JS and the active stylesheet; it and the older self-contained HTML use
obsolete localhost:5001 endpoints. No current entry, import, test, serving rule
or documented manual workflow requires these standalone files. Vite has a
single root HTML entry; Docker serves only its `dist`, not the source tree.
The duplicate page stylesheet has no import at all. Historical migration docs
record these files but do not require them to run. These four assets can be
removed. The active `connect4/connect4.css` must remain.

Already good: one shared token set; route-level main landmarks; native radio
opponent controls and disclosures; action buttons with `aria-pressed`; labeled
question/selector with helper associations; atomic live statuses; independent
game/analysis busy states; bounded source URLs; visible board focus; skip link;
responsive grids and wrapping analysis text. No animations or transitions exist.
No hook/component rewrite or dependency replacement is warranted.

Targeted findings: mobile nav can wrap to a third header row at 320px; root
package name `ui` and nonexistent `main: index.js` are placeholders; social text
metadata/favicon are absent; homepage “AI thinks” can imply internal reasoning;
`api-config.js` still calls React request code a controller. Box-sizing rules
are duplicated across pages, gameplay focus and font rules overlap the shell,
and shared page-band backgrounds repeat a literal. The reduced-motion rule on
the shell has no practical target; any motion safeguard should cover the actual
document. Disabled opponent cards need clearer visual state.

Follow-up reference audit: the unused `api/connect4/legacy/main.py` also names
the standalone HTML/static directory. `api/app.py` registers the current
`api.connect4` blueprint, never this standalone Flask app; Docker invokes
`api.app:app` and does not copy any UI source. No documented run workflow or
test invokes the legacy app. Its archival source is retained under the backend
scope exclusion, and does not justify shipping obsolete frontend assets.
The root `connect4.zip` includes old frontend files alongside research notebooks;
it is documented historical research provenance and excluded from Docker.
It is intentionally retained, not treated as a frontend cleanup target.

The current verified production API origin is
`https://board-game-ai-lab.onrender.com` (README, deployment and quickstart agree).
UI-3's recorded build command used a different hostname; UI-4 will verify the
correct current origin, without contacting production. Render's documented
`/*` → `/index.html` rewrite and nginx `try_files` preserve direct `/connect4`.

## Final changes and cleanup

- Mobile navigation uses four equal columns with 44px-tall links beneath the
  brand. At 320/375px all destinations remain visible in one row; the header is
  under 120px tall. No menu state, hidden destinations or JavaScript was added.
- Consolidated page box-sizing into a global reset, removed the redundant CTA
  declaration, and reused shared monospace and page-band surface tokens. Kept
  the existing color palette, layout breakpoints, gradients and board sizing.
  Feature-specific focus offsets are intentional: the board ring stays inside
  cells, while other controls and links use exterior rings.
- Raised select/textarea border contrast, added an exterior focus ring around
  keyboard-focused opponent cards, and dimmed disabled opponent cards with a
  matching disabled cursor. Native radio state remains authoritative.
- Restored keyboard-visible outlines on the skip destination and homepage
  anchor sections. The skip link itself still hides off-screen, appears on
  keyboard focus, and hides after focus leaves.
- Corrected analysis disclosure heading levels and added the methodology
  heading. Simplified optional-question instructions. Changed the hero line to
  “Explore how game-playing AI chooses moves.” Public/experimental, post-hoc,
  source/context/proof and agent-intent boundaries remain intact.
- Kept the existing title, description, viewport and theme color; added text-only
  Open Graph/Twitter metadata and a small geometric `public/favicon.svg`.
  No nonexistent social image, fixed route URL or third-party asset is referenced.
- Renamed package `ui` to `board-game-ai-lab-ui`, marked it private, added the
  actual repository/directory, and removed nonexistent `main: index.js`.
  Dependency versions and the entire lockfile dependency graph are unchanged;
  the lockfile changes only its two package-name fields.
- Updated current architecture context in `llm-explanations.md`, preserving
  its historical implementation record. Updated stale controller/board wording
  in source comments/test names. Shared unchanged browser analysis fixtures in
  `e2e/fixtures/analysis.js`; no existing behavioral tests were removed.
- Added six focused `polish.spec.js` cases and Firefox/WebKit smoke projects.
  Screenshot output defaults to ignored `playwright-report/portfolio` and can
  be overridden through `UI_SCREENSHOT_DIR`; it is portable across environments.

Removed after reference/build/workflow verification:

1. `ui/src/pages/connect4.css` — unused fixed-size duplicate stylesheet.
2. `ui/connect4/connect4.html` — obsolete standalone CDN/localhost UI.
3. `ui/connect4/connect4.js` — its obsolete standalone controller.
4. `ui/connect4/old/connect4_old.html` — older self-contained application.

Retained active `ui/connect4/connect4.css` despite the older directory name.
Retained historical docs, root research archive, and archived backend serving
code for the reasons above. No uncertain asset in the active frontend required
deletion. No backend, research, API, provider or deployment architecture changed.

## Accessibility and responsive review

Reviewed homepage, active game and populated analysis at **1440, 1024, 820, 768,
375 and 320px** using full-page captures and automated geometry checks. Board
aspect ratio/dominance, opponent cards, status/recovery, mode cards, fields,
counter, What If selector, long citations, evidence/context, disclosures and
long failure messages fit. No horizontal overflow remains in these states.
Native form styling and board proportions also passed Firefox/WebKit checks.

| Area | Result |
| --- | --- |
| Headings / landmarks | One main and one h1 per route; shell header/nav/footer; section and analysis headings; disclosure h3→h4 hierarchy corrected |
| Navigation / links | Labeled main nav, NavLink `aria-current="page"`, descriptive CTAs/source links, source new-tab notice and safe rel |
| Skip / route focus | Off-screen/focused/hidden-again contract passes; destination now visibly focused; route and section focus remain intact |
| Opponent state | Native labeled radio group, checked state agrees with selected card; keyboard card focus visible; settings apply on restart |
| Analysis state | Existing native action buttons and `aria-pressed`; native labels, helper associations, 500-character cap and counter; selector only mounts in What If |
| Status / loading | Atomic polite live feedback; board/analysis busy semantics; gameplay recovery and analysis loading remain independent |
| Board | Existing named native buttons, piece/color labels, Enter/Space play, legal/terminal/busy disabling and interior focus ring preserved |
| Disclosures / sources | Native details/summary support keyboard activation and visible focus; complete evidence/limitations remain accessible |
| Disabled state | Native disabling plus visual card/button treatment; no gameplay-wide overlay or dimming from analysis |
| Reduced motion | No active animations/transitions; document and shell descendants explicitly suppress motion/smooth scrolling under reduced-motion preference |

Contrast calculations against the darkest relevant raised surface: primary text
13.68:1, muted text 7.81:1, accent links/focus 9.31:1, contextual yellow 9.69:1,
primary CTA 10.42:1, playable badge 11.36:1, research badge 9.66:1. Placeholder
with its existing 0.7 opacity is 5.19:1; new form boundary is 3.60:1 against its
adjacent surface. Disabled controls are exempt from active-control contrast.
This is a deliberate DOM/keyboard/visual/contrast audit, not a claim of a manual
screen-reader session or complete WCAG certification.

## Production and performance review

Render production build uses **`https://board-game-ai-lab.onrender.com`**.
The built bundle contains this origin; API paths remain `/v1/connect4/...`.
The dev proxy and empty-base Docker/nginx behavior remain unchanged. Render
builds still reject absent/non-HTTPS origins and malformed URL paths/credentials.
Documented Render SPA rewrite and nginx fallback were inspected; local built
`/` and direct/refreshed `/connect4` navigation pass. No live Render setting,
production endpoint or deploy was touched; no Docker image rebuild was needed.

Final Render output: **277.96 kB JS (91.04 kB gzip), 23.54 kB CSS (5.47 kB
gzip), 1.34 kB HTML (0.56 kB gzip)**, plus the 375-byte SVG favicon. One JS and
one CSS asset, no duplicate bundles, remote fonts, source maps or test/dev harness.
The original JS was 277.87 kB (91.02 kB gzip): this pass adds negligible weight.
React/router/Axios remain the only runtime dependencies. No obvious render loop,
large unused dependency or expensive repeated work justified a performance rewrite.

Artifact checks reject local API endpoints/ports, the incorrect UI-3 hostname,
provider configuration names and test-loader/JSDOM markers. The remaining bare
`http://localhost` string is Axios's non-browser URL-resolution fallback
(`axios/lib/platform/common/utils.js:43`); browsers use `window.location.href`.
It is not the application's API endpoint. No secret values were added/read into
frontend configuration. Render build retained at `/private/tmp/ui-4-render-build`;
standard same-origin build restored in `ui/dist` after testing.

## Verification and browser scope

Safe local API, from repository root:

```sh
EXPLANATIONS_ENABLED=false OPENAI_API_KEY= CORS_ALLOWED_ORIGINS=http://127.0.0.1:4173 .venv/bin/gunicorn api.app:app --bind 127.0.0.1:8001 --workers 1 --threads 4
```

From `ui/`:

```sh
npm test
npm run build
VITE_API_BASE=https://board-game-ai-lab.onrender.com npm run build:render
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/board-game-ai-lab-playwright npx playwright install firefox webkit
VITE_API_BASE=http://127.0.0.1:8001 npm run build
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/board-game-ai-lab-playwright PLAYWRIGHT_BASE_URL=http://127.0.0.1:4173 PLAYWRIGHT_API_URL=http://127.0.0.1:8001 npm run test:e2e
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/board-game-ai-lab-playwright PLAYWRIGHT_BASE_URL=http://127.0.0.1:4173 PLAYWRIGHT_API_URL=http://127.0.0.1:8001 npx playwright test polish.spec.js
```

After browser checks, repeated the correct Render build/artifact inspection and
`npm run build` to restore the standard output. From repository root:

```sh
git diff --check
```

| Check | Result / scope |
| --- | --- |
| Node/JSDOM | **102/102 passed**, zero failed/cancelled/skipped; all existing contract and lifecycle coverage retained |
| Chromium 153.0.8010.12 | **49/49 passed**: all 43 existing regressions plus six final smoke/layout cases |
| Firefox 155.0 | **6/6 passed**: focused smoke at all six widths |
| WebKit 26.6 | **6/6 passed**: same focused smoke at all six widths |
| Combined full run | **61/61 passed**, no skipped cases |
| Final screenshot/fixture confirmation | **18/18 focused cases passed** after making captures portable and presentation text/citations coherent |
| Standard / Render / local API builds | All passed; correct HTTPS production origin verified |
| Lockfile / whitespace | Dependency graph unchanged; `git diff --check` passed |

The focused suite covers homepage/CTA/nav, metadata, direct game refresh, Random
start and real human/AI moves, all opponent radios, selected/busy semantics,
keyboard board play, optional question/counter, What If visibility, populated
mocked analysis, highlights, long curated citations, evidence, keyboard
disclosures/source focus and overflow. Full Chromium additionally retains full
Random/Negamax/MCTS games, restarts, independent sessions, initialization/AI
failures, expiry, cold/malformed starts, uncertain/lost-commit recovery, unmount
cancellation, strict explanation responses, stale rejection and disabled service
with gameplay independence. No paid/live provider calls: explanations are
intercepted, and the disposable backend explicitly disables them with an empty key.

Initial smoke failures were test-focus setup issues: route/anchor animation-frame
focus needed to settle, Firefox requires actual keyboard traversal for
`:focus-visible`, and macOS WebKit uses Option-Tab to include links. Tests now
exercise native traversal (Tab or Option-Tab) and Enter/Space; no tab-order shim
or browser-specific application code was introduced. Browser download DNS and
local process/browser sandbox restrictions were resolved using approved execution.

Backend Python tests were not run: backend files/contracts are unchanged.
Live deployment/provider smoke and manual screen-reader verification were not
performed. No tests point at production. Temporary API/preview servers were stopped.

## Final imperative DOM inventory

Only three occurrences in active application source:

| Location | Why it remains |
| --- | --- |
| `ui/src/main.jsx:7` | `document.getElementById("root")` supplies the React mount node |
| `ui/src/components/AppShell.jsx:14` | Looks up a hash destination for scroll and accessibility focus |
| `ui/src/components/AppShell.jsx:19` | Looks up the skip/route destination to focus it after navigation |

No active gameplay/analysis DOM queries, node creation, `replaceChildren` or
manual `addEventListener` remain. Test DOM queries/mutations and Playwright
geometry/focus helpers are verification utilities outside the production app.
AbortController references in the hooks are cancellation, not legacy controllers.

## Final portfolio screenshots and readiness

Captures are full-page PNGs in the ignored directory
`ui/playwright-report/portfolio/chromium/`:

1. `homepage-1440.png`
2. `homepage-375.png`
3. `connect4-active-1440.png`
4. `connect4-active-375.png`
5. `analysis-populated-1440.png`
6. `analysis-populated-375.png`

Equivalent captures at 1024/820/768/320 and Firefox/WebKit captures are retained
alongside them (57 captures total). Visually inspected the requested portfolio
states and breakpoint/narrow layouts. Games use real local Random replies;
analysis is a mocked presentation fixture with curated source references,
never a paid-provider result or strength claim.

**GO for review, commit and deployment of this frontend branch.** No identified
production frontend regression or completion blocker remains. Optional remaining
debt: manual assistive-technology verification, a future custom social preview
image, and archival backend cleanup outside this pass. Live deployment settings
and post-deploy smoke still belong to the later authorized deployment workflow.
No commit, push, deployment, README rewrite or resume work was performed.
