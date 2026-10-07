# UI-3: React analysis workspace

## Starting state and preserved invariants (recorded before implementation)

Verified `portfolio-ui-redesign`, clean working tree, and committed UI-2
`c6f9740`. React owns gameplay and board rendering. `ui/legacy/connect4.js`
is absent; `ExplanationPanel` actively mounts `ui/legacy/explanations.js`.
No commits, pushes, other-branch operations, backend or research changes.

The complete legacy controller and unit/browser explanation tests, UI-2 hook,
board/panel/styles, gameplay unit/browser tests, workspace tests, backend route,
response assembly and explanation contract were inspected.

- Separate state machines: analysis only reads a game snapshot and availability.
  Analysis errors never call gameplay POSTs, reconciliation GETs or recovery.
- Requests are explicit POSTs to `/v1/connect4/explain` with game ID, revision,
  mode and optional question; only `what_if` sends a zero-based column. Visible
  columns remain one-based. Only legal, nonterminal columns may be requested.
- Questions are optional, escaped text and capped at 500 characters.
- Abort plus generation/lifetime guards prevent stale success **and failure**
  from rendering after replacement, move initiation, revision changes, restart,
  game replacement, unavailability or navigation. HTTP abort alone is insufficient.
- All-or-nothing validation: response game/revision/mode/column binding,
  explanation object, nonempty confirmed tactical facts, at most three key facts
  contained in full facts with identical text, summary/focus/referenced fact IDs,
  at most one connected contextual strategic entry, additional context,
  limitations, references and at most 42 internally consistent square coordinates.
- Tactical facts are `confirmed_tactical`; concepts are `context_only` or
  `reference_only`. Context is not formal rule proof; analysis is post-hoc and
  does not expose intent, internal reasoning or search traces.
- Sources use a fixed trusted Allis PDF URL and backend chapter/section/page
  metadata with `_blank` and `noopener noreferrer`; arbitrary URLs are ignored.
- Full tactical evidence, reference context, preconditions and all limitations
  remain available in native disclosures. Text never becomes HTML.
- Highlights and results clear on a replacement request, failure, move/restart,
  stale snapshot or unmount. Highlight labels never alter pieces or legal moves.
- Analysis loading disables analysis inputs only. Human/AI play and restart
  remain governed exclusively by the gameplay hook.

Implementation and verification results are appended after the completion gate.

## Final architecture

`Connect4Page` composes the unchanged `useConnect4Game` and independent
`useConnect4Analysis(http, game, busy || uncertain)`. It passes the latter's
validated highlights to `GameBoard` and analysis state to `AnalysisPanel`.
`AnalysisControls` renders three native action buttons with `aria-pressed`, a
controlled question and controlled legal-column selector. `AnalysisResult`
composes `PrimaryExplanation`, `TacticalEvidence`, `StrategicContext` and
`AnalysisDetails`; the private `Concept` component centralizes classifications
and safe citations. No dependencies were added.

Analysis owns mode, question, hypothetical column, phase (`idle`, `loading`,
`success`, `error`), status and the validated revision-bound response. Highlights
are derived only from that response. A ref holds request generation, abort
controller, requested payload and mounted lifetime; another holds the committed
snapshot. Layout effects invalidate analysis before the new board is painted.
Results are also masked by current game/revision/availability during rendering.
Identical same-tick pending requests are suppressed; distinct replacement
requests abort the old controller and advance the generation. Both late success
and late rejection are ignored. Unmount and HTTP-client changes also invalidate.

Analysis only calls the existing explanation POST. It has no gameplay state
setter, POST or recovery GET, and never influences the gameplay busy flag.
Question text and legal column preferences remain separate from game snapshots.
No backend, provider, API contract, research, AlphaZero or deployment file changed.

## Validation and grounding

`ui/src/connect4/analysisValidation.js` extracts the legacy structural checks
without weakening them. It adds null guards for malformed fact, concept, square
and reference entries. `validateAnalysisResponse` binds game ID, revision, mode
and hypothetical column to the captured request before accepting the entire
explanation. Unsupported summary fact references, mismatched key evidence,
invalid classifications, inconsistent square coordinates and malformed curated
references reject the entire response, including highlights.

`confirmed_tactical`, `context_only`, and `reference_only` remain distinct.
Post-hoc and formal-proof boundaries are visible before any disclosure; all full
facts, additional concepts, reference preconditions and limitations remain in
native disclosures. React escapes all result/error/question text. Source links
use the fixed Allis PDF constant, never `entry.source.url`, retaining original
chapter, section and thesis/PDF-page metadata and safe new-tab attributes.

## Declarative board integration

The hook derives `highlights` from the validated response's `relevant_squares`.
`GameBoard` checks the validated names against each cell's existing column and
bottom-based row name, rendering `.explanation-square` and an `aria-hidden`
`.square-label`. Piece values, coordinates, legal moves and click behavior are
unchanged. Snapshot masking, request replacement/failure and layout invalidation
remove stale markers. Opponent-selection rerenders preserve current markers.

## Files

Added:

- `ui/src/connect4/useConnect4Analysis.js`
- `ui/src/connect4/analysisValidation.js`
- `ui/src/connect4/AnalysisPanel.jsx`
- `ui/src/connect4/AnalysisControls.jsx`
- `ui/src/connect4/AnalysisResult.jsx`
- `ui/tests/analysisValidation.test.js`
- `docs/ui-3-analysis-migration.md`

Modified:

- `ui/src/pages/Connect4.jsx`
- `ui/src/connect4/GameBoard.jsx`
- `ui/connect4/connect4.css`
- `ui/tests/explanations.test.js`
- `ui/e2e/explanations.spec.js`

Removed after parity tests passed:

- `ui/legacy/explanations.js`
- `ui/src/connect4/ExplanationPanel.jsx` (unused migration wrapper)

The active Connect 4 feature contains no DOM queries, DOM node creation or
imperative result/highlight mutation. Existing `main.jsx` looks up React's mount
root; `AppShell.jsx` looks up navigation destinations for scroll/focus. Those
unrelated shell accessibility operations remain unchanged. Unused standalone
Connect 4 HTML/JS/CSS and duplicate page CSS remain outside this migration.

## Behavioral parity

| Existing guarantee | React implementation and verification |
| --- | --- |
| Explicit last-move/position/what-if actions | Native action buttons; captured payload; unit and browser tests for all three modes |
| Context-aware last AI move | Snapshot previous-player label; real human/AI browser moves verify label changes |
| Optional question, 500-character limit | Controlled textarea, maxlength, count/helper, request-length guard; exact payload and browser cap tests |
| Legal hypothetical columns / numbering | Legal-only options and integer/membership/terminal guards; zero-based API and one-based labels/status |
| Loading leaves gameplay available | Independent phase; only analysis controls disable; board and restart verified enabled during loading |
| Disabled/unavailable/provider/rate-limit/stale/abort errors | Isolated retryable error state; no gameplay recovery calls; unit cases and real gameplay browser regressions |
| Move/restart/navigation cancellation | Layout/unmount invalidation, AbortController plus generation; unit signal assertions and delayed browser fixtures |
| Replacement cancellation | Distinct request aborts old signal; same-tick identical request suppression; late error and late success regressions |
| Old revision / wrong game / wrong mode / wrong column | Captured request binding plus current-snapshot guard; unit/browser rejection with empty results and no highlights |
| Whole-response integrity | Pure validator and 43 dedicated contract cases; malformed/null/HTML responses reject safely |
| Grounded primary explanation and bounded evidence | Summary and at most three matching verified key facts rendered in React; complete facts in details |
| Context versus proof | Classification badges, visible conceptual disclaimer, reference-only details, preserved preconditions and limitations |
| Citation provenance and XSS safety | Fixed PDF URL with metadata and safe rel; forged source URL ignored; HTML remains literal text |
| Highlight lifecycle | Props and JSX; completed/pending cleanup after moves, restart, navigation, replacement and failure |
| Gameplay independence | Unchanged gameplay hook/unit suite, immutable snapshot assertion, complete Random/Negamax/MCTS games and recovery browser suite |
| No live/paid provider traffic | Mock HTTP unit clients, intercepted browser explanation requests, local backend disabled and API key empty |

All original 42 unit cases remain (controller cases migrated to React); 59
meaningful contract/lifecycle cases were added. All original 30 browser cases
remain; 12 analysis, responsive, keyboard and failure cases were added.

## Verification

From `ui/`:

```sh
npm test
npm run build
VITE_API_BASE=https://board-game-ai-lab-api.onrender.com npm run build:render
VITE_API_BASE=http://127.0.0.1:8001 npm run build
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/board-game-ai-lab-playwright PLAYWRIGHT_BASE_URL=http://127.0.0.1:4173 PLAYWRIGHT_API_URL=http://127.0.0.1:8001 npm run test:e2e
```

The HTTPS origin was a build input only; no production request or deployment
was made. The local API-base build is the same verification variant used in UI-2.
Backend from repository root:

```sh
EXPLANATIONS_ENABLED=false OPENAI_API_KEY= CORS_ALLOWED_ORIGINS=http://127.0.0.1:4173 .venv/bin/gunicorn api.app:app --bind 127.0.0.1:8001 --workers 1 --threads 4
```

From repository root: `git diff --check`.

Final results: **101/101 unit tests, 42/42 Chromium browser tests**, all three
build variants and diff whitespace check passed. No skipped unit/browser cases.
Initial responsive checks needed keyboard modality in their test setup; corrected
Tab navigation and the final full suite pass. Local port binding required approved
sandbox escalation. No backend tests were run because backend files and contracts
are untouched; no live provider verification was attempted. Temporary servers
were stopped and the standard build restored after verification.

## Responsive/accessibility and visual review

Inspected full populated-result screenshots at 1440px and panel screenshots at
820px, 375px and 320px, including expanded detailed/methodology disclosures.
Desktop/tablet use three analysis action cards and two input columns; mobile
stacks cards and fields. Tactical cards flow to one column, source links and long
text wrap, and highlighted labels fit the narrow board. Automated overflow
checks pass for all four populated/expanded layouts and the long 320px error.
Loading and disabled-service screenshots at 320px show available gameplay with
only analysis controls locked. No workspace overlay or global dimming.

Buttons, textarea/select labels and helper associations, `aria-pressed`, atomic
live status, `aria-busy`, native details, visible keyboard focus and Enter/Space
activation are covered. Highlight labels are hidden from assistive technology;
analysis text conveys the squares. No manual screen-reader audit or non-Chromium
browser run was performed.

Screenshots (ignored generated artifacts, also copied to `/private/tmp/ui-3-analysis/`):

- Desktop full result: `ui/test-results/explanations-populated-ana-ff03b-d-disclosures-fit-at-1440px/analysis-desktop-result.png`
- Mobile full result: `ui/test-results/explanations-populated-analysis-and-disclosures-fit-at-375px/analysis-mobile-result.png`
- Tablet and narrow-mobile result/panel/expanded screenshots are in the corresponding responsive test directories.
- Loading, disabled and long-error captures: `ui/test-results/explanations-loading-disab-bb311-le-gameplay-stays-available/analysis-{loading,disabled,long-error}-320.png`

## Remaining debt and UI-4 recommendation

No known UI-3 behavioral blocker remains. Recommend UI-4 as a focused final pass
for cross-browser and assistive-technology verification, responsive spacing and
copy refinement across the shell/home/game/analysis, and an explicit audit of
unused standalone UI assets and duplicate CSS before any removal. Keep the
existing analysis and gameplay regression coverage and the no-provider test
setup. UI-4 has not begun. No commit or push was made.
