# Victor Research competition lab integration

October 9, 2026. Implemented in the MacBook checkout on
`victor-labs-and-research-ui`, based on verified remote main
`8a3c3e0e1e0d612ae121819d5fde4341aee8e3ab`. The initial tree was clean; fetching
main reconciled the local pre-merge release checkout without resetting anything.
No Mac Mini training environment, AlphaZero work, backend algorithm, benchmark
result, Render setting, production flag, or deployment was changed.

## Architecture and behavior

`connect4/researchCompetitor.js` defines Victor's identity, labels, strict flag,
known types, execution guard, and shared caveats. `researchAgent.js` retains the
Play-page adapter and re-exports that definition. Separating the definition from
payload adapters prevents circular module initialization. Existing classical
presets and defaults remain intact. Victor's canonical configuration is exactly
`{ type: 'victor_research' }`; unexpected fields and unknown types are rejected.
The Play view model still normalizes its existing inactive depth/simulation/turn
controls without sending any Victor search settings.

Match Lab offers **Victor Research (Experimental)** independently on Red and
Yellow, with a short explanation and no budget controls. Switching to Victor
clears inactive ordinary-agent settings. Human play, AI stepping, autoplay,
revision validation, history, timeline, and replay use the existing contracts.

Tournament Lab supports explicit Victor entrants in all four field sizes,
including duplicates and one optional Human. Strict entrant and completed-game
validation includes Victor. Bracket shuffle, color seeds, swapped rematches,
three-draw resolution, advancement, and champions are unchanged. Season Lab
supports explicit duplicate Victor entrants across its existing five sizes and
three games-per-pairing options, with the same balanced schedule, analytics,
ratings, pairwise results, and history. Victor is absent from both default fields.

## Flags, persistence, and recovery

`VITE_VICTOR_RESEARCH_ENABLED=true` or `1` exposes new Victor choices and enables
execution; the API separately requires `VICTOR_RESEARCH_ENABLED=true`. Production
already has both flags enabled. This implementation changes neither setting.

Saved Tournament v1 and Season v1 schemas, storage keys, and schedule version 1
are unchanged. Validation/replay of known Victor evidence is independent of the
UI flag. Completed records and exports remain readable with the flag off; new
Victor creation and mutation are blocked in controllers and UI controls. Existing
classical saved records remain compatible without migration.

All execution stays sequential: one locked request, authoritative history read,
then the next timer boundary. Pause settles the current request. Busy, failed,
disabled, and lost-response cases stop execution and reconcile by GET; they never
automatically repeat a POST. A failed history read keeps execution locked until
explicit refresh. Lost starts and expired sessions require explicit restart of
only the current game, preserving completed evidence. Tournament recovery also
permits explicit acknowledgment after a lost terminal response leaves no active
game. No ordinary recoverable error requires restarting the entire competition.

A visible warning explains Victor's slower native bounded search on Render's
Free-tier CPU and the cost of large seasons. The existing backend dedicated
nonblocking Victor reservation is untouched. The one-instance service's roughly
0.15 CPU cores / 512 MiB are hosting constraints, not local timing predictions.
Separate users or labs can still contend; `503 agent_busy` remains recoverable.

## Evaluation integrity

Ordinary seasons keep schema 1 / methodology 1 and their existing canonical
payloads and hashes. Victor seasons deliberately use schema 2 / methodology 2,
with a separate `victor_research: 1` reference policy identifier and explicit
wall-clock, seed-scope, and historical-replay caveats. Scoring, schedule, analytics,
canonicalization, and digest coverage are unchanged. The updated browser and CLI
verifiers reconstruct boards/schedules, reject extra/unknown configurations,
check hashes, and recompute every derived aggregate.

The current API manifest identifies the engine and classical agents. Per-game
manifests are retained verbatim; no old engine identifier is claimed to identify
Victor. The export panel and verifier explicitly warn that Victor's reference
identifier is not authentication of historical execution. Record backend source
commits/runtime details for campaigns. See [provenance](evaluation-provenance.md).
No canonical artifact under `benchmarks/connect4/results/` was modified.

## Website and research boundaries

The homepage shows four playable approaches only when the flag is on, with a
compatible SVG illustration for Victor and responsive four/two/one-column cards.
Its research entry covers Allis's nine-rule framework, conditional/narrower
verified guarantees, 1,722 selected exact book positions, native bounded proof
search, heuristics, and independent exact-oracle benchmarking, with documentation
links. It claims neither complete arbitrary nine-rule non-loss nor perfect play
nor exhaustive benchmark accuracy. DQN, neural MCTS, and AlphaZero-style work
remain experimental research, unavailable in public play.

Seeds determine bracket/schedule, colors, and seeded agents' randomness. Victor's
wall-clock budgets mean contention and hosting performance can change choices.
Recorded moves establish historical play, not identical future re-execution.

## Validation

Final verification:

| Check | Result |
| --- | --- |
| `cd ui && npm test` | 431 passed, no skips |
| Full configured Playwright suite, enabled build | 225 passed: Chromium 159, Firefox 33, WebKit 33; one disabled-only test intentionally skipped |
| Focused enabled Victor browser suite | 33 passed; one disabled-only test intentionally skipped (included in full run) |
| Focused disabled Victor browser suite | 7 passed; 27 enabled-only tests intentionally skipped |
| Production build, enabled and disabled | PASS |
| Render-mode build with explicit HTTPS API origin | PASS |
| Independent CLI, completed Victor Season v2 | PASS (test-generated evidence; not a strength benchmark) |
| Independent CLI, frozen canonical Season v1 | PASS; SHA-256 `e5845905b29c158cebed8579cd0a858b3a873a289d2b2e754c29601b7c9e407d` unchanged |
| Responsive checks | 1440, 820, 375, 320 px; homepage screenshots visually reviewed |
| Final diff review and `git diff --check` | PASS |

New coverage includes Victor/ordinary-agent and Victor/Victor histories on both
colors, Human tournament play, tournament advancement, all supported season
sizes/pairing options and balanced colors, full mocked seasons/analytics,
partial/complete exports, captured-core-provenance preservation, strict
configuration rejection, storage round trips, reloads, disabled execution,
busy/failed/disabled agent responses, committed lost responses, failed recovery
reads, pause during an in-flight request, and revision/duplicate-move assertions.

Local setup used an isolated API on 8001 (8000 was occupied) and production UI
previews on 3000 (enabled) / 3001 (disabled). Matching Playwright binaries were
downloaded into temporary storage. Initial setup runs found the dev proxy’s
fixed 8000 target and exhausted the reused test API’s bounded session store;
final verification used the explicit API origin and a fresh disposable store.
Initial old dropdown assertions were updated to account for the optional preset
and to select Human semantically. Final runs have no product assertion failures.

Example enabled browser command, from `ui/`, after starting the local API/UI:

```sh
PLAYWRIGHT_BROWSERS_PATH=/tmp/board-game-ai-lab-browsers \
VICTOR_LABS_UI=true VICTOR_RELEASE_UI=true VICTOR_RELEASE_API=true \
PLAYWRIGHT_BASE_URL=http://127.0.0.1:3000 \
PLAYWRIGHT_API_URL=http://127.0.0.1:8001 npm run test:e2e
```

For disabled-build checks use `VICTOR_LABS_UI=false`, port 3001, and
`npm run test:e2e -- --project=chromium e2e/victor-labs.spec.js`.
Restart the disposable API between full runs to avoid accumulating unrelated
sessions up to its normal 128-session limit.
Most new failure-path coverage is mocked; real smoke tests use only localhost
with Victor enabled and paid explanations disabled. No public load campaign or
paid API calls were performed. Backend code is unchanged, so no backend suite
was required for this frontend milestone.


## Recommendation and limits

**GO for the reviewed frontend milestone**, preserving the existing production
flags and configuration. This commit does not deploy anything. Victor remains
experimental and can lose; Render CPU contention can slow moves and change
wall-clock-budget choices. The verifier validates historical evidence and
analytics, not authenticated agent identity or deterministic future play. No
hosted performance guarantee or exhaustive mathematical strength claim is made.
