# Phase 4D.4K-B — AlphaZero frontend and release preparation

## 1. Executive summary

Release candidate completed on `phase4d4k-b-alphazero-frontend`, with independent
review, green bounded local ARM64 staging, full local browser regressions and
green remote native ARM64 CI. Ready for final review and separately approved
merge/deployment. **Not merged or deployed;
AlphaZero has not been enabled publicly.**

The experimental generation-6 CNN can be selected in Play, Match, Tournament and
Season Labs only when the backend reports it available. Safeguard Off preserves
the original baseline; On applies the same verified one-ply tactical selection
after unchanged PUCT. No model, search budget, guard algorithm or research
checkpoint changed. This is not an optimal solver or a state-of-the-art claim.

Across two CNNs against Negamax3/5, historical confirmation measured +1.302 pp
match score (95% [+0.521,+2.344]), below the prospectively declared +2pp practical
threshold. The UI states this context and does not promise stronger play in every
position. The safeguard is not a separately trained model.

## 2. Base, preflight and protected environment

Base: completed backend branch `phase4d4k-a-alphazero-backend`, commit
`5ac8387207fceb8d6852c79610a14b9246fefe4f`. Remote refs fetched; clean historical
checkout observed; ancestry against current main verified at
`8704142702da273e6571f0ca9df3fd4a793be285`. A new isolated worktree was created at
`experiment-output/phase4d4k-b-worktree`. No historical research branch was merged.

Reviewed K-A architecture/contract and the H-C, I-A, I-B, J-A and J-B reports
read-only from their committed refs. Inspected existing selectors, hooks,
controllers, export/verifier, persistence, deployment and CI before edits.

The live Mac mini container stayed at its original image, container ID, start
time and restart count. It remained on localhost port 8000 and was never used for
testing. No Services checkout, Render configuration, Funnel setting, credential,
production container or research job was changed. No active training was found
before bounded cached Docker builds. No training or research experiment ran.

## 3. Frontend architecture and discovery

`connect4/alphazero.js` owns the optional-agent contract, provenance validation,
identity comparison and shared per-HTTP-client discovery cache.
`useAlphaZero.js` subscribes via `useSyncExternalStore`, with effect cleanup for
focus listeners. Mount/focus reuse a 60-second cache; explicit refresh and
AlphaZero game starts recheck status. There is no polling and no discovery request
before every move. Sequence identities prevent an older response overwriting a
newer refresh. Unknown availability fails closed.

`GET /v1/connect4/agents/alphazero` is authoritative; no VITE flag controls
AlphaZero. Available status must have real booleans, consistent model/search
fields, supported versioned provenance and a boolean guard option defaulting Off.
404 means unsupported backend; disabled, enabled-but-unavailable, malformed and
network-failed responses hide the new option and leave legacy agents usable.
Availability messages and a normal accessible refresh control do not replace the
game status/error UI. Agent errors invalidate stale discovery. Saved configurations
remain identifiable even when continuation is unavailable.

## 4. Selector and tactical safeguard UX

Play offers **AlphaZero (Neural MCTS)** with an Experimental designation. Match
and field presets identify **AlphaZero (Experimental) · safeguard Off/On**.
`AlphaZeroSettings.jsx` uses existing controls, helper text and a collapsible
research explanation. It exposes only Off/On; no seed, simulation, temperature,
noise or exploration tuning. Payloads contain a real JSON boolean. Selection
starts Off. Strict payload/config validation rejects unsupported settings.

Existing typography, spacing, mobile layouts, focus rings and reduced-motion
behavior are retained. Home research copy now explains that optional AlphaZero
appears only when supported by the connected backend; it does not announce a
public deployment. The refresh control uses the existing accessible action style.

## 5. Play

The existing game hook and start/move endpoints serve both colors, both guard
settings, restart/new configuration, terminal games and ordinary analysis UI.
Settings apply to a new game; active-game settings are shown separately and never
mutated by the toggle. AI search uses the existing thinking state.

AlphaZero starts compare authoritative history with discovered identity and
expected canonical players before allowing play. Integrity failure quarantines
the board and offers restoring the original model or a verified new game.
Uncertain request recovery uses full AlphaZero history and identity validation,
not just an ordinary snapshot that could unlock mismatched evidence. Explicit
retry rechecks availability/identity and current revision. Position/last-AI
explanation UI is exercised with mocks; no paid provider call was made.

## 6. Match Lab

Shared payloads, labels, snapshot/history validation and move transport support
Human in either color, all eligible existing agents, AlphaZero versus itself,
and guarded versus unguarded configurations. Step, autoplay, pause, timeline,
terminal winner, replay and reset retain the existing hook's serialization.
Discovery/history pinning and busy cancellation share its lifetime/abort guards.
AlphaZero ignores `rng_seed`; this is stated in lab/provenance documentation.

## 7. Tournament Lab

Field presets include both guard variants when available. Entrant IDs distinguish
duplicates; exact boolean guard settings survive bracket planning, color changes,
match records, local replay, winner descriptions and persistence. Existing sizes,
seeding, draw/rematch/tiebreak rules and Human workflow are unchanged.

AlphaZero fields use local storage **version 2**, with a single pinned model/search
identity; legacy fields remain version 1. Existing storage keys let the validated
loader read both. Status is checked before each fixture start; history is checked
before accepting results. Model change/unavailability stops advancement without
replacing an entrant. Restore the original model or create a new tournament;
completed results remain reviewable.

## 8. Season Lab

Both guard presets work with unchanged balanced circle schedules, swapped-color
pairs, multi-game pairings, standings, ratings, intervals, pairwise summaries,
replay and exports. AlphaZero Seasons use local persistence version 2 and pin one
identity. Every completed AlphaZero game carries its captured identity; mixed or
missing identity is rejected. Ratings and scoring were not changed. The canonical
112-game Season and all prior benchmark/export bytes were not regenerated.

## 9. Versioned evaluation provenance

Evaluation **schema 3 / methodology 3** is used only when entrants include
AlphaZero. Backend core `/provenance` is unchanged. Available status adds
`provenance.version=1`; an AlphaZero session stores a deep copy at creation and
returns it as `alphazero_provenance` in history even if the runtime later becomes
unavailable. Before AlphaZero search, a different runtime identity is refused
without committing a move. This is the smallest explicit backend addition.

The captured identity includes type, implementation version, model ID,
architecture/parameters, artifact and source-checkpoint SHA-256, PUCT algorithm,
256 simulations, exploration 1.41, no root noise, maximum-visit final selection,
exact-position tie-break contract, ignored game seed, guard contract and source
hash. Entrant/player configs separately carry guard enablement. v3 evidence and
provenance include the pin; each completed AlphaZero game's evidence includes it.

Canonical sorted JSON and SHA-256 coverage remain unchanged in structure. v3 CSVs
add explicitly named provenance/guard columns. The verifier reconstructs schedules,
columns, results and analytics and rejects downgrade/missing/mixed provenance,
even after the digest is recomputed. Hashes prove internal consistency, not server
authentication or identical future floating-point search across every runtime.

## 10. Legacy compatibility and history/replay

Ordinary v1 and Victor v2 schema/methodology/provenance/CSV identities remain
frozen, with no silent upgrade. Independent base/current generation compared
JSON and both CSVs byte-for-byte for v1 and v2; both verified. Canonical fixtures
and the immutable 112-game benchmark still verify with their original digest.

History validation accepts optional AlphaZero model/guard/original-action/visit
evidence without changing move ordering, actor, revision or legal reconstruction.
It checks guard identity, integer 256 visits, model agreement and valid guard
reasons when that evidence is present. Compact competition storage keeps columns
and a small versioned provenance block, not duplicate boards and root vectors.
Full server history retains the move evidence; local replay remains column based.

## 11. Busy, unavailable and reconciliation

Only explicit 503 `agent_busy`/`game_busy` responses enter AlphaZero bounded busy
retry: at most two retries, 250/750 ms minimum delays, honoring Retry-After up to
five seconds. Invalid/long Retry-After requires explicit continuation. No uncertain
network failure, stale revision, rate limit or failed-agent response is blindly
resubmitted. Established controllers reconcile authoritative revision/history.

Pause epochs cancel pending retries; unmount abort/lifetime guards prevent old
hook retries. An in-flight accepted competition POST may settle/persist after
pause or route departure, under the existing sole mutation owner; it never queues
another ply. Timers and synchronous locks prevent overlapping autoplay, duplicate
advancement and replacing a busy session. Unavailable/invalid agents stop autoplay,
preserve completed results and never fall back. Expired/restarted sessions require
explicit new/restarted games. Saved evidence can be reviewed offline.

## 12. Docker deployment preparation

`deploy-mac-mini-api.sh [lite|alphazero]` defaults to lite. Both build and runtime
flags are explicit and override stale private env values. Full commit+mode image
tags, native ARM64 validation, existing private env handling and one-worker/four-
thread Gunicorn are retained. No shared search limit or Funnel topology changed.

Candidate creation records an immutable owned container ID; cleanup never removes
a foreign named candidate and handles start/port-binding failure. Localhost 8001
candidate checks include health, exact core/source provenance, mode-specific
status, real legacy-agent moves, optional Victor and both AlphaZero guard moves.
An unavailable model fails before cutover. Checks repeat after cutover; failure
restores the original container/environment/image and preserves rollback behavior.
**The actual production deployment script was never run.** Only mocked Docker
commands tested cutover/rollback logic.

## 13. Dependency supply chain and CI

The ARM64/Python3.11 runtime now has a verified nine-wheel CPU hash lock, including
all applicable torch transitive dependencies at already-tested versions. Native
resolver metadata, all nine official index digests, and force-reinstalled hashed
Docker downloads were independently checked. Lite still contains no torch.
See [dependency procedure](alphazero-runtime-dependencies.md) for exact update
commands and limits: optional AMD64 remains version-only; base OS/apt/ordinary API
packages are outside this narrow lock.

The new `alphazero-inference.yml` uses native `ubuntu-24.04-arm` (supported in the
[GitHub runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)),
relevant path filters including the lock, bounded timeouts, read-only repository
permissions and cached builds. It covers frontend unit/build, ARM64 image/model
integrity, CPU/thread settings, inference/guard/API/deployment mocks and Chromium
against its isolated service. It never deploys, contacts Mac mini, or publishes an
image. The existing main-only lite publishing workflow and triggers are unchanged.

## 14. Isolated staging architecture

Cached K-A images were inspected but final K-B images were built separately.
Owned identities/ports only:

| Service | Port | Purpose |
| --- | --- | --- |
| `bgai-k-b-staging` | localhost port 18114 | final hash-locked AlphaZero+Victor API |
| Vite | localhost port 18115 | React proxy via `BGAI_DEV_API_TARGET=...18114` |
| `bgai-k-b-lite` | localhost port 18116 | default torch-free/disabled API |
| earlier `bgai-k-b-locked` | localhost port 18117 | locked-build smoke, later removed |
| `bgai-k-b-unavailable` | localhost port 18118 | lite image enabled without torch, controlled unavailable |

Explanations disabled; explanation browser tests mock responses. No live port 8000,
production Services checkout, research fixture or public backend request was used.
Staging source-commit env used the base SHA as a declared local marker while the
working tree was under construction; image identities and final Git changes
identify actual candidate content. Production/CI use their exact committed SHA.
Cleanup completed after final verification: all three phase-owned staging
containers and both phase-owned Vite servers were removed/stopped. The temporary
lite venv, own dependency symlink (historical target left intact) and phase-specific
browser downloads were removed. Only the final `bgai-k-b-staging:lite-final` and
`:alphazero-final` images are retained for review; three obsolete K-B image tags
were removed. K-A and production images were preserved. Local logs/screenshots
remain under phase-prefixed `/tmp` paths as inspection evidence. Before/after
live container ID/image/start time/restart count still match exactly.

Only one real game per small Tournament/Season smoke was needed; large fields in
legacy browser tests use existing synthetic HTTP fixtures, not neural evaluation.

## 15. Regression results

| Check | Final result |
| --- | --- |
| Full torch-free backend |3077 passed,0 failed,17 skipped |
| Native targeted API/history/deployment after session pin |149 passed,0 failed |
| Final linux/arm64 inference/API/guard/provenance/deployment |122 passed,0 failed |
| v2 inference/contracts |90 passed,0 failed,14 deselected (training/generation/resume) |
| Frontend unit/contract suites (final) |463 passed,0 failed,0 skipped |
| Production Vite build |PASS |
| Deployment mock suite |13 passed,0 failed |
| AlphaZero Chromium matrix before added reverse-Human/explanation cases |22 passed,0 failed |
| Corrected14 legacy browser failures targeted rerun |14 passed,0 failed |
| Full Chromium before final harness rerun |183 passed,0 failed,1 skipped (disabled-Victor UI case run separately) |
| Final configured full Chromium |183 passed,0 failed,1 skipped; all Victor test flags matched frontend/API |
| Firefox |57 passed,0 failed,0 skipped |
| WebKit |57 passed,0 failed,0 skipped |
| Disabled-Victor UI, separate localhost UI |3 passed,0 failed |
| Final isolated lite/enabled/unavailable builds and release smoke |PASS; no torch in lite; missing runtime refuses AlphaZero |
| Remote native ARM64 workflow |SUCCESS:463 frontend;122 native inference;24 Chromium; build/smoke/cleanup passed |

Remote [run 38599224110](https://github.com/andrewcukierwar/board-game-ai-lab/actions/runs/38599224110)
tested exact implementation commit `0dda537cf373d32daabeee7a5564701f95b93b1a`.
Both jobs concluded success on 2026-10-14; the native ARM64 job took 7m29s.
There were zero failed/skipped tests in its 463/122/24 suites. Core/source smoke,
CPU-only import/thread configuration and final container cleanup passed. Startup
health briefly returned connection reset during the bounded readiness loop,
then passed normally. GitHub emitted a non-blocking Node20 action-runtime
deprecation annotation and ran those actions with Node24; application tests/build
used configured Node22. These annotations did not prevent either job from passing.

The 17 torch-free skips cover optional torch-dependent tests and existing platform-specific tests. No training,
self-play campaign, optimizer experiment, paid provider traffic or sealed input
was run to eliminate skips. Bounded v2 tests deliberately deselect 14 training,
generation and resume cases. Existing optional native-platform skips remain.

Test iteration failures are retained rather than omitted: initial frontend
integration 16 failures were discovery-unaware mocks/lifetime timing/status-role
issues, fixed with 440/440 legacy units preserved; first new focused suite 3 failed
on boolean-null handling and test move-config fixtures, fixed. Docker 11 failures
were inherited service flags contaminating pytest, corrected by sanitizing the
subprocess env (also in CI). Initial 22 browser launches failed before execution
because Chromium was absent, then 5/2 native-test interaction/replay assertions
were corrected, ending 22/22. Initial broad legacy browser run 77 failed/82 passed/
1 skipped due catch-all mocks treating discovery as history plus obsolete labels/
presets. Subsequent full run 14 failed/169 passed/1 skipped isolated remaining
stale selectors and one legacy Match error-message regression; all 14 passed after
specific corrections. Backend mock tests initially 5 failed because a Python path
with spaces broke generated shebangs; 13 now pass with a portable temp interpreter
symlink. One concurrent staging release probe saw 503 search contention; serial
ready-server probes pass. These are resolved iterations, not excluded failures.

The first combined Firefox/WebKit attempt reused sessions left by earlier runs
and exhausted the unchanged 128-session staging capacity: 54 passed, 6 failed,
1 interrupted and 53 not run. One Firefox busy/pause failure additionally exposed
pointer scheduling ambiguity in the test; the explicit busy response is now held
until Pause completes. No application concurrency/session limit was raised.
Resetting only the phase-owned staging container between browser projects gives
each suite a clean session store. Firefox then passed all 57 cases; WebKit also passed all 57 cases. Separate output directories prevent concurrent
test artifact cleanup. This initial failed attempt is retained as test evidence.

A later final Chromium rerun accidentally omitted `VICTOR_LABS_UI=true` and
`VICTOR_RELEASE_UI=true` from its test process while Vite had Victor enabled. It
finished with 149 passed, 5 failed and 30 skipped: two preset-count assertions and
three Victor availability assertions expected the disabled UI, while the enabled
Victor cases were skipped. The final full rerun with all test flags matching frontend/API is
recorded above; the separate disabled UI run uses its own port and flags. This
was a test invocation correction, with no implementation/assertion relaxation.
The first correction also omitted `VICTOR_RELEASE_API=true`; it was stopped
early after 77 passed, 1 interrupted and 106 not run, before the known-mismatched
API expectation. The final invocation explicitly sets all three flags:

```sh
VICTOR_LABS_UI=true VICTOR_RELEASE_UI=true VICTOR_RELEASE_API=true \
PLAYWRIGHT_BROWSERS_PATH=/tmp/bgai-k-b-browsers \
PLAYWRIGHT_BASE_URL=http://127.0.0.1:18115 \
PLAYWRIGHT_API_URL=http://127.0.0.1:18114 \
npm --prefix ui run test:e2e -- --project=chromium \
  --output=/tmp/bgai-k-b-chromium-all-flags-results
```

Existing gameplay, sequence/revision, schedule, draw, seed, compact-history and
winner assertions were retained. Browser mock backends now explicitly answer
optional discovery. Legacy browser selectors were updated to the backend branch's
existing Classic MCTS names/custom budgets/preset counts, preserving exact request
budgets and duplicate-configuration/bracket assertions. Frozen export fixtures were
not edited. Full browser suites are rerun, not replaced by focused success alone.

## 16. Browser scenario coverage

The AlphaZero matrix covers available/disabled/unavailable/404/malformed/network
discovery; Human in both colors with each guard; restart/fixed active config;
AlphaZero versus Human/Negamax/Classic MCTS/Tactical MCTS/Victor/AlphaZero; guarded
versus unguarded; step/autoplay/pause/busy cancellation; timeline/replay; uncertain
committed response reconciliation; active unavailability/discovery refresh; small
Tournament and Season with both variants; persisted reload/local replay; v3 JSON
export+self-verification; mobile 320/375px and native keyboard typeahead. Added
reverse-Human Match and position/last-AI mocked explanation cases complete the
matrix. Legacy canonical, seeded comparisons, ratings and larger synthetic fields
remain covered by their existing unit/browser suites.

## 17. Independent review

[Independent report](phase4d4k-b-independent-review.md) was written by a reviewer
who wrote none of implementation/tests. Three initial findings were fixed and
re-reviewed: major Play provenance recovery unlock; major AlphaZero-to-Victor
Play switching; minor Match mismatch messaging. A later minor lock-only CI path
filter omission was fixed/re-reviewed. Candidate-ID cleanup, all nine upstream
hashes/dependency closure, CI lifecycle, and per-session backend pinning received
additional targeted review. The reviewer independently reproduced v1/v2 byte
compatibility, v3 tamper/rehash rejection, real both-color AlphaZero moves, busy
pause/unmount, unavailable/mismatched session pinning, 86 final frontend tests and 13
mocked deployment tests. No unresolved material finding was identified. The
reviewer's documented limits (including remote CI/full browser evidence) are
retained. The reviewer independently inspected the final local browser logs and
remote run status/logs; this is evidence verification, not a claim that the
reviewer reran every full suite. Final controller recovery, legacy Match error wording, homepage/guard
copy and busy/pause browser harness corrections received targeted re-review.
For both controllers, the reviewer independently preserved completed-game bytes
and two validated unfinished plies through a model mismatch, blocked restart
under the changed identity, then explicitly restarted only the unfinished game
after restoring the original model and seed. No unresolved major issue remains.

## 18. Historical preservation

No research/campaign/checkpoint/benchmark path was written. All 17 other worktrees
(including the original historical checkout) were observed with 0 tracked changes.
Selected artifact SHA remains `4c3cf26aa58495f43094436f231486969658d70b94b246a912a5bb54756dc528`;
manifest SHA `004e76c685ad011d0bcb4d67aeedee10372069d7621600c717ac27aa6f3449cc`;
guard SHA `d2ee54ba1fe9ddbcc4641538e2f5dd89a2435234ff194d6909bed72096632d04`.
The canonical 112-game benchmark, old export fixtures, all K-A evidence and
production model paths are absent from the branch diff. Original H-C generation-6
checkpoint hashes were checked read-only: seed 42
`ba2bde2ec71b3ac7486046d1cc26ab591bb59d6b764193ece08c6d6a6f24f58b`;
seed 314159 `2646d7cdee7d7b069412688bf95fd9794bea30402b4952120c904dbcb399401d`.
Live container read-only
before/after identity observations match. Original campaign/checkpoints remain
in their untouched historical worktrees. Sealed fixtures were never opened or
searched; the older broad preservation script was not run because it hashes
sealed payloads. No claim is made to a new per-file sealed audit.

## 19. Remaining limitations and launch recommendation

All required final regression, staging and remote CI evidence is green. No
unresolved material correctness, provenance, deployment or review issue remains.
Recommendation: approve this branch for final merge review and a separately
approved controlled public deployment using section 20. No public launch occurred.
There is one AlphaZero search per process; UI busy handling bounds demand but does
not establish per-user fairness. Neural floating-point runtime differences may
change future decisions; game seeds ignore AlphaZero and do not reproduce research
streams. Compact replay keeps columns/model identity rather than full diagnostics.
Optional AMD64 and base OS/API supply-chain limitations are explicitly documented.
No public latency/availability claim can be made before separately approved
production verification. No unresolved review finding is bypassed to finish.

## 20. Future production release and rollback (approval required)

1. Review this branch, final test evidence and native ARM64 CI. Separately approve
   merging and separately approve public deployment; neither occurred here.
2. In the **approved Services release checkout**, fetch and select the exact
   approved release SHA; confirm clean status and available resource headroom.
   Preserve the private env file, CORS/Funnel/explanation settings and search limits.
3. Run the reviewed `bash scripts/deploy-mac-mini-api.sh alphazero`. Omission or
   `lite` remains torch-free/disabled. Candidate localhost port 8001 must pass all
   probes before replacing the live container. Keep the stopped rollback container.
4. After approval, release that same reviewed SHA through the existing Render
   Static Site: root directory `ui`, build command `npm ci && npm run build:render`,
   publish directory `dist`, existing `/*` to `/index.html` Rewrite. Preserve
   `VITE_API_BASE` as the current approved HTTPS Funnel **origin**, without `/v1`,
   and the existing Victor flag. Optionally set `VITE_EVALUATION_SOURCE_COMMIT`
   to the exact approved SHA. No AlphaZero VITE availability flag exists. Verify
   public discovery, both Play colors/guards,
   small Labs/export, mobile/CORS/health and resource use without paid explanations.
5. Keep rollback until public behavior is verified. On failure, the script restores
   the old container automatically; manual approved rollback is:

   ```sh
   docker rm -f bgai-api
   docker rename bgai-api-rollback bgai-api
   docker start bgai-api
   curl -fsS http://127.0.0.1:8000/v1/connect4/health
   ```

   Funnel still points at 8000. Sessions reset on cutover/rollback. If frontend
   rollback is needed, release the previous static UI revision; it tolerates an
   absent AlphaZero endpoint. Inspect public core/source provenance and existing
   agents before declaring recovery. Remove rollback deliberately only after
   stable verification. These commands are future operator instructions only.

## 21. Publication and changed-file inventory

Implementation commit: `0dda537cf373d32daabeee7a5564701f95b93b1a`, pushed to
`origin/phase4d4k-b-alphazero-frontend`. Final evidence/documentation updates are
published as a subsequent documentation-only commit; no implementation changes
are concealed by that update. Remote CI evidence is recorded in section 15.

### Added (15 files)

- `.github/workflows/alphazero-inference.yml`
- `docs/alphazero-runtime-dependencies.md`
- `docs/phase4d4k-b-alphazero-frontend.md`
- `docs/phase4d4k-b-independent-review.md`
- `requirements-alphazero-arm64.lock`
- `scripts/smoke-api-release.py`
- `tests/test_deploy_mac_mini.py`
- `ui/e2e/alphazero.spec.js`
- `ui/src/connect4/AlphaZeroSettings.jsx`
- `ui/src/connect4/alphazero.js`
- `ui/src/connect4/busyRetry.js`
- `ui/src/connect4/useAlphaZero.js`
- `ui/tests/alphazero.test.js`
- `ui/tests/alphazeroPlay.test.js`
- `ui/tests/fixtures/alphazero.js`

### Changed (47 files)

- `README.md`
- `api/connect4/__init__.py`
- `api/connect4/alphazero.py`
- `api/connect4/state.py`
- `docker/api.Dockerfile`
- `docs/evaluation-provenance.md`
- `docs/mac-mini-deployment.md`
- `scripts/deploy-mac-mini-api.sh`
- `tests/test_alphazero_api.py`
- `ui/e2e/connect4.spec.js`
- `ui/e2e/fixtures/season.js`
- `ui/e2e/fixtures/tournament.js`
- `ui/e2e/home.spec.js`
- `ui/e2e/match-lab.spec.js`
- `ui/e2e/polish.spec.js`
- `ui/e2e/season.spec.js`
- `ui/e2e/tournament.spec.js`
- `ui/playwright.config.js`
- `ui/src/connect4/AgentSelector.jsx`
- `ui/src/connect4/CompetitorSelector.jsx`
- `ui/src/connect4/competitorConfig.js`
- `ui/src/connect4/gameSnapshot.js`
- `ui/src/connect4/matchRecord.js`
- `ui/src/connect4/matchTransport.js`
- `ui/src/connect4/researchAgent.js`
- `ui/src/connect4/researchCompetitor.js`
- `ui/src/connect4/useConnect4Game.js`
- `ui/src/connect4/useConnect4Match.js`
- `ui/src/evaluation/ExportPanel.jsx`
- `ui/src/evaluation/csv.js`
- `ui/src/evaluation/export.js`
- `ui/src/evaluation/provenance.js`
- `ui/src/evaluation/verify.js`
- `ui/src/pages/Connect4.jsx`
- `ui/src/pages/Home.jsx`
- `ui/src/pages/MatchLab.jsx`
- `ui/src/pages/SeasonLab.jsx`
- `ui/src/pages/TournamentLab.jsx`
- `ui/src/season/MatchViewer.jsx`
- `ui/src/season/controller.js`
- `ui/src/season/model.js`
- `ui/src/tournament/MatchViewer.jsx`
- `ui/src/tournament/controller.js`
- `ui/src/tournament/model.js`
- `ui/tests/connect4.test.js`
- `ui/tests/matchLab.test.js`
- `ui/vite.config.js`
