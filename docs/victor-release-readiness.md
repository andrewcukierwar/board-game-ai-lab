# Victor Research release readiness

October 9, 2026. Local preparation only. Started clean on
`victor-astra-second-pass` at `1c9269378bf17694811d6e13df6ec22bf10b1e00`;
`victor-research-release` is based directly on that commit. No push, merge,
deployment, production configuration change, live traffic test, paid explanation
call, AlphaZero campaign change or canonical research artifact modification.

**Recommendation: conditional GO for Stage A with Victor disabled.** Local
engineering checks pass. Before execution, verify current Render configuration
and rollback targets and obtain explicit production approval. **NO-GO for public
Victor enablement until an authorized Stage B measurement passes and Stage C is
approved.** Experimental play can lose; this release does not establish perfect play.

## Build, packaging and proof checks

- Built `docker/api.Dockerfile` locally with `--platform linux/amd64`.
  Image ID: `sha256:44bb981779c6490a4eb6b0bbd0fe29630d3490f82dad560c9c8c254049b03818`.
  The builder compiles C using GCC; the runtime receives the source/platform-keyed
  shared object and virtual environment. `.dockerignore` excludes host binaries,
  virtual environments, test oracles, dependencies and local environment files.
- Runtime reports Linux x86_64/glibc 2.41 and `native.available() == True`.
  The disabled container also loads the binary and book. No `cc`, `gcc` or
  `victor_validation` package is installed in the runtime. C source remains
  packaged because it supplies the loader's source digest; no runtime compilation.
- All **1,722** book entries load and replay to an `exact` lookup. Integrity digest
  checking remains active. The book was not extended or regenerated.
- Six frozen independent-oracle positions cover win/draw/loss in both colours.
  Complete serial native searches return exactly the frozen move values.
  Zero/128-node interrupted searches with tiny tables enclose those values;
  zero-time searches enter zero nodes. A 1 ms native budget returns `time_budget`
  with valid intervals in **1.15 ms**. Incomplete searches retain bounds rather
  than acquiring exact labels; existing proof-integrity regressions also pass.
- Allocation exhaustion (`table_entries=0`), an actually absent shared library in
  a fresh interpreter, and an injected native exception all select legal moves.
  The original binary is restored in the disposable check container. No oracle
  or compiler is used by this harness.

No production-code or packaging defect was found; no solver, budget, Dockerfile
or feature default needed changing. Added a small frozen-label fixture,
standalone Linux smoke script, two backend regressions and browser flag/recovery
checks. The fixture copies 26 existing labelled positions without changing their
sources. This is a release check, not another benchmark campaign.

## Regression and runtime evidence

| Check | Result |
| --- | --- |
| Full backend, requested environment | **1,475 passed, 15 skipped**, 102.61 s |
| Frontend unit suite | **404 passed**, zero failed/skipped |
| Full browser suite, fresh disabled container | **189 passed, 3 intentionally skipped**, 2.7 min |
| Both flags enabled, Chromium | **4 passed**: selection, busy retry, complete games as both colours, lost-response recovery |
| Frontend on / backend off, Chromium | **1 passed, 3 intentionally skipped** |
| Frontend off / backend on, Chromium | **1 passed, 3 intentionally skipped** |
| Frontend production builds | Off, on, mismatch and backend-only builds passed |
| Whitespace validation | `git diff --check` passed |

Backend command: `EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests -q`.
Frontend: `cd ui && npm test`. Browser runs use downloaded browsers in
`/tmp/victor-release-browsers`, explicit loopback frontend/API URLs, and distinct
`--output` directories. The broad suite covers Chromium plus configured Firefox
and WebKit smoke tests, including Match, Tournament, Human Tournament, Season,
Evaluation and mocked explanations. Existing optional skips remain unchanged.

Initial browser launch failed because browsers were absent. Parallel runs then
collided in their shared trace directory; isolated artifact directories resolved
that runner issue. Reusing the same API for full-suite reruns exhausted its
existing 128-session capacity (`503 session_limit`); the final run starts with a
fresh disposable API. No test assertions or application limits were weakened.

Local containers use Render-style `PORT=10000`, one Gunicorn gthread worker and
four threads. Health returns HTTP 200/`OK`. Complete real HTTP games against a
seeded legal human policy finish in **7 and 10 plies**, with exact history replay,
successive revisions and public fields only. Browser games verify both colour
choices and recover from a response lost after the server commits.

Both flags default off. Disabled selection and API errors retain existing
behavior; backend-only enablement leaves the UI unchanged. The frontend-only
case reports that Victor is not enabled and permits choosing a normal opponent.
The enabled description says experimental, not perfect play, and can lose.
Legacy `victor` remains separate; saved data and other labs keep their existing
public player types. Bounds, certificates and research labels remain internal.

## Concurrency, deadlines and resources

Deterministic Linux checks block a native step while the Victor reservation is
held: a second game immediately receives `503 agent_busy`, unchanged state and
history; MCTS and health still succeed. Explicit retry succeeds after release.
An injected whole-agent failure returns `503 agent_failed` without changing the
board/revision/history and releases its reservation. A lost success response is
reconciled; replaying its old revision returns `409 stale_revision`. Turning the
backend flag off during a session rejects the AI move without state mutation.
Browser busy/error checks confirm recovery without duplicate AI requests.

Three real Gunicorn HTTP trials launch two Victor requests, MCTS and health
together. All complete; health median **20.9 ms**, maximum **25.4 ms**. These quick
opening moves produced no busy rejection, so the deterministic blocked-native
test supplies the reservation/retry evidence. No queue or isolation added.

| Local Linux measurement | Result |
| --- | --- |
| 20 fixed public-profile moves, median / p95 / max | **15.5 / 492.3 / 1,001.0 ms** |
| Whole-move deadline hits | **1/20**; no response over 1.5 s |
| Optimal selected moves on this small frozen sample | **20/20**, not a general accuracy claim |
| Nine game AI HTTP moves, median / p95 / max | **5.2 / 6.3 / 6.3 ms** |
| Cold library / book load after Python imports | **0.95 / 2.15 ms** |
| Smoke interpreter peak RSS, including native/search/failure tests | **121.4 MiB** |
| Observed API container memory, disabled / enabled | **77.3 / 97.2 MiB** |
| Observed enabled Gunicorn worker high-water RSS | **90.1 MiB** |

Docker Desktop runs an ARM Linux VM with 10 CPUs and 7.75 GiB available; this
amd64 image uses architecture translation. Measurements establish Linux binary
operation, not Render performance, a hard deadline, a memory ceiling or a load
capacity guarantee. Timings are modest samples; cooperative Python cover work
can still overrun a deadline. Full imports/server startup are not included in
the cold library/book numbers. Existing session capacity is finite and process
state disappears on restart. Preserve one worker and one service instance.

To reproduce the bounded container check after building the image above:

```sh
docker run --rm --platform linux/amd64 --network container:victor-release-on \
  -e PYTHONPATH=/app \
  -v "$PWD/scripts/validate_victor_release.py:/tmp/check.py:ro" \
  -v "$PWD/tests/fixtures/victor_release.json:/tmp/fixtures.json:ro" \
  victor-release-api:local python /tmp/check.py \
  --fixtures /tmp/fixtures.json --api http://127.0.0.1:10000
```

The named disposable local API must run this image with Victor enabled,
explanations disabled and no provider key. Omit `--api` for proof/fallback and
deterministic in-process concurrency checks alone. The script rejects nonlocal
HTTP origins. Do not point automated traffic at production.

## Render inspection and required preflight

The Render connector is installed and authenticated enough to list workspaces.
Service listing returns `no workspace selected` and explicitly requires the
user to confirm a workspace before retrying. It lists **Andrew's Workspace**,
`tea-csp659jtq21c73eao5og`; confirmation was requested but not received during
preparation. No service, deploy, log, metric or environment was inspected. This
is an access-context gap, not proof of any production setting.

Repository deployment documentation describes the API as an image-backed Web
Service at `board-game-ai-lab.onrender.com` using GHCR `:main`, and the frontend
as a Static Site at `board-game-ai-lab-ui.onrender.com`, branch `main`, root `ui`,
build `npm ci && npm run build:render`, output `dist`. These are documented
expectations, **not newly verified live findings**. The committed workflow builds
linux/amd64, publishes `:main`, `:latest` and `:sha-…`, then calls the API deploy
hook on a main push or manual dispatch. It runs no regression suite and does not
deploy the frontend. Do not dispatch it for read-only validation.

After workspace confirmation, inspect service details, the last deploys, errors
and CPU/memory limits via Render tools, or Dashboard → each service → Settings,
Environment, Events, Logs and Metrics. Record privately:

1. API/frontend service IDs, live deploy IDs, UI commit, API immutable image digest
   and any associated commit. Image-backed services may not expose a Git commit;
   match digest to GHCR provenance rather than inferring from the moving tag.
2. Image URL, Docker command override, one worker/four threads, one instance,
   region, CPU/memory tier, health path `/v1/connect4/health` and port binding.
3. Names/presence of relevant variables; confirm the nonsecret Victor flags are
   absent/false. Privately verify CORS, frontend API origin and existing
   explanation configuration. Never copy secret values into logs or this report.
4. Static build settings, SPA rewrite, auto-deploy behavior, image auto-deploy and
   hook destination, recent build/runtime failures, available rollback deploys
   and an accessible immutable old backend digest.

If tools cannot expose environment settings or command overrides, verify them in
Dashboard; do not use the environment-update tool to discover values.

## Exact rollout and rollback

**A — approved disabled deployment.** Complete Render preflight and record
rollback targets first. Obtain approval to push/merge this reviewed release into
main and perform the resulting publication/hook deployment and frontend deploy.
If frontend auto-deploy would break sequencing, approval must also cover pausing
it. Ensure `VICTOR_RESEARCH_ENABLED=false` (or absent) and build-time
`VITE_VICTOR_RESEARCH_ENABLED=false` (or absent); changing settings can restart or
deploy, so do so only within approval. Preserve current explanation settings,
CORS, one worker/four threads and one instance. Publish the reviewed image,
record its digest and ensure the API deploy actually uses it, then deploy the
frontend. Verify health, normal Random/Negamax/MCTS gameplay, history and labs.
Use an authorized service shell or isolated check to confirm native loading and
1,722 book entries; health alone does not establish these. Stop/rollback on
startup, gameplay or resource regressions.

**B — separately authorized controlled measurement.** Use an isolated same-tier
nonproduction Render runtime with the exact new image and the committed fixture;
enable Victor there only. Run the local-only script inside that environment
against loopback, with modest traffic. Record native load, correct intervals,
fallbacks, cold startup, whole-move/HTTP median, p95, max, deadline hits, process
and service memory, two-Victor overlap, MCTS coexistence and health responsiveness.
Production-shell execution or creation of that test environment also needs
explicit authorization; preparation does not create it. No public stress test.
Investigate repeated moves above roughly 1.5 s; if necessary lower the public
budget conservatively, rerun local checks and remeasure before exposure. Do not
silently accept Python fallback as native performance evidence.

**C — separately approved enablement only after B passes.** Set API
`VICTOR_RESEARCH_ENABLED=true`, allow its deployment/restart and verify API
independently. Set Static Site `VITE_VICTOR_RESEARCH_ENABLED=true` and rebuild.
Confirm experimental labeling, complete games in both colours, revisions/history,
busy retry, lost-response recovery and unchanged normal agent selection. Inspect
errors, resource use and fallbacks through authorized internal diagnostics; the
public API intentionally does not expose proof status. Do not add a diagnostic
public endpoint for this rollout.

**D — approved rollback.** Disable the API flag first, then rebuild the frontend
with its flag false; a frontend setting alone does not change an existing bundle.
Retained research sessions reject subsequent AI moves with `409 invalid_agent`
and unchanged state, and the UI explains the unavailable agent. Offer a new game
with a normal opponent; an already-running move may finish before the restart.
Deploy/restart loses process-local games. For a code regression, restore the
privately recorded immutable API image digest and previous frontend deploy/commit
with Victor off. Verify health, CORS and normal gameplay. Flag-only rollback is
preferred when code remains healthy; do not rely on moving `:main`/`:latest` tags
or automatically assume an older image contains this agent.

Remaining gates are current Render configuration/rollback verification, explicit
Stage A production approval, hosted Stage B performance evidence and separate
Stage C approval. No production changes are authorized by this report.
