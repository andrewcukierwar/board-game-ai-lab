# Production deployment — Phase 3C preparation

**Status update (October 3, 2026):** The user subsequently confirmed explanations have been enabled and manually tested publicly. Earlier statements below about pending rollout describe the original Phase 3 preparation, not current public status. Source defaults remain disabled. Phase 4A.2 MCTS integration is local, pending review/deployment; it does not change explanation settings or make paid calls. See [the MCTS handoff](phase4a2-mcts-integration.md).

Phase 2 public deployment is complete. On October 2, 2026, the user confirmed manually verified public Connect 4 gameplay, including successful frontend/backend CORS configuration. The deployed topology is a **Render Static Site** for React/Vite and a **Render image-backed Web Service** for Flask/Gunicorn using GHCR. Local Compose remains a separate, same-origin setup.

**Verified production URLs:** [public frontend](https://board-game-ai-lab-ui.onrender.com/), [Connect 4](https://board-game-ai-lab-ui.onrender.com/connect4), [backend API origin](https://board-game-ai-lab.onrender.com), and [API health endpoint](https://board-game-ai-lab.onrender.com/v1/connect4/health).

Phase 3A is committed at `37dc55e`; Phase 3B and the Phase 3B.1 refinement are present on `phase3b-llm-explanations` at `45894d1`. The user reports successful local testing of the revised explanations with real GPT-6 Luna requests. Phase 3C reviews production readiness with mocked responses only and prepares this procedure; public explanation deployment and enablement remain pending. See [the final readiness review](phase3c-readiness.md). No production configuration, credentials or deployment are changed during preparation.

Local Phase 2 results on October 2, 2026: **74 backend tests, 15 frontend tests, and all 8 Chromium browser tests against each of Compose, Vite development and separate-origin static/API hosting passed**. Compose startup, production/Render builds, direct `/connect4` refresh, health, Render-style port 10000/one-worker operation, and actual API restart/browser session recovery passed. An initial missing-Chromium runner failure was resolved by installing Chromium. Cold-start delay/HTML and lost-response scenarios were simulated locally. The reusable manual checklist below covers detailed production checks; the user’s gameplay/CORS confirmation does not assert that every individual checklist item was exercised. Existing npm advisories were not changed.

## Configuration by environment

| Environment | Frontend API routing | CORS |
| --- | --- | --- |
| Docker Compose | Empty API base; Nginx proxies `/v1/` to `api:8000` | Not needed for same-origin requests |
| Vite development | Empty API base; Vite proxies `/v1` to `localhost:8000`, even if `VITE_API_BASE` is set | Not needed |
| Render Static Site | `VITE_API_BASE` is the HTTPS API origin, compiled into the bundle | API explicitly allows the actual frontend origin |

`ui/.env.production` now has an empty default. The former hardcoded API value was unused by the UI; the verified production API origin is now explicitly configured through the Render build’s `VITE_API_BASE`. The Docker build explicitly sets `VITE_API_BASE=` and excludes local env files, so a developer's deployment configuration cannot redirect Compose gameplay to production.

Vite embeds `VITE_*` values at **build time**. Changing the API origin requires rebuilding the Static Site. Never put secrets in these variables. `npm run build:render` selects mode `render` and requires an explicit HTTPS origin; it does not load `.env.production`. `npm run build` retains a same-origin default and accepts HTTP origins for local verification. See [Vite environment documentation](https://vite.dev/guide/env-and-mode).

## Exact Render settings

The values below use the verified production origins. For another deployment, substitute its actual Render/custom-domain origins. An origin includes scheme and hostname (and port if needed), with **no trailing slash or path**.

### API: Web Service → Existing Image

| Setting | Value |
| --- | --- |
| Image URL | `ghcr.io/andrewcukierwar/board-game-ai-lab:main` (or retain `:latest`, which this workflow also publishes) |
| Image platform | `linux/amd64` (explicit in GitHub Actions) |
| Docker command override | Empty: use the image's CMD |
| `PORT` | `10000` (Render default; image also supports another port) |
| `CORS_ALLOWED_ORIGINS` | `https://board-game-ai-lab-ui.onrender.com` |
| Health check path | `/v1/connect4/health` |
| Instance count | **1**, no autoscaling |
| Gunicorn | **1 worker, 4 threads**, enforced by image CMD |
| Registry credential | None if GHCR package is public; otherwise GitHub username + PAT with `read:packages` and access to the package |

Leave `GUNICORN_CMD_ARGS` unset and remove any old command override that changes worker count. Compose defaults to port 8000; Render's `PORT` controls only the listener, not the frontend origin (which uses public HTTPS without `:10000`). Do not add a disk or database for this phase.

For multiple deliberately supported domains, set a comma-separated exact list, for example `https://your-site.onrender.com,https://games.example.com`. Add custom domains individually, and rebuild the frontend if its API hostname changes. Wildcards, paths and `null` origins are rejected at API startup. No cookies/credentials are used. CORS limits browser access; it is not API authentication or protection against arbitrary non-browser callers.

See Render's [image-backed deployment](https://render.com/docs/deploying-an-image), [port binding](https://render.com/docs/web-services#port-binding), and [health check](https://render.com/docs/health-checks) documentation.

### Backend environment-variable checklist for Phase 3C

Set these on the **backend Web Service only**, during a separately approved rollout. Names and parsing are verified against `api/app.py`, `api/gunicorn_config.py` and `api/connect4/explanations.py`. These production overrides do not change source defaults or `.env.example`.

| Variable | Initial production value | Purpose |
| --- | --- | --- |
| `PORT` | `10000` | Existing Render listener configuration. |
| `CORS_ALLOWED_ORIGINS` | `https://board-game-ai-lab-ui.onrender.com` | Preserve the existing exact origin, without a slash or wildcard. |
| `EXPLANATIONS_ENABLED` | `false` | Keep disabled through image and gameplay verification. Only a separately approved change sets `true`. |
| `OPENAI_API_KEY` | Leave unset/empty for the disabled rollout | Required and nonempty only for enablement; enter privately in backend Render configuration. Do not copy a value into this document or a command. |
| `OPENAI_EXPLANATION_MODEL` | `gpt-6-luna` | Backend provider model. |
| `OPENAI_EXPLANATION_REASONING_EFFORT` | `none` | Backend reasoning effort. |
| `EXPLANATION_MAX_OUTPUT_TOKENS` | `800` | Generated-token cap per attempt. |
| `EXPLANATION_TIMEOUT_SECONDS` | `40` | Provider socket timeout/deadline; OS DNS caveat below. |
| `EXPLANATION_GLOBAL_LIMIT` | `10` | Reserved provider attempts per process/window, across all games/clients. |
| `EXPLANATION_GAME_LIMIT` | `3` | Reserved provider attempts over one game session's lifetime. |
| `EXPLANATION_CLIENT_LIMIT` | `5` | Reserved provider attempts per observed network client/window, across games. |
| `EXPLANATION_WINDOW_SECONDS` | `3600` | Process/client counter window. |
| `EXPLANATION_MAX_CONCURRENT` | `1` | One provider attempt in flight, leaving gameplay threads available. |

The following source defaults can remain unset. If existing backend overrides are present, reconcile them to these values for this rollout:

| Variable | Retained default |
| --- | --- |
| `EXPLANATION_MAX_QUESTION_LENGTH` | `500` |
| `EXPLANATION_CACHE_SIZE` | `256` |
| `EXPLANATION_CACHE_TTL_SECONDS` | `1800` |
| `EXPLANATION_CLIENT_CAPACITY` | `1024` |

`VICTOR_RESEARCH_ENABLED` (October 2026) gates the opt-in `victor_research` API agent and defaults to `false`. Leave it **unset** in production: it has not been enabled, deployed or measured on Render. The Static Site has a matching build-time flag, `VITE_VICTOR_RESEARCH_ENABLED` (default off), which only shows the "Victor Research (Experimental)" opponent on the Connect 4 page. Both must be `true` for the option to work; with only the UI flag set, the page reports that the server has not enabled it. See the [readiness checklist](victor-opening-and-app-readiness.md#controlled-render-trial-readiness-checklist) before any separately approved trial.

For the Victor release, use the [release readiness report](victor-release-readiness.md) and its staged rollout. Stage A requires both Victor flags off and preserves the existing explanation configuration; the older Phase 3C explanation rollout below is not part of this release. Linux container validation is complete locally; current Render configuration and hosted Victor performance still require verification before live changes.

Keep `GUNICORN_CMD_ARGS` unset, the Docker command override empty, one worker/four threads and one instance. `GAME_SESSION_CAPACITY` and `GAME_SESSION_TTL` are application config entries, **not environment overrides read by this code**; retain 128 sessions/1800 seconds.

Do not put `OPENAI_API_KEY` or any other explanation setting in the Static Site configuration, a `VITE_*` variable, a tracked file, a build argument, an image layer or documentation examples. The frontend only needs the API origin and Node version listed below. An already provisioned backend key may remain private; the disabled flag prevents its use.

### Spending safeguards and their limits

All three quotas apply together: at most three provider attempts per game, five per observed network client/window across games, and ten per API process/window across all visitors. The mocked production-profile regression exhausts all three and checks that failures consume allowance. Reservations and the one-call concurrency gate share a lock; duplicate in-flight requests for the same game fail with 409, other uncached requests fail with 429 while capacity is occupied. Those rejections do not reserve attempts. Provider I/O holds no game lock, so gameplay can continue. Stale results, malformed responses, provider failures and timeouts retain their reservation; there is no automatic retry or alternate-model fallback.

Successful responses are cached by game, revision, mode, hypothetical column, trimmed question, model and effort. The bounded LRU cache retains successes for 1800 seconds. A valid current-revision hit returns before quota/concurrency checks and uses no provider allowance, even if that game has reached its cap. Changing game/revision/question creates a distinct request; failures are not cached.

The global fixed window starts with the process; each client window starts with its first reservation. They renew on request activity after 3600 seconds, rather than on wall-clock hour boundaries. Adjacent windows can allow bursts. The game cap does not renew hourly; restarting a game creates a new session, but does not bypass existing client/global counts. Client identity is `request.remote_addr`; forwarded-IP headers are not trusted. Render proxies may aggregate visitors, making the client cap more restrictive than expected. Verify actual proxy behavior before considering any future identity changes; keep the global cap regardless.

**Counters, cache and games are process-local and reset after deploys, restarts or free-service spin-down.** Extra workers/instances, or overlapping old/new processes during a deploy, have independent allowances. These limits are not a durable hourly account limit or dollar budget. Provider-side project usage/budget monitoring and spending alerts remain required before enablement; confirm their actual enforcement rather than treating an alert as a hard stop. This application does not retain provider usage/timing metadata. The output cap does not bound input-token cost.

The 40-second provider setting controls socket timeout and a monotonic watchdog that interrupts slow response reads and releases the permit. Connection setup consumes that deadline, but the watchdog starts only after connection setup: OS DNS resolution or sequential address/TCP/TLS connection stages can exceed the total interval before the elapsed-deadline check runs. It is not an absolute 40-second wall-clock guarantee for all network failures. Browser cancellation does not cancel an already dispatched provider request or undo its cost. The frontend's existing 90-second timeout accommodates normal bounded requests and cold starts, but can abort first in an exceptional setup delay. The image runs Gunicorn's threaded worker; its worker-liveness timeout is not a per-request 30-second cutoff. No Gunicorn timeout override is needed for this provider setting.

### UI: Static Site

| Setting | Value |
| --- | --- |
| Repository | `andrewcukierwar/board-game-ai-lab` |
| Branch | `main` |
| Root directory | `ui` |
| Build command | `npm ci && npm run build:render` |
| Publish directory | `dist` |
| `NODE_VERSION` | `22` (compatible with the local Docker build; use a release ≥22.12) |
| `VITE_API_BASE` | `https://board-game-ai-lab.onrender.com` |
| Redirect/rewrite rule | Source `/*`, destination `/index.html`, action **Rewrite** |
| Auto-deploy during rollout | Disable until configuration/review is complete; re-enable deliberately afterward |

The rewrite serves React on direct navigation and refresh at `/connect4`; existing static assets are served normally. Nginx already provides the equivalent fallback locally. Render uses dashboard rewrite rules; no Nginx server is deployed for this Static Site. See [Render rewrites](https://render.com/docs/redirects-rewrites) and [Node version configuration](https://render.com/docs/node-version).

## Sequential Phase 3C rollout (future approval required)

Preparation stops before step 1. No push, merge, workflow dispatch, deploy hook, Render setting save, credential change or paid call is authorized by the readiness review. The initial Phase 2 rollout is complete; the following procedure applies to the new explanation release.

1. Obtain approval for the **disabled deployment**, identifying the reviewed local commit and intended merge commit. Privately record the current backend deploy ID and immutable image digest, frontend deploy ID/commit, nonsecret settings and rollback target. Ensure the old digest remains accessible in GHCR. Inspect actual current service settings; this review does not assert live configuration inspection.
2. Before any merge/push, apply the backend checklist with `EXPLANATIONS_ENABLED=false`, preserving exact CORS and one instance/worker. Leave a missing API key unset. Saving environment settings can deploy/restart the service; verify the currently running API returns 200/`OK` afterward. Disable frontend auto-deploy for sequential rollout. Do this only under the deployment approval.
3. Confirm the workflow's image target/access and hook destination privately. `.github/workflows/docker-lite.yml` builds `linux/amd64`, publishes `:main`, `:latest` and `:sha-<short-sha>`, and then calls the API Render deploy hook. It runs on pushes to `main` **and manual dispatch**. It does not run regression tests or deploy the UI. Do not dispatch it merely to check CI, and never expose the hook or registry credentials.
4. After the disabled deployment approval, merge/push the reviewed release to `main`. **This invokes build, image publication and the API deploy hook.** Keep the flag false before this trigger. Watch build/push/hook outcomes and record the merge SHA and new GHCR digest. Hook success only requests deployment. If the service is pinned to an old digest/SHA, the current unparameterized hook pulls that old reference; deliberately select the new digest and deploy it once available. Avoid redundant overlapping deploys.
5. Verify Render reports the new API image live, its digest matches the approved build, the health endpoint returns HTTP 200/`OK`, and logs show one worker using port 10000/four threads with no import/startup errors. Health alone does not establish image identity or explanation availability. Inspect the backend setting privately and confirm the flag is still false.
6. Deploy the frontend's reviewed commit using the documented Static Site settings and `npm ci && npm run build:render`. Preserve `VITE_API_BASE=https://board-game-ai-lab.onrender.com` and the `/*` → `/index.html` Rewrite. Verify homepage, direct `/connect4` and refresh, intended API host, exact-origin preflight/error responses, independent sessions, restart, and full games against Random/Negamax.
7. With a valid current game/revision, click **Analyze Position** once while disabled. Expect HTTP 503 with `code=explanations_disabled`; gameplay must continue. Do not infer this result from an invalid-game request or from health. Check the browser bundle/network for credential exposure. Record the disabled API/frontend deploy IDs, image digest, commit and smoke results.
8. Stop for **separate explicit approval to enable explanations**. Require healthy disabled-image/gameplay results and provider-side budget/usage monitoring. If a key is needed, privately provision it only on the backend under that approval. Review every quota/model/token/timeout value before changing `EXPLANATIONS_ENABLED=true`; do not widen limits. Saving/redeploying resets process counters and sessions.
9. After enablement, verify the same image is healthy. Only with an explicit paid smoke-call allowance, check the three modes in a fresh game: position, last AI move and one legal hypothesis, then repeat one unchanged request to verify `cached=true`. Cap that smoke test at three provider attempts, including failures; stop on an error. Reuse the unchanged revision for the cache check, and verify the hypothetical mode did not mutate the game. Public traffic shares these quotas, so do not assume the allowance is reserved for the operator.
10. Record enablement approval, nonsecret configuration, smoke outcomes, cache evidence, provider-side usage and remaining allowance. Observe failures/429s and provider spending without probing limits using paid requests. Re-enable frontend auto-deploy deliberately only after successful verification; keep credentials out of logs/screenshots/docs. Use rollback below if any gate fails.

Render supports digest-based image references and **Manual Deploy → Deploy latest reference**. See [image deployment behavior](https://render.com/docs/deploying-an-image) and [deploy hooks](https://render.com/docs/deploy-hooks).

## Rollback procedure

1. For an explanation-only failure or unexpected spending, set backend `EXPLANATIONS_ENABLED=false` under the rollback authorization and save/deploy that setting. Keep the current image initially and preserve CORS/quotas. Wait until the disabled process is live, confirm health and a valid explanation request returns `explanations_disabled`, then verify gameplay. An in-flight paid call may finish; disabling cannot undo it.
2. For an API/gameplay regression, select the previously recorded **immutable GHCR digest** in the image-backed service and deploy it with the flag false and the existing exact CORS. Do not rely on a previous `:main`/`:latest` tag: pulling it again can retrieve the broken new image. Confirm old-image availability/access, deploy identity, health and gameplay. Keep the bad image from being redeployed by a later workflow hook; inspect the service's persistent image reference as well as its current deploy.
3. If the frontend is implicated, roll the Static Site back to the recorded successful deploy, or redeploy the previous known-good commit with its existing API origin and Rewrite. Verify direct navigation, refresh and both opponents. Keep automatic frontend deploys controlled until the cause is resolved.
4. Recheck the effective environment after any Dashboard rollback: it can reuse the target deploy's environment for that rollback while leaving persistent service settings unchanged. Keep explanations false in both effective and persistent configuration, and preserve exact CORS/instance/worker settings. Record restored deploy IDs/digest and results. Deploys/rollbacks lose active games and reset quotas; ask visitors to start fresh. Explanation re-enablement requires new approval after the fix passes review.

Render's [rollback documentation](https://render.com/docs/rollbacks) explains environment/configuration reuse and mutable-tag behavior. Retain old registry digests; Render must be able to pull an image again for image-backed rollback.

[Run 37034008283](https://github.com/andrewcukierwar/board-game-ai-lab/actions/runs/37034008283) for `830bfdc` succeeded at image build/push and hook invocation. Read-only inspection confirmed those step outcomes. The workflow still does not run regression suites or deploy the UI; local checks should precede future deployments. No hook was invoked during Phase 2 implementation.

## Public search presets and turn order

The current source offers human-first (default) or AI-first play. Settings apply
only at start/restart; Player 1 is always red. An AI opener is a separate move
POST after accepting revision 0, with the same GET reconciliation and explicit
retry behavior as later AI turns.

Corrected Negamax accepts integer depths 1–8; UI presets are 1/2/4/6/8, default 2.
MCTS accepts only 50/100/250/400/800 simulations; the UI offers 100/400/800, default
100. The old 50/250 API budgets remain compatible. MCTS still uses full terminal
random rollouts, with the same process-wide nonblocking reservation and no queue.
Both searches are synchronous. [Local latency measurements](public-agent-strength-and-turn-order.md)
justify conservative caps but do not establish Render latency. After an authorized
deploy, measure warm and cold move latency at depth 8 and 800 simulations, both
player orders and concurrent independent games. Reduce caps if service latency
is unacceptable; do not widen timeouts to mask CPU contention.

## Network behavior and session limits

The frontend waits up to **90 seconds per request**, shows a wake-up message, and locks controls while waiting. Render free services can take about a minute to wake after idle; proxies/platform errors can still arrive sooner. See [Render free-service behavior](https://render.com/docs/free#spinning-down-on-idle).

No POST is retried automatically. After an uncertain move result, the UI reads the authoritative game before another move. If that read fails, the board remains locked behind **Refresh game**. An unfinished AI turn offers explicit **Retry AI move**; if the AI already committed, the refreshed revision prevents replay. Revision checks also guard late requests. If the session is missing after a restart/spin-down/expiry, the UI offers a new game.

Sessions remain process-local: **one instance and one worker** are mandatory. Deploys, restarts and free-service spin-down lose all games. Capacity is 128 sessions and idle expiry is 30 minutes. A lost initial start/restart response cannot recover the newly created game ID; explicitly starting again can leave an abandoned session until expiry. There is no durable resume/account system in this phase.

## Local verification of separate hosts

After `docker compose up --build -d`, start a temporary API with Render-style port binding:

```sh
docker run --rm --name board-game-phase2-api \
  -e PORT=10000 -e CORS_ALLOWED_ORIGINS=http://localhost:4173 \
  -p 127.0.0.1:8001:10000 board-game-ai-lab-api
```

In another terminal:

```sh
cd ui
VITE_API_BASE=http://localhost:8001 npm run build
npm run preview -- --host localhost --port 4173 --strictPort
```

Open http://localhost:4173/connect4 and refresh it. Browser network requests must go to `http://localhost:8001/v1/connect4/...`, include the frontend Origin, and receive an exact CORS response. This verifies separate origins locally without contacting Render. Local HTTP verification uses the general build command because `build:render` deliberately requires HTTPS.

In another terminal, from `ui`:

```sh
PLAYWRIGHT_BASE_URL=http://localhost:4173 \
PLAYWRIGHT_API_URL=http://localhost:8001 npm run test:e2e
```

Install Chromium first with `npx playwright install chromium` if needed. Stop temporary servers with Ctrl+C; Compose can remain running. Regression tests intentionally reject non-local URLs. Do not run them against production.

## Production smoke-test checklist (manual, reusable)

Phase 2 public gameplay and CORS have been manually verified by the user. The boxes below are a template for future smoke-test runs, not outstanding Phase 2 completion requirements.

- [ ] API health responds 200/`OK`; image digest matches the intended build; one Gunicorn worker and one instance are running.
- [ ] Homepage loads over HTTPS; **Play Connect 4**, direct `/connect4`, and refresh at `/connect4` all work. JS/CSS assets load correctly.
- [ ] DevTools shows API calls to the actual HTTPS API origin, not the Static Site's `/v1` route. JSON POST preflights and game/error responses allow only the exact frontend origin; no wildcard, credentials or mixed-content errors.
- [ ] Start and finish games against Random, Negamax and MCTS in both player orders. Verify exactly one AI opener, human-yellow winner copy, terminal result and disabled board; restart mid-game and switch opponent, budget and side. Measure depth-8/800-simulation latency on the actual instance.
- [ ] Open two independent browser contexts; moves/restarts in one do not alter the other. Double-click a column; only one human move and one AI reply occur.
- [ ] Try a first request after the API has been idle. The wake-up message is visible and controls stay locked; if it fails, explicit retry recovers without reloading.
- [ ] Use DevTools offline mode during a move. Reconnect and use **Refresh game**/**Retry AI move** as offered; verify the board against its revision and no duplicate human/AI move. Also test a failed start, a failed AI opener and an opening response lost after server commit.
- [ ] After an intentional API restart/redeploy (only when approved), an old game offers a fresh start and new games work. Check natural 30-minute expiry when feasible.
- [ ] Check browser console and API logs for unexpected errors. Record frontend/API URLs, tested image digest, date and any failures for each smoke-test run.
