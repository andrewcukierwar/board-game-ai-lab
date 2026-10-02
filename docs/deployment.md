# Phase 2 — Public deployment

Implementation and local verification are complete; public deployment and smoke testing remain pending review. No production settings, hooks or pushes were changed. The intended topology is a **Render Static Site** for React/Vite and a **Render image-backed Web Service** for Flask/Gunicorn using GHCR. Local Compose remains a separate, same-origin setup.

Local results on October 2, 2026: **74 backend tests, 15 frontend tests, and all 8 Chromium browser tests against each of Compose, Vite development and separate-origin static/API hosting passed**. Compose startup, production/Render builds, direct `/connect4` refresh, health, Render-style port 10000/one-worker operation, and actual API restart/browser session recovery passed. An initial missing-Chromium runner failure was resolved by installing Chromium. Cold-start delay/HTML and lost-response scenarios were simulated locally; real Render cold starts and public routing still require the manual smoke checks below. Existing npm advisories were not changed.

## Configuration by environment

| Environment | Frontend API routing | CORS |
| --- | --- | --- |
| Docker Compose | Empty API base; Nginx proxies `/v1/` to `api:8000` | Not needed for same-origin requests |
| Vite development | Empty API base; Vite proxies `/v1` to `localhost:8000`, even if `VITE_API_BASE` is set | Not needed |
| Render Static Site | `VITE_API_BASE` is the HTTPS API origin, compiled into the bundle | API explicitly allows the actual frontend origin |

`ui/.env.production` now has an empty default. The former `https://board-game-ai-lab.onrender.com` value was unused by the UI; confirm that hostname in the API's Render dashboard before using it. The Docker build explicitly sets `VITE_API_BASE=` and excludes local env files, so a developer's deployment configuration cannot redirect Compose gameplay to production.

Vite embeds `VITE_*` values at **build time**. Changing the API origin requires rebuilding the Static Site. Never put secrets in these variables. `npm run build:render` selects mode `render` and requires an explicit HTTPS origin; it does not load `.env.production`. `npm run build` retains a same-origin default and accepts HTTP origins for local verification. See [Vite environment documentation](https://vite.dev/guide/env-and-mode).

## Exact Render settings

Replace the two hostname placeholders with the actual Render/custom-domain origins. An origin includes scheme and hostname (and port if needed), with **no trailing slash or path**.

### API: Web Service → Existing Image

| Setting | Value |
| --- | --- |
| Image URL | `ghcr.io/andrewcukierwar/board-game-ai-lab:main` (or retain `:latest`, which this workflow also publishes) |
| Image platform | `linux/amd64` (explicit in GitHub Actions) |
| Docker command override | Empty: use the image's CMD |
| `PORT` | `10000` (Render default; image also supports another port) |
| `CORS_ALLOWED_ORIGINS` | `https://<actual-frontend-host>` |
| Health check path | `/v1/connect4/health` |
| Instance count | **1**, no autoscaling |
| Gunicorn | **1 worker, 4 threads**, enforced by image CMD |
| Registry credential | None if GHCR package is public; otherwise GitHub username + PAT with `read:packages` and access to the package |

Leave `GUNICORN_CMD_ARGS` unset and remove any old command override that changes worker count. Compose defaults to port 8000; Render's `PORT` controls only the listener, not the frontend origin (which uses public HTTPS without `:10000`). Do not add a disk or database for this phase.

For multiple deliberately supported domains, set a comma-separated exact list, for example `https://your-site.onrender.com,https://games.example.com`. Add custom domains individually, and rebuild the frontend if its API hostname changes. Wildcards, paths and `null` origins are rejected at API startup. No cookies/credentials are used. CORS limits browser access; it is not API authentication or protection against arbitrary non-browser callers.

See Render's [image-backed deployment](https://render.com/docs/deploying-an-image), [port binding](https://render.com/docs/web-services#port-binding), and [health check](https://render.com/docs/health-checks) documentation.

### UI: Static Site

| Setting | Value |
| --- | --- |
| Repository | `andrewcukierwar/board-game-ai-lab` |
| Branch | `main`, after approved merge/push |
| Root directory | `ui` |
| Build command | `npm ci && npm run build:render` |
| Publish directory | `dist` |
| `NODE_VERSION` | `22` (compatible with the local Docker build; use a release ≥22.12) |
| `VITE_API_BASE` | `https://<actual-api-host>`; verify whether the existing API is `https://board-game-ai-lab.onrender.com` |
| Redirect/rewrite rule | Source `/*`, destination `/index.html`, action **Rewrite** |
| Auto-deploy during rollout | Disable until configuration/review is complete; re-enable deliberately afterward |

The rewrite serves React on direct navigation and refresh at `/connect4`; existing static assets are served normally. Nginx already provides the equivalent fallback locally. Render uses dashboard rewrite rules; no Nginx server is deployed for this Static Site. See [Render rewrites](https://render.com/docs/redirects-rewrites) and [Node version configuration](https://render.com/docs/node-version).

## Approval-gated rollout steps

1. Review the local changes on `phase2-public-deployment`. Confirm the actual API and desired frontend hostnames in Render. Public gameplay has not yet been verified.
2. After approving production changes, configure the existing image-backed API with the settings above, including the exact future frontend origin. Obtain the Static Site origin by configuring that site; if its initial build deploys before CORS is ready, keep the URL unannounced until both services are verified. Configuration saves can trigger deploys.
3. Review GHCR package visibility/access and image selection. GitHub's `GITHUB_TOKEN` builds/pushes with explicit `contents: read` and `packages: write`; `RENDER_DEPLOY_HOOK_URL` is a repository Actions secret for this API service. The workflow publishes `:main`, `:latest` and `:sha-<short-sha>`. Do not expose the hook URL or PAT.
4. Only after approval, merge/push to `main`. **That push builds/pushes the API image and invokes the Render deploy hook.** It can also deploy the Static Site if auto-deploy is enabled. Do not separately run the workflow/hook unless a redeploy is intended.
5. Verify the new workflow run, GHCR image digest and Render deploy events/logs. With `:main` or `:latest`, the hook pulls that tag's current image. If the service is pinned to an old SHA/digest, update its reference deliberately; an unparameterized hook will otherwise redeploy the old image. For a manual retry, use **Manual Deploy → Deploy latest reference**. Hook success means deployment was requested, not that the service is healthy.
6. Confirm `GET https://<actual-api-host>/v1/connect4/health` returns HTTP 200 and `OK`, and Gunicorn logs show one worker. Record the image digest and deploy result.
7. Deploy the Static Site with the specified build env and rewrite rule. Confirm the build used `build:render`. Run the manual checklist below, then record actual demo URLs and public verification results in README/project plan. Re-enable frontend auto-deploy only when desired.

[Run 37034008283](https://github.com/andrewcukierwar/board-game-ai-lab/actions/runs/37034008283) for `830bfdc` succeeded at image build/push and hook invocation. Read-only inspection confirmed those step outcomes. The workflow still does not run regression suites or deploy the UI; local checks precede this approved rollout. No hook was invoked during Phase 2 implementation.

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

## Production smoke-test checklist (manual)

- [ ] API health responds 200/`OK`; image digest matches the intended build; one Gunicorn worker and one instance are running.
- [ ] Homepage loads over HTTPS; **Play Connect 4**, direct `/connect4`, and refresh at `/connect4` all work. JS/CSS assets load correctly.
- [ ] DevTools shows API calls to the actual HTTPS API origin, not the Static Site's `/v1` route. JSON POST preflights and game/error responses allow only the exact frontend origin; no wildcard, credentials or mixed-content errors.
- [ ] Start and finish games against Random and Negamax. Verify terminal result and disabled board; restart mid-game and switch opponent/depth.
- [ ] Open two independent browser contexts; moves/restarts in one do not alter the other. Double-click a column; only one human move and one AI reply occur.
- [ ] Try a first request after the API has been idle. The wake-up message is visible and controls stay locked; if it fails, explicit retry recovers without reloading.
- [ ] Use DevTools offline mode during a move. Reconnect and use **Refresh game**/**Retry AI move** as offered; verify the board against its revision and no duplicate human/AI move. Also test a failed start.
- [ ] After an intentional API restart/redeploy (only when approved), an old game offers a fresh start and new games work. Check natural 30-minute expiry when feasible.
- [ ] Check browser console and API logs for unexpected errors. Record frontend/API URLs, tested image digest, date and any failures before marking Phase 2 complete.
