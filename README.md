# Board Game AI Lab

Play Connect 4 against Random or depth-limited Negamax in a Flask + React/Vite application.

**Status:** Phase 1 is completed and manually verified. Phase 2 deployment support is implemented and locally verified; public rollout remains pending review. The production architecture is a Render Static Site calling a Render Web Service backed by the GHCR API image. See [deployment settings, manual steps and smoke tests](docs/deployment.md).

## Run locally with Docker

Install Docker Desktop (or Docker Engine with the Compose plugin), start Docker, then run from the repository root:

```sh
docker compose up --build
```

Open **http://localhost:3000** and select **Play Connect 4**. No `.env` file, API key, Python install, or Node install is required. The first build downloads public images and dependencies. Ports 3000 and 8000 must be free. The UI waits for the API health check before starting.

Choose Random or Negamax (depth 1–4, default 2), then **Start game**. You play red and move first. Use **Start new game** at any time to restart; choose another opponent first to switch. Two tabs have independent games. Failed requests show recovery controls instead of automatically repeating moves.

Stop the stack with `Ctrl+C`, then `docker compose down` to remove its containers/network.

## Development and tests

Use Python **3.11** and Node **22.12+** (Vite also supports Node 20.19+).

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
gunicorn api.app:app --bind 127.0.0.1:8000 --workers 1 --threads 4
```

In a second terminal:

```sh
cd ui
npm ci
npm test
npm run build
npm run dev
```

Open the Vite URL printed in the terminal. Its `/v1` proxy forwards to port 8000. Keep the API on **one worker** because session storage is process-local.

To run real-browser regression tests against the Docker stack at port 3000:

```sh
cd ui
npx playwright install chromium
npm run test:e2e
```

For Vite instead, set `PLAYWRIGHT_BASE_URL=http://localhost:5173` when running `npm run test:e2e`. The tests start games and simulate request failures on the **local** server; never point them at production.

## Scope and limitations

- The web API exposes Human, Random, and Negamax only; the UI is human-versus-AI. Negamax is capped at depth 4. Search strength is not formally benchmarked.
- Sessions have random game IDs, per-game locks and revision checks. The server retains at most 128 sessions, with 30-minute idle expiry. Expired sessions are reclaimed on subsequent requests. At capacity it rejects new sessions; restarting your existing game replaces it without consuming another slot.
- Games disappear on server restart. Reloading or leaving the gameplay page starts a new browser interaction; abandoned server sessions expire. Game IDs isolate games but are not authentication credentials for an account system.
- Separate-host frontend builds use `VITE_API_BASE`; Render uses `npm run build:render` and an explicit HTTPS API origin. Docker builds and Vite development retain their local `/v1` proxies. API CORS uses an exact `CORS_ALLOWED_ORIGINS` allowlist; no wildcard is enabled by default.
- Frontend requests allow 90 seconds for cold starts and never automatically repeat a POST. Recovery reads the authoritative board before another move. Deployments/free-service spin-down lose sessions; keep one API instance and one Gunicorn worker.
- API inference requires only `requirements-api.txt`. The optional `requirements.txt` retains dependencies for historical ML/training code. MCTS, neural MCTS, Victor and historical DQN need further work and are not advertised as playable web agents.
- LLM explanations and Mancala web play remain later phases. Public deployment has not been performed by this change. Pushing to `main` invokes the API build/push/deploy-hook workflow and requires rollout approval.

See [the quickstart](docs/quickstart.md) for the API contract and manual checks, and [the project plan](project_plan.md) for project status and resume acceptance criteria.
