# Board Game AI Lab

Play Connect 4 against Random, depth-limited Negamax or bounded MCTS in a Flask + React/Vite application.

**[Play the public application](https://board-game-ai-lab-ui.onrender.com/)** · **[Play Connect 4](https://board-game-ai-lab-ui.onrender.com/connect4)**

Backend API origin: [https://board-game-ai-lab.onrender.com](https://board-game-ai-lab.onrender.com) · [API health endpoint](https://board-game-ai-lab.onrender.com/v1/connect4/health).

**Status:** Phases 1–3 are complete; the user confirms explanations have been enabled and manually tested publicly, following the earlier local Phase 3C readiness review. Source defaults remain disabled and key-free. Phase 4A.1 standalone MCTS correctness is complete at `c213ddf`. Phase 4A.2 API/UI integration is implemented locally, pending review and deployment; public MCTS availability is not yet verified. See [MCTS integration and verification](docs/phase4a2-mcts-integration.md), [the explanation contract](docs/llm-explanations.md), and [deployment guidance](docs/deployment.md).

## Run locally with Docker

Install Docker Desktop (or Docker Engine with the Compose plugin), start Docker, then run from the repository root:

```sh
docker compose up --build
```

Open **http://localhost:3000** and select **Play Connect 4**. No `.env` file, API key, Python install, or Node install is required. The first build downloads public images and dependencies. Ports 3000 and 8000 must be free. The UI waits for the API health check before starting.

Choose Random, Negamax (depth 1–4, default 2), or MCTS (Quick — 50 simulations; Standard — 100 simulations, default; Deeper — 250 simulations), then **Start game**. You play red and move first. Use **Start new game** at any time to restart; choose another opponent first to switch. Two tabs have independent games. Failed requests show recovery controls instead of automatically repeating moves.

Stop the stack with `Ctrl+C`, then `docker compose down` to remove its containers/network.

## Development and tests

Use Python **3.11** and Node **22.12+** (Vite also supports Node 20.19+).

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
EXPLANATIONS_ENABLED=false OPENAI_API_KEY= python -m pytest -q
EXPLANATIONS_ENABLED=false OPENAI_API_KEY= gunicorn api.app:app --bind 127.0.0.1:8000 --workers 1 --threads 4
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

The explanation panel supports **Analyze Last AI Move / Analyze Last Move**, **Analyze Position**, and legal **What If?** simulations. Last-move analysis describes verified consequences; it does not reveal agent intent, MCTS search traces or UCB scores. To configure the backend locally, copy `.env.example` to ignored `.env`; `python-dotenv` loads it automatically. Explanations require `EXPLANATIONS_ENABLED=true` and a backend-only `OPENAI_API_KEY`. The backend defaults to `OPENAI_EXPLANATION_MODEL=gpt-6-luna` and `OPENAI_EXPLANATION_REASONING_EFFORT=none`; both stay server-side. Automated/browser verification uses mocks with live requests forbidden. Use provider-side budget/usage monitoring before any separately approved enablement or live test. See [all defaults](docs/llm-explanations.md) and [the exact production overrides](docs/deployment.md#backend-environment-variable-checklist-for-phase-3c).

The initial production profile reserves at most **10 provider attempts per process/hour, 3 per game and 5 per observed network client/hour**, with **1 concurrent request**, **800 output tokens** and a **40-second provider timeout**. Successful cache hits do not spend allowance; failures and stale results do. Counters reset on process restart/deployment, and proxy visitors can share a client counter. These request limits are not a durable dollar budget. The user confirms public explanations are already enabled and tested. The [deployment guide](docs/deployment.md) retains the historical initial rollout and rollback instructions; this MCTS change does not alter explanation settings.

To run real-browser regression tests against the Docker stack at port 3000:

```sh
cd ui
npx playwright install chromium
npm run test:e2e
```

For Vite instead, set `PLAYWRIGHT_BASE_URL=http://localhost:5173` when running `npm run test:e2e`. The tests start games and simulate request failures on the **local** server; never point them at production.

## Scope and limitations

- The local web API exposes Human, Random, Negamax and MCTS; the UI is human-versus-AI. Negamax is capped at depth 4. MCTS uses UCT selection, random rollouts, alternating-player reward backpropagation and final visit-count selection, with immediate-win and immediate-loss-avoidance root guards. Public simulation presets are 50/100/250 (default 100); strength is not formally benchmarked.
- At most one MCTS search runs per process. A competing request immediately receives HTTP 503 `agent_busy` without changing its board, revision or history; recovery refreshes the board before an explicit retry. Human, Random and Negamax do not use this guard. Searches are synchronous; simulation limits bound work, not a strict wall-clock deadline. Retain one worker/four threads and one instance on Render Free; this is not a distributed limit or fairness queue.
- Sessions have random game IDs, per-game locks and revision checks. The server retains at most 128 sessions, with 30-minute idle expiry. Expired sessions are reclaimed on subsequent requests. At capacity it rejects new sessions; restarting your existing game replaces it without consuming another slot.
- Games disappear on server restart. Reloading or leaving the gameplay page starts a new browser interaction; abandoned server sessions expire. Game IDs isolate games but are not authentication credentials for an account system.
- Separate-host frontend builds use `VITE_API_BASE`; Render uses `npm run build:render` and an explicit HTTPS API origin. Docker builds and Vite development retain their local `/v1` proxies. API CORS uses an exact `CORS_ALLOWED_ORIGINS` allowlist; no wildcard is enabled by default.
- Frontend requests allow 90 seconds for cold starts and never automatically repeat a POST. Recovery reads the authoritative board before another move. Deployments/free-service spin-down lose sessions; keep one API instance and one Gunicorn worker.
- API inference requires only `requirements-api.txt`. The optional `requirements.txt` retains dependencies for historical ML/training code. MCTS, Random and Negamax require neither PyTorch nor neural checkpoints. MCTS-NN, VictorAgent and historical DQN remain experimental and unavailable through the web API; neural correctness/training is deferred.
- The backend records immutable session move history and can build deterministic, revision-bound evidence internally. Formal Allis rule applications are not implemented; the thesis rules are curated reference knowledge.
- Phase 3B.1 explanations use constrained model selections over verified relationships, supporting facts and relevant thesis concepts, with per-game/client/global limits, bounded concurrency and caching. Formal rules remain reference-only; freeform strategic reasoning is not generated. The user confirms successful public explanation testing; current automated verification uses mocks and makes no provider calls. Mancala web play remains deferred. Pushing to `main` or manually dispatching the image workflow builds/publishes the API image and invokes its Render deploy hook.

See [the quickstart](docs/quickstart.md) for the API contract and manual checks, and [the project plan](project_plan.md) for project status and resume acceptance criteria.
