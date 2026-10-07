# Board Game AI Lab

Play Connect 4 against Random, depth-limited Negamax or bounded MCTS in a Flask + React/Vite application.

**[Play the public application](https://board-game-ai-lab-ui.onrender.com/)** · **[Play Connect 4](https://board-game-ai-lab-ui.onrender.com/connect4)**

Backend API origin: [https://board-game-ai-lab.onrender.com](https://board-game-ai-lab.onrender.com) · [API health endpoint](https://board-game-ai-lab.onrender.com/v1/connect4/health).

**Status:** Public Play and Phase 5A Match Lab are deployed and manually verified at main commit `ed8cd20533fa6ecbad550a960365550b08f29250`. Play supports Random, corrected Negamax, MCTS, both turn orders, and grounded AI Analysis. Match Lab supports independent competitors, manual stepping, safe autoplay/pause, and authoritative replay. Phase 5B adds **Tournament Lab** locally at `/connect4/tournament`: seeded 8/16/32/64-player AI brackets, sequential execution, browser-local persistence, and compact completed-game replay. Tournament Lab is pending review and deployment. See [Tournament Lab architecture and verification](docs/phase5b-tournament-lab.md), [Match Lab](docs/phase5a-match-lab.md), and [deployment guidance](docs/deployment.md).

## Run locally with Docker

Install Docker Desktop (or Docker Engine with the Compose plugin), start Docker, then run from the repository root:

```sh
docker compose up --build
```

Open **http://localhost:3000** and select **Play Connect 4**. No `.env` file, API key, Python install, or Node install is required. The first build downloads public images and dependencies. Ports 3000 and 8000 must be free. The UI waits for the API health check before starting.

Choose Random, Negamax (depth presets 1/2/4/6/8, default 2), or MCTS (Quick — 100 simulations, default; Balanced — 400; Deep — 800), then **Start game**. Choose **You go first** (default, red) or **AI goes first** (you play yellow). AI-first games automatically make one opening move. Use **Start new game** at any time to restart; choose another opponent first to switch. Two tabs have independent games. Failed requests show recovery controls instead of automatically repeating moves.

Open **http://localhost:3000/connect4/match-lab** to configure Red and Yellow independently. **Start match** leaves the empty board paused; use **Next move** for one AI ply or **Autoplay** to watch sequential moves. Human turns wait for a board click. **Previous**, **Next**, the move list, and **Return to live** inspect history without undoing the server game.

Stop the stack with `Ctrl+C`, then `docker compose down` to remove its containers/network.

## Development and tests

Use Python **3.11** and Node **22.12+** (Vite also supports Node 20.19+).

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
EXPLANATIONS_ENABLED=false OPENAI_API_KEY= python -m pytest tests -q
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

- The local web API exposes Human, Random, Negamax and MCTS; Play is human-versus-AI; Match Lab supports Human/Human, Human/AI, and AI/AI. Corrected Negamax is capped at depth 8 (API integers 1–8; UI presets 1/2/4/6/8). It uses exact dominant terminal values, zero draws, center-first ties and bound-typed transposition entries. MCTS uses UCT selection, random rollouts, alternating-player reward backpropagation and final visit-count selection, with immediate-win and immediate-loss-avoidance root guards. UI simulation presets are 100/400/800 (default 100); the API also retains benchmarked 50/250 for compatibility. Local latency measurements selected these caps; playing strength is not formally benchmarked. See [the performance and turn-order report](docs/public-agent-strength-and-turn-order.md).
- At most one MCTS search runs per process. A competing request immediately receives HTTP 503 `agent_busy` without changing its board, revision or history; recovery refreshes the board before an explicit retry. Human, Random and Negamax do not use this guard. Searches are synchronous; simulation limits bound work, not a strict wall-clock deadline. Retain one worker/four threads and one instance on Render Free; this is not a distributed limit or fairness queue.
- Sessions have random game IDs, per-game locks and revision checks. The server retains at most 128 sessions, with 30-minute idle expiry. Expired sessions are reclaimed on subsequent requests. At capacity it rejects new sessions; restarting your existing game replaces it without consuming another slot.
- Games disappear on server restart. Reloading or leaving the gameplay page starts a new browser interaction; abandoned server sessions expire. Game IDs isolate games but are not authentication credentials for an account system.
- Separate-host frontend builds use `VITE_API_BASE`; Render uses `npm run build:render` and an explicit HTTPS API origin. Docker builds and Vite development retain their local `/v1` proxies. API CORS uses an exact `CORS_ALLOWED_ORIGINS` allowlist; no wildcard is enabled by default.
- Frontend requests allow 90 seconds for cold starts and never automatically repeat a POST. Recovery reads the authoritative board before another move. Deployments/free-service spin-down lose sessions; keep one API instance and one Gunicorn worker.
- API inference requires only `requirements-api.txt`. The optional `requirements.txt` retains dependencies for historical ML/training code. MCTS, Random and Negamax require neither PyTorch nor neural checkpoints. MCTS-NN, VictorAgent and historical DQN remain experimental and unavailable through the web API; neural correctness/training is deferred.
- The backend records immutable session move history and can build deterministic, revision-bound evidence internally and exposes a read-only history endpoint for Match Lab replay. Formal Allis rule applications are not implemented; the thesis rules are curated reference knowledge.
- Phase 3B.1 explanations use constrained model selections over verified relationships, supporting facts and relevant thesis concepts, with per-game/client/global limits, bounded concurrency and caching. Formal rules remain reference-only; freeform strategic reasoning is not generated. The user confirms successful public explanation testing; current automated verification uses mocks and makes no provider calls. Mancala web play remains deferred. Pushing to `main` or manually dispatching the image workflow builds/publishes the API image and invokes its Render deploy hook.

See [the quickstart](docs/quickstart.md) for the API contract and manual checks, and [the project plan](project_plan.md) for project status and resume acceptance criteria.
