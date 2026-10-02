# Board Game AI Lab — Project Plan

**Repository:** https://github.com/andrewcukierwar/board-game-ai-lab  
**Plan updated:** October 2, 2026  
**Status:** Resuming development; Phase 1 Dockerization completed locally, Phase 2 public deployment partially completed.  
**Guiding objective:** Build a polished, publicly playable AI game laboratory and make the three intended resume bullets accurate and defensible. Prefer shipping a compelling hands-on application over expanding infrastructure or running formal agent benchmarks.

## 1. Product vision

Board Game AI Lab lets someone select a board game, play against different AI approaches, and ask an LLM for grounded explanations of positions, moves, threats, and plausible alternatives. The point is to bring together classical search, deep/reinforcement learning, and generative AI in an interactive experience—not just to collect algorithms in scripts.

**Primary MVP:** A publicly accessible, reliable **Connect 4** experience with agent selection and an LLM-powered analysis feature. Extend the same experience to **Mancala** after the Connect 4 vertical slice works end-to-end.

### MVP user journey

1. Open the public site and select Connect 4.
2. Configure a human-versus-agent game and choose from the agents that actually work (including adjustable difficulty where appropriate).
3. Make legal moves, receive AI responses, see the board/status update correctly, and restart or switch opponents.
4. Ask a question about the current position or the last move (for example, *Why did the agent choose this column?*, *What threat should I respond to?*, or *What changes if I play another legal column?*).
5. Receive an explanation grounded in the actual game state and available engine output, with appropriate uncertainty when the selected agent cannot expose a definitive reason for its choice.
6. Later, repeat the workflow with Mancala.

## 2. Resume-driven acceptance criteria

The current resume language is a **target specification**, not proof that every feature is already complete:

> **Board Game AI Lab | Personal ML/AI Project**  
> Jan 2024 – Present
>
> - Implemented minimax, Deep Q-learning, and MCTS + Neural Net agents for Connect 4/Mancala in PyTorch.
> - Deployed one-command Docker Compose stack (Flask API + React/Vite UI) for live play against AI agents.
> - Integrated an LLM-based explanation layer to analyze positions, adding interpretability to gameplay AI.

| Resume objective | Required evidence before claiming completion |
| --- | --- |
| Multiple AI approaches | Minimax/Negamax, Deep Q-Learning (DQN), and neural-guided MCTS exist in working, selectable implementations for the games to which they are attributed. Agents execute legal moves and can be played against. State precisely which approach is implemented for which game; do **not** imply every approach is in both games if it is not. PyTorch supports the applicable learned models. |
| Containerized, playable app | The documented `docker compose up --build` workflow starts the relevant local services and permits a complete game. Separately, a public frontend URL communicates successfully with the deployed API. Local Compose and public Render deployment are distinct claims. |
| LLM explanation layer | A real, callable backend feature passes structured board state and available move/engine evidence to an LLM and displays useful position/move analysis in the frontend. The system does not invent inaccessible agent reasoning. |

**Definition of done for the resume:** A new visitor can use the deployed application; a developer can reproduce it locally from the README; the advertised algorithms and explanation feature can be demonstrated in code and in the UI. Final wording should reflect what was actually shipped.

## 3. Confirmed repository baseline (as inspected October 2, 2026)

- Last commit on `main`: [`dd92677`](https://github.com/andrewcukierwar/board-game-ai-lab/commit/dd926773cc79cdf51a6139e06e798ce09ed568fd), **August 7, 2025** — enabled a Render deploy-hook step in the Docker-lite workflow. A current successful CI run or healthy live deployment was **not** independently verified.
- Docker Compose defines a Flask/Gunicorn API on port `8000` and a React/Vite UI served by Nginx on local port `3000`; the original local Dockerization phase had been completed.
- Connect 4 Flask endpoints exist under `/v1/connect4/`, including `start_game`, `make_move`, and `health` (`/v1/connect4/health`). The React page is `ui/src/pages/Connect4.jsx` and currently bridges to legacy frontend JavaScript.
- Connect 4 source includes Random, Negamax, MCTS, MCTS + NN, and Victor agent classes, plus model checkpoint files under `models/connect4/`.
- Mancala source includes Human, Random, Simple, Minimax, and Negamax agents, but does not yet have an equivalent integrated web-play experience.
- The executable source tree did **not** reveal a DQN implementation; audit historical code, branches, or archives before deciding whether to recover or implement one. Do not count DQN as delivered on present evidence.
- No LLM-based explanation layer was identified in the current application.
- `README.md` is effectively empty and `docs/quickstart.md` is unfinished.
- `.github/workflows/docker-lite.yml` builds/pushes the **API** image to GHCR and triggers a Render hook; it does not deploy the React UI or establish a separate automated test workflow.

### Known issues to investigate rather than assume resolved

| Area | Observation / likely consequence |
| --- | --- |
| Production frontend routing | The UI uses relative `/v1/connect4/...` API URLs. These work through the local Nginx proxy but need an explicit API routing/base-URL strategy when the frontend is hosted separately. |
| Learned-model packaging | `docker/api.Dockerfile` copies `api/` and `games/`, not `models/`; `.dockerignore` excludes `*.pt`. An MCTS-NN agent loading `models/connect4/connect4_model_iter_100.pt` may fail inside the production container. |
| Per-user game state | Flask currently stores game/agent instances in process-global variables. Multiple visitors can overwrite each other's sessions; introduce session-isolated or explicitly stateless game handling before inviting public traffic. |
| CI/CD | Inspect GHCR permissions, the Render deploy-hook secret, image tags, Render configuration, and a full run. Enabling a hook in source is not itself evidence of successful deployment. |
| Repo hygiene | `.env` (currently comments only) and dependency directories such as `node_modules/` were committed; update ignores and untrack generated content. Never commit real keys. |
| Documentation | Complete setup and architectural instructions; remove stale or truncated documentation. |
| Code quality | Check imports, legacy/React coupling, error handling, package compatibility, and missing focused tests. Avoid broad refactors unless they unblock core functionality. |

## 4. Intended architecture

**Local development:** Docker Compose orchestrates an API container (Python / Flask / Gunicorn / game engines / trained artifacts) and UI container (React/Vite build served by Nginx). Nginx proxies `/v1/` to the API service.

**Production (previously chosen approach):**

```text
Visitor's browser
    |
    v
Render Static Site — React + Vite
    |
    | HTTPS requests to configured API origin
    v
Render Web Service — Flask + Gunicorn, Docker image from GHCR
    |
    +-- Connect 4 / Mancala game logic and isolated sessions
    +-- Classical and learned agents / packaged model checkpoints
    +-- LLM analysis service (server-side API key only)
```

The frontend must use a production-compatible API base URL (or an intentional deployment proxy). Restrict production CORS to configured frontend origins. API keys and model-provider calls belong **only in the backend**, never in the browser bundle. The GitHub Actions image-build pipeline should be verified rather than assumed healthy.

The prior intended **Render Static Site** settings were root directory `ui`, build command `npm ci && npm run build`, and publish directory `dist`. Verify their current compatibility before applying them.

## 5. Implementation phases

Work sequentially, but keep each phase bounded and demonstrable. Do not start a major new feature while the preceding end-to-end vertical slice is broken.

### Phase A — Focused audit and local stabilization (**next**)

**Goal:** Establish what works today and the smallest set of changes required to continue confidently.

**Tasks**

- Review code paths for both game engines, agent factory/loading, API endpoints, React/legacy UI, Docker files, workflow, and checkpoint artifacts.
- Trace each resume claim to working code; specifically find whether DQN exists outside the current executable source (history/archives/branches) or needs implementation.
- Run existing local builds and a manual Connect 4 game; fix only actual blockers. Add a handful of focused regression/smoke tests for game rules, API contract, and illegal/terminal moves.
- Check whether the production model weights can be loaded from the built API image; choose an explicit checkpoint-packaging approach (and keep unnecessary training artifacts out of production).
- Fix `.gitignore` and stop tracking generated dependencies/configuration. Provide an `.env.example` with variable names but no secrets.
- Document observed failures, changes made, and remaining work. Avoid rewriting the game engines or swapping frameworks merely for modernization.

**Exit criteria:** Clean local startup, one complete Connect 4 game against at least one available agent, trustworthy status of each claimed AI approach, and a short prioritized blocker list.

### Phase B — Finish deployment and playable Connect 4 (**highest shipping priority**)

**Goal:** Make the existing game playable by anyone at a public URL.

**Tasks**

- Verify/repair GitHub Actions → GHCR → Render API deployment, including permissions, secrets, health checks, and image selection.
- Create/finish the separate Render Static Site for React/Vite using the architecture above.
- Implement a consistent local/production API configuration and the necessary backend CORS policy.
- Resolve model-checkpoint placement/loading, or clearly disable an unavailable agent until it is deployable rather than leaving a broken selector.
- Replace process-global game state with a minimal per-game/session design; check behavior across simultaneous visitors and service restarts as applicable.
- Ensure move validation, game-over/draw behavior, AI thinking/loading feedback, error states, restart, and agent selection are usable from the browser.
- Complete README quickstart and explain the difference between local Docker Compose and deployed frontend/API hosting.

**Exit criteria:** The public site loads, the API is reachable, separate visitors do not share a game, and a full Connect 4 match can be completed against a selectable AI. Local `docker compose up --build` is also repeatable.

### Phase C — Complete the advertised AI approaches and improve gameplay

**Goal:** Make the agent-selection experience substantive and align it with the first resume bullet.

**Tasks**

- Verify existing Minimax/Negamax, MCTS, MCTS-NN, and other agents correctly generate legal moves and handle terminal positions.
- Locate/recover DQN if it previously existed; otherwise add a focused DQN implementation and a compatible trained checkpoint. Specify which game supports it. Do not assert DQN is already present.
- Expose only usable agent types and appropriate tunable difficulty/simulation parameters in the UI; avoid choices that routinely time out in the public demo.
- Make the trained-model load path stable locally and in deployment, with clear failure handling.
- Use manual play to discover obvious agent/gameplay problems and fix the highest-impact ones. Add regression tests for discovered defects.
- Improve game UI only where it makes interaction, agent differentiation, or move feedback clearer.

**Exit criteria:** Each advertised AI approach has a working code path and can be challenged in the application for its supported game. The resume bullet accurately names each game/algorithm pairing.

### Phase D — LLM-powered strategy explanation (**essential resume feature**)

**Goal:** Let users ask useful questions about an actual game, grounded in board state and available engine evidence.

**Recommended MVP**

- Add a backend analysis endpoint that accepts a validated game/session, board state, side to move, last move (when relevant), and a user question or a small set of presets.
- Provide the LLM with structured context: game rules, current board, legal actions, recent moves, immediate tactical threats, and *available* agent/engine signals. Add lightweight deterministic analysis when practical.
- Support at least: **Analyze this position**, **Explain the last move**, and **What if I play [legal move]?**
- Differentiate a faithful report of a recorded agent signal from a post-hoc strategic explanation. When the system cannot observe the agent's internal reason, say so.
- Integrate the explanation into the React gameplay screen with loading/error states and enough visual context to match explanations to the board.
- Keep model-provider key server-side; add reasonable token, timeout, request-size, and cost controls. Avoid unnecessary multi-agent orchestration.
- Manually check several tactical positions, invalid inputs, and a finished game for grounding and useful explanations. Formal benchmarking is not required.

**Exit criteria:** A real user can ask why a move matters or explore a legal alternative during a live match and receive a relevant, bounded explanation rather than a static demo or generic chatbot response. This supports the third resume bullet.

### Phase E — Mancala integration and portfolio polish

**Goal:** Expand beyond the Connect 4 demonstration without destabilizing the MVP.

**Tasks**

- Expose existing Mancala rules and applicable agents through the API; add a playable React board using the same session and API design.
- Verify the Mancala-specific gameplay flow (legal pit selection, captures, extra turns, termination, scoring) through manual play and focused correctness tests.
- Reuse the LLM analysis abstraction with Mancala-specific rules/board representation if it adds value without excessive scope.
- Finish the README: what the app does, algorithms actually available by game, architecture, screenshots, quickstart, environment setup, public demo URL, and technical tradeoffs.
- Remove stale legacy assets and unfinished documentation when safe. Add a concise explanation of how the agents differ and what is or is not measured.
- Recheck and, if necessary, revise the final resume bullets to avoid overstating functionality.

**Exit criteria:** Connect 4 and Mancala are presented clearly; the published demo, documentation, and final resume accurately reflect the code.

## 6. Non-goals and scope control

**Explicitly deferred:** formal agent tournaments, Elo/rating systems, statistically powered strength comparisons, extensive training sweeps, complex observability, heavy infrastructure, new board games beyond the immediate plan, and generalized multi-agent orchestration.

Manual play **is sufficient for the present product goal**: check perceived difficulty, detect obvious blunders, and assess whether opponents are enjoyable. It is not evidence of small strength differences or formal improvement claims, so avoid such claims in documentation/resume. Revisit rigorous benchmarking only if a future research question or public claim actually requires it.

Additional guardrails:

- Favor an end-to-end usable feature over an elaborate but disconnected subsystem.
- Preserve the existing Flask / React / Vite / Docker foundation unless there is a concrete blocker.
- Keep changes incremental, reviewable, and preferably one phase per PR/commit series.
- Favor focused automated correctness checks over a benchmark platform.
- Do not publish placeholder agent options, fabricated explanations, or untested deployment claims.

## 7. Practical delivery checklist

- [ ] Repo audit completed and DQN status established.
- [ ] Existing Connect 4 game works locally, including terminal/invalid-move behavior.
- [ ] Local Docker Compose builds and permits a complete match.
- [ ] Production API image includes all required inference artifacts.
- [ ] GHCR/Render backend deploy verified with working health and gameplay endpoints.
- [ ] React frontend deployed separately; production API routing and CORS work.
- [ ] Per-user game/session isolation implemented.
- [ ] Selectable advertised agents work end-to-end, with viable public-demo defaults.
- [ ] LLM analysis endpoint and frontend interface work on real game states.
- [ ] Provider credentials remain server-side; basic cost/error safeguards in place.
- [ ] Mancala playable via the public UI, with applicable agents.
- [ ] README, screenshots, quickstart, and demo link updated.
- [ ] Final resume wording revalidated against shipped features.

## 8. Immediate next assignment for Codex

Start with **Phase A only**. Inspect the current `main` branch and compare the repository to Sections 2–3 of this plan. Determine what actually runs and what is merely present in source. Return:

1. A concise implementation inventory by game/agent, explicitly investigating DQN.
2. A reproducible local run/build result (including relevant failures).
3. The minimum prioritized blockers to reach a playable public Connect 4 deployment.
4. A small, bounded proposed change set for Phase A and the smoke tests needed.

Do not implement new game modes, LLM analysis, heavy benchmarking, a major refactor, or unrelated modernization during this initial assignment. Do not perform paid API calls or trigger deployments without explicit approval.

---

**Project finish line:** A prospective employer can click a link, select an AI opponent, play a real game, ask a grounded strategy question, and see concrete implementations corresponding to the project's three resume claims.
