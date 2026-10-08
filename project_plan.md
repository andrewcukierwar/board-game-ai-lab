# Board Game AI Lab — Project Plan

**Repository:** https://github.com/andrewcukierwar/board-game-ai-lab  
**Plan updated:** October 8, 2026
**Current status:** The public Connect 4 system and competition/evaluation system are feature-complete on `main` at `c7d0ce65e0a2a23dc6f398164429bd1f2162228f`. The canonical benchmark is complete and verified: 8 agents, 112 games, seed 20261008. Final portfolio packaging is complete locally on `final-portfolio-packaging`, pending review/merge and deployment checks; no commit or push is part of this assignment.
**Current objective:** Present the completed public search/evaluation system clearly, retain reproducible evidence and scientific caveats, and keep experimental research separate.

## Current delivery status

- [x] Public Connect 4: Random, corrected Negamax, bounded MCTS, grounded post-hoc explanations.
- [x] Match Lab, Tournament Lab (including human participation), Season Lab, and evaluation exports/verifier complete.
- [x] Canonical benchmark against the frozen source completed; [preserved evidence and results](benchmarks/connect4/results/canonical-season-v1/README.md).
- [x] Final README, architecture diagram, curated application screenshots, and detailed benchmark report packaged.
- [x] Local backend/frontend, production builds, browser regression, and canonical integrity verification completed; [verification record](docs/final-portfolio-packaging.md).
- [ ] Review and merge packaging; check deployed application freshness and complete release smoke verification separately.
- [ ] Resume wording is a subsequent task; no resume edits in this packaging assignment.

**Research boundary:** AlphaZero/neural checkpoints and other historical agents remain experimental, outside the public API and canonical benchmark. `phase4d3b-alphazero-v2-preflight` and frozen campaign artifacts are not modified. No new training, agent, opening suite, benchmark, Mancala integration, or feature phase is scheduled here.

## Historical plan and development records

The sections below preserve earlier objectives, acceptance criteria, phase reports, and approval boundaries as historical context. Their older “current assignment,” deferred-feature lists, agent budgets, and pending statuses do not supersede the delivery status above. In particular, tournaments/Elo/benchmarking were later delivered, and Mancala remains outside the completed public scope. Detailed subsequent implementation and research records also remain in `docs/`.

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

## 3. Audited baseline and current status (October 2, 2026)

The following baseline records the initial audit; Phase 1 and Phase 2 changes are recorded below. Phase 1 at `830bfdc` is completed and manually verified by the user.

- Audited baseline on `main`: `b93f1a8`, **October 2, 2026**. [API CI run 37023754328](https://github.com/andrewcukierwar/board-game-ai-lab/actions/runs/37023754328) successfully built/pushed the API image and invoked the Render deploy hook. This verifies execution of those pipeline steps, not Render readiness, frontend deployment, or public gameplay.
- Docker Compose defines a Flask/Gunicorn API on port `8000` and a React/Vite UI served by Nginx on local port `3000`; the original local Dockerization phase had been completed.
- Connect 4 Flask endpoints exist under `/v1/connect4/`, including `start_game`, `make_move`, and `health` (`/v1/connect4/health`). The React page is `ui/src/pages/Connect4.jsx` and currently bridges to legacy frontend JavaScript.
- Connect 4 source includes Random, Negamax, MCTS, MCTS + NN, and Victor agent classes, plus model checkpoint files under `models/connect4/`.
- Mancala source includes Human, Random, Simple, Minimax, and Negamax agents, but does not yet have an equivalent integrated web-play experience.
- **Historical Connect 4 DQN recovered:** `Connect 4/DQN.ipynb` in commit `cfc2a6f` contains the same PyTorch prototype as `connect4/DQN.ipynb` inside `connect4.zip`. It includes replay memory and a target network, but lacks playable-agent integration and saved DQN weights. Alternating-player targets and legal-action masking need review before recovery. Historical tabular Q-learning also exists, including Mancala experiments; it is not DQN. DQN is not delivered in the current application.
- No LLM-based explanation layer was identified in the current application.
- At the audit baseline, `README.md` was effectively empty and `docs/quickstart.md` unfinished; local setup, API contract, tests, and manual checks are now documented.
- `.github/workflows/docker-lite.yml` builds/pushes the **API** image to GHCR and triggers a Render hook; both steps succeeded for the audited baseline. It does not deploy the React UI or run gameplay regression tests.
- Repository cleanup is complete: `.env`, `node_modules/`, and generated build output are ignored and no longer tracked. The audit found that Compose still required an untracked `.env`, breaking a fresh checkout; that unnecessary requirement has now been removed.
- Vite 7 requires **Node.js 20.19+ or 22.12+**, not the old quickstart’s Node ≥18. Prefer a compatible Node 22 LTS release for frontend development.

### Remaining issues and resolved local blockers

| Area | Observation / likely consequence |
| --- | --- |
| Production frontend routing | Implemented locally: Render builds require `VITE_API_BASE` as an HTTPS API origin; Docker builds and Vite development retain their local proxies. Exact-origin API CORS and Render SPA routing support the functioning public application (user-confirmed Phase 2 completion). |
| Learned-model packaging | Models are not packaged in the API image. Phase 4A.2 adds bounded MCTS to the local API/UI alongside Human, Random and bounded Negamax (UI: human-versus-AI). Neural agents remain unavailable until their correctness and packaging are repaired. |
| Per-user game state | Resolved locally: app-owned game-ID store, per-game locks, revision checks, 128-session capacity and 30-minute idle expiration. Single-worker operation remains required; restarts lose games. |
| CI/CD | [Run 37034008283](https://github.com/andrewcukierwar/board-game-ai-lab/actions/runs/37034008283) for `830bfdc` confirms API image build/push and hook invocation. Phase 2 adds explicit package permissions, `linux/amd64`, and a bounded hook request. Phase 2 was committed at `cbf8db5`, and the user confirms public gameplay works. The Phase 3A implementation assignment did not trigger a production workflow or deployment. |
| Repo hygiene | Cleanup is complete and Compose no longer requires `.env`. Docker context excludes nested dependencies, local environments and test/build output. |
| Documentation | Local setup, session/API behavior, tests and manual checks are documented. Grounding schema and thesis references are documented in `docs/allis-grounding.md`; screenshots and final portfolio wording remain later work. |
| Code quality | Local request validation, frontend lifecycle/recovery and focused tests are implemented. Advanced-agent/search/training defects from the audit remain deferred. |

### Phase 1 local reliability verification (completed)

- API supports Random and Negamax depth 1–4 (default 2), strict JSON/configuration/move validation, atomic board commits, terminal rejection and authoritative snapshot recovery. Neural imports/checkpoints are not required for the local API.
- Frontend has an obvious homepage entry, one human-versus-AI flow, request locking, explicit AI retry, expired-session recovery, and restart/opponent switching. Navigating away cleans up listeners and cancels requests.
- **55 backend tests, 9 frontend controller tests, and 5 Chromium end-to-end tests passed.** Browser checks completed games against both opponents, restarted with different opponents, checked two independent sessions, and exercised startup/AI/invalid-session recovery.
- Vite production build passed. Compose built and started successfully from a clean source copy without `.env`, host dependencies or build output. Nginx/API health and browser gameplay were verified against the containers. An initial local Docker image-metadata timeout was overcome using an isolated client configuration; the subsequent standard-client Compose build/start also passed.
- API runtime dependencies are separated from optional historical ML dependencies. Docker keeps one Gunicorn worker/four threads, uses Node 22 for the frontend build, and waits for API health before starting Nginx.
- `npm` reports 13 existing dependency advisories (1 low, 1 moderate, 11 high); dependency upgrades remain outside this narrowly scoped reliability change.
- No production deployments, paid API calls, formal benchmarks, training experiments or later-phase features were performed. The user subsequently reviewed and manually verified Phase 1, and authorized Phase 2 implementation/local verification.

### Phase 2 implementation and local verification (historical record; rollout subsequently completed)

- Environment-aware API configuration is implemented: Docker builds explicitly use Nginx's `/v1/` proxy, Vite development always uses its development proxy, and Render builds require an explicit HTTPS `VITE_API_BASE`. Removed the dormant hardcoded production default and excluded developer env files from Docker contexts.
- API CORS allows only configured exact `CORS_ALLOWED_ORIGINS`, supports JSON preflight and error responses, and defaults to no cross-origin access. Wildcard/path/invalid origins fail startup. Health remains `/v1/connect4/health` with HTTP 200/`OK`.
- API image honors Render's `PORT`, retains port 8000 locally, and enforces one Gunicorn worker/four threads. Deployment must also retain one instance. The workflow has explicit GHCR package permissions, `linux/amd64` output and a bounded hook call; no workflow or hook was triggered during this assignment.
- Requests allow 90 seconds for cold starts, display wake-up feedback, reject HTML/non-snapshot responses safely, and retain request locking/revision reconciliation. Moves are never automatically replayed after uncertain responses.
- **74 backend tests, 15 frontend tests, and 8 Chromium end-to-end tests in each of three configurations passed** (24 browser test executions): Docker/Nginx, Vite development with a deliberately set remote API variable, and a production-built static frontend at port 4173 calling an exact-CORS API at port 8001. Tests cover complete games against both opponents, independence, switching/restart, direct navigation/refresh, slow/HTML startup, failure/expiry recovery and lost post-commit responses.
- Compose rebuild/start and default/Render/cross-origin production builds passed. Missing Render API configuration fails the build as intended. The temporary API listened on Render-style port 10000 with one worker. Actual temporary-container restart recovery was verified in Chromium: old session reported missing, fresh start succeeded.
- README and quickstart are updated; [docs/deployment.md](docs/deployment.md) contains exact Render settings, approval-gated manual steps and the production smoke checklist. Existing npm advisories remain outside this scope.
- The original implementation stopped before rollout. **Subsequent Phase 2 completion (October 2, 2026):** the user confirms public deployment is complete and public Connect 4 gameplay has been manually verified, including successful frontend/backend CORS configuration. This records the user’s verification, not a new production smoke test by this documentation update.
- **Verified production URLs:** [public frontend](https://board-game-ai-lab-ui.onrender.com/), [Connect 4](https://board-game-ai-lab-ui.onrender.com/connect4), [backend API origin](https://board-game-ai-lab.onrender.com), and [API health endpoint](https://board-game-ai-lab.onrender.com/v1/connect4/health).

### Phase 3A implementation and local verification (committed at `37dc55e`)

- Created local `phase3-allis-grounding` from freshly fetched `origin/main` at `cbf8db5`. No credentials changed, pushes, production mutations, or deployments.
- Added immutable per-session move records with player/agent, column, before/after board, ply/revisions, and outcome. Existing locks, atomic candidate commits, revision errors, expiration, replacement, capacity, and gameplay JSON contracts are preserved.
- Added reusable deterministic position/move analysis, complete session-history replay, revision-bound context assembly, and deterministic retrieval of 15 curated primary-source entries (six concepts and all nine formal rules).
- Read [Allis’s actual 1988 thesis](https://tromp.github.io/c4/connect4_thesis.pdf). References distinguish thesis pagination from 1-based PDF pages and zero-based indices. The supplied edition has no visible body-page folios.
- Tactical evidence covers immediate wins, forced defensive responses, gravity-playable completion squares, legal alternatives and simple square creation/removal. **Formal Allis rule applications remain unsupported**: no local pattern is promoted into a rule, Zugzwang, compatibility, coverage, or long-term result proof.
- Reviewed the experimental VictorAgent and corrected its complete-solver/Allis-classification claims without behavior changes. Negamax is not attributed Allis reasoning.
- **119 backend tests, 15 frontend tests and 8 Chromium end-to-end tests passed.** Browser tests used this branch’s local API at port 8001 and production-built frontend at port 4173 with exact CORS. Default and separate-origin Vite builds passed.
- Evidence schema, research references, applicability boundaries and regression examples: [docs/allis-grounding.md](docs/allis-grounding.md). No LLM call, explanation HTTP endpoint or frontend panel has been added.

### Phase 3B local verification (awaiting review)

- Implemented all three explanation modes under `/v1/connect4/explain`, using Phase 3A's detached/replay-verified evidence, constrained model selections, validated citations and explicit unknowns.
- Backend-only `.env`/environment configuration, disabled-by-default behavior, bounded provider request/output/timeout, per-game/client/global attempts, concurrent/duplicate protection and successful-response caching are implemented.
- Added the explanation panel within the existing React/legacy controller architecture. Gameplay remains independent of explanation loading/errors; moves, restarts and navigation cancel/clear stale explanations.
- **188 backend tests, 25 frontend tests and 13 Chromium tests passed**, including existing complete games against Random and Negamax. Default, separate-origin and Render-mode builds passed. All explanation provider responses were mocked. Live model compatibility/quality remain unverified.
- No pushes, deployment, Render setting changes or paid requests. Detailed contract, limits and a separately approved three-call manual-test proposal: [docs/llm-explanations.md](docs/llm-explanations.md).

### Phase 3B final review and configuration verification

- Review fixes at `cd9eb0b` preserve the provider deadline/cleanup and exact hypothetical-column response validation. The `d35b9f0` documentation commit remains in history.
- Backend defaults are now `OPENAI_EXPLANATION_MODEL=gpt-6-luna` and `OPENAI_EXPLANATION_REASONING_EFFORT=none`, using the Responses API's nested reasoning parameter. Both remain server-side/environment-configurable; cache identity includes effort. Token/time caps and unsupported-output rejection are preserved.
- **224 backend tests passed on macOS and Linux, 26 frontend tests and 42 Chromium executions passed**, with all provider responses mocked. Default/separate-origin/Render-mode and Docker builds passed. See [the final review record](docs/phase3b-review.md).
- The user authorizes a local commit after passing checks. No paid calls, push, merge, deployment, credentials or Render configuration changes.

## 4. Intended architecture

**Local development:** Docker Compose orchestrates an API container (Python / Flask / Gunicorn / game engines / trained artifacts) and UI container (React/Vite build served by Nginx). Nginx proxies `/v1/` to the API service.

**Production (deployed topology):**

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

The **Render Static Site** settings are root directory `ui`, build command `npm ci && npm run build:render`, publish directory `dist`, `NODE_VERSION=22`, and an explicit HTTPS `VITE_API_BASE`. Configure `/*` → `/index.html` as a Rewrite. API settings: GHCR image, `PORT=10000`, `/v1/connect4/health`, exact `CORS_ALLOWED_ORIGINS`, one instance and the image CMD with one worker/four threads. See [the deployment guide](docs/deployment.md) for full settings and manual rollout steps.

## 5. Implementation phases

Work sequentially, but keep each phase bounded and demonstrable. Do not start a major new feature while the preceding end-to-end vertical slice is broken.

### Phase 1 — Focused audit and Reliable Local Connect 4 (**complete; manually verified**)

**Goal:** Establish what works today and the smallest set of changes required to continue confidently.

**Tasks**

- Review code paths for both game engines, agent factory/loading, API endpoints, React/legacy UI, Docker files, workflow, and checkpoint artifacts.
- Trace each resume claim to working code; specifically find whether DQN exists outside the current executable source (history/archives/branches) or needs implementation.
- Run existing local builds and a manual Connect 4 game; fix only actual blockers. Add a handful of focused regression/smoke tests for game rules, API contract, and illegal/terminal moves.
- Check whether the production model weights can be loaded from the built API image; choose an explicit checkpoint-packaging approach (and keep unnecessary training artifacts out of production).
- Preserve the completed ignore/untracking cleanup. Remove the unnecessary Compose `.env` requirement; provide an environment example only when actual configuration variables are needed.
- Document observed failures, changes made, and remaining work. Avoid rewriting the game engines or swapping frameworks merely for modernization.

**Exit criteria:** Clean local startup, one complete Connect 4 game against at least one available agent, trustworthy status of each claimed AI approach, and a short prioritized blocker list.

### Phase 2 — Public Deployment (**complete; public application functioning, user-confirmed**)

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

### Deferred — Complete the advertised AI approaches and improve gameplay

**Goal:** Make the agent-selection experience substantive and align it with the first resume bullet.

**Tasks**

- Verify existing Minimax/Negamax, MCTS, MCTS-NN, and other agents correctly generate legal moves and handle terminal positions.
- Locate/recover DQN if it previously existed; otherwise add a focused DQN implementation and a compatible trained checkpoint. Specify which game supports it. Do not assert DQN is already present.
- Expose only usable agent types and appropriate tunable difficulty/simulation parameters in the UI; avoid choices that routinely time out in the public demo.
- Make the trained-model load path stable locally and in deployment, with clear failure handling.
- Use manual play to discover obvious agent/gameplay problems and fix the highest-impact ones. Add regression tests for discovered defects.
- Improve game UI only where it makes interaction, agent differentiation, or move feedback clearer.

**Exit criteria:** Each advertised AI approach has a working code path and can be challenged in the application for its supported game. The resume bullet accurately names each game/algorithm pairing.

### Phase 3 — Allis-grounded LLM explanations (**complete; public enablement/testing confirmed by user**)

**Goal:** Let users ask useful questions about an actual game, grounded in board state and available engine evidence.

**Phase 3A — Deterministic grounding (implemented, locally verified and committed)**

Record immutable session history; verify tactical facts and legal alternatives; curate the original Allis thesis; assemble a revision-bound payload separating facts, supported rule applications, observations and unknowns. No nine-rule detector is claimed until its full applicability and interaction conditions are implemented and tested. See the implementation record above.

**Phase 3B — LLM integration and frontend (implemented, including Phase 3B.1; revised local live test completed by user)**

- Add a backend analysis endpoint that accepts a validated game/session, board state, side to move, last move (when relevant), and a user question or a small set of presets.
- Provide the LLM with structured context: game rules, current board, legal actions, recent moves, immediate tactical threats, and *available* agent/engine signals. Add lightweight deterministic analysis when practical.
- Support at least: **Analyze this position**, **Explain the last move**, and **What if I play [legal move]?**
- Differentiate a faithful report of a recorded agent signal from a post-hoc strategic explanation. When the system cannot observe the agent's internal reason, say so.
- Integrate the explanation into the React gameplay screen with loading/error states and enough visual context to match explanations to the board.
- Keep model-provider key server-side; add reasonable token, timeout, request-size, and cost controls. Avoid unnecessary multi-agent orchestration.
- Manually check several tactical positions, invalid inputs, and a finished game for grounding and useful explanations. Formal benchmarking is not required.

**Exit criteria:** A real user can ask why a move matters or explore a legal alternative during a live match and receive a relevant, bounded explanation rather than a static demo or generic chatbot response. This supports the third resume bullet.

**Phase 3C — Production Deployment & Verification (historical preparation; subsequent public testing confirmed)**

- Confirm clean starting branch/commit and Phase 3A/3B/3B.1 ancestry; review tracked files/history and production images for private artifacts and credential exposure.
- Verify the proposed profile using mocks: GPT-6 Luna/`none`, 800 output tokens, 40-second timeout, 10 attempts/process/window, 3/game, 5/network client/window, 3600-second windows, one concurrent request. Preserve implementation defaults.
- Document exact backend variable names, unchanged exact-origin CORS, one-instance/worker requirement, process-local restart/window/proxy limitations, cache behavior and provider-side budget/usage monitoring.
- Verify the disabled Linux amd64 API image and frontend builds/tests. Label last-move analysis accurately as **Analyze Last AI Move**, preserving the endpoint and analysis behavior.
- Prepare sequential disabled-image rollout, explicit separate enablement approval, bounded paid smoke testing only with its own allowance, and rollback by retained immutable image digest.

**Exit criteria:** Local readiness verification completed; the user subsequently confirmed public explanations are enabled and manually tested. This records user confirmation, not a new production inspection; detailed deployment IDs/settings were not collected in Phase 4A.2. See [the readiness review](docs/phase3c-readiness.md) and [the rollout/rollback procedure](docs/deployment.md).

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

- [x] Repo audit completed and historical DQN prototype status established.
- [x] Existing Connect 4 game works locally, including terminal/invalid-move behavior (Random/bounded Negamax).
- [x] Local Docker Compose builds and permits a complete match.
- [x] API image includes required code/dependencies for public Random and bounded Negamax; no learned checkpoints are needed for these agents. Neural agents remain disabled/deferred.
- [x] Public backend supports functioning Connect 4 gameplay (user-confirmed Phase 2 completion).
- [x] Separate-origin frontend API routing and restricted CORS implemented, verified locally and manually verified in production by the user.
- [x] Public React frontend and API are functioning (user-confirmed Phase 2 completion; historical detailed smoke checklist retained in the deployment guide).
- [x] Per-user game/session isolation implemented for local single-worker Connect 4.
- [ ] Selectable advertised agents work end-to-end, with viable public-demo defaults.
- [x] Phase 3A deterministic grounding and evidence schema implemented, locally tested and committed at `37dc55e`.
- [x] Phase 3B/3B.1 endpoint and frontend implemented; user reports successful revised local GPT-6 Luna testing. Automated verification uses mocks; subsequent public enablement and manual testing are confirmed by the user.
- [x] Provider credentials remain server-side; disabled-by-default functionality, request/token/time limits, per-game/client/global caps, duplicate protection and cache implemented.
- [x] Phase 3C local readiness review, proposed spending-profile verification and disabled-first deployment/rollback documentation prepared.
- [x] Public explanations enabled and manually tested, as subsequently confirmed by the user.
- [ ] Archive exact production image/deploy IDs and operational monitoring evidence when supplied; this assignment did not inspect production.
- [x] Phase 4A.1 standalone MCTS correctness completed at `c213ddf`.
- [x] Phase 4A.2 MCTS integration implemented and verified locally.
- [ ] Phase 4A.2 review, manual acceptance and approved deployment/public smoke checks.
- [ ] Mancala playable via the public UI, with applicable agents.
- [x] README, quickstart and verified public demo links updated.
- [ ] Portfolio screenshots added.
- [ ] Final resume wording revalidated against shipped features.

## 8. Historical assignment and approval boundary (Phase 4A.2)

The current assignment implements **Phase 4A.2 — Public Connect 4 MCTS integration** on local branch `phase4a2-mcts-integration`, based on freshly fetched `origin/main` at `c213ddfbe2e39ec29a5f4f013d5a49b800643e88`. The starting tree was clean. Phase 4A.1 is complete at that commit. Phases 1–3 are preserved; the user confirms explanations have been enabled and manually tested publicly.

MCTS uses UCT selection, random rollouts, reward from each node's previous-player perspective, and final visit counts, plus immediate root tactical guards. The API accepts only 50, 100 (default) or 250 simulations. One process-wide nonblocking reservation permits one synchronous MCTS search; another game receives retryable HTTP 503 `agent_busy` without board/history/revision mutation. Other agents remain unaffected. The simulation count is not a wall-clock guarantee; one instance/worker with four threads remains required, and Render Free performance still needs an approved rollout check.

See [Phase 4A.2 verification, API contract and rollout checklist](docs/phase4a2-mcts-integration.md). MCTS-NN, DQN and VictorAgent remain experimental/unavailable; Mancala and neural-model work are deferred. **Do not commit without separate approval; do not push, merge, deploy, trigger workflows, change Render settings or credentials, or make paid provider calls.** Stop after local implementation and verification.

---

**Project finish line:** A prospective employer can click a link, select an AI opponent, play a real game, ask a grounded strategy question, and see concrete implementations corresponding to the project's three resume claims.

### Phase 3B.1 — Explanation quality (October 3, 2026)

The user completed the initial local live evaluation with GPT-6 Luna / reasoning effort `none`. All three modes worked, but the report was too verbose and the model mostly reordered templated content. Phase 3B.1 adds verified explanatory relationships, a model-selected primary paragraph, bounded relevant evidence, position-connected Allis concepts, expandable full analysis/methodology, and verified-square labels. Primary source Chapters 3–8 and original diagrams were consulted directly. Formal rule applications remain unsupported. No new paid request, production change, push, merge, training or Mancala work is authorized. See [the audit, examples, changed-file inventory and final verification](docs/phase3b1-quality.md).

### Phase 3C — Readiness review (October 3, 2026)

Started from clean `phase3b-llm-explanations` at `45894d1`, preserving all prior commits. The user reports the revised Phase 3B.1 feature now works with real GPT-6 Luna requests. This review uses mocks only and does not repeat live testing. The exact proposed production profile passes configuration, quota/cache/failure/window regressions; implementation defaults remain unchanged. Full verification and remaining operational limits are recorded in [docs/phase3c-readiness.md](docs/phase3c-readiness.md). [docs/deployment.md](docs/deployment.md) contains the backend checklist and sequential disabled deployment, separate enablement and rollback instructions. The user subsequently confirms public explanations are enabled and manually tested. The preceding procedure is a historical readiness record; exact production image identity was not re-inspected during Phase 4A.2.
