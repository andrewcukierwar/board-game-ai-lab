# Connect 4 quickstart

From a fresh checkout with Docker running:

```sh
docker compose up --build
```

Open http://localhost:3000. No `.env` file is required. The UI is served by Nginx, which proxies `/v1/` to the Flask/Gunicorn API. Health is available at http://localhost:8000/v1/connect4/health. See the README for non-Docker development (Python 3.11; Node 20.19+ or 22.12+).

## API contract

All POST requests must use `Content-Type: application/json`. Error responses are JSON objects with actionable `error` text and a stable `code`.

- `POST /v1/connect4/start_game`: optional `player1` and `player2` objects (defaults: human and Negamax depth 2). Types: `human`, `random`, `negamax`, `mcts`; only Negamax accepts `depth`, an integer 1–8 (default 2). Only MCTS accepts `simulation_limit`, an integer in 50/100/250/400/800 (default 100; UI presets 100/400/800). All other configuration fields and non-integer budgets, including booleans/null, are rejected. Optional `replace_game_id` replaces a known previous session atomically after validation; an expired/unknown previous ID starts a fresh session. Returns 201.
- `GET /v1/connect4/games/<game_id>`: returns the authoritative board for recovery. Returns 404 for missing or expired sessions.
- `GET /v1/connect4/games/<game_id>/history`: read-only ordered immutable move records, with `game_id`, `revision`, `players`, `moves`, and a matching authoritative `state` snapshot captured under the same session lock. See [Match Lab](phase5a-match-lab.md).
- `POST /v1/connect4/make_move`: requires `game_id` and the latest integer `revision`. On human turns include integer `column` (0–6); on AI turns omit `column`. Each request advances exactly one ply and increments revision. Returns 200.
- `POST /v1/connect4/explain`: requires `game_id`, current `revision`, and `mode` (`last_move`, `position`, `what_if`); only `what_if` requires a legal `column` (0–6). Optional `question` is capped at 500 characters by default. Returns a revision-bound concise summary, key evidence, relevant square coordinates, full facts, Allis context/citations and limitations without mutating the game. Disabled by default. See [the complete explanation contract and configuration](llm-explanations.md).

Player 1 / X / red always moves first; Player 2 / O / yellow moves second. The setup choice applies only at start/restart. For AI-first play, send the AI config as `player1` and human as `player2`. The Play frontend accepts the revision-0 start snapshot, then requests a separate revision-0 AI move and accepts revision 1. A lost opening response is reconciled by GET before an explicit retry; POSTs are never blindly replayed. Last Move, Current Position and What If use the accepted revision and actual player types. See [benchmarks and selection rationale](public-agent-strength-and-turn-order.md).

Match Lab is a separate local route at `/connect4/match-lab`. Configure each player independently; Start leaves revision 0 paused even when Red is AI. Next move requests one AI ply. Autoplay waits at Human turns and resumes after a human click. Pause stops future scheduling while accepting any in-flight move safely. Rewind is local inspection: it pauses autoplay, disables moves, and preserves the live server revision. Return to live restores that board. Match Lab omits historical AI Analysis.

Successful game responses include `game_id`, `revision`, `board` (six rows of seven `X`, `O`, or space characters), `players`, `currentPlayer` (0 or 1), `gameOver`, `legalMoves`, and `winner` (`null`, `Player 1`, `Player 2`, or `Draw`). Responses are not cacheable.

Invalid inputs return 400; expired/missing sessions 404; stale revisions, busy games and terminal moves 409; request bodies above 4 KiB 413; session capacity, agent failures and MCTS contention (`agent_busy`) 503. At most one MCTS search runs per process; contention does not queue. Failed moves leave the board, revision and history unchanged. After an uncertain network response, GET the game before issuing another move: a timed-out POST may already have succeeded.

The default store holds 128 games with a 1,800-second idle timeout. Internal Flask config keys `GAME_SESSION_CAPACITY` and `GAME_SESSION_TTL` can be overridden when constructing an app for tests. No external service is required. Keep **one Gunicorn worker**; per-game locks allow independent sessions on its four threads. Restarting the API loses all games.

## Manual acceptance checks

1. Open the homepage, follow **Play Connect 4**, choose Random and start. Make moves until a win or draw. Check that the final board is disabled and the result is visible.
2. Change to Negamax, choose a depth from 1/2/4/6/8 and click **Start new game**. Choose **You go first** and confirm an empty board. Restart with **AI goes first** and confirm exactly one red AI piece before your yellow turn, then complete another game. Also try restarting mid-game. Repeat with MCTS Quick (100), Balanced (400) and Deep (800); MCTS controls should appear only for MCTS, and changes apply only on start/restart.
3. Open a second browser tab and start another game. Moves and restarts in one tab must not affect the other.
4. Double-click a playable column. Only one human move and one AI response should occur. A full column must be disabled.
5. Stop the API with `docker compose stop api`, then try starting/moving. Controls must recover and show an error. Start it again with `docker compose start api`; use **Refresh game** or **Start game** as offered. The previous game should be reported as expired/missing after the server restart.
6. For natural expiration, leave a game idle for 30 minutes and then move. It should offer a fresh game. Focused tests use a controlled clock so they do not need to wait.
7. Refresh `/connect4` directly; the page should load. Return Home and re-enter without duplicate handlers or console errors.

Explanation runtime settings are backend-only: `OPENAI_EXPLANATION_MODEL` defaults to `gpt-6-luna` and `OPENAI_EXPLANATION_REASONING_EFFORT` defaults to `none`. The complete configuration and limits are in [the explanation guide](llm-explanations.md).

## Public deployment

Phase 2 public deployment is complete. [Play the public application](https://board-game-ai-lab-ui.onrender.com/) or [open Connect 4 directly](https://board-game-ai-lab-ui.onrender.com/connect4). The user verified public gameplay and CORS on October 2, 2026. The [API health endpoint](https://board-game-ai-lab.onrender.com/v1/connect4/health) is at the verified backend origin. Phase 3A is committed at `37dc55e`; Phase 3B and 3B.1 are implemented at `45894d1`, and the user reports successful revised local GPT-6 Luna testing. The user subsequently confirms explanations are enabled and manually tested publicly. Phase 3C preparation used mocked verification only. The current verified public main commit is `1d0874102c8a78935ab68d8cd470f24f66431e9f`, including Random, corrected Negamax, MCTS, turn order, and AI Analysis. Match Lab is implemented locally in Phase 5A, pending review and deployment. Follow [the deployment guide](deployment.md) for the backend variable checklist, disabled-first rollout, separate approval for enablement, rollback and manual smoke checks. No production rollout or new paid calls occur during preparation.

The Static Site uses root `ui`, build `npm ci && npm run build:render`, publish `dist`, `NODE_VERSION=22`, and `VITE_API_BASE=https://board-game-ai-lab.onrender.com`. Add a **Rewrite** from `/*` to `/index.html` for direct `/connect4` and `/connect4/match-lab` navigation. The API uses the GHCR image, `PORT=10000`, health path `/v1/connect4/health`, and `CORS_ALLOWED_ORIGINS=https://board-game-ai-lab-ui.onrender.com` (no slash/path/wildcard). Keep one instance and one worker/four threads, with the image CMD and no Render command override.

Docker explicitly builds with an empty API base; Vite development always uses its local proxy. Render's API origin is compiled into the browser bundle and requires rebuilding when changed. Requests allow 90 seconds for wake-up, with locked controls and explicit recovery. No move POST is automatically retried. After API restart/spin-down or expiration, start a fresh game.

Do not push `main`, run the workflow or invoke its Render hook unless a future deployment is explicitly authorized. Automated gameplay tests must target local servers only.
