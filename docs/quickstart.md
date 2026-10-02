# Connect 4 quickstart

From a fresh checkout with Docker running:

```sh
docker compose up --build
```

Open http://localhost:3000. No `.env` file is required. The UI is served by Nginx, which proxies `/v1/` to the Flask/Gunicorn API. Health is available at http://localhost:8000/v1/connect4/health. See the README for non-Docker development (Python 3.11; Node 20.19+ or 22.12+).

## API contract

All POST requests must use `Content-Type: application/json`. Error responses are JSON objects with actionable `error` text and a stable `code`.

- `POST /v1/connect4/start_game`: optional `player1` and `player2` objects (defaults: human and Negamax depth 2). Types: `human`, `random`, `negamax`; only Negamax accepts `depth`, an integer 1–4. Optional `replace_game_id` replaces a known previous session atomically after validation; an expired/unknown previous ID starts a fresh session. Returns 201.
- `GET /v1/connect4/games/<game_id>`: returns the authoritative board for recovery. Returns 404 for missing or expired sessions.
- `POST /v1/connect4/make_move`: requires `game_id` and the latest integer `revision`. On human turns include integer `column` (0–6); on AI turns omit `column`. Each request advances exactly one ply and increments revision. Returns 200.

Successful game responses include `game_id`, `revision`, `board` (six rows of seven `X`, `O`, or space characters), `players`, `currentPlayer` (0 or 1), `gameOver`, `legalMoves`, and `winner` (`null`, `Player 1`, `Player 2`, or `Draw`). Responses are not cacheable.

Invalid inputs return 400; expired/missing sessions 404; stale revisions, busy games and terminal moves 409; request bodies above 4 KiB 413; session capacity and agent failures 503. Failed moves leave the board and revision unchanged. After an uncertain network response, GET the game before issuing another move: a timed-out POST may already have succeeded.

The default store holds 128 games with a 1,800-second idle timeout. Internal Flask config keys `GAME_SESSION_CAPACITY` and `GAME_SESSION_TTL` can be overridden when constructing an app for tests. No external service is required. Keep **one Gunicorn worker**; per-game locks allow independent sessions on its four threads. Restarting the API loses all games.

## Manual acceptance checks

1. Open the homepage, follow **Play Connect 4**, choose Random and start. Make moves until a win or draw. Check that the final board is disabled and the result is visible.
2. Change to Negamax, choose a depth from 1–4 and click **Start new game**. Confirm an empty board, and complete another game. Also try restarting mid-game.
3. Open a second browser tab and start another game. Moves and restarts in one tab must not affect the other.
4. Double-click a playable column. Only one human move and one AI response should occur. A full column must be disabled.
5. Stop the API with `docker compose stop api`, then try starting/moving. Controls must recover and show an error. Start it again with `docker compose start api`; use **Refresh game** or **Start game** as offered. The previous game should be reported as expired/missing after the server restart.
6. For natural expiration, leave a game idle for 30 minutes and then move. It should offer a fresh game. Focused tests use a controlled clock so they do not need to wait.
7. Refresh `/connect4` directly; the page should load. Return Home and re-enter without duplicate handlers or console errors.

## Public deployment

Phase 2 public deployment is complete. [Play the public application](https://board-game-ai-lab-ui.onrender.com/) or [open Connect 4 directly](https://board-game-ai-lab-ui.onrender.com/connect4). The user manually verified public gameplay and successful frontend/backend CORS configuration on October 2, 2026. The [API health endpoint](https://board-game-ai-lab.onrender.com/v1/connect4/health) is available at the verified backend origin. Phase 3A is implemented and committed at `37dc55e`; Phase 3B remains unimplemented. Follow [the deployment guide](deployment.md) for Render settings, future rollout steps, separate-origin local verification, and the reusable production smoke checklist.

The Static Site uses root `ui`, build `npm ci && npm run build:render`, publish `dist`, `NODE_VERSION=22`, and `VITE_API_BASE=https://board-game-ai-lab.onrender.com`. Add a **Rewrite** from `/*` to `/index.html` for direct `/connect4` navigation. The API uses the GHCR image, `PORT=10000`, health path `/v1/connect4/health`, and `CORS_ALLOWED_ORIGINS=https://board-game-ai-lab-ui.onrender.com` (no slash/path/wildcard). Keep one instance and one worker/four threads, with the image CMD and no Render command override.

Docker explicitly builds with an empty API base; Vite development always uses its local proxy. Render's API origin is compiled into the browser bundle and requires rebuilding when changed. Requests allow 90 seconds for wake-up, with locked controls and explicit recovery. No move POST is automatically retried. After API restart/spin-down or expiration, start a fresh game.

Do not push `main`, run the workflow or invoke its Render hook unless a future deployment is explicitly authorized. Automated gameplay tests must target local servers only.
