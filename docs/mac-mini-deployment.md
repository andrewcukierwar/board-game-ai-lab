# Mac Mini public backend: hardening and controlled deployment

Production topology: Render static React UI -> public Tailscale Funnel HTTPS URL ->
Mac Mini port 8000 (Docker bound to `127.0.0.1`) -> Flask/Gunicorn
**one worker, four threads**. Deployment initially operated at
`bf0b15ab17317f5411f5abac5c3375471357ed09`.

This procedure changes ONLY a separate `~/Services/board-game-ai-lab-api`
checkout and the `bgai-api` Docker container. **Do not modify, fetch into,
rebase, merge, or run builds in the AlphaZero training worktree**. Do not alter
`benchmarks/connect4/results/canonical-season-v1` or other frozen research
evidence. Do not push `main` during rollout: the existing
`.github/workflows/docker-lite.yml` still deploys Render on a main push.

## Protections

- `PUBLIC_RATE_LIMIT_ENABLED=true`: constant-memory, process-wide token buckets.
  All `/v1/connect4/*` methods count except OPTIONS (browser preflight) and
  `/health` (monitoring). Default total API rate: **60 requests/second**
  with a **160-request burst**. A separate start-game rate: **8/second**
  with a **40-start burst**. Token depletion returns JSON
  `{"code":"rate_limited",...}`, HTTP **429**, and `Retry-After`.
  Preflight and successful/error responses still use the exact existing CORS
  allowlist. Tokens also count for invalid requests.
- `PUBLIC_SEARCH_CONCURRENCY=2`: shared nonblocking semaphore around expensive
  Negamax, MCTS, and Victor searches. Search saturation retains the
  `agent_busy` 503 contract and does **not** commit a move. Existing one-at-a-time
  MCTS and Victor reservations remain. Random and human moves do not reserve
  a compute slot. The shared semaphore is released on all failure paths.
- No per-IP identity is inferred from `X-Forwarded-For` or other untrusted
  proxy headers. Tailscale Funnel can aggregate network identities; global
  limits avoid false per-client partitioning or spoof-based bypass. These are
  availability *bounds*, not authenticating users, network DDoS prevention,
  per-person fairness, or a hard OpenAI spending limit.
- Existing explanation quotas and game-session limits are unchanged. A
  process restart resets all of these in-memory limits and live sessions.

Defaults can be tuned using exactly these optional backend-only environment
variables (none may be placed in `VITE_*` or committed with secrets):

| Key | Default |
| --- | --- |
| `PUBLIC_RATE_LIMIT_ENABLED` | `true` |
| `PUBLIC_REQUEST_RATE_PER_SECOND` | `60` |
| `PUBLIC_REQUEST_BURST` | `160` |
| `PUBLIC_START_RATE_PER_SECOND` | `8` |
| `PUBLIC_START_BURST` | `40` |
| `PUBLIC_SEARCH_CONCURRENCY` | `2` |

Global budgets intentionally favor serving the small public lab over allowing
unlimited benchmark traffic. Run bulk research benchmarks against a separate
local test instance if you need different limits. Avoid changing public limits
merely to make a stress test pass; distinguish normal and adversarial load first.

## Stage without modifying production

1. Confirm there is no ongoing AlphaZero training process before starting a
   resource-intensive Docker image build. Do not stop or modify a training job.
2. In a new Mac Mini Terminal:

   ```sh
   cd ~/Services/board-game-ai-lab-api
   git status --short
   git fetch origin hardening/mac-mini-public-api
   git switch --track origin/hardening/mac-mini-public-api
   git rev-parse HEAD
   ```

   This checks out the dedicated hardening branch in the **Services clone**,
   not the research checkout. No merges or pushes are required.
3. Review your ignored private environment file at
   `~/.config/board-game-ai-lab/api.env` without pasting any API key. Preserve
   `CORS_ALLOWED_ORIGINS=https://board-game-ai-lab-ui.onrender.com`,
   `VICTOR_RESEARCH_ENABLED=true`, and your existing explanation flags and
   quotas. Append only overrides if desired; all new hardening defaults are
   active even if no `PUBLIC_*` keys are present.
4. Run backend tests (Python 3.11; optional PyTorch skip behavior unchanged):

   ```sh
   python3.11 -m venv .venv
   .venv/bin/python -m pip install -r requirements-dev.txt
   .venv/bin/python -m pytest tests/test_public_api_limits.py \
     tests/test_connect4_api.py tests/test_connect4_mcts_api.py -q
   .venv/bin/python -m pytest tests -q
   ```

   To confirm the frontend remains compatible, from `ui/` run `npm ci` and
   `npm test` if Node 22.12+ is installed. Tests mock LLM provider traffic;
   they do not require paid calls.
5. Verify clean checkout and remaining CPU/memory headroom. The deployment
   script does a local-only test on port **8001** before touching the live API.

## Deploy and verify

From the Services checkout, after tests pass:

```sh
bash scripts/deploy-mac-mini-api.sh
```

The script refuses a dirty checkout or existing `bgai-api-rollback`;
builds a commit-specific **linux/arm64** image; launches a private candidate
on `127.0.0.1:8001`; checks health, provenance and game creation; and only
then replaces `bgai-api` on port 8000. If cutover fails before final health
verification, it restores the original container automatically.

Run these checks after a deployment:

```sh
docker ps --filter name=bgai-api
curl -fsS http://127.0.0.1:8000/v1/connect4/health
curl -fsS http://127.0.0.1:8000/v1/connect4/provenance
tailscale funnel status
docker stats --no-stream bgai-api
```

Verify the **public** Funnel health/provenance, CORS preflight, Play
(human/Negamax/MCTS/Victor), Match, Tournament and a small Season Lab with
exports. If explanations were enabled previously, use a small explicit
provider-call budget to test after the game-agent smoke test. Inspect browser
Network to confirm the unchanged Render UI still calls the expected `.ts.net`
API origin. Do **not** rerun/overwrite the canonical 112-game result.

## Rollback

If a live deployment causes a regression, in the Services checkout:

```sh
docker rm -f bgai-api
docker rename bgai-api-rollback bgai-api
docker start bgai-api
curl -fsS http://127.0.0.1:8000/v1/connect4/health
```

This restores the **original container**, including its original environment
and image (not merely an old tag). Funnel continues to point to port 8000.
Existing active sessions reset on any cutover/restart. Review public API and
provenance after rollback. The script deliberately keeps
`bgai-api-rollback` after a successful cutover; only remove it when the new
deployment has proven stable:

```sh
docker rm bgai-api-rollback
```

An unexpected public outage can also be mitigated by restoring Render UI's
`VITE_API_BASE` to `https://board-game-ai-lab.onrender.com` and rebuilding
the Static Site, provided the old Render backend is still active.

## Follow-up work

Add independent uptime checks, evaluate latency and memory under ordinary
traffic, then replace the legacy Render deploy hook in
`.github/workflows/docker-lite.yml` in a separately reviewed change.
Do not automatically pull/rebuild/restart the Mac Mini on every main push while
it also runs long experiments.
