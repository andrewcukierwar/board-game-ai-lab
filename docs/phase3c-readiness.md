# Phase 3C — Production-readiness review

**Status update (October 3, 2026):** The user subsequently confirmed explanations have been enabled and manually tested publicly. Earlier statements below about pending rollout describe the original Phase 3 preparation, not current public status. Source defaults remain disabled. Phase 4A.2 MCTS integration is local, pending review/deployment; it does not change explanation settings or make paid calls. See [the MCTS handoff](phase4a2-mcts-integration.md).

Reviewed October 3, 2026, starting from clean `phase3b-llm-explanations` at `45894d1`. The user reports successful real GPT-6 Luna tests of the revised Phase 3B.1 feature. This review makes **no paid provider requests**, push, merge, deployment, Render configuration change or credential change. Local disposable images/servers use empty keys or mock-only sentinels; all explanation responses in verification are mocked.

## Git, implementation and artifact review

- Starting branch and HEAD match the assignment; `git status --short` was empty. History contains Phase 3A `37dc55e`, Phase 3B `695b872`, transport fixes `cd9eb0b`, model/effort configuration `c5fc0f3`, and Phase 3B.1 `45894d1`; prior documentation commit `d35b9f0` is preserved. No history is rewritten.
- Phase 3A deterministic analysis, immutable history and revision-bound evidence feed the Phase 3B three-mode endpoint, bounded backend provider and frontend panel. Phase 3B.1 verified focus selection, concise summary/evidence, relevant-square highlights and expandable analysis are present. Formal Allis applications and recorded Negamax intent remain unsupported.
- Tracked environment files are only `.env.example` (empty key/configuration template) and `ui/.env.production` (public empty API-base default). Private root `.env`, local env overrides, research PDFs, virtualenvs, browser outputs and build directories are untracked/ignored. No tracked research/private-development directory or PDF was found.
- Scanned tracked files, all 67 members of the historical `connect4.zip` archive, and 2060 reachable historical blobs for provider/GitHub credential patterns and private-key material. Also compared the privately read local provider key against historical blobs without printing it. No matches were found. The archive contains historical source/macOS metadata; existing learned checkpoints are historical project assets, not explanation dependencies. Neither archive nor checkpoints are copied into production images. This is a scoped artifact/credential review, not a comprehensive security audit.
- `docker/api.Dockerfile` installs `requirements-api.txt` into a virtualenv and copies it into runtime alongside `api/` and `games/`. The explanation provider uses standard-library `http.client`, `socket` and threading, with no OpenAI SDK dependency. `python-dotenv` is included for the app import; NumPy, Flask and Gunicorn remain required. Runtime import checks and `pip check` passed in a freshly built **Linux amd64** image, executed locally under emulation.
- Image inspection confirmed amd64, the production one-worker/four-thread CMD, disabled/key-free defaults, GPT-6 Luna/`none`, and no `.env` or PDF in `/app`. Runtime source includes the grounding/provider/focus modules. UI build artifacts contain neither backend setting names/model nor mock sentinel/local credential values. No source defaults, credential files, Dockerfiles or workflow behavior were changed.

## Spending-profile review

The exact proposed environment profile parses and passes startup validation: `gpt-6-luna`, effort `none`, 800 output tokens, 40-second timeout, 10 global attempts, 3 per game, 5 per observed network client, 3600-second windows, and one concurrent request. It is a set of production environment overrides; source defaults and `.env.example` remain unchanged. The initial flag stays false.

| Safeguard | Actual behavior and evidence |
| --- | --- |
| Disabled-first operation | Valid explanation requests return 503/`explanations_disabled` before any provider reservation/call, even with a configured mock key. Default image is disabled and key-free. |
| Atomic attempt accounting | Service lock covers concurrency checks and game/client/global reservations before external I/O. Failures, timeouts, malformed output and stale results consume allowance. No refunds, automatic retries or fallback calls. Existing race/stale/failure regressions pass. |
| Combined proposed quotas | New production-profile regression permits 3 attempts in a game and rejects a fourth, exhausts 5 across one client's games, then reaches 10 across clients including a failed tenth attempt. A new client/game cannot bypass the global cap; spoofed forwarded headers do not bypass the observed-client cap. |
| Cache | Successful current-revision hits return before quota/concurrency checks without another reservation. The profile test checks hits at exhausted game/global caps. Cache identity includes game/revision/mode/column/trimmed question/model/effort; LRU size 256 and TTL 1800 seconds remain defaults. Errors are not cached. |
| Concurrency/gameplay | One in-flight game prevents another uncached provider call; duplicate same-game requests receive 409, others 429. External I/O releases game locks and the permit releases in `finally`. Existing concurrency tests and actual one-worker/four-thread runtime verification pass. |
| Window semantics | Global window begins at service creation; each client window begins at its first attempt. Expiration is checked on later request activity. The production-profile regression renews client/global counts at 3600 seconds, retaining the lifetime game count. Cache expiration alone does not restore allowance. |
| Provider deadline | Socket timeout plus monotonic deadline/watchdog, bounded reads and resource cleanup prevent a slow response stream from retaining a permit. Mocked transport/socket regressions pass. The watchdog begins after connection setup: OS DNS or sequential address/TCP/TLS stages can exceed the total interval before the elapsed-deadline check, so this is not an absolute wall-clock bound. The service forwards the proposed 800/40 caps unchanged. |

Counters, cache and games reset on process restart, deploy or free-service spin-down. Each worker/instance, including overlapping deployment processes, has separate allowances; retain one instance/worker. Fixed-window boundaries can allow bursts. A new game resets its game cap but not existing process/client caps. The client identity is `request.remote_addr`; proxies may group visitors under one five-attempt counter. These limits are not authentication, a persistent account budget or a dollar cap. Output caps do not limit input-token cost, and browser aborts cannot undo an already dispatched request. Provider-side usage/budget monitoring and spending alerts remain required; do not assume alert thresholds enforce a hard spending stop. The app does not retain provider timing/token metadata.

## Final local verification

| Check | Result |
| --- | --- |
| Host Python 3.11 backend suite | **240 passed**. Live HTTPS is forbidden in explanation tests. |
| Disposable Linux amd64 production API backend suite | **240 passed**; only pytest was installed additionally in the disposable test container. |
| Frontend/controller tests | **29 passed**. |
| Chromium, separate-origin production build/API | **15 passed** with exact local CORS and the proposed disabled backend profile. |
| Chromium, Docker/Nginx same-origin images | **15 passed** after correcting the test harness's proxy port. |
| Frontend builds | Default, Render mode with the production HTTPS API origin, and separate-origin local production build passed. Render output was built/scanned only, never used to contact production. |
| Production images/runtime | Linux amd64 API and local UI images built; runtime imports/`pip check`, defaults/artifact isolation, Render-style port 10000 health and real valid-game disabled-endpoint/exact-CORS smoke passed. |
| 40-second mocked Gunicorn request | **Passed in 40.0 seconds**: one concurrent request, duplicate/capacity rejection, gameplay/health responsiveness past 30 seconds, sanitized timeout and permit release; the next mocked request succeeded. No extra worker boot occurred during the held request. |
| Integrity | `git diff --check` and local secret/artifact checks passed; commit/tree status checked after local commit. |

Browser coverage includes complete Random/Negamax games, session isolation/restart/refresh, initialization/failure/lost-response recovery, all explanation modes with fixtures, cancellation/stale/hypothetical binding, revised labels, native detail disclosure and square highlights. The real local disabled endpoint was also checked independently of intercepted browser fixtures.

The first Docker browser run returned 502 because the test harness bound the API to Render's port 10000 while unchanged Nginx expects `api:8000`. Recreated the isolated same-origin API at port 8000 and the complete suite passed. Render-style/separate-origin checks retained port 10000. A sandbox-blocked local preview listener was rerun with authorized loopback access. Neither issue required implementation changes.

## Changes and remaining deployment gates

Changed the UI label to **Analyze Last AI Move** (or **Analyze Last Move** for the human/no-move case) and updated its unit/browser assertions. The `last_move` mode/endpoint remains unchanged. Added two mocked backend tests for the exact production profile and combined quotas/cache/failure/window behavior. Updated README, project plan, quickstart, explanation status, historical-audit cross-reference and deployment instructions. No implementation deployment blocker was confirmed, so no backend refactor/default change was made.

**Remaining gates are operational:** obtain separate approval for push/merge and disabled production rollout; confirm live service settings/image digest/build/logs; verify public gameplay/CORS and the actual disabled endpoint; obtain separate enablement approval and an explicit paid smoke-call allowance; verify provider-side monitoring and public proxy identity behavior. No new production requests or live account/configuration inspection were performed during this review. User-reported local live success does not substitute for healthy production-image verification, and no live latency/usage metrics are invented. Existing dependency-advisory and generic request-flood considerations remain recorded in the [earlier review](phase3b-review.md); no broader dependency/security work is included.

Use [the exact backend checklist and sequential rollout/rollback](deployment.md). A push to `main` or workflow dispatch builds/publishes the API image **and invokes the Render deploy hook**. Deploy disabled first, record healthy-image/gameplay results, then stop for separate enablement approval. Retain immutable old image digests for rollback; mutable `main`/`latest` tags can pull the newer broken image again.
