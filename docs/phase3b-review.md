# Independent Phase 3B readiness review

Reviewed October 2, 2026 on local `phase3b-llm-explanations`, starting at `695b872`. The working tree was initially clean. Review scope was the complete branch diff against `main`, the three requested planning/grounding documents, actual implementation and tests, and the existing gameplay/deployment paths. Findings below reference the final working tree; original locations are identified where changed.

**Assessment: ready for a limited local live test after review of these uncommitted fixes. Public explanation enablement still needs separate operational and live-provider validation.** No confirmed P0 defect remains, and both confirmed P1 defects have been fixed. No paid requests, public-site tests, deployment, Render setting changes, credential changes, pushes or commits were performed.

## Findings, ordered by severity

### P0 — no confirmed defect

Keys remain backend-only; disabled defaults, atomic attempt reservations, cross-game client/global quotas, duplicate exclusion and validated model selections hold under the reviewed code and regressions. This is a scoped code review, not a guarantee against all possible abuse. Operational limitations are listed below.

### P1 — socket timeout did not bound a slow provider response (fixed)

[Provider deadline and socket interruption](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/api/connect4/openai_provider.py:28). Originally `generate()` created a socket timeout at line 16 and called `response.read()` at line 29, closing only the connection at line 47. Each read could receive another byte before the socket timeout, allowing headers or a body to occupy a request thread and explanation permit far beyond the configured interval. A detached `Connection: close` response also needed explicit response closure.

Added a monotonic deadline and bounded watchdog that shuts down the captured socket, timeout classification for interrupted HTTP reads, explicit response/connection cleanup, and timer cancellation/join. DNS resolution remains subject to the OS resolver; connection setup uses socket timeouts and consumes the request deadline. No request is retried.

[Three real socket/parser regressions](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/tests/test_connect4_explanations.py:429) exercise slow headers, fixed-length bodies and chunked bodies, including socket closure. All three fail against the original provider and pass against the fix on macOS and Linux.

### P1 — hypothetical response column was not verified (fixed)

[Response identity validation](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/ui/legacy/explanations.js:105). The UI checked game ID, revision and mode, but accepted a response for a different hypothetical column and labeled it with the requested column. The backend currently returns the correct column; this was a missing client grounding check for unusable/mismatched responses.

The client now requires the exact hypothetical column, or null for other modes, before displaying any facts. Added a [controller regression](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/ui/tests/explanations.test.js:136) and a [browser regression](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/ui/e2e/explanations.spec.js:98). The controller regression demonstrably fails against the original code. Fixtures now match the actual API's explicit null column contract.

### P2 — dependency advisory backlog (reported, unchanged)

[Frontend dependencies](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/ui/package.json:18) and [development dependencies](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/ui/package.json:24). Current `npm audit` reports 13 affected dependency entries: 11 high, one moderate and one low. This backlog predates Phase 3B; dependency files are unchanged by this branch's implementation and this review. The audit exits 1 because advisories exist. It is not proof of an exploitable Phase 3B path: the app uses browser Axios, fixed navigation links, and static production assets, while many advisories involve Node adapters, SSR, untrusted build inputs or dev servers. Full advisory triage and upgrades remain separate work. Keep live-test development servers local.

Raw audit output: [local JSON report](/private/tmp/phase3b-review-npm-audit.json).

### P2 — evidence computation precedes disabled/cache/quota checks (reported, unchanged)

[Explanation service entry](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/api/connect4/explanations.py:249). Full replay and tactical analysis happen before returning disabled, cached, duplicate or rate-limit responses. Boards/history are bounded, but paid-call quotas are not CPU/request-flood protection. Repeated requests can consume application work even without provider access. An early lightweight session/revision/configuration check would be a later efficiency improvement; preserving current input/session error semantics matters.

### P2 — stale deployment status text (reported, unchanged)

[Deployment guide](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/docs/deployment.md:7) still says Phase 3B is unimplemented, conflicting with current implementation documents. This does not affect runtime behavior.

## Verified controls and grounding boundaries

| Area | Review result and evidence |
| --- | --- |
| Key isolation/defaults | Backend key used only in provider Authorization. No config/key returned in gameplay or explanation results. Sentinel regression checks prompt/results; production frontend build with a synthetic backend key contains no sentinel. Default configuration is false/empty. Built image has no root `.env` and starts key-free/disabled. |
| Repeat paid calls | No gameplay-triggered generation, provider retry, alternate model or frontend automatic retry. Successful cache reuse is detached. Failures and discarded stale results consume attempts. Explicit repeated failures can cost money until caps are reached; cancellation does not undo a provider request. |
| Cross-session spending | Per-game allowance resets with a new session, but client and global counters survive session creation/replacement within the process. [Atomic reservation](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/api/connect4/explanations.py:259) prevents quota races; eight simultaneous requests across games yield exactly one provider attempt with a one-attempt client or global allowance. Distinct network addresses still cannot bypass the global allowance. |
| Duplicate/concurrency gates | One in-flight call per game, global maximum two. Eight simultaneous duplicates reserve one attempt, and failures release the permit. Completed cached responses can be returned without a paid call. Locks are released before provider I/O. |
| Client identity | [Endpoint](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/api/connect4/__init__.py:162) uses `request.remote_addr`; no ProxyFix or client-supplied forwarded-IP parsing was introduced. Spoofed X-Forwarded-For does not evade the tested client quota. Proxies can aggregate visitors; their topology has not been changed or live-verified. |
| Evidence/session binding | [Evidence capture](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/api/connect4/evidence.py:16) requires a live bearer game ID and current revision, captures under the game lock, then replays history on a detached snapshot. Game IDs authorize possession-based access; they do not authenticate a user identity. Client board/history fields are rejected. |
| All three modes | `position` uses actual current board; `last_move` uses the recorded player/agent/column and post-hoc transition, including no-history/terminal cases; `what_if` simulates a legal move for the current side on detached boards without changing board, revision or history. |
| Unsupported claims/injection | [Selection validation](/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab/api/connect4/explanations.py:195) accepts only exact fields, bounded unique IDs from supplied evidence. Question data cannot expand the renderer's claim vocabulary. Arbitrary prose, fabricated moves, formal applications and model citations are rejected in all three modes. All essential facts are retained even if omitted by the model. |
| Allis provenance/agent signals | Display text/citations come from deterministic facts and the curated knowledge catalog, never model prose. Allis rule applications remain empty and retrieved rules reference-only. No Random/Negamax search trace is recorded; limitations explicitly identify post-hoc analysis and unavailable intent. |
| Staleness/cache | Final server revision/existence recheck discards changed/replaced results. Client generation/cancellation and game/revision/mode/column validation prevent late rendering; move initiation, restart, recovery and navigation clear stale explanations. Cache identity includes game, revision, mode, column, normalized question and model; new regressions exercise each dimension. |
| API/errors | Strict fields/types, 4096-byte body cap, question/column/revision validation, current-session checks even on cache, stable sanitized error codes and no-store responses. Malformed/refused/incomplete provider results are errors, with attempts counted and gameplay unchanged. |
| Provider compatibility | Standard-library HTTPS, not an OpenAI SDK. Default `gpt-4o-mini`, `/v1/responses`, `text.format` JSON Schema, required fields, enum arrays, item bounds and `additionalProperties: false` align with the [official model documentation](https://developers.openai.com/api/docs/models/gpt-4o-mini) and [Structured Outputs contract](https://developers.openai.com/api/docs/guides/structured-outputs). This is documentation/transport validation, not a live account/model-access check. |
| Deployment/runtime | API Docker builder installs `requirements-api.txt`, including python-dotenv, and copies its virtualenv into runtime. Runtime imports and pip check passed. CMD retains one Gunicorn worker/four threads; image health passed with local Render-style PORT=10000. Nginx/same-origin and separate-origin exact-CORS browser flows passed. No Render configuration was changed. |

## Changes made during this review

- Provider deadline, HTTP interruption handling, response/connection/timer cleanup.
- Frontend hypothetical-column binding.
- Seventeen additional backend cases for deadlines, simultaneous client/global quotas, duplicates, failure release, unsupported claims in every mode, cache dimensions, disabled defaults and key isolation.
- One additional controller test and one additional browser test; fixtures updated to reflect the real column field.
- Explanation documentation updated for standard-library transport and the deadline/DNS boundary.
- This review report. P2 issues were not repaired or expanded into a refactor.

## Verification results

| Verification | Result |
| --- | --- |
| Original baseline | 188 backend tests and 25 frontend tests passed before edits. |
| Final backend | 205 passed on host Python 3.11; 205 passed in an ephemeral Linux production-image container. Only that disposable test container received pytest. |
| Final frontend controller/config | 26 passed. |
| Chromium | 14 passed against Docker/Nginx; 14 against separate-origin built frontend/API; 14 against Vite development (42 executions). Includes complete games against Random and Negamax. Vite used a temporary local proxy configuration and a deliberately unused remote API variable. |
| Real disabled browser smoke | Unintercepted local API returned 503/explanations_disabled, and gameplay remained enabled. |
| Actual Gunicorn concurrency | Production image, one worker/four threads, HTTPS-blocked temporary mock provider: two slow explanations, extra call rejected, concurrent move completed in ~2 ms, both timeout permits released, next explanation succeeded. |
| Builds | Default, separate-origin and Render-mode Vite builds passed. Render-mode build was not served or used for remote requests. Local Compose built/started with Docker's Node 22 and API Python 3.11. |
| Runtime/security checks | Runtime dependency imports and pip check passed; no root `.env`; explicit disabled/key-free review containers; backend sentinel absent from browser assets; local PORT=10000 health returned OK. |
| Regression strength | Three timeout regressions and hypothetical-column controller regression fail against original code and pass after fixes. |
| Diff hygiene | `git diff --check` passed. No commits. |
| Dependency audit | 13 advisory entries; nonzero exit expected. No dependency upgrades. |

All provider behavior was mocked. Slow-drip tests use a socket pair and real HTTP parsing, not a provider connection. Temporary mock runtime explicitly forbids HTTPS. Browser servers/API targets were local; paid model requests were never made.

## Remaining limits and local live-test decision

A limited local live test can now proceed after the user reviews the uncommitted diff: use a dedicated project/key, game/client/global allowances of three, concurrency one, 400 output tokens and 20-second timeout, with no restart during the test. The documented three-mode/cache procedure is suitable. Failures count toward the three attempts. This review does not enable or authorize those calls itself.

Live schema acceptance, actual account access, provider latency, refusal frequency and useful model ordering still need observation. The model orders/selects trusted content; it does not author unrestricted explanations or reveal an agent's actual rationale.

In-memory games/quotas reset on process restart, and multiple workers/instances multiply allowances. Fixed-window boundaries permit adjacent-window bursts. These are application request-count limits, not durable dollar budgets. Provider alerts/project budgets must be assessed separately for actual enforcement. The public API has no user authentication; bearer IDs and CORS are not substitutes for it. Proxy identity aggregation, generic request flooding and dependency advisories remain operational considerations before public enablement. DNS resolution can exceed the socket deadline; browser aborts do not cancel a paid call already sent. Docker verification used the local native Linux architecture, not a new Render amd64 deployment.

Stop for review; do not commit yet.
