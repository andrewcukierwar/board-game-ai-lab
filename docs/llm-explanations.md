# Phase 3B: Allis-grounded explanations

Implemented locally on `phase3b-llm-explanations`; the preceding `d35b9f0` documentation commit is preserved. Nothing has been pushed or deployed. The user completed initial local live testing before Phase 3B.1; development verification uses mocks and makes no paid model calls. Random/Negamax gameplay and existing game responses are unchanged. The feature is disabled by default.

## Files changed

- Backend: `api/app.py`, `api/connect4/__init__.py`, `api/connect4/state.py`, `api/connect4/evidence.py`, new `api/connect4/explanations.py` and `api/connect4/openai_provider.py`.
- Frontend: `ui/src/pages/Connect4.jsx`, `ui/legacy/connect4.js`, `ui/connect4/connect4.css`, new `ui/legacy/explanations.js`.
- Tests: new `tests/test_connect4_explanations.py`, `ui/tests/explanations.test.js`, `ui/e2e/explanations.spec.js`.
- Configuration: `requirements-api.txt`, new `.env.example`.
- Documentation: `README.md`, `project_plan.md`, `docs/allis-grounding.md`, `docs/quickstart.md`, new `docs/llm-explanations.md`.

## Request and response

`POST /v1/connect4/explain`, JSON, maximum body size 4096 bytes:

```json
{
  "game_id": "<ID from start_game>",
  "revision": 6,
  "mode": "what_if",
  "column": 3,
  "question": "Does this leave an immediate reply?"
}
```

- `game_id`: nonempty string, at most 64 characters, identifying a live session.
- `revision`: required nonnegative integer; booleans are rejected. Must match the current session.
- `mode`: `last_move`, `position`, or `what_if`.
- `column`: required integer 0–6 only for `what_if`. The UI labels these columns 1–7. Full columns and terminal positions cannot be simulated. A simulation always uses the current side to move, including an AI side when requested directly.
- `question`: optional string, default empty, maximum 500 characters by default. Whitespace at its ends is stripped for cache identity. It guides emphasis within the available evidence, rather than providing a general chat interface. Unknown fields are rejected.

Success (HTTP 200) has `game_id`, `revision`, `mode`, `column` (null outside `what_if`), `cached`, and `explanation`:

- `summary`: trusted concise paragraph, selected `focus_id`, supporting `fact_ids`, and deterministic analyzer `evidence_paths`.
- `key_facts`: at most three verified observations related to the selected focus; model ordering can change their emphasis.
- `relevant_squares`: verified named-square/column/matrix-row objects from the selected relationship. These are visual markers, including hypothetical coordinates, and do not modify the board.
- `facts`: complete records `{id, classification: "confirmed_tactical", text}` derived exclusively from deterministic evidence.
- `strategic_context`: selected curated concepts with `concept_id`, `classification: "context_only"`, `title`, `text`, a position-specific `connection`, `preconditions`, `limitations`, and `source` containing the thesis title, URL and verified reference records. Each reference includes chapter/section, thesis page range, 1-based PDF range and zero-based indices.
- `additional_context`: selected reference material without a verified connection to the primary focus, available in detailed analysis; formal rules remain reference-only.
- `supported_allis_rule_applications`: always `[]`.
- `limitations`: explicit post-hoc/agent-intent, horizon, formal-rule and question limitations.

The revision refers to the actual game, including for `what_if`. A simulation neither appends history nor advances that revision. `last_move` identifies the actual played column, player/agent and resulting position; before any move it explicitly says there is no last move. Terminal positions can be analyzed and their actual last move explained.

Errors retain the existing `{error, code}` contract:

| HTTP | Codes |
| --- | --- |
| 400 | `invalid_request`, `invalid_game_id`, `invalid_revision`, `invalid_mode`, `invalid_move`, `invalid_question` |
| 404 | `session_not_found` (including expiry/replacement) |
| 409 | `stale_revision`, `game_busy`, `game_over` (hypothetical only), `explanation_busy` |
| 413 | `request_too_large` |
| 429 | `explanation_game_limit`, `explanation_rate_limited`, `explanation_capacity` |
| 502 | `invalid_explanation` (invalid selection, malformed/incomplete response, refusal, oversized output) |
| 503 | `explanations_disabled`, `explanation_unavailable` (missing key, unavailable model, provider rejection/outage) |
| 504 | `explanation_timeout` |

Session and input errors are checked even when disabled or when a matching cached result exists. Responses use `Cache-Control: no-store`; successful results are cached internally only. No provider message, API key, stack trace or sensitive configuration is returned to the client. Failed explanations never alter gameplay or call its recovery/move endpoints automatically.

## Grounding and LLM boundary

`get_explanation_context()` captures the game under its existing lock and performs full Phase 3A replay/deterministic analysis after releasing it. For `what_if`, `analyze_move()` and `analyze_position()` run on detached boards. The prompt contains one relevant board, verified readable facts, four recent move records without board snapshots, verified explanatory relationships (with fact IDs and analyzer paths), selected catalog entries, coordinate conventions and explicit unknowns/horizon limitations. No full historical board list, game ID, key or sensitive configuration goes into model content.

Phase 3B.1 retrieval uses verified relationships rather than generic tactical/parity keywords: accessible or inaccessible winning squares link to `winning_square`; mandatory defense and exhaustive next-reply losses link to `tactics`. Quiet openings have no relevant Allis concept in the default view. At most two explicitly named catalog terms in a question can add reference material to details, never applicability. No formal detector, compatibility/coverage check, Zugzwang-control proof, search trace or game-theoretic evaluation is added.

The single provider response now has exactly `focus_id`, `fact_ids` (1–3), and `concept_ids` (0–1). Strict JSON Schema restricts all values to supplied IDs. The model chooses the verified relationship that answers the mode/question and orders its supporting observations. Multiple candidates can represent different immediate wins, inaccessible completion squares, or risky alternatives; decisive wins, mandatory defense and unavoidable next-reply consequences take precedence over quiet context. The backend validates exact fields, types, lengths, uniqueness and membership, then checks the relationship between the selected facts/concepts and focus. Unrelated allowed facts are omitted from key evidence; omitted supporting evidence is restored. Unrelated selected concepts go to details, with a deterministic relevant-concept fallback where necessary. No arbitrary text, inferred sequence or model-authored citation is accepted.

Summary wording, facts, square coordinates, concept connections and citations all originate in deterministic analysis or the existing curated catalog. The model is an editorial selector, not a new tactical analyzer. Its selection now affects the visible primary paragraph and key evidence, rather than merely reordering a report containing every fact. On positions with only one meaningful focus its contribution remains small. Unsupported questions still cannot receive a novel freeform answer. Full facts remain in **Detailed analysis**, while integrity disclosures, reference preconditions and limitations remain in **Methodology and limitations**. Relevant square labels clear on a new request, board change, move initiation, restart and cleanup.

Questions are a JSON `question_untrusted` field in a user message; fixed grounding constraints are separate instructions. An injection can at most influence selections among verified records. Allis concepts are labeled contextual; formally unimplemented rules can only appear as reference material in details. Random and Negamax are never described as internally using Allis rules. Sources retain chapter/section and verified thesis/PDF pages from the Phase 3A catalog.

The provider uses Python's standard-library HTTPS client (no OpenAI SDK dependency) for one request to the fixed OpenAI Responses endpoint, strict `text.format`, `store: false`, a bounded response body (64 KiB), socket timeouts and a deadline watchdog for request/response I/O, with no automatic retries or alternate models. The watchdog interrupts slow header/body streams and explicitly closes both the response and connection. Connection setup consumes the deadline, but OS DNS resolution cannot be interrupted by this socket watchdog; TCP/TLS setup also uses socket timeouts. See the [official Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs) and [GPT-6 Luna model contract](https://developers.openai.com/api/docs/models/gpt-6-luna). Parsing rejects incomplete output and refusals as well as malformed JSON. `ExplanationService.provider.generate()` and the provider connection factory are independently mockable.

No game/session or service lock is held during network I/O. The server rechecks session existence/revision after a successful provider response and discards stale/replaced results. The response still binds to its captured revision; a move can race after the final check, so clients must compare revisions too. The panel additionally cancels on move initiation, restart or navigation, ignores late responses, and clears prior explanations when the board changes. Browser cancellation does not cancel an already-running provider request; its attempt still counts and may cost money.

The existing client is standard-library HTTPS, so there is no SDK version to upgrade or install. For `POST /v1/responses` the request sends `"reasoning": {"effort": "none"}` alongside `text.format`; the flat `reasoning_effort` parameter belongs to Chat Completions and is not sent. [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) documents support for the six configured efforts and Structured Outputs. The strict schema now includes one enumerated relationship ID alongside the bounded fact/concept arrays.

The [reasoning API contract](https://developers.openai.com/api/docs/guides/reasoning) counts reasoning tokens within `max_output_tokens`. The default `none` avoids spending the small 400-token allowance on reasoning. Higher efforts may exhaust the existing 64–2000 token cap or take longer than the existing provider deadline; neither cap is raised automatically. Incomplete output (including a `max_output_tokens` stop), refusal or invalid selections remain sanitized errors, count as attempts, release the permit and are never cached or displayed. Completed output may contain reasoning items; those are ignored while the single structured message is parsed and validated. No reasoning text is exposed. Arbitrary model overrides must support the selected effort; provider rejection remains a safe unavailable error with no retry or fallback.

## Configuration and safeguards

Copy `.env.example` to an ignored root `.env` for host development; Python loads it with `python-dotenv` without overriding exported environment variables. Never commit credentials or put them in `VITE_*` variables. Docker/hosted services receive configuration through their backend runtime environment; the image excludes `.env`, and the default Compose setup remains key-free and disabled. For opt-in local Docker work, use a local untracked Compose override to set `api.env_file: [.env]`. Do not alter public settings until rollout is separately approved.

| Backend variable | Default | Purpose |
| --- | --- | --- |
| `EXPLANATIONS_ENABLED` | `false` | Explicit opt-in (`true`/`1`, `false`/`0`). |
| `OPENAI_API_KEY` | empty | Backend project key; required only when enabled. |
| `OPENAI_EXPLANATION_MODEL` | `gpt-6-luna` | Backend-only model; choose one supporting Responses, Structured Outputs and the configured effort. Availability is not live-verified. No fallback. |
| `OPENAI_EXPLANATION_REASONING_EFFORT` | `none` | Backend-only effort: `none`, `low`, `medium`, `high`, `xhigh`, `max`. Invalid values fail startup. |
| `EXPLANATION_MAX_OUTPUT_TOKENS` | 400 | Total generated-token cap, including reasoning and visible output, configurable 64–2000. |
| `EXPLANATION_TIMEOUT_SECONDS` | 20 | Socket timeout and request/response deadline, configurable 1–60 seconds; OS DNS caveat above. |
| `EXPLANATION_MAX_QUESTION_LENGTH` | 500 | Character cap, configurable 1–1000; current UI caps at 500. |
| `EXPLANATION_GAME_LIMIT` | 8 | Total reserved provider attempts over this session's lifetime, including failures/stale results. |
| `EXPLANATION_CLIENT_LIMIT` | 20 | Attempts per network client per time window, across games. |
| `EXPLANATION_GLOBAL_LIMIT` | 100 | Total attempts per process per time window. |
| `EXPLANATION_WINDOW_SECONDS` | 3600 | Fixed-window duration; a boundary can permit two adjacent windows' allowances. |
| `EXPLANATION_MAX_CONCURRENT` | 2 | Concurrent provider requests, hard upper bound 2 to preserve gameplay threads. |
| `EXPLANATION_CACHE_SIZE` | 256 | Bounded LRU successful-result entries. |
| `EXPLANATION_CACHE_TTL_SECONDS` | 1800 | Successful result reuse duration. |
| `EXPLANATION_CLIENT_CAPACITY` | 1024 | Bounded active client counters; reject new clients when full until counters expire. |

Limits are reserved atomically before I/O. Only explicit button/API requests call the model; gameplay never does. Simultaneous requests for the same game (identical or different) return `explanation_busy`, without a second call. Global concurrency also rejects excess calls. Cache identity includes game, revision, mode, hypothetical column, normalized question, model and reasoning effort. Identical successes reuse a detached cached result without another paid attempt, after validating that the session is still current. Failures are not cached and consume attempts. Exhausting an explanation allowance never stops gameplay.

Network client identity uses `request.remote_addr`; untrusted `X-Forwarded-For` is ignored. Behind a proxy this may aggregate visitors under the proxy address. Do not enable arbitrary forwarded-header trust; review the hosting platform's trusted-proxy topology before changing this. The global cap still bounds calls if client identities vary or users create new sessions. Limits are request counts, not exact dollar accounting; model price/input length affect actual cost.

All limits, caches and games are in memory. Restart resets them. Keep one worker/instance; multiple processes multiply independent allowances and do not share sessions. New/replaced sessions get new game allowances, but share client/global limits. Abandoned client counters expire lazily, and expired cache entries are removed on explanation requests. These safeguards are a bounded MVP, not durable abuse prevention or authentication.

Use a **separate OpenAI project/API key**, restrict model access as appropriate, and configure an **external spending alert and project budget** before enabling anything public. Confirm the provider's actual budget enforcement: an alert/budget may not be a hard stop. Application throttling cannot replace provider-side spending controls. No database, queue, PDF ingestion, embeddings or additional service is required.

## Automated verification

All provider behavior is mocked; explanation tests additionally forbid live HTTPS requests. Run:

```sh
python -m pytest -q
cd ui
npm test
npm run build
```

Initial implementation verification on October 2, 2026: **188 backend tests, 25 frontend tests and 13 Chromium tests passed**. Browser tests used the production-built frontend at port 4173 against the disabled-provider local API at port 8001, with exact-origin CORS. Default, separate-origin and Render-mode production builds passed. `git diff --check` passed. An initial browser run caught a React unmount/listener-cleanup issue; the fix and a focused DOM-removal regression test are included, and the full browser suite subsequently passed. No provider API calls, deployments or public-site tests were performed.

Final reviewed configuration verification on October 2, 2026: **224 backend tests passed on macOS and Linux, 26 frontend tests and 42 Chromium executions passed** (14 each against Docker/Nginx, separate-origin hosting and Vite). Default, separate-origin, Render-mode and local Docker builds passed. Runtime defaults were verified as `gpt-6-luna`/`none`, disabled and key-free; model/effort/key settings were absent from frontend assets. All provider responses were mocked; the real disabled-endpoint browser smoke also passed. Deadline, exact hypothetical-column binding and quota regressions remain green. Detailed findings and remaining P2 issues are in [the independent review](phase3b-review.md).

The backend tests cover all modes, actual AI identity, absent history, terminal/illegal hypotheses, strict revisions/inputs, full columns and immediate replies, detached evidence/cache values, session isolation/replacement/expiry, prompt injection, unsupported outputs, missing keys/disabled configuration, timeout/refusal/incomplete/malformed/unavailable provider responses, rate limits/cache expiry, duplicate requests, concurrency and gameplay during provider I/O. Frontend tests cover explicit request payloads, legal columns, structured/source display, loading/error/retry, terminal controls, invalid responses, cleanup, cancellation and stale results.

For browser tests, run a local API with `EXPLANATIONS_ENABLED=false` and `OPENAI_API_KEY` empty. Build the UI against it and serve locally. Every browser explanation request, including disabled-provider failures, is intercepted with fixtures; real endpoint configuration errors are covered by backend tests. Existing complete-game Random/Negamax tests remain part of the suite. The Playwright configuration rejects non-local targets.

## Proposed manual test — requires separate approval, maximum three live calls

1. Review this implementation first. In a separate OpenAI project set a small budget/spending alert, confirm model access, and create a dedicated key. Use only a local backend, never the public app. Enter the key privately in ignored `.env`; do not paste it into chat or browser tools. Set `EXPLANATIONS_ENABLED=true`, `OPENAI_EXPLANATION_MODEL=gpt-6-luna`, `OPENAI_EXPLANATION_REASONING_EFFORT=none`, game/client/global limits to **3**, output cap **400**, timeout **20**, concurrent cap **1**. Start fresh so the global allowance begins at zero; do not restart until testing is complete.
2. After explicit approval for up to **three** live calls, request **Analyze Position** once, then repeat exactly the same request and check `cached: true` with no extra provider usage. Confirm facts, limitations, schema acceptance, citations and actual usage in the provider dashboard. A failed request consumes one of the three attempts; do not automatically retry it.
3. Play one human/AI turn locally (no model call), request **Explain AI Move** once, and verify the recorded column/piece/result and post-hoc caveat. Use the final allowance for one legal **What If?**; compare against a manually inspected detached result and verify board/revision did not change. Any malformed/unavailable response should remain a safe error; stop to review it instead of adding calls.
4. Disable explanations and stop the local servers. Review project usage/spend, revoke the temporary key if appropriate, and keep all credentials out of tracked files. No deployment or provider request is authorized by these instructions alone.

The user reports the original three-mode local integration works. The revised Phase 3B.1 schema and model selection quality have not been live-tested; historical provider latency/usage logs are unavailable. Advanced agents, Mancala and multiworker persistence remain deferred. See [the Phase 3B.1 audit, examples, inventory and verification](phase3b1-quality.md).
