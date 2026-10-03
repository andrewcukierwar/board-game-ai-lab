# Phase 4A.2 — Public Connect 4 MCTS integration

October 3, 2026. Implemented locally on `phase4a2-mcts-integration`, pending review,
manual acceptance and deployment. Freshly fetched `origin/main` was
`c213ddfbe2e39ec29a5f4f013d5a49b800643e88`, the completed Phase 4A.1 reference.
The starting branch was `phase4a-mcts` and its working tree was clean. The new
branch starts at current remote main; the older local `main` was left intact.
No commit, push, workflow, deployment, credential or Render setting change occurred.

The user subsequently confirmed Phase 3 explanations are enabled and manually
tested publicly. This supersedes the earlier documentation's pending-rollout
status; it is user confirmation, not a new production inspection. All verification
here used mocked explanations or a disabled/key-free backend; no paid calls occurred.

## API and UI contract

`POST /v1/connect4/start_game` accepts MCTS in either `player1` or `player2`:

```json
{
  "player1": {"type": "human"},
  "player2": {"type": "mcts", "simulation_limit": 100}
}
```

- `simulation_limit` may be omitted and normalizes to 100 in returned `players`.
  When supplied it must be a JSON integer in **50, 100, 250**. Boolean, float,
  string, null, nonpositive, unsupported counts and extra fields return
  HTTP 400 `invalid_agent`, before replacing any existing game.
- Only MCTS accepts `simulation_limit`; MCTS rejects `depth`. Human and Random
  accept only `type`; Negamax accepts only `type` and optional integer `depth`
  1–4 (default 2). Start-game defaults remain Human / Negamax depth 2.
- AI requests remain `POST /v1/connect4/make_move` with `game_id` and current
  integer `revision`, omitting `column`. Human turns require `column` 0–6.
- The public UI always starts with the human first. MCTS options are
  **Quick — 50 simulations**, **Standard — 100 simulations (default)** and
  **Deeper — 250 simulations**. Negamax keeps its separate depth selector.
  Selection changes only take effect on start/restart. Only the selected agent's
  fields are sent. The existing explanation panel, navigation cleanup, disabled
  controls, board reconciliation and explicit retry flow are preserved.

## Algorithm and resource boundaries

Phase 4A.1's algorithm is unchanged: UCT selection, expansion, random rollouts,
backpropagation from each node's previous-player perspective, and final selection
by child visit count. Immediate root guards take a win or exclude moves allowing
an immediate opponent win when an alternative exists. Finite random rollouts can
still miss deeper tactics; no perfect-play or measured-strength claim is made.
The standalone default of 1,000 is unchanged; the API explicitly passes its public
preset and creates a fresh tree per move.

A module-level `BoundedSemaphore(1)` is shared across Flask app instances in a
process. MCTS reserves it nonblockingly after revision/terminal/turn validation
and before constructing the agent. If occupied, another game receives:

```json
{
  "code": "agent_busy",
  "error": "Another MCTS search is running. Retry the AI move shortly."
}
```

The status is **503**. That request does not change board, revision or history.
The usual session access timestamp may refresh. Same-game overlap continues to
return **409 `game_busy`** through the existing per-game lock. The reservation is
released in `finally` after search/output validation, including constructor or
search exceptions and invalid output. The per-game lock remains held through the
atomic board/history commit. Detached inputs and legal-output checks remain in
place. Search errors return the existing sanitized **503 `agent_failed`**.

Human, Random and Negamax do not acquire this reservation. Other Python work can
still compete for CPU; the guard is not a fairness/rate-limit queue or a memory
cap. Searches are synchronous and simulation-bounded, **not subject to a strict
wall-clock deadline**. Aborting a browser request does not cancel an active server
search. Preserve **one instance, one Gunicorn worker, four threads** on Render Free.
Each additional process would have an independent reservation. No external queue,
Redis, service or new dependency is introduced.

## History and explanation boundaries

MoveRecord is unchanged and records MCTS as `agent_type="mcts"` with no depth.
The session's normalized player configuration retains the simulation preset;
individual history records do not add simulation statistics. Complete histories
are deterministically replayed and checked against boards, turns, revisions,
agent types and outcomes. MCTS games pass this existing contract, including
terminal games and corruption rejection.

The explanation prompt and limitations explicitly include MCTS. Evidence remains
verified board consequences; no actual MCTS tree, search trace, UCB scores, hidden
intent or Allis-rule use is supplied or inferred. All three explanation modes
pass with mocked selections. MCTS-NN, DQN and VictorAgent remain experimental and
unavailable through the API; Mancala and neural code are unchanged.

## Verification

- **352 backend tests passed** in 6.20 seconds with Python 3.11.17, versus the
  Phase 4A.1 baseline of 311. Coverage includes all presets/defaults, strict
  rejection, both positions, real complete games, full columns, wins/draws,
  detached inputs, stale revisions, independent sessions, history replay,
  constructor/search/output failures and resource release. Event-controlled
  concurrent tests verify the process-wide guard across games and app instances,
  plus unaffected Human/Random/Negamax moves.
- **35 frontend unit tests passed**, including six new MCTS cases covering all
  preset payloads, conditional controls, switching/restart and busy/lost-response
  recovery without automatic POST replay.
- Default Vite production build and Render-mode build passed. Render build used
  `VITE_API_BASE=https://board-game-ai-lab.onrender.com`; building performs no
  requests to that service.
- Local Docker Compose build/start passed with a healthy API. A separate
  **Linux amd64 production API image** built and passed all three public AI move
  smoke checks with networking disabled. Both native and amd64 checks asserted
  torch was uninstalled, rejected any attempted torch import, and found no
  `.pt`/`.pth` checkpoints in `/app`. Dependencies and Dockerfiles are unchanged.
- **16 Playwright/Chromium tests passed** against Docker/Nginx in 10.1 seconds,
  including a complete MCTS game at 50 simulations, restart/navigation, and all
  seven existing Phase 3 browser tests.
- `git diff --check` passed.
- Backend tests now forbid real HTTPS connection establishment and requests
  suite-wide. Provider transport tests use injected fakes/local sockets. Browser
  explanation tests intercept requests; the actual Compose API is disabled and
  key-free. No provider call was made.

The initial backend run had three new assertions looking for the wrong disclaimer
wording (348 passed, 3 failed); they were corrected to check the actual decision-
process/search-trace boundary. The initial browser attempt could not launch because
matching Chromium was absent; Chromium was installed in a temporary directory.
The installed-browser rerun and final diff validation passed as recorded above.

## Informal local observations

macOS arm64, Python 3.11.17. Each row summarizes three fresh processes with empty
boards and seeds 42/43/44. Elapsed time surrounds `choose_move` only (imports
excluded). Memory uses process peak RSS (`resource.getrusage`, bytes on macOS),
including the interpreter and imported engine dependencies; it is not tree-only
allocation or a server memory bound.

| Simulations | Search latency range | Median | Peak process RSS | Increase over pre-search peak |
| --- | --- | --- | --- | --- |
| 50 | 45.6–73.8 ms | 68.3 ms | 34.11–34.23 MiB | 0.11 MiB |
| 100 | 116.5–147.8 ms | 128.0 ms | 34.52–34.83 MiB | 0.19–0.21 MiB |
| 250 | 258.7–285.1 ms | 269.1 ms | 34.69–34.80 MiB | 0.46–0.47 MiB |

These small local samples are not representative of Render Free, midgame states,
concurrent traffic, cold starts, long-lived-process memory or playing strength.

## Remaining rollout checks

1. Review the local diff and manually play each preset; try switching/restarting,
   full columns, terminal play and explicit recovery after a busy/failed request.
   Local Compose explanations remain disabled; use mocks for explanation checks.
2. Obtain separate commit/push/deployment approval. Coordinate API deployment
   before frontend exposure of MCTS; an older API rejects the new selector type.
   Retain rollback image/commit references and account for session loss.
3. During an authorized rollout, verify deployed image identity, health, exact
   CORS, one instance/worker/four threads, all presets and complete public games.
   Observe actual Render latency/memory and simultaneous-visitor busy recovery.
4. Preserve existing public explanation settings and limits. Any new paid
   explanation smoke test requires its own approval; none is part of this task.
   Verify grounded MCTS wording with mocked responses before any paid test.

No outstanding correctness blocker is identified by local verification. Actual
Render capacity and public MCTS availability remain unverified until rollout.

## Changed files

| Files | Purpose |
| --- | --- |
| `api/connect4/__init__.py` | Strict preset normalization, explicit MCTS dispatch and process-wide reservation. |
| `api/connect4/explanations.py`, `games/connect4/grounding/context.py` | Include MCTS in the existing post-hoc analysis/intent limitations; no schema changes. |
| `ui/src/pages/Connect4.jsx`, `ui/legacy/connect4.js` | Selector, conditional presets, disabled states and agent-specific start payloads. |
| `tests/test_connect4_mcts_api.py` | 37 focused API/atomicity/concurrency/gameplay/replay cases. |
| `tests/test_connect4_evidence.py`, `tests/test_connect4_explanations.py` | MCTS immutable-history and mocked three-mode explanation coverage. |
| `tests/conftest.py` | Move live-provider blocking from one module to the entire backend suite and block connection establishment too. |
| `ui/tests/connect4.test.js`, `ui/e2e/connect4.spec.js` | Six MCTS unit cases and complete 50-simulation browser regression. |
| `README.md`, `project_plan.md`, `docs/quickstart.md` | Current phase status, public contract, limits and manual checks. |
| `docs/deployment.md`, `docs/llm-explanations.md`, `docs/phase3c-readiness.md` | Brief status notices superseding historical pending-public-explanation statements. |
| `docs/phase4a2-mcts-integration.md` | This verification and rollout handoff. |
