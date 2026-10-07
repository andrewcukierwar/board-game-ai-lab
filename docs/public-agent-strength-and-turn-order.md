# Public agent strength and turn order

October 7, 2026. Work branch: `public-agent-strength-and-turn-order`, based on
remote `main` at `1d3f12d` (the local main ref was stale). No commit, push, merge,
rebase, cherry-pick, deployment or provider call was performed. The AlphaZero
preflight branch and frozen campaign artifacts were only inspected read-only.

## Decision

**GO for local review; production rollout remains conditional on Render smoke
verification.** Negamax UI: **1 / 2 / 4 / 6 / 8**, default **2**; API accepts every
integer **1–8**. MCTS UI: **100 Quick / 400 Balanced / 800 Deep**, default **100**.
API accepts only **50 / 100 / 250 / 400 / 800**; 50/250 remain for existing clients.
Booleans, floats, strings, null, unknown settings and unbounded budgets are rejected.

Previous public settings: Negamax integers 1–4, default 2; MCTS 50/100/250,
default 100; frontend human-first only. No MCTS simulation-depth control was added.

## Corrected Negamax

The read-only Phase 4 reference and its oracle/cutoff tests on
`phase4d3b-alphazero-v2-preflight` established the legacy unsafe-cache and terminal
heuristic defects. The public implementation now uses fresh per-decision
EXACT/LOWER/UPPER entries keyed by both piece bitboards, mover and remaining
depth. Cutoff bounds are never treated as exact under another window.

A terminal win/loss is ±(1,000,000 + remaining depth) from the current mover's
perspective, checked before depth-limited leaves; draws are exactly zero. This
prefers faster wins and slower losses. The existing 1/3/9 open-window heuristic
is retained for nonterminal leaves (69 windows; even the 81 four-piece weight
cannot approach the terminal magnitude). Center-first move ordering and exact
tie selection are deterministic. Every legal root move receives a full-window
value, with one shared, bound-typed table per decision.

Compact seven-bit-per-column bitboards, shift-based four detection and popcounts
reduce Python search overhead without changing rules or the heuristic. Independent
array/engine minimax tests check every legal root value, including the Phase 4
cache reproduction, seeded legal positions, immediate wins/blocks, both terminal
perspectives, draws, transpositions, lower/upper cutoffs, table reuse and exact ties.
Caller boards are detached and play/undo cleanup is protected by `finally`.
The standalone MCTS algorithm file is unchanged.

## Local benchmark method

Hardware: MacBook Pro Mac17,2, **Apple M5**, 10 CPU cores (4 performance / 6
efficiency), **32 GB RAM**. Runtime: CPython **3.11.17**, native arm64, macOS
**26.6.2**. Node **25.9.0** was used for frontend verification. No GPU or neural
model is involved. Timings exclude HTTP/network and service wake-up.

Run `python -m scripts.benchmark_public_agents --output /tmp/public-search.json`
from the repository root. Each row has one discarded warm-up and three fresh-agent
samples; tables below show **median / maximum milliseconds**. Negamax algorithm
and source stayed identical across all depth measurements, including a second
complete run (first-run worst depth-8 218 ms, depth-10 1,560 ms). MCTS repetitions
use different fixed seeds 701/702/703; repeatable by rerunning the same seeds.
Every returned column was checked legal; Negamax moves and node counts repeated
exactly. The [raw samples](public-agent-benchmarks.json) retain legal columns,
node/table/hit/cutoff counts, histories and implementation SHA-256 hashes.

Five fixed legal positions: empty board (7 legal root moves); near-opening
`[3,2,4,3]` (7); midgame after 16 plies (7); a second, varied 16-ply midgame with seven safe root replies (7); late after 30 plies (3). Histories are
in the raw file. The first midgame has immediate losing opponent replies after every root move; the second avoids that tactical shortcut. Supplemental midgame rows were measured separately, with the same algorithm hashes and idle test servers. These are a small latency sample, not a worst-case bound, p95
estimate or playing-strength tournament. MCTS performs its full budget on these
positions, rather than exiting via an immediate root win.

### Negamax

| Depth | Empty | Near-opening | Midgame (tactical) | Midgame (varied) | Late |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 2.5 / 2.5 | 2.9 / 2.9 | 1.7 / 1.7 | 2.9 / 3.0 | 0.1 / 0.1 |
| 6 | 17.5 / 17.6 | 26.0 / 54.9 | 12.3 / 12.4 | 12.5 / 12.6 | 0.3 / 0.3 |
| 8 | 103.3 / 105.1 | 216.9 / 218.0 | 58.1 / 58.3 | 43.2 / 43.4 | 0.5 / 0.5 |
| 10 | 530.3 / 530.9 | 1546.6 / 1548.1 | 193.2 / 194.4 | 114.1 / 115.6 | 1.0 / 1.0 |

Depth 8's broad near-opening case visits 80,742 nodes and keeps 22,621 table
entries; depth 10 visits 646,784 nodes and keeps 178,705 entries. Depth 10's
roughly 1.55-second local latency and much larger CPU/table footprint leave too
little confidence for a smaller shared Render instance. Depth 8 is the chosen
cap; it is about seven times faster on that case. Depth 10 is rejected despite
being measurable locally. Default depth 2 stays unchanged.

### MCTS

| Simulations | Empty | Near-opening | Midgame (tactical) | Midgame (varied) | Late |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 | 45.2 / 45.6 | 33.7 / 35.3 | 23.0 / 23.9 | 26.0 / 27.5 | 21.8 / 23.3 |
| 100 | 84.3 / 87.1 | 65.6 / 68.8 | 43.7 / 43.7 | 48.2 / 52.8 | 43.4 / 44.8 |
| 250 | 206.5 / 208.4 | 159.6 / 162.4 | 99.6 / 100.3 | 116.7 / 121.1 | 105.2 / 116.7 |
| 400 | 326.7 / 327.6 | 248.5 / 259.6 | 151.1 / 161.3 | 179.0 / 186.9 | 164.8 / 171.1 |
| 800 | 651.5 / 653.7 | 498.5 / 511.0 | 296.5 / 298.0 | 349.4 / 370.1 | 320.5 / 327.2 |
| 1000 | 814.0 / 819.5 | 619.7 / 637.4 | 365.2 / 368.8 | 434.3 / 458.2 | 395.8 / 399.7 |

800 simulations completes within 654 ms in the measured cases. 1,000 is locally
usable at up to 820 ms, but adds about 25% work without measured strength evidence.
The simpler 100/400/800 ladder leaves more production headroom. API legacy budgets
50/250 are also measured and retained. The process-wide reservation still rejects
competing MCTS requests immediately; no queuing or algorithm changes were made.

## Turn order and recovery

Accessible native radios offer **You go first** (default) and **AI goes first**.
Selection stays separate from the accepted game's player configuration; changes
apply only on Start new game. Payload ordering puts human/AI in player1/player2
or swaps them. Player 1/X/red always starts, Player 2/O/yellow always moves second.

Start accepts a validated revision-0 snapshot, then invokes the existing AI-turn
function once under the same synchronous busy lock, issuing a separate revision-0
move POST. Only a valid matching-game revision-1 result is accepted. HTTP failures,
lost responses and malformed/mismatched snapshots enter GET reconciliation;
failed GET leaves board and analysis locked behind Refresh game. An authoritative
AI turn offers explicit retry, which GETs again before any move POST. A committed
opener recovered at revision 1 enables the human without replay. Unmount aborts
both requests and ignores late responses. There is no render effect that can
issue a second opener. Snapshot validation also prevents accepting stale GETs,
wrong-game responses and unexpected mutation revisions.

Fixed assumptions: reducer and status winner copy; active-opponent index and
budget note; board ownership legend and hover color; page instructions and idle
message; homepage color ownership and MCTS budgets. Winner ownership now derives
from the winning player's configured type. Before a game, the legend describes
Player 1/red and Player 2/yellow rather than assigning pending selection to an
active board. Existing turn eligibility and last-AI-move analysis wording already
used player type correctly. Game controls and backend explanation semantics needed
no changes. Analysis tests cover Last Move immediately after the opener, Current
Position and What If, both orders and all three opponents, without board mutation.

## Verification

- Full current backend suite: `EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest tests -q -rs` — **455 passed**, **14 skipped** in 10.27 s. All public API, engine, grounding, explanation, Negamax and MCTS tests ran. Skips are optional PyTorch-dependent research modules; the API-only virtualenv has no torch. No claim is made that those optional modules were verified.
- From `ui/`, `npm test` — **127 passed**, no skips or failures.
- `VITE_API_BASE=http://127.0.0.1:8001 npm run build` — passed (production build used for browser tests).
- `VITE_API_BASE=https://board-game-ai-lab.onrender.com npm run build:render -- --outDir /tmp/public-render-build` — passed. Building the production origin made no network/provider requests.
- Full Playwright run — **67 passed** in 43.2 s: **55 Chromium** regressions, **6 Firefox** and **6 WebKit** layout/accessibility/gameplay smoke tests. Both player orders and keyboard side changes were verified in all three engines at 1440/1024/820/768/375/320 px. Chromium covers all AI openers, highest presets, failure/retry, lost response after commit, navigation, pending side changes and all analysis actions. Desktop and 320-px screenshots were visually inspected.
- Local browser API: fresh Gunicorn process on 127.0.0.1:8001, one worker/four threads, explanations explicitly disabled, provider key empty, exact CORS for localhost:4173. The pre-existing service on port 8000 was not changed. Browser explanations used deterministic fixtures; backend explanation tests used a fake provider plus the existing live-HTTPS prohibition.
- The revision-36 browser analysis regression now reaches the advanced board via authoritative stale-revision recovery, rather than fabricating a revision-36 start response. Its evidence/highlight/gameplay assertions remain intact.
- `git diff --check` — passed. Final agent SHA-256 hashes match all 50 recorded benchmark rows. MCTS source is unchanged. No frozen artifact or historical phase report was edited.

## Files changed

- Production: `api/connect4/__init__.py`, `games/connect4/agents/negamax_agent.py`.
- React gameplay: `ui/src/connect4/useConnect4Game.js`, `AgentSelector.jsx`, `GameBoard.jsx`, `GameStatus.jsx`; `ui/src/pages/Connect4.jsx`, `Home.jsx`; `ui/connect4/connect4.css`.
- Tests: `tests/test_connect4_negamax.py`, `test_connect4_api.py`, `test_connect4_mcts_api.py`, `test_connect4_explanations.py`; `ui/tests/connect4.test.js`; `ui/e2e/connect4.spec.js`, `explanations.spec.js`, `polish.spec.js`.
- Documentation and evidence: `README.md`, `docs/quickstart.md`, `docs/deployment.md`, this report, `docs/public-agent-benchmarks.json`, `scripts/benchmark_public_agents.py`.

The existing analysis hook/validation/components, game controls, MCTS algorithm,
backend move transactions, reservation and explanation semantics needed no changes.

## Production boundary

No Render performance claim is made. Synchronous CPU work can block or contend
across API threads; caps bound search work, not a strict wall-clock deadline.
Keep one worker/four threads and one instance. On an authorized future deployment,
measure warm and cold depth-8/800-simulation turns on the actual instance, both
player orders, failed/lost openers, simultaneous independent sessions, reservation
rejection and analysis after the opening. Reduce caps if necessary. Search depth
and simulation count alone do not prove a particular playing strength. Existing
initial-start-response-loss and process-local session-loss limitations remain.
