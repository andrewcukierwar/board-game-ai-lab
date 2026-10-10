# MCTS v3 strength research — progress log

Durable checkpoint for the autonomous Connect Four MCTS strength-per-compute
study. Another agent must be able to resume from this file and GitHub alone.

## Fixed context

| Item | Value |
| --- | --- |
| Repository | `andrewcukierwar/board-game-ai-lab` |
| Branch | `research/mcts-v3` (dedicated worktree `board-game-ai-lab-mcts-v3`) |
| Starting SHA | `60b99b0d42145da149f433907585b809060c946d` (= `origin/main` at start) |
| Python | CPython 3.11.17, worktree-local `.venv` from `requirements-dev.txt` (no PyTorch) |
| Hardware | MacBook Pro, Apple M5, 10 cores, 32 GB |
| Benchmark lock | `$HOME/.cache/bgai-laptop-benchmark.lock` via `scripts/mcts_v3/with_benchmark_lock.sh` |

Out of scope and untouched: `main`, the `research/negamax-v3` worktree and
branch, AlphaZero worktrees, the external SSD, Mac Mini services, Tailscale,
Render, Docker deployment, live APIs, public MCTS presets (100/400/800), API
caps, provenance versions, `canonical-season-v1`, and all prior evidence
directories. The production `MCTSAgent` and `mcts_bitboard` are not edited.

## Status

| Field | Value |
| --- | --- |
| Current phase | B complete → C (pilots) |
| Current experiment | preflight (smoke + cost projection), then pilot 1 |
| Design | [DESIGN.md](DESIGN.md), pushed at `65f4211` before any strength game |
| Last validated commit | Phase B implementation commit (see log) |
| Last pushed commit | see `git log origin/research/mcts-v3` |
| Next exact action | `scripts/mcts_v3/with_benchmark_lock.sh .venv/bin/python -m scripts.mcts_v3.harness run --study preflight`, then declare and run `pilot1` |
| Incomplete work | pilots, confirmatory phase, report |
| Blockers | none |

## How to resume

1. `cd` to the worktree, confirm `git status` is clean and the branch is `research/mcts-v3`.
2. Create `.venv` if missing: `/opt/homebrew/bin/python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`.
3. Read this file's Status table and the last Log entry.
4. Every declared study lives in `docs/search-mcts-v3/<study>/` with `study.json`
   (the frozen plan), `results.jsonl` (one fsynced line per finished game),
   `run-log.jsonl`, and `analysis.json`. Resume any unfinished study with
   `scripts/mcts_v3/with_benchmark_lock.sh .venv/bin/python -m scripts.mcts_v3.harness run --study <study> --max-seconds 600`
   and analyse with `.venv/bin/python -m scripts.mcts_v3.analysis --study <study>`.
   A study refuses to resume if its pinned sources changed; declare a new one instead.
5. New studies are functions in `scripts/mcts_v3/studies.py`, committed before their first game.

## Log

### 2026-10-10 — session start, worktree verification

- `pwd` and Git root: `/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab-mcts-v3`.
- Branch `research/mcts-v3`, HEAD `60b99b0d42145da149f433907585b809060c946d`,
  clean tree, merge-base with `origin/main` is the same SHA.
- Worktrees: main checkout on `search/negamax-v2` (`19f9379`), this worktree,
  and `board-game-ai-lab-negamax-v3` on `research/negamax-v3` (other agent; not touched).
- Remote `origin` = `https://github.com/andrewcukierwar/board-game-ai-lab.git`.
  `origin/research/mcts-v3` did not exist yet; local upstream pointed at
  `origin/main`, so every push names the branch explicitly
  (`git push origin research/mcts-v3`) and never pushes to `main`.
- Benchmark lock directory absent at start (free).
- No `.venv` in this worktree; created one with `/opt/homebrew/bin/python3.11`
  and `requirements-dev.txt`. Other worktrees' environments were not modified.

### 2026-10-10 — audit of the existing MCTS

Read `mcts_agent.py`, `mcts_bitboard.py`, both v2 reports, the v2 strength
design, the evaluation/analysis/profiling scripts, and the MCTS tests.

- **Tree:** `Node` stores reward for the player who made the incoming move
  (`player_just_moved`); win 1, draw 0.5, loss 0. Maximising child UCB is
  therefore correct at every depth.
- **Selection:** UCB1, `c = 1.41`, unvisited children are infinite, ties
  broken through the injected RNG (one `rng.choice` per selection step).
- **Expansion:** one uniformly random untried move per simulation; legal order
  is centre-first `(3, 2, 4, 1, 5, 0, 6)`.
- **Rollout:** uniform random legal columns to the end of the game, on local
  bitboard integers; no tactical knowledge of any kind.
- **Backpropagation:** every node on the path gets one visit and its
  mover-perspective reward.
- **Root guards:** play an immediate win without searching; otherwise restrict
  root expansion to moves that do not allow an immediate opponent win (falling
  back to all moves when none are safe). Guards exist at the root only.
- **Final move:** most-visited root child, ties through the RNG.
- **Known profile (v2):** rollouts are 47–63% of time on broad early trees,
  selection 15–28%; about 80k simulations/s from the empty board.
- **Known strength curve (v2):** versus MCTS 400, budgets 800 / 2,000 / 5,000 /
  10,000 score 54.5% / 60.9% / 65.8% / 70.3%. Doubling simulations is worth
  roughly 4–5 points, which is the yardstick any slower-per-simulation variant
  has to beat.

Observed weaknesses that motivate the candidates: no tactics below the root,
rollouts that ignore one-move wins and blocks, no reuse of proven terminal
results in the tree, and an untuned exploration constant.

### 2026-10-10 — Phase A: design

`DESIGN.md` predeclares candidates (R1 decisive rollouts, R2 + gift avoidance,
S solver with tactical expansion, E centre-first expansion, C exploration
constant), the equal-simulation and equal-time comparisons, the development
(96) and held-out (256) opening sets, seeds, the cluster bootstrap, a fixed
family of 8 for the Bonferroni adjustment, decision gates, and compute caps.
Pushed at `65f4211` before any game.

### 2026-10-10 — Phase B: research-only implementation

New files (production agent and bitboard module untouched):

- `games/connect4/agents/mcts_research_agent.py` — `ResearchMCTSAgent`,
  `ResearchConfig`, `ResearchNode`, `winning_cells`, `tactical_rollout`.
  Not imported by the agent factory or the API.
- `scripts/mcts_v3/harness.py` — openings, study declaration, serial resumable
  runner with pinned source hashes.
- `scripts/mcts_v3/analysis.py` — opening-cluster bootstrap, sign-flip test, timing.
- `scripts/mcts_v3/studies.py` — declared studies. `scripts/mcts_v3/throughput.py` — fixture latency/memory.
- `docs/search-mcts-v3/openings.json` — frozen dev / holdout / preflight / empty sets.
- `tests/test_connect4_mcts_research.py` (119 tests), `tests/test_mcts_v3_harness.py` (7 tests).

What the tests establish:

- Default configuration reproduces the production agent exactly: same move,
  same final RNG state, same complete tree fingerprint, on 5 fixtures × 4
  budgets × 2 seeds.
- `winning_cells` agrees with four-detection on every empty cell of random
  positions from 4 to 36 plies.
- Both tactical rollouts match an independent array-engine statement of the
  policy in outcome and RNG consumption (84 positions × 6 seeds each).
- Solver: every proven node in the tree, at any depth, matches an exhaustive
  memoised solve of late positions; a proven root always plays a move that
  achieves the exact value; a proven-lost move is never chosen when an
  alternative was searched; the search stops early once the root is proven.
- All configurations: legal moves, no caller mutation, seeded reproducibility,
  no tree retained between decisions, terminal rejection, and the production
  root guards (immediate win, forced block, floating-threat avoidance).

A real defect was caught by the rollout oracle before any game was played:
gift detection in R2 shifted unmasked sentinel-row bits back onto the board
and wrongly excluded top-row moves. Fixed by masking with `BOARD_MASK`.
