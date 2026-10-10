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
| Current phase | A — experimental design and preflight |
| Current experiment | none started |
| Last validated commit | `60b99b0` (starting SHA; no changes yet) |
| Last pushed commit | see `git log origin/research/mcts-v3` |
| Next exact action | Write and push `DESIGN.md`, then implement research-only variants |
| Incomplete work | everything after the audit |
| Blockers | none |

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
