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
| Current phase | Complete: phases A–E, declared secondaries, and follow-ups A–C |
| Outcome | F1 (R2 rollouts + solver, c = 0.5) and F2 (R1 rollouts + solver) are accepted research improvements; see [REPORT.md](REPORT.md) |
| Design | [DESIGN.md](DESIGN.md): original at `65f4211`; amendments 1–3 and the finalist freeze each pushed before the games they govern |
| Last validated commit | the commit that adds `REPORT.md`; full suite 2,882 passed, 15 skipped; 130 focused tests; replay audit 440 games, 0 mismatches |
| Last pushed commit | `git log origin/research/mcts-v3 -1` |
| Production changes | none. `MCTSAgent`, `mcts_bitboard`, presets, API caps, provenance versions untouched |
| Incomplete work | none of the declared work. Open research items are listed under "Next research" in the report |
| Blockers | none |
| Next exact action | Human decision on a separately authorized integration phase for F1. If continuing research instead, start with a fresh development/held-out split and the positional-knowledge candidates in the report |

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

### 2026-10-10 — Preflight (excluded from inference)

`preflight/`: 384 smoke games in 63 s with no errors; fixture throughput in
`preflight/throughput.json`. Per-decision time versus the baseline at equal
simulations: R1 ×1.4–1.5, R2 ×1.7–1.9, S ×1.1, E and C ×1.0. Traced peak memory
is unchanged (0.21 MiB at 400, 1.03 MiB at 2,000). Negamax 4 / 6 / 8 medians
1.0 / 7.0 / 36 ms, so depth 8 is included as an opponent. Host note: a
pre-existing `BTLEServer` process holds roughly one core throughout (load
average ≈ 3.5–4); it predates this session and was left alone.

### 2026-10-10 — Phase C: pilots on the development set (exploratory)

All on the 96 development openings, 192 games per condition, under the lock.
Pilot search time 1,614 s of the 3,600 s cap. Raw games, plans and analyses are
in `pilot1/` … `pilot3e/`. Pooled over budgets 400 and 2,000, versus baseline:

| Configuration | Equal simulations | Equal time |
| --- | ---: | ---: |
| A/A control | 50.7% | — |
| R1 | 60.3% | 59.6% |
| R2 | 64.7% | 59.2% |
| S | 58.3% | 57.7% |
| E | 51.4% | not run |
| C 0.5 / 0.7 / 1.0 / 2.0 | 57.9% / 57.8% / 52.6% / 47.7% | — |
| R1+S | 65.1% | 61.0% |
| R2+S | 68.2% | 61.6% |
| R2+S, c = 0.5 | 65.8% | 63.3% |

Every equal-time pilot match realised a time ratio between 0.876 and 0.951,
so the 0.90 handicap achieved parity with margin.

Pilot decisions, each by the predeclared rule:

- **E rejected at pilot stage:** 51.4% pooled, below the 52% pass rule, and
  no better than the A/A control's noise.
- **R1, R2, S pass** to equal time; all stay above 50% at both budgets.
- **Amendment 1** (recorded before pilot 3): pilot both R1+S and R2+S because
  R1 and R2 were indistinguishable at equal time.
- **Constant:** on R2+S, c = 0.5 scores 53.6% head-to-head against 1.41 and
  replaces it. c = 2.0 is worse everywhere.
- **Finalists frozen:** F1 = R2+S c = 0.5, F2 = R1+S. See the freeze section
  of DESIGN.md.

Caveat carried forward: the top four configurations sit within about four
points of each other at equal time, which is inside pilot noise. The pilots
picked what to confirm; they do not rank the finalists.

### 2026-10-10 — Phase D: confirmatory evaluation (held-out set)

Finalists and equal-time budgets were frozen at `9e98cac` before the first
held-out game.

- `confirm_primary` (4,096 games, 719 s): all eight comparisons have
  family-adjusted lower bounds above 56%. Equal time: F1 70.8% (187 vs 400)
  and 65.7% (1,023 vs 2,000); F2 66.1% and 60.6%. Realised time ratios
  0.935–0.970.
- `confirm_negamax` (3,456 games): at equal time F1 gains +21.6 to +26.0
  points over the baseline against Negamax 4 / 6 / 8; F2 +9.1 to +19.3.
- `confirm_scaling` (768 games, budgets calibrated by `pilot4` on the
  development set): F1 at equal time scores 71.3% / 66.8% / 66.2% against the
  baseline at 100 / 800 / 5,000.
- `confirm_empty` (256 games): 79.3% and 72.3%, but realised time ratios of
  1.109 and 1.121, so time parity failed from the empty board.
- `finalists/throughput.json`: fixture latency and traced memory. Memory gate met.

### 2026-10-10 — Phase E and follow-ups

- **Follow-up A, strict latency (amendment 2).** Budgets cut to 151 / 821 by a
  rule fixed beforehand. Fresh empty-board seeds (`empty2`): 75.4% and 73.4%
  at time ratios 0.900 and 0.918. Held-out: 69.2% and 67.6% at time ratios
  0.789 and 0.783, with 95th-percentile latency within 1–6% of the baseline.
- **Follow-up B, tactical audit.** Depth-8 Negamax re-scoring of 512 held-out
  games: F1 makes 0.49% / 0.16% provable blunders per decision against the
  baseline's 2.15% / 1.22%, and none at the three-ply horizon. Most F1 losses
  contain no provable blunder, so the residual weakness is positional.
- **Follow-up C, compute equivalence (amendment 3).** F1 at 187 beats the
  baseline at 2,000 (58.8%) and F1 at 1,023 beats the baseline at 10,000
  (63.5%), each in about one-fifth of the time.
- **Replay audit.** 440 recorded games from all 16 studies replayed with zero
  mismatches.
- **Gates.** Every declared gate passes for F1 and F2 at both budget tiers.
  E is rejected; single components and constants are promising but not
  individually confirmed.
- **Validation.** Full backend suite 2,882 passed, 15 skipped
  (`validation/backend-tests-final.txt`).

Totals: 20,288 games, 4,013 s of game time, every phase inside its declared
compute cap. The benchmark lock was acquired for every timed run and released
after each batch; it was never found held by the other agent.
