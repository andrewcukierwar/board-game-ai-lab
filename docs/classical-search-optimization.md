# Classical search optimization: Negamax and MCTS

October 9, 2026. Branch `search/negamax-v2`, based on `main` at `b3ce5b7`.
This page summarizes seven completed phases and links to their full reports.

**Status: implemented and tested, not deployed.** The public deployment may
still run an older backend until it is updated separately, by hand. No public
preset, search depth, simulation budget or API contract changed.

## What changed

The Connect 4 Negamax and MCTS agents compute the same decisions as before with
less work. Four production files changed; everything else on the branch is
tests, benchmark harnesses and evidence.

| File | Change |
| --- | --- |
| `games/connect4/agents/negamax_agent.py` | Immediate wins searched first, then a same-depth transposition-table (TT) move hint, then the existing center order; leaf heuristic maintained incrementally during play/undo; TT keys and entries packed into integers |
| `games/connect4/agents/negamax_tt.py` (new) | Lossless key/entry packing; an experimental bounded table used only by benchmark scripts |
| `games/connect4/agents/mcts_agent.py` | Tree nodes hold compact immutable bitboard states; child selection computes the parent logarithm once |
| `games/connect4/agents/mcts_bitboard.py` (new) | Bitboard state and uniform random rollouts, standard library only |

Preserved exactly: every Negamax root score for its depth, the full-window
search of every root move, center-first tie-breaking, terminal-before-heuristic
scoring, EXACT/LOWER/UPPER bound semantics, a fresh table per decision, the
MCTS rollout policy, UCT arithmetic, root tactical guards and random-number
consumption. Seeded MCTS games and all Negamax games therefore replay
identically. For that reason the implementation identifiers in
[evaluation provenance](evaluation-provenance.md) stay at `negamax: 2` and
`mcts: 2`.

## Measured results

All measurements come from one laptop (Apple M5, CPython 3.11.17) with other
host activity present. They are fixed-position engineering comparisons, not
production latency, endpoint throughput or worst-case guarantees.

| Phase | Commit | Result | Report |
| --- | --- | --- | --- |
| 1. Negamax ordering and TT hints | `81aeb76` | Depth-10 latency down 22.5% (empty), 80.8% (near-opening), 93.8% (tactical midgame), 60.2% (varied midgame); a tiny late board regressed by about 0.1 ms | [report](search-negamax-v2/REPORT.md) |
| 2. MCTS bitboards | `fadd61c` | 53.6–72.3× faster at equal simulation budgets across 48 comparisons; traced tree memory down 70.0–71.4% at 1,000 simulations | [report](search-mcts-v2/REPORT.md) |
| 2B. MCTS strength and budgets | `babaac7` | 1,664 games; see playing strength below | [report](search-mcts-v2/strength-evaluation/REPORT.md) |
| 3A. Deeper Negamax strength | `93ee943` | 512 games; see playing strength below | [report](search-negamax-v2/deeper-search/REPORT.md) |
| 3B.1. Incremental evaluation | `f2f57b4` | 1.83× geometric speedup on 28 broad trees, 1.47× over all 76 conditions, identical scores and search counters | [report](search-negamax-v2/incremental-evaluation/REPORT.md) |
| 3B.2. Packed TT | `2e79b15` | Retained TT memory down 56.0% on 28 large tables, for 4.83% more wall time | [report](search-negamax-v2/transposition-table/REPORT.md) |
| 3B.3. Iterative deepening, cross-depth hints | `27dc320` | All variants rejected; production search unchanged | [report](search-negamax-v2/iterative-deepening/REPORT.md) |

Speed and memory are separate results. Packed TT entries are a memory
reduction that costs time; they are not a speedup. The 1.83× figure belongs to
the incremental evaluator and was not re-measured after the TT change.

For public settings: at 800 simulations on the empty board, MCTS fell from
659.9 ms to 9.8 ms in one paired run. Negamax depth 8 on the empty board
measured 110.1 ms before the branch and 46.8 ms at its end, and 217.9 ms versus
41.6 ms on the near-opening board. Those Negamax pairs come from separate
sessions with different sample counts, so treat them as indicative.

## Playing strength

Faster search does not make the public agents stronger: their depths and
budgets are unchanged and their moves are identical. Two evaluations asked what
larger settings would buy.

- **MCTS budgets.** Against MCTS 400, the score was 54.5% at 800 simulations
  (inconclusive after multiplicity adjustment), 60.9% at 2,000, 65.8% at 5,000
  and 70.3% at 10,000. 10,000 was not clearly better than 5,000. MCTS 10,000
  still lost 71 of 256 games to MCTS 400.
- **Negamax depth.** Depth 8 scored 52.3% against depth 6 and depth 10 scored
  54.7% against depth 8. Both intervals include 50%, so neither result
  establishes an improvement. Depth 10 was materially more expensive.

Neither evaluation yields Elo or a ranking, and neither changed a preset.

## Rejected experiments

These stayed out of ordinary search because they failed criteria declared
before measurement. Their code lives in `scripts/` and is imported by no
production module.

| Experiment | Outcome |
| --- | --- |
| Threat-count move ordering | Fewer nodes on some trees but slower overall; kept as an internal ablation switch that is off by default |
| Bounded direct-mapped TT, modulo and mixed indexing | 30–92% memory saving depending on capacity, at 1.2–1.8× geometric latency; every capacity failed the latency limits |
| Cross-depth move hints | 1.13× on broad trees, 0.92× over all conditions; failed regression and memory limits |
| Iterative deepening without reuse | 0.74×; added work for identical final-iteration counters |
| Iterative deepening with hints | 0.86× over all conditions |

The negative results apply to the specific schedules and table designs tested.

## Limitations

- Timings use one machine, a small number of samples per condition and a
  finite set of positions. No confidence intervals or percentiles are claimed.
- Small, terminal-heavy Negamax searches got slower by fractions of a
  millisecond, because incremental bookkeeping costs more than the leaf scans
  it removes.
- Deep Negamax correctness beyond tractable oracle depths rests on exact
  parity with the previous implementation, which was itself oracle-checked.
- MCTS equivalence is measured on tested seeds and one Python version, not
  proven for all seeds.
- Depth 10 and 12 and simulation budgets above 800 appear only in experiments.
  The API still accepts Negamax depths 1–8 and MCTS budgets
  50/100/250/400/800, and the UI presets are unchanged.
- [`canonical-season-v1`](../benchmarks/connect4/results/canonical-season-v1/README.md)
  is frozen at backend commit `c7d0ce6` and was not re-run. Its timings
  describe the earlier implementation.

## Verifying

```sh
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest tests -q -rs
cd ui && npm test && npm run build
```

Result at the head of this branch: 2,752 backend tests passed with 15 PyTorch
skips, and 431 frontend tests passed.

The benchmark harnesses compare against earlier phase commits. Those commits
are missing from single-commit checkouts such as CI, so
`scripts/pinned_git_source.py` falls back to byte-identical committed copies
and checks every load against a pinned SHA-256. Each report lists the commands
to repeat its experiment in a fresh directory.

## Deployment

Merging to `main` runs CI and publishes one immutable image,
`ghcr.io/andrewcukierwar/board-game-ai-lab:mac-arm64-sha-<commit>`. Nothing
activates that image: the workflow contains no Render hook, SSH, Tailscale or
self-hosted step and does not move the `:main` or `:latest` tags. The Mac Mini
API changes only when someone runs `scripts/deploy-mac-mini-api.sh` there; see
[Mac Mini deployment](mac-mini-deployment.md).
