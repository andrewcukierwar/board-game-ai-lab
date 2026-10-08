# Victor benchmarking, improvement and integration readiness

October 8, 2026. Branch `victor-performance-and-integration`, based on
`victor-functional-solver` at `e55dd32`. Builds on the
[functional solver report](victor-functional-solver.md). **The solver is not
perfect play.** No proof claim in this report depends on a coverage witness
alone. No AlphaZero branch, training code or artifact, frozen season,
canonical benchmark or older experiment artifact was changed. Nothing was
pushed, merged, enabled or deployed.

## Summary

| Measure (same suite, same games) | Original solver | Improved solver |
| --- | ---: | ---: |
| Optimal moves, 371 decisive oracle-solved positions | 70.4% | **79.0%** |
| Optimal moves, 131 positions where Negamax-4 errs | 35.1% | **54.2%** |
| Optimal moves, late game (9–16 empty cells) | 93.7% | **100%** |
| Game score, 112 games vs Random/Negamax-4/Negamax-6/MCTS-400 | 0.790 | **0.844** |
| Oracle-judged move errors in those games | 8.1% | **5.2%** |
| Worst serial move, early game (full Victor) | 1.49 s | 1.02 s |
| Worst move inside games (full Victor, no deadline) | 8.45 s | 3.55 s |

Paired over identical openings, opponents and colours, full Victor's game score
rose by **+0.054 [95% CI +0.006, +0.101]** (12 games better, 3 worse).

The ablation ladder is monotone and paired-significant on the improved solver:
exact search beats Negamax-4, CL/BI/VE adds to exact, and composite rules add to
CL/BI/VE. White threat contexts added nothing measurable in games.

The opt-in `victor_research` API agent is implemented behind
`VICTOR_RESEARCH_ENABLED` (default **false**), with a 1-second cooperative
deadline (worst observed 1.081 s), a dedicated non-blocking reservation and a
guaranteed legal fallback. It has not been enabled or deployed.

## Methodology

### Independent ground truth

[`tests/victor_validation/c4_oracle.c`](../tests/victor_validation/c4_oracle.c)
is a separate win/draw/loss solver in C: bitboards, non-losing-move generation,
threat-count ordering and a mirror-keyed transposition table. It shares no code
with Victor. It is compiled on demand by
[`native_oracle.py`](../tests/victor_validation/native_oracle.py) and reports
`unknown` instead of partial values when its node limit is reached. Checks:

- It reproduces the known theorem that **every Black reply to White's centre
  opening loses** (1.75 billion nodes, 58 s).
- It agrees on every root move value with the existing independent Python oracle
  (150 endgames) and with Victor's exact engine (100 positions at 14–20 empty
  cells). The test suite repeats both comparisons on fresh positions.
- Throughput is about 25 million nodes/s on this machine. Every position used
  in this report was solved exactly; none was left unknown.

### Position suite

[`suite.json`](victor-performance/suite.json): 4,494 game prefixes from a seeded
mix of random, ε-Negamax-2 and ε-Negamax-4 play. That gave 4,258 distinct
positions (mirror-deduplicated), all solved, in 155 s. Only **decisive**
positions are kept: at least one legal move must be worse than optimal, so
trivial positions cannot inflate accuracy.

| Phase (stones played) | Natural (White/Black) | Negamax-hard (White/Black) |
| --- | ---: | ---: |
| Early, 6–13 | 40 / 40 | 25 / 25 |
| Middle, 14–25 | 40 / 40 | 25 / 25 |
| Late, 26–33 | 40 / 40 | 20 / 11 |

The **natural** subset is drawn in sampled order. The **Negamax-hard** subset
contains quiet positions where Negamax-4, which is also Victor's fallback,
chooses a suboptimal move. That subset is adversarial to Negamax-4 by
construction, so Negamax-4 scores 0% there by definition; the subset measures
how often the other layers rescue the fallback. Late-game hard positions are
scarce because Negamax-4 rarely errs there. All 114 tactical positions (an
immediate win or forced block exists) were solved correctly by every
configuration, so the benchmark's difficulty comes from its 257 quiet positions.

### Configurations

Every Victor configuration uses identical budgets: exact search with 200,000
nodes, **no wall-clock cut-offs** (so decisions are reproducible on any host),
10,000 cover nodes, up to 7 strategic children, depth-4 Negamax fallback, and a
20,000-node policy audit capped at 10 empty cells.

| Name | What it adds |
| --- | --- |
| `negamax4` | Existing public Negamax, depth 4 (Victor's fallback) |
| `exact_fallback` | Exact search, tactics and Negamax-4; strategic rules disabled |
| `three_rule` | Plus CL/BI/VE certificates and covers |
| `nine_rule` | Plus all nine Allis rules (composite covers and policies) |
| `victor_full` | Plus restricted White threat contexts |

The baseline ran the original `e55dd32` solver, frozen in a separate worktree,
with its original 14-cell exact cap. That worktree also contained only the
`rules` ablation field and this benchmark tooling. The improved solver uses a
24-cell exact cap (the existing hard ceiling).

### Games

Sixteen seeded two-move openings, each played with Victor as White and as Black,
against Negamax-4, Negamax-6 and MCTS-400 (32 games each), plus 8 openings
against Random (16 games): 112 games per configuration, 1,120 in total. Every
Victor move made with at most 36 empty cells was judged by the oracle: did it
keep the game-theoretic value? Score intervals are normal-approximation 95%
intervals over games. Games against deterministic opponents from a fixed opening
are deterministic, so the intervals describe this opening sample, not an
independent population. Paired comparisons use identical
(opening, opponent, colour) triples.

### Reproducing

```sh
B="PYTHONPATH=tests .venv/bin/python -m victor_validation.performance_benchmark"
$B suite --output docs/victor-performance/suite.json            # ~3 min, 6 workers
$B positions --suite docs/victor-performance/suite.json --output improved-positions.json
$B games --exact-remaining 24 --opponents random negamax:4 negamax:6 mcts:400 \
   --openings 16 --random-openings 8 --adjudicate --output improved-games.json   # ~5 min
$B composite --pool 12000 --output composite.json                 # ~75 s, 7 workers
$B latency --suite docs/victor-performance/suite.json --output improved-latency.json
$B public-latency --suite docs/victor-performance/suite.json --output public-latency.json
$B report --positions improved-positions.json --games improved-games.json ...
```

The baseline artifacts were produced by the same commands with `--exact-remaining
14`, run in a `git worktree` of `e55dd32` containing only the `rules` ablation
field and this tooling. Raw artifacts are in
[`docs/victor-performance/`](victor-performance/) (3.1 MB in total, compact JSON).
Hardware: Apple M5 (4 performance and 6 efficiency cores), 32 GB, macOS 26.6.2,
CPython 3.11.17. No Mac Mini or AlphaZero process was involved.

## Optimal-move accuracy and ablations

Accuracy is the share of positions where the chosen move keeps the exact
game-theoretic value; brackets are Wilson 95% intervals.

### Original solver (exact cap 14)

| Group | negamax4 | exact_fallback | three_rule | nine_rule | victor_full |
| --- | ---: | ---: | ---: | ---: | ---: |
| All (n=371) | 56.6% | 61.7% | 62.8% | 68.2% | 70.4% |
| Early (n=130) | 48.5% | 48.5% | 50.0% | 54.6% | 54.6% |
| Middle (n=130) | 52.3% | 52.3% | 53.8% | 63.8% | 66.2% |
| Late (n=111) | 71.2% | 88.3% | 88.3% | 89.2% | 93.7% |
| Natural (n=240) | 87.5% | 87.5% | 88.3% | 89.6% | 89.6% |
| Negamax-hard (n=131) | 0.0% | 14.5% | 16.0% | 29.0% | 35.1% |
| White to move (n=190) | 54.7% | 61.1% | 61.1% | 61.1% | 65.3% |
| Black to move (n=181) | 58.6% | 62.4% | 64.6% | 75.7% | 75.7% |

### Improved solver (exact cap 24)

| Group | negamax4 | exact_fallback | three_rule | nine_rule | victor_full |
| --- | ---: | ---: | ---: | ---: | ---: |
| All (n=371) | 56.6% [51.5, 61.6] | 73.0% [68.3, 77.3] | 73.9% [69.2, 78.1] | 78.7% [74.3, 82.6] | **79.0%** [74.5, 82.8] |
| Early (n=130) | 48.5% | 48.5% | 50.0% | 56.9% | 56.9% |
| Middle (n=130) | 52.3% | 74.6% | 75.4% | 82.3% | 83.1% |
| Late (n=111) | 71.2% | 100% | 100% | 100% | 100% |
| Natural (n=240) | 87.5% | 90.0% | 90.8% | 92.5% | 92.5% |
| Negamax-hard (n=131) | 0.0% | 42.0% | 42.7% | 53.4% | **54.2%** [45.7, 62.5] |
| White to move (n=190) | 54.7% | 72.6% | 72.6% | 76.3% | 76.8% |
| Black to move (n=181) | 58.6% | 73.5% | 75.1% | 81.2% | 81.2% |

### Tactical failures

Failure counts over all 371 positions, by type:

| Solver / config | Win→draw (missed win) | Win→loss | Draw→loss |
| --- | ---: | ---: | ---: |
| Original `negamax4` | 46 | 73 | 42 |
| Original `victor_full` | 29 | 44 | 37 |
| Improved `exact_fallback` | 26 | 41 | 33 |
| Improved `victor_full` | 21 | 28 | 29 |

**No configuration missed an immediate win or a forced defence** (114/114
tactical positions correct). In games, oracle-judged errors for full Victor fell
from 71 missed wins, 17 win→loss and 40 draw→loss (128 of 1,575 decisions) to
47, 7 and 24 (78 of 1,507).

### Does strategic coverage change moves, and are the changes improvements?

"Changed" means the move differs from `exact_fallback` under the same budgets,
which is exactly the counterfactual without strategic rules.

| Solver | Config | Strategic decisions (optimal) | Changed | Better | Worse | Neutral |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original | three_rule | 17 (12) | 5 | 4 | 0 | 1 |
| Original | nine_rule | 56 (48) | 28 | 24 | 0 | 4 |
| Original | victor_full | 69 (61) | 36 | 32 | 0 | 4 |
| Improved | three_rule | 10 (8) | 4 | 3 | 0 | 1 |
| Improved | nine_rule | 45 (36) | 31 | 22 | 1 | 8 |
| Improved | victor_full | 47 (38) | 32 | 23 | 1 | 8 |

- **Composite rules carry most of the strategic value.** Relative to
  CL/BI/VE, the nine-rule system adds 35 strategic decisions (improved) and
  changes far more moves (5 → 28 on the original solver, 4 → 31 on the
  improved one).
- **No strategic Black move chose a losing move** in either solver. Every
  non-optimal strategic Black decision was win→draw: a cover establishes at
  least a draw, not a win, so it can settle for a draw when a win exists.
- Strategic coverage is less frequent on the improved solver (47 vs 69
  decisions) because exact search now resolves the middle game before strategy
  is consulted.
- The one worse change is a White refutation-avoidance case (`e0-006`; see
  below).

### Gameplay strength

Score over 112 games; W/D/L; oracle error rate per judged Victor decision.

| Config | Original: score [95% CI], W/D/L, error rate | Improved: score [95% CI], W/D/L, error rate |
| --- | --- | --- |
| negamax4 | 0.625 [0.544, 0.706], 60/20/32, 10.7% | (unchanged agent) |
| exact_fallback | 0.643 [0.563, 0.723], 62/20/30, 10.7% | 0.719 [0.644, 0.793], 71/19/22, 8.9% |
| three_rule | 0.723 [0.649, 0.798], 72/18/22, 8.8% | 0.777 [0.703, 0.850], 83/8/21, 6.5% |
| nine_rule | 0.763 [0.699, 0.828], 72/27/13, 8.6% | 0.844 [0.785, 0.902], 87/15/10, 5.2% |
| victor_full | 0.790 [0.729, 0.852], 76/25/11, 8.1% | **0.844 [0.785, 0.902], 87/15/10, 5.2%** |

Improved full Victor by opponent: Random 16/0/0; Negamax-4 21/6/5 (0.750);
Negamax-6 19/9/4 (0.734); MCTS-400 31/0/1 (0.969). As White it scored 0.839;
as Black, 0.848.

Paired ablation on the improved solver (96 games, Random excluded):

| Comparison | Mean score difference [95% CI] | Games better / worse |
| --- | ---: | ---: |
| exact_fallback − negamax4 | +0.109 [+0.051, +0.168] | 13 / 0 |
| three_rule − exact_fallback | +0.068 [+0.009, +0.127] | 15 / 5 |
| nine_rule − three_rule | +0.078 [+0.004, +0.153] | 15 / 7 |
| victor_full − nine_rule | +0.000 | 0 / 0 (103 of 112 games identical) |
| victor_full − negamax4 | +0.255 [+0.166, +0.345] | 36 / 5 |

**The nine-rule system measurably improves decisions and play.** That evidence
is empirical; it is not a proof that the composite strategy is sound.

## Improvements, each driven by a measurement

1. **Composite compatibility checks (largest bottleneck).** Under cProfile, a
   five-stone Black decision spent about 87% of its time in `conflict_masks` →
   `failed_constraints`: one million pairwise checks, each re-hashing nested
   frozen dataclasses. The §7.4 constraints are now evaluated with
   integer bitmasks for whole candidate classes:
   - overlap ⇒ C1/C4 failure;
   - per-column part buckets ⇒ C3;
   - claimeven-below-inverse-top index ⇒ C2;
   - explicit C4 checks for disjoint inverse pairs.

   Coverage uses a per-square target-group bitmask instead of rebuilding sets.
   Measured without a profiler on the same 12 early positions, a nine-rule
   cover search went from 0.296 s to **0.027 s (≈11×)**, and empty-board
   enumeration with coverage from 0.103 s to 0.016 s. Masks are
   bit-identical to the pairwise reference on random candidate sets, and
   coverage is identical on 65,965 candidates (both are tests). If
   `compatibility._CHECKS` is substituted, as the existing mutation tests do,
   the pairwise reference path is used, so those tests still prove each
   constraint is load-bearing.
2. **Exact search reach.** All 102 suite positions with 15–24 empty cells were
   *unsolved* only because of the 14-cell cap; even the old engine solved most
   of them within 200,000 nodes. The engine was rewritten on integer bitboards,
   keeping the same API, budgets and all-or-nothing contract:
   - non-losing move generation;
   - threat-count ordering;
   - a mirror-symmetric table key;
   - null-window WDL root children.

   Nodes fell 3–8× (median at 23 empty cells: 34,597 → 12,444) and suite time
   5× (17.5 s → 3.6 s). Every suite position with ≤24 empty cells now solves
   within 200,000 nodes (maximum 175,108). The default cap is now the hard
   24-cell ceiling; the ceiling and its test are unchanged. Effect: middle-game
   accuracy 66.2% → 83.1%, late game 93.7% → 100%.
3. **White refutation avoidance.** Offline, over 93 White positions with >24
   empty cells (633 moves, 3,983 cover searches), a Black cover reachable after
   some Black reply never occurred after a winning White move:

   | Refutation of a White move | Loses | Draws | Wins |
   | --- | ---: | ---: | ---: |
   | CL/BI/VE certificate | 26 | 9 | 0 |
   | Nine-rule cover only | 109 | 14 | 0 |
   | None found | 231 | 82 | 152 |

   White now scans candidates in Negamax order and plays the first unrefuted
   move, preferring one that also has a White threat cover. These decisions are
   labelled `exploratory_unrefuted`; a certified refutation proves only that the
   refuted move cannot win. Result: 13 changed White moves (8 better, 4 neutral,
   1 worse). In the final games, all 38 oracle-judged `exploratory_unrefuted`
   moves kept the game value. Known failure mode
   (`e0-006`): in a **drawn** position the refuted move is a draw, and an
   unrefuted alternative can lose.
4. **Tie-break among equal exact values.** After change 2, full Victor
   unexpectedly *lost more* games (17 vs 11; intermediate run score 0.781,
   80/15/17, not kept as an artifact). Tracing each loss showed that Victor was
   already theoretically lost at the divergence point and the exact engine then
   played the first column in centre order. The original solver's heuristic
   played a resilient move instead, and the fallible opponent later erred (4 of
   6 such baseline games ended drawn or won). Equal-value exact moves are now
   ranked by the bounded Negamax score: still exact-optimal, but practically
   resilient. Full Victor then reached 0.844 (87/15/10).
5. **Smaller fixes.**
   - Strategic scans skip moves Negamax proves lost inside its horizon (a
     terminal-only fact).
   - A retained-plan session reuses its own exact search instead of repeating
     it.
   - A `deadline` field bounds whole-analysis wall time.
   - A `rules` field enables the ablations.

**Measured but not adopted:**

- **Deeper fallback.** On positions with >24 empty cells, Negamax at depths 4–8
  scored 85, 82, 89, 84 and 87 out of 109 natural positions, and hard-subset
  results were non-monotone. Depth was left at 4.
- **Per-child partial proofs.** With the 24-cell cap, full root solves almost
  always complete, so the experiment was removed.
- **"Prefer Negamax-proven wins" guard.** All 8 strategic missed wins already
  equalled Negamax's top choice, so the guard was not added.
- **"All White moves refuted ⇒ bound."** No suite position exercised it, so it
  was not shipped.

### Latency (serial, idle machine; seconds: median / p90 / max)

| Config | Early, original | Early, improved | All, original | All, improved |
| --- | ---: | ---: | ---: | ---: |
| nine_rule | 0.265 / 0.835 / 1.357 | 0.083 / 0.158 / 0.163 | 0.000 / 0.321 / 1.357 | 0.001 / 0.108 / 0.163 |
| victor_full | 0.273 / 0.982 / 1.489 | 0.082 / 0.949 / 1.016 | 0.000 / 0.365 / 1.489 | 0.001 / 0.207 / 1.016 |
| exact_fallback | 0.003 / 0.003 / 0.004 | 0.003 / 0.003 / 0.004 | 0.000 / 0.003 / 0.004 | 0.001 / 0.004 / 0.043 |

Full Victor's remaining early-game cost is the White refutation scan: up to
about 49 grandchild cover searches. Exact search now tops out near 1 s at
200,000 nodes (about 310,000 nodes/s).

## Composite-rule validation

The default search tries Claimeven-family candidates first, so Highinverse was
almost never selected: zero Highinverse covers in a 300-sample pilot. The
`composite` experiment therefore also searches with CL/BI/VE plus **one**
composite rule whenever that rule has useful candidates, forcing witnesses that
execute it. Over 11,934 White-to-move positions with 8–30 empty cells it found
2,271 distinct covers. Each composite cover was then:

- solved by the oracle;
- checked by complete adversarial replay of its concrete policy (≤16 empty
  cells, 3 million nodes);
- played out four times against Negamax-6 and ε-Negamax-4 White.

| Rule | Covers using it | Targeted | Essential | Oracle White wins | Policy verified | Audit beyond cap | Unsupported | Playouts B/D/W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aftereven | 975 | 0 | 544 | 0 | 351 | 624 | 0 | 3900/0/0 |
| Lowinverse | 268 | 29 | 60 | 0 | 42 | 226 | 0 | 796/276/0 |
| **Highinverse** | **96** | 82 | 89 | 0 | 27 | 69 | 0 | 277/107/0 |
| **Baseclaim** | **27** | 2 | 17 | 0 | 5 | 21 | **1** | 95/13/0 |
| Before | 379 | 0 | 327 | 0 | 114 | 265 | 0 | 1272/244/0 |
| Specialbefore | 80 | 33 | 57 | 0 | 29 | 51 | 0 | 288/32/0 |

"Essential" means no cover exists without that rule (for targeted covers: none
with CL/BI/VE alone).

- **No oracle contradiction among 1,512 composite covers**: 1,401 positions are
  Black wins and 111 draws. No White win occurred in 6,048 playouts. 528
  concrete policies were completely verified non-losing; 983 lie beyond the
  audit cap and remain unknown.
- **Reproducible counterexample to the concrete spare policy (not to the
  cover).** Baseclaim c4/a5/b5 plus Claimeven f5/f6, history
  `[0,1,1,1,4,6,1,4,3,3,3,3,5,2,4,4,0,4,0,4,3,5,5,0,6,6,2,2,5,3]`. After White g4,
  Black's spare g5 and White g6, every playable square is a forbidden role
  square, so the policy reports `no_permitted_spare`. The position is a Black
  win, so the cover's claim stands. It is preserved as
  `test_baseclaim_policy_spare_gap_is_preserved_and_not_a_cover_contradiction`.
  This is concrete evidence of the documented missing piece: there is no
  zugzwang/spare argument for composite collections. In play, a retained plan
  that cannot select is discarded and re-analysed, and child covers whose audit
  reports `unsupported_policy` are never preferred.
- The CL/BI/VE certificate, its verifier, `strategy.py` and the established
  guarantees were not modified. All prior certificate, assurance and
  independent-audit tests pass unchanged.

## Application integration

**Status: implemented opt-in, disabled by default, not enabled or deployed.**

| Concern | Implementation |
| --- | --- |
| Distinct identity | New `victor_research` agent ([`victor_research_agent.py`](../games/connect4/agents/victor_research_agent.py)). Legacy `VictorAgent`, its `victor` factory identifier and behaviour are untouched; the factory gains a separate `victor_research` entry. |
| Feature flag | `VICTOR_RESEARCH_ENABLED`: strict `true/false/1/0`, default `false`. When off, the API's accepted types and error text are byte-identical, and the agent module is not imported. Moves are rechecked at request time (409 if disabled later). |
| Labelling | `LABEL = 'Victor research solver (experimental; not perfect play)'`. |
| Resource limits | Fixed public profile: exact 150,000 nodes / 0.4 s / 24 cells; 10,000 cover nodes; 4 White contexts; audit 10,000 nodes / 0.1 s; **1.0 s whole-move deadline**; no client-settable budgets. |
| Non-blocking | Dedicated `BoundedSemaphore(1)`: a concurrent Victor move returns `503 agent_busy` immediately without mutating the game. MCTS keeps its own reservation, so neither blocks the other. |
| Legal fallback | Solver result → depth-4 Negamax → first legal column. Any solver exception or illegal result degrades to a legal move. |
| Proof claims | Labels stay on `agent.last_decision` (tests and logs). History records `{'type': 'victor_research'}` only; snapshots and outcomes are unchanged. |
| Sessions and history | Stateless per move (deterministic under node budgets); existing revision, lock, history and validation code paths are reused unchanged. |

Measured latency of the exact public profile (`public-latency`, serial, all 371
suite positions plus a seeded opening sample, n=562): median 0.070 s, p95
1.013 s, p99 1.055 s, **maximum 1.081 s**. 62 moves hit the deadline. The
earlier sweep under full benchmark load peaked at 1.164 s.

**Remaining integration limits:**

- The deadline is *cooperative*. It is checked between bounded steps, each a
  single cover search (≈≤0.1 s), so there is no hard preemption.
- A CPU-bound Python move still competes with the API's three other threads
  under the GIL, as MCTS-800 already does.
- Production latency on Render's shared CPU is unmeasured and likely several
  times slower. Before any trial, measure there and consider a lower deadline
  or process isolation.
- The UI validator (`gameSnapshot.js`) rejects unknown player types, so UI
  exposure needs separate `AgentSelector`, `competitorConfig` and validator
  changes. None were made.
- `docs/deployment.md` records that the flag must stay unset.

## Remaining gaps to perfect play

1. **Early game.** With 25–36 empty cells, accuracy is 56.9%; most errors are
   depth-4 heuristic decisions with no strategic coverage. That is the frontier.
   Exact search cannot reach it under interactive budgets in Python.
2. **No composite soundness theorem.** There is no global zugzwang/spare or
   parity proof for composite collections, as the preserved Baseclaim spare gap
   shows concretely. Pairwise §7.4 compatibility is not a substitute.
3. **White.** Restricted White contexts are conditional and do not form a
   White-winning strategy; they added no measured game value. White's winning
   play beyond 24 empty cells is heuristic plus refutation avoidance.
4. **Covers prove at least a draw for Black, never a win.** Black's strategic
   moves can turn wins into draws; this accounts for most strategic
   non-optimal decisions.
5. **Sample limits.** The opening sample is fixed (16 openings); deterministic
   games make the intervals descriptive. The suite is an exposure sample, not
   the distribution of human games.

## Portfolio-worthy accomplishments

- Built an **independent C ground-truth oracle** that reproduces the classical
  result that White's centre opening wins, and cross-validated it against two
  Python solvers.
- Designed a **stratified, decisive-only, ablation benchmark** (371 positions,
  1,120 oracle-adjudicated games, paired comparisons, uncertainty intervals)
  that isolates exact search, CL/BI/VE, composite rules and White contexts.
- Raised optimal-move accuracy from **70.4% to 79.0%** and on Negamax-hard
  positions from **35.1% to 54.2%**. Game score improved from 0.790 to 0.844
  (paired +0.054). Errors fell from 8.1% to 5.2%.
- Achieved about **11× faster composite cover search** with tested-identical
  output (bitmask §7.4 constraints and coverage), and 3–8× fewer exact-search
  nodes.
- Found and fixed a **counter-intuitive regression** (more losses after a
  strength gain) by tracing lost games to arbitrary moves in already-lost
  positions.
- Ran **targeted Highinverse/Baseclaim exposure** (96 and 27 covers, zero
  contradictions), which found and preserved a real executable-policy gap.
- Delivered a **safe, flag-gated API integration** with a measured ≈1 s bound,
  a guaranteed legal fallback and research claims kept out of public outcomes.

## Next milestone recommendation

**Opening-phase strength with a sound evidence chain.** Concretely:

1. Add an opening book generated offline by the C oracle (positions up to ~8
   stones, stored as exact move values). This removes the early-game frontier
   for real play with mathematically exact labels and costs no runtime search.
2. Prove or refute a spare/zugzwang lemma for the composite rules. Start from
   the preserved Baseclaim case: either extend the spare policy (for example
   proactive role acquisition) and re-audit all 528+ verified policies, or
   restrict composites that lack it.
3. Measure the flag-gated agent on the actual Render instance before any public
   trial, then add the UI entry with an explicit "experimental" badge.
