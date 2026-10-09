# Victor opening strength and application readiness

October 8, 2026. Branch `victor-opening-strength-and-app`, based on
`victor-performance-and-integration` at `383f260`. Builds on the
[benchmarking and integration report](victor-performance-and-integration.md).

**The solver is not perfect play.** The opening book adds exact values for 1,722
early positions; everything beyond it is unchanged in kind:

- exact search at 24 or fewer empty cells;
- conditional strategic rules;
- a depth-4 heuristic fallback.

Nothing was pushed, merged, deployed or enabled. No AlphaZero branch, training
code or artifact, frozen season, canonical benchmark or earlier experiment
artifact was changed. The frozen 371-position suite was used read-only.

## Summary

| Measure | Previous Victor | Improved Victor |
| --- | ---: | ---: |
| Optimal moves, frozen 371-position suite | 79.0% (293) | **84.1% (312)** — in-sample, see below |
| Same, book restricted to non-benchmark entries | 79.0% | 79.2% (294) |
| Early game (6–13 stones), frozen suite | 56.9% (74/130) | 71.5% (93/130) — in-sample |
| Negamax-hard subset, frozen suite | 54.2% (71/131) | 66.4% (87/131) — in-sample |
| Fresh held-out positions not in the frozen suite (n=259) | 87.6% (227) | 87.6% (227) |
| Errors in Victor's own empty-board games, ≤8 stones | 8 of 72 decisions | **0 of 71** |
| All oracle-judged errors, 48 empty-board games | 21 of 748 | **10 of 706** |
| Game score, frozen 112-game protocol | 0.844 (87/15/10) | 0.821 (86/12/14) |
| Game score, 48 empty-board games | 0.948 (45/1/2) | 0.906 (43/1/4) |
| Public agent, 562-move sweep: median / p95 / max | 0.070 / 1.013 / 1.081 s | **0.004 / 0.319 / 1.003 s** |
| Public agent moves reaching the deadline | 62 | 2 |

In short:

- **Opening book.** It is exact, verified and fast. Wherever it applies, Victor
  now plays a mathematically optimal move. It is a play-closure book, **not a
  complete opening book.**
  - It covers every position with at most 2 stones.
  - It covers every position Victor can face through 8 stones when it plays
    from the empty board and follows the book, as White or Black, against any
    opponent.
  - It also includes 46 early positions of the frozen benchmark suite.
- **The headline accuracy gain is in-sample.** It comes from those 46 entries.
  On positions not built into the book, accuracy is unchanged.
- **Games did not get measurably stronger.** The book removed every in-book
  error and halved the errors in Victor's own games. Results still turn on
  heuristic decisions between 9 and about 18 stones, beyond the book and
  before exact search.
- **White refutation avoidance.** I reproduced `e0-006` and explained it. Three
  alternative policies were measured on a fresh development set; all were
  worse, so the previous policy stays, with `e0-006` pinned as a known failure.
- **Baseclaim gap.** It is fixed by a narrowly justified "retiring spare". The
  preserved counterexample now verifies non-losing under complete adversarial
  replay. No composite theorem is claimed.
- **UI.** "Victor Research (Experimental)" is implemented behind matching
  build- and run-time flags. Both are off by default.

## 1. Exact opening book

### Cost estimate and pilot

Mirror-distinct non-terminal positions
([`position-counts.jsonl`](victor-opening/position-counts.jsonl)):

| Stones | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Positions | 1 | 4 | 25 | 121 | 568 | 2,144 | 8,231 | 27,109 | 91,295 | 129,498 |
| In the book | 1 | 4 | 25 | 24 | 24 | 146 | 151 | 822 | 525 | 1,722 |

Pilot runs with the committed C oracle (about 25–30 million nodes/s per core):

- **Random positions:** about 9 s each at 4 stones, about 1 s at 6 and about
  0.3 s at 8.
- **Shallow positions dominate the cost:**
  - The empty board took 168 s (4.26 billion nodes).
  - The four mirror-distinct 1-stone positions took 78–203 s each.
  - Realistic centre-heavy 6–8-stone positions took 0–5 s each.

Mean nodes per solved position in the actual build were 1.5 billion (0–2
stones), 345 million (3–4), 70 million (5–6) and 12 million (7–8).
Extrapolated, a comprehensive 8-stone book needs about 2.4 trillion nodes:
**roughly 3–4 hours on 8 cores**, far beyond the 20-minute budget. It was not
attempted.

### What was built

[`opening_book_builder.py`](../tests/victor_validation/opening_book_builder.py)
solves three tiers. An entry can belong to several.

| Tier | Definition | Entries |
| --- | --- | ---: |
| `shallow` | Every position with ≤2 stones (all first and second moves) | 30 |
| `closure` | Every Victor-to-move position reached from the empty board when Victor plays the book move (as White, or as Black after any first move) and the opponent plays any legal move, up to 8 stones | 1,659 |
| `benchmark` | The 46 early positions (≤8 stones) of the frozen suite: **in-sample** | 46 |

The closure is expanded from Victor's *actual* choice. That is
`preferred_move`, the same function the solver uses: a Negamax-4 ranking among
equally optimal moves. Expansion happens in the board's real orientation, and
solving is mirror-deduplicated. A runtime test walks the closure again: every
Victor decision through 8 stones from the empty board, as either colour,
against every opponent reply, is an exact book hit.

Generation used a hard wall budget (`--wall-budget 1200`) and per-ply node
limits from 2 to 30 billion:

| Batch | Positions | Wall | Oracle nodes |
| --- | ---: | ---: | ---: |
| 0–2 stones (+ benchmark tier) | 76 | 379 s | 44.1 B |
| 3–4 stones | 48 | 136 s | 16.6 B |
| 5–6 stones | 281 | 156 s | 19.8 B |
| 7–8 stones | 1,317 | 138 s | 16.2 B |
| **Total** | **1,722 exact, 0 unresolved** | **810 s** | **96.7 B** |

It ran on 8 workers on an Apple M5 (10 cores), alongside about 3 cores of
macOS background activity (`mediaanalysisd`). The file
([`opening_book.json`](../games/connect4/victor/data/opening_book.json)) is
78 KB, 14 KB gzipped.

### Exactness guarantees

- **Entries come only from completed searches.** A position becomes an entry
  only when the oracle printed `ok`: every root child was solved by a completed
  terminal-only search. `unknown` (node limit), a killed process (wall budget)
  or any error is written to `unresolved` and never given a value. A smoke run
  with a 25-second budget produced 76 unresolved positions and 0 entries. This
  run had none unresolved.
- **Every entry stores:**
  - the canonical key (`X + mask + bottom`, minimised with its mirror) and the
    player to move;
  - the mover-relative W/D/L value;
  - all seven move values (`+ = - x`), so **every** optimal move is recorded;
  - a canonical replayable history and its source tiers.
- **The file records** the oracle source SHA-256, compiler, base commit,
  generation settings, per-batch statistics and a completion table.
- **Verification** ([`book-verify.json`](victor-opening/book-verify.json)):
  - All 1,722 entries replay legally to their keys. Their mirror images look
    up with mirrored columns.
  - 166 parent/child links are minimax-consistent.
  - A fresh oracle process (different transposition-table state) re-solved 290
    sampled entries with ≥3 stones: **0 mismatches, 0 unknown** (489 s).
  - The pilot's independent solves of the empty board and all 1-stone
    positions match their entries.
  - The empty-board entry reproduces the published theorem: centre wins,
    columns c and e draw, the rest lose.
- **Runtime lookup is pure Python and fails closed.** It needs no compiler and
  no `tests/` code; it ships in `games/`, which the Docker image copies.
  - A schema or SHA-256 digest mismatch disables the whole book.
  - Every hit replays its stored history and must reproduce the board, its legal
    columns, and a value equal to its best move value. Otherwise it returns
    `invalid`, and the solver uses its normal policy.
  - Absent, `unresolved`, `invalid` and `disabled` lookups never influence a
    move.
  - Load takes 1.3 ms; a lookup about 0.1 ms; a complete book decision,
    including the tie-break, about 3.5 ms.
- **Limit.** The runtime cannot detect a *consistently* wrong but well-formed
  value. Only the oracle cross-check can; a test demonstrates that. A test also
  fails if the oracle source changes without regenerating the book.

### Integration

`analyze_position` and `VictorSolver.select_move` consult the book first, ahead
of exact search and any retained Black plan. A hit returns move kind
`opening_book`: a justified exact move, with `exact_value` set and no bound.
Ties among optimal moves use the existing Negamax-4 ranking, which keeps the
project's split between mathematical optimality and practical resilience.
Positions outside the book follow the unchanged policy. `SolverBudget` gained
`opening_book` (default on) and `book_exclude` (used for the held-out
measurement).

## 2. Decision accuracy

Artifacts: [`positions.json`](victor-opening/positions.json),
[`heldout-positions.json`](victor-opening/heldout-positions.json),
[`heldout-overlap.json`](victor-opening/heldout-overlap.json). Budgets are as
before: node-only and deterministic.

- `previous_full` is the previous solver's configuration in the current code.
  It reproduces the committed previous results **move for move**: 371/371
  positions and 112/112 games.
- `heldout_book` excludes entries whose only source is the benchmark tier.

### Frozen 371-position suite

| Group | Negamax-4 | Previous | Improved | Improved, held-out book | `certified_only` White |
| --- | ---: | ---: | ---: | ---: | ---: |
| All (n=371) | 56.6% | 79.0% | **84.1%** | 79.2% | 82.7% |
| Early (n=130) | 48.5% | 56.9% | **71.5%** | 57.7% | 70.8% |
| Middle (n=130) | 52.3% | 83.1% | 83.1% | 83.1% | 80.0% |
| Late (n=111) | 71.2% | 100% | 100% | 100% | 100% |
| Negamax-hard (n=131) | 0% | 54.2% | **66.4%** | 55.0% | 61.8% |
| White / Black to move | 54.7 / 58.6% | 76.8 / 81.2% | 83.7 / 84.5% | 77.4 / 81.2% | 81.1 / 84.5% |

Errors (improved vs previous): missed wins 18 vs 21, win→loss 24 vs 28,
draw→loss 17 vs 29.

**Covered vs uncovered openings (frozen suite, early phase):**

- **Book hits:** 46 positions, all in-sample. Previous Victor was optimal on
  27/46; improved Victor on 46/46.
- **Uncovered:** 84 positions, 47/84 for both.
- **Held-out book** (benchmark-only entries excluded): 4 hits, all of which
  previous Victor already played correctly. Total 294 vs 293.

### Held-out sample

I built a second suite with the identical methodology and a new seed
(`20261104`). **112 of its 371 positions repeat frozen-suite histories exactly.**
The near-greedy samplers revisit the same short prefixes, so a new seed is not
an independent test set. On the full second suite, the improved solver scores
310 vs 301 — but 16 of its 17 book hits are those repeats. Restricted to the
**259 fresh positions**:

| Group (fresh only) | Negamax-4 | Previous | Improved | `certified_only` |
| --- | ---: | ---: | ---: | ---: |
| All (n=259) | 196 | 227 | 227 | 217 |
| Early (n=96) | 61 | 71 | 71 | 65 |
| Negamax-hard (n=55) | 0 | 27 | 27 | 20 |

There was **one** fresh book hit, which the previous solver also played
correctly. **The book does not generalise to arbitrary sampled openings.** Its
value lies in the lines Victor itself reaches from the empty board.

## 3. Games

### Frozen protocol

16 seeded two-move openings × both colours against Negamax-4, Negamax-6 and
MCTS-400, plus 8 against Random: 112 games, oracle-adjudicated
([`games.json`](victor-opening/games.json)).

| Config | W/D/L | Score [95% CI] | Judged error rate |
| --- | --- | --- | ---: |
| Previous | 87/15/10 | 0.844 [0.785, 0.902] | 5.18% |
| Improved | 86/12/14 | 0.821 [0.757, 0.886] | 4.92% |

The paired difference (96 non-Random games) was −0.026 [−0.079, +0.027]: 5
games better, 7 worse and 69 move-for-move identical. **No significant
difference.**

- The 133 book decisions had 0 errors.
- Previous Victor was suboptimal in 32 of 112 decisions at book positions.
- In each of the six ply-2 divergences that changed a result, the previous move
  was theoretically wrong — for example, the losing column 2 after opening (2,1),
  where only column 5 wins. It happened to work against deterministic Negamax.
  The improved solver played the winning move, left the book (the closure does
  not extend from fixed two-move openings) and later lost the win heuristically.

### Empty-board games

8 seeded games per colour against MCTS-400, ε-Negamax-4 and ε-Negamax-6
(ε=0.1): 48 per configuration, paired by seed
([`book-games.json`](victor-opening/book-games.json)).

| Config | W/D/L | Judged errors | Errors at 6–13 stones | Errors ≤8 stones | Max move |
| --- | --- | ---: | ---: | ---: | ---: |
| Previous | 45/1/2 | 21 / 748 | 19 / 192 | 8 / 72 | 2.32 s |
| Improved | 43/1/4 | 10 / 706 | 9 / 189 | **0 / 71** | 1.14 s |

The paired score difference was −0.042 [−0.099, +0.016] (0 games better, 2
worse). Both extra losses were to MCTS-400. In each, the book handed Victor a
**won** position, and a single heuristic move outside the book (at 9 and 12
stones) lost it. Decision quality clearly improved, while results did not
measurably change in this sample.

## 4. Strategic-policy findings

### White refutation avoidance (`e0-006`)

**Reproduced.** In this drawn position, only column 1 draws, and Negamax-4
ranks it first. One Black reply to column 1 reaches a nine-rule cover, which
claims only "Black ≥ draw" — consistent with the draw. The scan therefore skips
column 1 and plays column 6, the first move with no refutation found. Column 6
loses. Absence of a refutation is not evidence of a win, and no deeper Negamax
horizon (up to 10) shows column 6 losing.

**Measured on a fresh development set, not the frozen suite**
([`white-policy-dev.json`](victor-opening/white-policy-dev.json); seed
20261101; 768 oracle-solved White positions with >24 empty cells; 449
decisive). Every candidate's refutation status and White-cover evidence was
recorded once, then each policy was replayed:

| Policy | Optimal (decisive) | Changed vs Negamax | Better / worse |
| --- | ---: | ---: | ---: |
| `off` (Negamax-4 move) | 318 / 449 | — | — |
| **`first_unrefuted`** (previous default) | **362 / 449** | 63 | 47 / 2 |
| `certified_only` (only a verified CL/BI/VE refutation skips a move) | 334 / 449 | 27 | 17 / 1 |
| `positive_evidence` (switch only to a move with a White threat cover) | 328 / 449 | 17 | 10 / 0 |

- **Nine-rule refutations are usually right.** They drove 32 improvements
  against the 2 regressions, which look structurally identical to the
  improvements.
- **`certified_only` was worse everywhere.** On the frozen suite it scored 307
  vs 312, and on the fresh positions 217 vs 227.
- **Decision: no policy change.** The default stays `first_unrefuted`;
  `white_refutation` is now an explicit `SolverBudget` field for ablations.
- **No exploratory cover becomes a refutation claim.** Statuses stay
  `nine_rule` vs `certified`. The moves are labelled `exploratory_unrefuted`,
  with no bound, and are not justified moves.
- **The regression is pinned.** `e0-006` is preserved in
  `test_e0_006_refutation_avoidance_failure_is_preserved_and_labelled`.

### Composite policy gap (Baseclaim + Claimeven)

**Diagnosis.** After White g4, spare g5 and White g6, every playable square
(a5, b5, c4, f5) belongs to a rule. This is an **incomplete spare-move policy**,
not a missing global invariant for this board. A Black stone on b5 lies in both
Baseclaim target groups (a6-b5-c4-d3 and a5-b5-c5-d5); a Black stone on f5 lies
in the Claimeven's group (f3–f6). Either one blocks every group its rule is
responsible for, which is exactly the existing retirement condition in
`_prune`.

**Fix: retiring spare, last resort only.** When no ordinary spare exists, Black
may play a rule square if **every** active rule whose obligations use that
square retires after the move. Rule squares are disjoint (§7.4 C1), so no other
rule is touched. For example, c4 is rejected: it blocks only one Baseclaim
group. The legacy behaviour remains available as
`NineRulePolicy(..., retiring_spares=False)`.

**Validation** ([`composite-recheck.json`](victor-opening/composite-recheck.json)):

- **Baseclaim case:** complete adversarial replay over all White continuations
  (12 empty cells) now gives `verified_policy_nonloss`. The oracle value is a
  Black win.
- **Stored 1,512 composite covers:**
  - all 528 previously verified policies remain verified;
  - the 1 unsupported case is now verified;
  - 983 remain beyond the 16-cell audit cap;
  - 0 oracle White wins and 0 White wins in 6,048 playouts.
- **Fresh 1,034 covers (seed 20261103):** 373 verified under both policies,
  661 beyond the cap; no case needed a retiring spare. 0 oracle White wins and
  0 White wins in 4,136 playouts.

**Limits.** This is local soundness plus bounded verification of one board. It
is **not** a zugzwang or spare theorem for composite collections. Retiring
spares occurred only in the single known case. Composite covers are not
promoted to proven non-loss, and unaudited policies keep their exploratory
labels. On both suites, retiring spares changed no decision.

## 5. Runtime and resource behaviour

### Real worst-case blocking

The deadline is cooperative: it is checked between bounded steps. Step timing
on the public profile (562 positions,
[`public-latency.json`](victor-opening/public-latency.json)):

- **Longest non-preemptible step:** one nine-rule cover search (max 0.123 s).
- **Other steps:**
  - certificate searches, max 0.008 s;
  - White cover searches, max 0.030 s;
  - exact search, max 0.401 s at its own 0.4 s cap, checked per node.

Worst-case blocking is therefore *deadline + one cover search*. I added two
checks — between the certificate and cover searches in the refutation scan,
and before the root certificate. They can only skip work after the deadline.
With the book removing the heaviest early analyses:

- the maximum fell from 1.081 to **1.003 s**;
- deadline hits fell from 62 to 2;
- the heaviest moves now occur just past the book, at about 10 stones.

**Local contention tests (not Render measurements;
[`stress.json`](victor-opening/stress.json),
[`stress-efficiency-cores.json`](victor-opening/stress-efficiency-cores.json)):**
85 public-profile decisions under each condition; every move was legal.

| Condition | Median | p95 | Max | Deadline hits |
| --- | ---: | ---: | ---: | ---: |
| Idle | 0.004 s | 0.418 s | 0.882 s | 0 |
| 4 CPU-bound competitors | 0.004 s | 0.439 s | 1.015 s | 1 |
| 10 competitors (= cores) | 0.007 s | 0.670 s | 1.007 s | 4 |
| 16 competitors | 0.016 s | 1.015 s | 1.029 s | 6 |
| Efficiency cores only (`taskpolicy -c background`) | 0.008 s | 0.781 s | 1.021 s | 4 |

On slower CPUs, wall time stays bounded while quality degrades: more moves
reach the deadline and fall back to heuristics. Deadline-truncated decisions
are therefore host-dependent; node-only benchmark decisions are not.

**GIL interference (in-process).** During five ~1.0 s research moves,
concurrent `/health` requests (38,656 of them) had p50 0.06 ms, p95 0.07 ms and
max 76 ms. A research move slows but does not block other request threads.

**No heavier mitigation was needed.**

- What exists: one reservation per agent (`503 agent_busy`, no queueing),
  stateless moves, a hard legal fallback, and the book making openings nearly
  free.
- What was not added: process isolation, background workers or queues.
- MCTS keeps its own reservation; AlphaZero and other agents are untouched.

## 6. UI integration

**Status: implemented, hidden by default, not enabled or deployed.**

| Concern | Implementation |
| --- | --- |
| Flag | `VITE_VICTOR_RESEARCH_ENABLED` (build time; only `true`/`1`; default off) for the UI, and `VICTOR_RESEARCH_ENABLED` for the API. Both must be on. |
| Scope | Connect 4 play page only ([`researchAgent.js`](../ui/src/connect4/researchAgent.js)). Match, tournament and season labs, saved records and exports keep the public four agent types, so stored data and its validators are unchanged. |
| Label | "Victor Research (Experimental)", described as experimental, not perfect play, able to lose, with moves taking about a second. It is never called perfect or unbeatable. |
| Validation | `validateSnapshot` accepts `victor_research` only when the play page opts in; it is rejected everywhere else. The payload is `{type: 'victor_research'}` with no settings. |
| Failures | `agent_busy` (503), disabled server (`invalid_agent` at start or 409 at move) and `agent_failed` use the existing reconcile-then-explicit-retry path, with clear messages. There is never an automatic replay of an uncertain POST. A UI-only enablement shows "not enabled on this game server". |
| Hidden data | No certificates, bounds, exact values or move kinds reach the API response, history or UI. The history records `{type: 'victor_research'}` only. |
| Dependencies | No new LLM dependency or paid call. |

A local browser run (system Chrome, both flags on) played a full game with the
AI moving first: worst round trip 682 ms, no page errors. With the default
build, the option is absent.

## 7. Remaining barriers to perfect play

1. **Middle opening (9 to ~18 stones).** Neither the book nor exact search
   reaches it. Every changed game result traced in this report was decided
   there by a heuristic move.
   - Extending the closure to 10 stones would need about 9,000 more solves,
     roughly an extra 5–10 minutes at the measured rates. That would put total
     generation at about 18–24 minutes, at or beyond this milestone's budget,
     so it was not run. It is the most direct next step.
   - A comprehensive 8-stone book costs about 3–4 hours of 8-core time.
2. **No composite soundness theorem.** Retiring spares fix one demonstrated gap.
   General zugzwang and parity arguments for composite collections remain
   open. 983 + 661 composite policies lie beyond the audit cap.
3. **White.** Refutation avoidance is a strong heuristic (+44 optimal on the
   development set) with a known failure mode in drawn positions. It is not a
   White-winning strategy.
4. **Covers prove at least a draw for Black, never a win.**
5. **Measurement.** The new-seed suite overlaps the frozen one. Fixed-opening
   games are deterministic and small. The empty-board sample has 48 games per
   configuration.

## 7a. Reproducing

```sh
B="PYTHONPATH=.:tests .venv/bin/python -m victor_validation"
$B.opening_book_builder estimate
$B.opening_book_builder build --output games/connect4/victor/data/opening_book.json   # 810 s, 8 workers
$B.opening_book_builder verify --sample 300 --min-resolve-plies 3
$B.performance_benchmark positions --suite docs/victor-performance/suite.json \
   --configs negamax4 previous_full victor_full book_only heldout_book no_book certified_only \
   --workers 8 --output docs/victor-opening/positions.json
$B.performance_benchmark suite --seed 20261104 --workers 8 --output docs/victor-opening/heldout-suite.json
$B.performance_benchmark positions --suite docs/victor-opening/heldout-suite.json \
   --configs negamax4 previous_full victor_full certified_only --output docs/victor-opening/heldout-positions.json
$B.performance_benchmark games --configs previous_full victor_full --opponents random negamax:4 negamax:6 mcts:400 \
   --openings 16 --random-openings 8 --adjudicate --workers 8 --output docs/victor-opening/games.json
$B.opening_app_benchmark white-policy --pool 1200 --output docs/victor-opening/white-policy-dev.json
$B.opening_app_benchmark book-games --configs previous_full victor_full \
   --opponents mcts:400 epsnegamax:4:0.1 epsnegamax:6:0.1 --games 8 --output docs/victor-opening/book-games.json
$B.opening_app_benchmark composite-recheck --fresh-pool 8000 --output docs/victor-opening/composite-recheck.json
$B.opening_app_benchmark stress --loads 0 4 10 16 --output docs/victor-opening/stress.json
$B.performance_benchmark public-latency --suite docs/victor-performance/suite.json --output docs/victor-opening/public-latency.json
```

Hardware: Apple M5 (4 performance + 6 efficiency cores), 32 GB, macOS, CPython
3.11.17. No Mac Mini or AlphaZero process was involved.

## Controlled Render trial readiness checklist

Nothing below has been done; each item needs separate approval.

1. **Deploy with both flags off first.** Confirm that the image includes
   `games/connect4/victor/data/opening_book.json` (the Docker image copies
   `games/`). From a shell on the instance, run
   `python -c "from games.connect4.victor.opening_book import default_book; print(len(default_book()))"`
   and expect `1722`. `None` means a missing or corrupt book: the agent still
   works, but without book moves.
2. **Measure on the actual Render instance before exposure.** Time the
   research agent over the 562-position public sweep
   (`performance_benchmark public-latency`) or a smaller fixed subset, and
   record median, p95, max and deadline hits. Proceed only if the max stays
   under about 1.5 s. Otherwise lower `PUBLIC_BUDGET.deadline` (for example to
   0.6 s) and re-measure.
3. **Check concurrent behaviour.** Run two simultaneous research games and one
   MCTS game. Expect `503 agent_busy` for the second research move, MCTS
   unaffected, and `/health` responsive.
4. **Check the free-tier lifecycle.** After a cold start, the first move
   includes the book load (about 1 ms locally). Confirm the 90-second client
   timeout is never approached.
5. **Enable the API flag only** (`VICTOR_RESEARCH_ENABLED=true`). Verify that
   `start_game` with `victor_research` succeeds and that the public agent
   lists and errors for other agents are unchanged.
6. **Then rebuild the Static Site** with `VITE_VICTOR_RESEARCH_ENABLED=true`.
   Play one complete game as each colour and check the history endpoint.
7. **Watch logs** for `Connect 4 agent failed` and for any `fallback` kind.
   Keep `EXPLANATIONS_ENABLED=false` unless separately approved.
8. **Roll back** by unsetting both flags and rebuilding the Static Site. A game
   in progress then receives `409 invalid_agent`, which the UI reports.

**GO/NO-GO:** **GO for a controlled, flag-gated trial** on the live service once
steps 1–4 pass on Render. **NO-GO** for presenting it as a strong or perfect
opponent, or for enabling it before Render latency is measured.
