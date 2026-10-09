# Phase 3A predeclared design — October 9, 2026

This declaration precedes all Phase 3A search measurements and strength games.
The application search code, public caps, UI, prior evidence, and production
services remain unchanged. All games run serially in the laptop checkout without
providers. The source/configuration hashes bind the experiment to this baseline.

Primary comparisons are Negamax 8 vs 6 and 10 vs 8, targeting **64 opening pairs
each**. Secondary comparisons are 10 vs 6, 8 vs MCTS 2,000, 10 vs MCTS 2,000,
and 10 vs MCTS 5,000, targeting **32 pairs each**. Every opening has both agent
colors. Challenger is the first/deeper Negamax named. All comparisons share a
prefix of one opening pool, enabling paired shared-opponent contrasts.

The pool has equal random and agent-play cohorts. Random openings reproduce
Phase 2B's uniform legal-column process at 2/5/8/11/14/17/20/23 plies; terminal
histories are rejected. Board-plus-mover identity is deduplicated across main
and preflight, rather than merely deduplicating histories. No mirror augmentation.
Each block has four independently generated random boards and four snapshots
from one independently seeded agent trajectory at 5/14/23/32 plies. Agent play
starts with two random moves, then uses 80% alternating Negamax 4 / MCTS 400
policy and 20% uniformly legal exploration. A trajectory that ends before ply
32 or duplicates a board is regenerated in its entirety, never selected by
test-agent performance. This survival-conditioned, exploratory policy is a
plausible-play cohort, not a representative human or perfect-play distribution.
Full histories, generator seeds and rejected-attempt counts are preserved.

Stage labels are early (≤8), midgame (9–23), late (≥24). Independently label
tactical when the mover has an immediate win or any move permits an immediate
opponent win, otherwise quiet. These labels use array-engine-independent MCTS
bitboard root guards, with correctness already covered by the existing suite;
all histories and outcomes are verified using the separate array Connect4
engine. Tactical/quiet overlap chronological stages and are reported separately.
Slow searches and forced losses stay included. Report cohort × stage results
and tactical/quiet summaries before pooling the 50/50 cohort mixture.

The result-excluded preflight comprises one independently seeded eight-opening
block (four random, four related agent snapshots), both colors of all six
conditions: **96 games**. It has a 600-second cumulative cap. Separate traced
searches at 6/8/10 on every preflight board estimate peak allocations, retained
TT bytes, and node counts. These diagnostics and opening generation are outside
the main cap. Do not inspect main results before freezing the design.

The main cap is **1,800 cumulative seconds**. Before main, estimate target
cost as each matchup's mean preflight paired-game cost × its target pair count.
Apply a **1.75× safety multiplier**, reserving **300 seconds**. Choose the largest
of primary/secondary **64/32, 48/24, 32/16, 16/8** fitting 1,500 seconds after
that multiplier. If even 16/8 is infeasible, do not start main: document a new
fixed design first. Save candidate, preflight, memory measurements, and final
experiment manifest; never change sample count/cap after main starts. Smaller
designs sharply limit power, particularly their small agent-trajectory count.
Signals enforce the remaining wall budget inside games; partial games have
explicit incomplete records and are never losses or inferential observations.
An exhausted cap is not enlarged. Completed-pair inference with missing games
can be selected by runtime and must be labeled incomplete/conditional.

Each completed game is fsynced, with a per-move active checkpoint and cumulative
budget checkpoint. Game-boundary resume validates source hashes/Python/config,
plan fields and independent engine replay before skipping observations. Interrupted
games restart with original seeds and prior work counts toward the cap. A corrupt
JSONL fails closed. Use one writer. Abrupt termination during a first search may
lose uncheckpointed wall time; orderly signals and elapsed status avoid that
normal case. Role-specific independent MCTS RNGs persist within a game and are
reinitialized per game; their seeds are shared across colors/conditions for an
opening, without claiming identical rollout trajectories after divergent play.
Negamax uses no RNG. Every move records CPU/wall time and existing Negamax
nodes/entries/hits/search-loop cutoffs. Full histories, outcomes, hashes, final
MCTS RNG fingerprints, and configuration are saved. No games invoke public APIs.

Inference averages both color games, then resamples entire trajectory families
within cohort, retaining all correlated snapshots and comparisons. Random boards
each form their own cluster. Use 10,000 bootstrap draws with declared seed;
report nominal 95% and Bonferroni **97.5%** percentile intervals for the two
primary claims. A primary bootstrap improvement requires the adjusted lower
bound above 0.5, with explicit small-cluster/degeneracy caveats. Also report a
conservative bounded-variable 97.5% interval under independent-family sampling
to guard against degenerate percentile estimates; uniqueness conditioning and
survival selection still qualify its interpretation. Secondary/subgroup/paired
contrasts are exploratory. One MCTS seed pair per opening mixes position and
seed variability; no separate seed-variance claim. No Elo or universal ranking.

Profile depths 4/6/8/10 on the predeclared manifest positions: existing empty,
near-opening, two midgames, late fixtures; immediate-win, forced single reply,
double opponent threat/forced-loss, dense endgame; plus every quiet midgame
in the preflight block, selected by board tactics rather than runtime/outcomes.
Each condition has one discarded warm-up and three rotating-depth production
timing samples with fresh TT. Separately run cProfile and tracemalloc with a
diagnostic table reference. Compare every diagnostic move, full score vector
and counter vector against production timing. cProfile counts leaf calls and
disjoint heuristic/terminal/ordering/play-undo subtrees; recursive-body residual
contains TT work. Separate intrusive exclusive line tracing at depth 6 estimates
TT/bound bookkeeping vs recursive overhead. It is approximate and not assumed
to extrapolate unchanged to depth 10. Raw line/function data stay available.
Traced peak excludes interpreter/native/RSS overhead; retained TT reference is
diagnostic only, because production drops its per-decision table. Profiling has
a separate 900-second planning target, with all positions retained if it overruns.

Review Victor's terminal-only Python/C engines for reusable geometry and compact
bounded TT patterns, without importing WDL-only pruning/score semantics. Rank
incremental windows, compact/bounded TT, cross-iteration hints, iterative
deepening, aspiration/PVS, conservative tactical pruning, and optional native
kernel. Implement none in Phase 3A. Preserve exact depth-sensitive terminals,
69-window heuristic, every full-window exact root score, typed depth/mover
bounds, and stable center ties; different depths may choose different moves.

Run independent harness/regression tests and relevant backend integration tests.
Application source changes would require the complete backend suite; none are
planned. Review new artifacts only, commit and push this feature branch, verify
remote SHA. No production, main, deployment, training, or frozen evidence writes.

## Profiling supplement declared after preflight, before main

The generated base set has two quiet midgame fixtures (midgame-wide and
quiet-5). To meet the requested range with three such fixtures, also profile
the **first quiet midgame in candidate main-pool order absent from the base
profile set**, using the identical 4/6/8/10 procedure. Selection uses only
preexisting labels and board identity, before any main game or score inspection.
The original candidate and preflight evidence remain unchanged. The separate
`profiling-supplement/candidate.json` binds this addition to the original
candidate/source hashes and records its declaration time. Its position also
remains in strength evaluation: performance measurements cannot change either
agent or the strength design. Do not count this overlapping board as another
strength observation. The initial design digest in host-start.json precedes
this explicitly recorded supplement.
