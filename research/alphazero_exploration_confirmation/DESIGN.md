# Phase 4D.4E prospective confirmation design

Frozen before new confirmatory games; prior 4D data and runtime-only feasibility
may inform this design. This document must be committed and pushed before main
collection. No neural training, campaign resume, runner change, main merge or
deployment. Start revision 08656faa1efb9d76fb9ae546f894164011e0b453.

## Question, population and endpoint

For each complete empty-board self-play game, count executed actions at
zero-based plies >=8 that reduce exact mover-relative achievable W/D/L. State
value V(s) and action value -V(s') use +1 win, 0 draw, -1 loss. Separate win->draw,
win->loss, draw->loss; loss->loss and win->win are neutral, regardless of realized
outcome or win distance. Primary effect: extended minus original mean counts
per COMPLETE game. All eligible actions are attempted, not only tactical flags.
The completed game, paired by identity, is the statistical unit.

## Fixed collection and identities

Generation-20 inference checkpoints for seed 42 and 314159; unchanged weights,
CPU float32, deterministic algorithms, intra/inter-op one, OMP/MKL/OpenBLAS/
VECLIB/NUMEXPR thread settings one, existing M4 virtualenv. Accepted pool-eight
collector, exactly 256 simulations, unchanged PUCT/root noise/network/targets.
Original ((0,1.0),(8,0.0)); extended ((0,1.0),(16,0.0)). Independent-game protocol
connect4-az-independent-games-v1. Experimental generation identity 22. Seed-42
cohorts use base seeds 4404511,4404512; seed-314159 cohorts 4404521,4404522;
each has ids 0..63, matched across schedules. These identities do not occur in
4D. 128 games/checkpoint/schedule, 512 complete games, no optional supplement.
Cohort-major, rotate schedule order. Collection ceiling 650 seconds. A deadline
preserves complete games and excludes only whole unmatched checkpoint/cohort
blocks, explicitly reported; no adverse game is selectively excluded. No sample
size change after primary collection begins. Prior measured throughput (~1 game/s)
fits this ceiling; old-position feasibility needs no new pilot games.

## Exact solver and resource limits

Use the project's independent native C oracle from pinned reference 8704142702da273e6571f0ca9df3fd4a793be285,
tests/victor_validation/c4_oracle.c, copied byte-identically. Compile with system
cc -O3; no downloaded solver/package. Terminal wins/draws handled by engine:
new mover value -1 on a win and 0 on draw. Validate native legal columns, values,
terminal actions, reflection, and representative positions against Python
BitboardSolver (and exhaustive minimax on bounded endgames). Neither exact solver
enters AlphaZero search. Native full vectors prove V(s) and each action's exact
value; successful vectors and their successors populate a canonical board+actor
cache, with correct reflected column mapping. Reuse the native transposition
table. Original source refuses terminal root inputs; never send them.

Pass 1: each unique eligible canonical pre-state in descending ply, tie by
canonical key, at 200,000 nodes/root, wall <=0.3 sec/root. Pass 2: unresolved
canonical states in the same order, 2,000,000 nodes/root, wall <=0.5 sec/root.
Whole solver invocation (load/validation/write included) ceiling 650 sec; stop
launching queries after 600 sec, preserve every result. Per-position wall kills
and restarts the process, discarding TT only then. Node aborts keep valid completed
TT proofs, never partial root values. Exact cache may resolve a state despite an
earlier aborted root. No result-dependent additional retry or stage exclusion.
The feasibility pilot uses only old games, separately stored/excluded.

An immediate winning legal action proves the pre-state scalar value +1 by
engine scan, even if the native full vector aborts. Confirmed-count differences
are coverage-dependent descriptive contrasts, not lower bounds on the true
difference; the sensitivity limits are the principal effect range.

Each decision is exactly adjudicated by full pre-state vector, or exact pre/next
state values; pre=-1 proves neutral by minimax even if successor unresolved;
executed immediate terminal win proves neutral (pre=1/action=1). If exact action
value 1 is known, it also proves neutral. Otherwise unresolved stays unknown.
Do not call an unknown safe or an error. Store histories, queries, reasons,
values, per-decision classifications, coverage by ply/stage/checkpoint/schedule/
action category (max-visit vs sampled non-max, tactical categories), and all
unresolved cases. Denominator is every complete game and eligible action.

## Uncertainty, missingness and decisions

For each game, lower count = confirmed value-changing actions; upper count =
lower + unresolved actions (each can add at most one). Report observed confirmed
rates as lower bounds, never full-population rates with unresolved cases.
Per pair, lower difference = extended lower-original upper; upper difference =
extended upper-original lower. Report identification interval of their means.
95% uncertainty interval = mean(lower difference)-1.96*SE(lower difference) to
mean(upper difference)+1.96*SE(upper difference). SE uses paired game-level sample
variance, stratified by checkpoint; pooled mean weights checkpoints equally and
SE^2 sums checkpoint SE^2/4. Report checkpoint-specific intervals too. This is
an approximate paired normal/jackknife interval, not finite-sample certification;
rare-event precision and achieved half-width must be stated. No position-based
standard errors or claim that an interval covering zero establishes equivalence.

Single material-harm margin: +0.05 exact value-changing actions/game (one extra
irreversible result downgrade in 20 games). This is a permissive practical
screening budget before a small learning experiment, not a universal safety
standard; it is ~116% of prior flagged-event baseline 0.043. That prior estimate
is incomplete and post hoc. Prior n=256 paired half-width ~0.044 indicates the
suggested 512-game study may remain inconclusive even at full coverage.

- MATERIAL HARM: pooled conservative lower 95% bound > +0.05.
- NO MATERIAL HARM WITHIN THE DECLARED MARGIN: pooled AND each checkpoint's
  conservative upper 95% bound < +0.05; at least 99% exactly adjudicated in
  each checkpoint/schedule/stage with eligible actions; all correctness gates pass.
- INCONCLUSIVE: otherwise (including sparse solving or incomplete matching).

These are fixed before primary results. No secondary endpoint overrides them.
Report pooled and checkpoint estimates and cohort descriptive differences.
Uncertainty concerns game RNG under these two frozen networks, not new checkpoints.

## Sampling counterfactual and secondary endpoints

For extended decisions at plies 8..15 use recorded visits from SAME search.
M = all maximum-visit columns. Report non-max selection frequency. When executed
and every M action value is exact, report worse than all/equal all/better than all/
mixed tie values; also expected regret = uniform mean of M values minus executed
value and fractions worse/equal/better among M. Any missing required value makes
this comparison unresolved. No added search. For exact executed errors label
sampling-attributable only when every M action is exact and preserves V(s);
search-attributable if every M action is exact and downgrades it; mixed ties
ambiguous, missing M values unresolved attribution. Include stages >=16 where
execution is greedy; no causal alternative trajectory is claimed.

Use retained 4D definitions/code for duplicate positions at >=8, mirror duplicates,
collision probability, unique 12/16/20/24 prefixes and full trajectories, one-reply
errors, immediate-win/block misses, outcome and first/second-player W/D/L, lengths,
normalized-root-visit target entropy, unique tactical states. Paired game jackknife
95% intervals for secondary differences are descriptive; no multiplicity claim.
Diversity replication requires lower duplicates, mirror duplicates and collision
in both checkpoints, lower duplicate rate in each cohort, and pooled/checkpoint
paired duplicate intervals below zero. More diversity proves no learning gain.

## Correctness, preservation and execution budget

Replay all games legally with terminal winners/actor targets checked; rebuild
record hashes; check visits=256, normalized targets/distributions/schedule,
frozen checkpoints (all eight) and model hashes, independent RNG ownership,
identical full decision records before ply8 in matched identities. Default versus
explicit original regression on retained 4C records, using already validated
collector and a bounded repeat. Focused tests: exact classifications, timeout/
node unknowns, reflection, terminal handling, missingness bounds, three-way rules.
Independent reviewer examines implementation, evidence and interpretation.
Verify all 150 official files against retained inventory and declaration hash
c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716 before/after.
Keep all previous results/checkpoints unchanged and historical checkout clean.

Measured experimental execution: feasibility, validation collection, main
collection and exact analysis, maximum aggregate 1,800 sec; target ~1,200 sec.
Reserve <=120 sec for collector regression/repeat and <=100 sec for final analysis
and preservation. Solver launch ceiling can only be reduced to respect aggregate
budget, never increased after results. Reanalysis from recorded proofs does no
new solving. If exact coverage inadequate, conclude INCONCLUSIVE, assess whether
a tightly bounded learning pilot is worthwhile, and do not automatically propose
another frozen-checkpoint study. Next training phase is DESIGN ONLY: per-generation
schedule provenance/validation plus an explicitly experimental batched runner;
unchanged optimizer/replay at warm start and matched updates/compute, network-only
and PUCT strength with separate tactical guard and resumable generation boundaries.
