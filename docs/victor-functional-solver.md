# Functional Victor research solver

Built on `794e2f1992fdae9587a4ccdb30bfd7c2165081fe` on branch
`victor-functional-solver`, 2026-10-08. Primary source: the local
[Allis thesis](../research/references/allis-1988-connect4.pdf), chapters 6–9.
The previous [nine-rule report](victor-nine-rule-implementation.md) describes
the unchanged enumeration, conditional coverage and compatibility definitions.

The package now evaluates positions, selects legal moves and completes games.
It combines terminal-only exact search, the established three-rule Black
non-loss theorem, replayable conditional nine-rule responses, restricted White
threat contexts and a deterministic Negamax fallback. **It is not perfect play.**
No public agent, factory, API, UI, explanation contract, deployment setting,
canonical benchmark or AlphaZero training artifact was changed.

## Running it

From the repository root:

```sh
.venv/bin/python -m games.connect4.victor.cli --white victor --black random
.venv/bin/python -m games.connect4.victor.cli --white negamax:4 --black victor
.venv/bin/python -m games.connect4.victor.cli --white victor --black mcts:32
.venv/bin/python -m games.connect4.victor.cli --white victor --black victor
.venv/bin/python -m games.connect4.victor.cli --white human --black victor
```

Human input uses columns **1–7**. Python moves and CLI `--moves` opening
replays use **0–6**. A game prints JSON with its replay, winner (`0` White/X,
`1` Black/O, `-1` draw), decision labels, exact-search resources and duration.
Use `--nodes`, `--seconds`, `--remaining`, `--cover-nodes` and
`--fallback-depth` to change settings. `--cover-nodes 0` disables strategic
search, useful for an exact-plus-heuristic ablation.

```python
from games.connect4.victor import (
    SearchBudget, SolverBudget, VictorSolver, analyze_position, select_move,
)

board = [[" "] * 7 for _ in range(6)]  # top row first; detached on entry
budget = SolverBudget(exact=SearchBudget(
    nodes=100_000, seconds=0.25, max_remaining=14, table_entries=100_000))
result = analyze_position(board, 0, budget)
assert result.move in range(7)
print(result.move, result.move_kind, result.exact_value, result.bound)

# Same result shape, with a move suitable for play:
result = select_move(board, 0, budget)
# Keep one instance per player/game to retain original Black rule instances:
session = VictorSolver(budget)
result = session.select_move(board, 0)
```

`VictorSolver.choose_move(game)` is a research adapter for the existing engine.
It does not replace `VictorAgent`. Terminal analysis returns no move; invalid
gravity, counts, turn or terminal winner raises `ValueError`. A snapshot has
basic legality checks, not a proof of historical reachability. The CLI's full
legal replay supplies history for the actual games it plays.

## Architecture and decision policy

| File | Responsibility |
| --- | --- |
| `exact.py` | Bitboard alpha-beta, bound-typed transpositions, hard resource accounting |
| `execution.py` | Original-witness Black policy, residual obligations, full adversarial policy replay |
| `white.py` | Odd threats, three threat-combination cases, restricted regions and conditional covers |
| `solver.py` | One result shape, evidence boundaries, tactical decisions, retained game session |
| `cli.py` | Complete games with Random, existing Negamax/MCTS, Victor or a human |

The existing candidate enumerator, 69-group geometry, compatibility matrix,
cover backtracker and independent nine-rule verifier are reused. No second
certificate format was introduced. `certificate.py`, its producer, its
independent verifier and `strategy.py` were not modified.

Priority is completed exact search, immediate win, necessary tactical defense,
applicable established Black bound, then exploratory strategic preference and
depth-4 Negamax fallback. A Black move into a three-rule certified child is
non-losing under the established theorem. A move into a nine-rule cover is
exploratory unless the concrete policy's full adversarial replay completes.
White may prefer a move into a restricted threat cover; that preference is
explicitly exploratory. A retained policy is also checked for immediate
tactical defeat before it is used; a failed plan is discarded and reanalysis
chooses a legal move.

| `move_kind` | What it establishes |
| --- | --- |
| `exact` | Optimal W/D/L move from completed exact search, or a proved unavoidable immediate loss |
| `terminal_win` | Actual four-in-a-row on this move; exact mover value +1 |
| `forced_defense` | Only move avoiding defeat on the next reply; no longer-term non-loss claim |
| `strategic_nonloss` | Independently checked CL/BI/VE theorem, including retained original `sigma_R` responses |
| `verified_policy_nonloss` | Entire concrete Black policy replayed against every White continuation to terminals |
| `exploratory_nine_rule` | Verified coverage plus a conditional executable Black policy |
| `exploratory_white_context` | Restricted White source cover; no White outcome theorem |
| `heuristic` | Deterministic bounded Negamax choice, no optimality guarantee |

`exact_value` is -1/0/+1 from the mover's perspective, or `None`. `bound`
identifies Black explicitly and never substitutes for an exact value.
`justified_move` includes necessary tactical defenses, which must not be counted
as guaranteed non-losing moves. `exact.status` reports every search cutoff;
coverage and continuation-cover fields remain inspectable even when they cannot
establish an outcome. Exact results can return early without collecting rules.

## Nine-rule execution

Policy construction accepts only an independently verified `NineRuleWitness`.
It retains the exact initial board and each selected candidate's original
index. Public selection replays the complete continuation and rejects illegal
moves, moves after a terminal and prior Black deviations. Residual obligations
are attached to their original instance, rather than re-enumerating a new rule
and losing conditional information.

| Rule | Implemented execution |
| --- | --- |
| Claimeven | Forbid proactive lower; answer lower with upper; discharge when Black owns upper |
| Baseinverse | Answer either playable endpoint with the other; proactive acquisition discharges the pair |
| Vertical | Answer lower with upper; proactive lower is allowed, including even-upper Before components |
| Aftereven | Execute all component Claimevens; preserve the original group and timing coverage; stop immediately on completion |
| Lowinverse | Forbid both lowers initially; after White takes one, answer its upper and retain the other pair as a Vertical |
| Highinverse | Forbid both lowers initially; answer lower with middle; create a Claimeven on the other middle/upper; add the exposed upper/other-lower Baseinverse only if that lower was playable on the original board |
| Baseclaim | For roles first/second/third: first → third plus second/above-second Claimeven; second → third plus first/above-second Baseinverse; third → second plus the same Baseinverse |
| Before | Execute mixed Claimeven/Vertical components; Black takes the group square or its successor, keeping the original timing clause |
| Specialbefore | Execute other components and the playable group/extra-square Baseinverse; group square → extra, extra → group square |

All simultaneous demands are collected. Equal responses coalesce; distinct
responses produce `conflicting_obligations`. A gravity-inaccessible response
produces `unplayable_obligation`. No legal permitted spare produces
`no_permitted_spare`. These statuses do not become game values. Retirement
requires actual Black blocking of every originally covered group; timing rules
can stay active until the game ends. Terminal states stop all obligations.

Spares prefer even landing rows, then deterministic center order. Mixed Before
Verticals permit odd lower acquisition, so the conditional policy allows an odd
spare if no even spare exists. **No general parity proof justifies this spare
policy.** Source-defined forbidden squares and residual pairs are implemented,
but the §7.1 even-release invariant has not been proved globally for every
overlapping compatible collection. Pairwise §7.4 compatibility is never used
as a substitute for this missing argument.

`policy.audit(SearchBudget(...))` explores every White move and the fixed Black
response until terminal boards. Safe completed subtrees are memoized. A full
completion proves non-loss for that mathematical starting position and that
particular deterministic policy. It does not prove optimality, the general
nine-rule theorem, or all possible spare choices. Cutoffs report unknown;
White-winning branches retain their entire counterexample continuation.
Unsupported response states also retain the offending continuation.

All six composite response mechanisms are executable, but **none has been
promoted to a generally certified composite strategy**. The finite policy check
is the sound bounded route for practical small cases. Longer games can use
these strategies only with an exploratory label. The old three-rule strategy
continues through its original verifier and response function unchanged.

## White contexts

Evaluation respects §9.2's opponent-to-move requirement: Black whole-board
coverage is evaluated with White to move; White restricted coverage with Black
to move. A White move can be evaluated through its Black-to-move child.

An odd threat is a White group with three stones and one empty, nonplayable odd
square. Each threat gets a separate context, reserving its column. Black groups
needing the threat square or any square above it are conditionally excluded.
White's reserved-column odd claims exclude the lowest square if it is odd and
directly playable, exactly as §8.2 specifies. Other threats stay on the
remainder of the board. Diagrams 8.1 and 8.2 reproduce restricted covers.

A threat combination needs two White groups with exactly two stones each:
two odd holes in one group, a shared nonplayable odd crossing hole in the other,
and an even hole immediately above/below the other odd hole, in another column.
Both columns are reserved. The implementation distinguishes:

1. Even above odd: crossing odd exclusions; both-above exclusion;
   crossing-successor/other-odd exclusion; the playable-other-odd height limit;
   conditional bottom Baseinverse; reserved-column Vertical pairs.
2. Odd above even, even not playable: crossing odd exclusions; both-above
   exclusion; conditional bottom Baseinverse; reserved-column Vertical pairs.
3. Odd above even, even playable: only crossing odd and both-above exclusions.

Applying the reserved bottom Baseinverse shifts the Vertical pairing start by
one square. Tests pin these clauses, playable exceptions, reserved columns,
relevant Black groups and reflection on visually transcribed diagrams 8.3/8.7.
The complete context inventory is available; `white_contexts` explicitly limits
how many are searched. The result exposes total root context count so a partial
context search cannot look exhaustive.

Every empty square used by a nine-rule candidate must be in the permitted
region outside the reserved columns. The shared enumerator uses White as
defender only *after* this context has been derived; it is not a whole-board
Black search with colors reversed.

**White limitations:** these are conditional source claims and restricted covers,
not independently certified White outcomes. White reserved-column execution and
its composition with all remaining rules are not implemented as a guaranteed
White policy. In particular, the general evaluator uses the explicit §8.4
claims; it does not generalize additional diagram-specific conclusions from
§8.3 (such as g6 acquisition) into new unrestricted axioms. Combination covers
can therefore miss wins. A detected odd threat or even a complete restricted
cover alone never supplies `exact_value=+1`. Exact search remains the supported
way to establish White wins beyond immediate tactics.

## Exact search and budgets

The solver uses seven-bit columns (six cells plus sentinel), shift-based four
detection and terminal-only negamax alpha-beta. Center ordering and immediate
threat ordering are deterministic. Transposition entries carry exact/lower/upper
flags; only completed nodes enter the table. Every root child receives a full
window, so all returned root move values are exact. Finding an immediate win
inside a subtree proves its maximum WDL without exploring irrelevant children.

Budgets bound recursive entries (including cache hits), wall-clock duration,
table entries and remaining cells. The default is 200,000 nodes, one second,
200,000 table entries and at most 14 remaining cells. The hard remaining ceiling
is 24, preventing accidental opening/empty-board solving. Interruption discards
all partial root values and moves. There are no heuristic or strategic leaves
in the exact engine. Set `seconds=None` for reproducibility under node budgets;
deadline stopping points naturally vary with host load.

Budgets are per operation. `cover_nodes` bounds the existing cover DFS; its
finite standard-board enumeration/conflict preprocessing is not deadline
interruptible. Each configured strategic child and White context can require
another cover search. These limits are visible in `SolverBudget`; the API does
not claim one global wall-clock budget across all analysis. Fallback depth is
restricted to 1–6 and uses the existing Negamax implementation.

## Validation and experiments

Reproduce the small experiment (no paid APIs, training or canonical artifacts):

```sh
PYTHONPATH=tests .venv/bin/python -m victor_validation.functional_experiment \
  > /tmp/victor-functional-results.json
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests -q
git diff --check
```

The checked-in [raw experiment](victor-functional-results.json) contains all
starting replays, per-position resources, game replays and decision labels.
Sampling deliberately avoids terminal random moves; an additional quiet sample
excludes immediate wins for either player. This is an adversarial exposure
sample, not a naturally occurring position distribution.

Measured on Python 3.11.17 / macOS 26.6.2 arm64. Exact search used 100,000 nodes,
0.25 seconds and a 22-cell cap; games used 100,000 nodes, 0.5 seconds and a
12-cell cap, depth-4 fallback, three strategic children and four White contexts.
Host load affects time cutoffs and durations.

| Position experiment | Result |
| --- | ---: |
| Positions tested, both turns | 190 |
| Exactly solved / optimal decisions | 188 / 190 |
| Independent bitboard oracle comparisons, all root move values | 106, zero disagreements |
| Independent uncached matrix oracle comparisons | 20, zero disagreements |
| Black opponent-to-move contexts | 164 |
| Nine-rule covers | 19 / 164 (11.6%) |
| Covers containing composites | 5 |
| Concrete policies completely verified non-losing | 15 |
| Covered policies beyond the audit's 12-cell cap | 4, unknown |
| Refuted policies / coverage-versus-exact contradictions | 0 in this sample |
| Decisions justified by exact solving | 188 |
| Heuristic decisions after time cutoffs | 2 |

The 160 ordinary exposure positions at 4–16 remaining all solved; many were
tactical, so their very short search times must not be generalized. The 30
additional quiet positions had these results:

| Quiet remaining cells | Exact / tested | Total nodes | Total exact seconds |
| --- | ---: | ---: | ---: |
| 10 | 6 / 6 | 422 | 0.003 |
| 11 | 6 / 6 | 2,354 | 0.010 |
| 14 | 6 / 6 | 3,557 | 0.015 |
| 18 | 6 / 6 | 96,028 | 0.408 |
| 22 | 4 / 6 | 184,130 | 0.813 |

Across all 190 searches: **307,125 recursive entries**, **1.335 seconds** of
exact search, median 0.000026 seconds, maximum 0.250 seconds. Maximum entries
in a single transposition table: **27,412**; maximum recursive entries in one
search: **58,929**. Cover evaluation totaled 1.189 seconds, median 0.0024 seconds
per applicable context. These report entries and elapsed time, not a measured
process RSS. The full experiment, including analyses and eight games, took
**36.97 seconds**. Opening cover construction is substantially more expensive
than endgame enumeration; complete-game durations below include it.

| White | Black | Start | Result | Plies | Seconds |
| --- | --- | --- | --- | ---: | ---: |
| Victor | Random | Empty | White win | 7 | 2.014 |
| Random | Victor | Empty | Black win | 18 | 1.828 |
| Victor | Negamax depth 4 | Empty | White win | 39 | 2.620 |
| Negamax depth 4 | Victor | Empty | Black win | 38 | 5.610 |
| Victor | MCTS 32 | Empty | White win | 17 | 3.177 |
| MCTS 32 | Victor | Empty | Black win | 24 | 6.981 |
| Victor | Victor | Empty | Black win | 42 | 9.620 |
| Negamax depth 4 | Victor | Thesis diagram 6.10 | Black win | 38 total | 0.360 |

Victor won both colors against each external opponent in this **one-game-per-
color** sample. Different settings and seeds may change results. These wins do
not prove strength generally or perfect play. The full-game decision records
include exploratory White covers and retained composite Black responses; the
diagram 6.10 game made 11 exploratory Black decisions and four exact decisions.
The sampled covers used Aftereven, Before and Lowinverse; Highinverse and
Baseclaim have explicit local transition tests, but this experiment does not
provide broad full-cover exposure for them. Specialbefore is exercised in the
thesis game and response tests.

An additional quiet-sampler trap (seed 6105, ten remaining) is preserved in
`test_exact_avoids_a_quiet_tactical_trap_that_greedy_fallback_loses`. There is no
immediate win for either player. Columns 0, 1, 3 and 5 draw; column 6 loses.
Depth-1 greedy fallback selects 6; the functional exact solver selects a draw.
This is a regression for the fallback's limitation, not a contradiction of a
certified cover. No composite-policy contradiction was discovered in the
bounded sampled covers.

Final backend validation: **1,388 passed, 14 skipped**. The neural/DQN skips
remain due to unavailable torch. `git diff --check` passes. The old Victor
certificate, executable strategy, independent audits and nine-rule tests pass
unchanged.

Tests additionally cover every composite response branch, Highinverse's
original-playability gate, mixed/even-upper Before components, Specialbefore's
extra square, conflicts, unplayable demands, forbidden spares, original-instance
replay, mismatched games, terminal stopping, exact cutoff boundaries, table/time
cutoffs, invalid budgets, tactical wins/defenses on both sides, both independent
oracles, no-cover draws, retained established `sigma_R` and seven kinds of full
games. No unrelated test was weakened.

Constraint 2 remains the documented **entirely above** interpretation. The
existing literal-overlap mutation fixture is retained and passes: relaxing it
creates a cover on a position independently won by White. The prior report's
48 exact counterexamples remain relevant evidence against relaxation. A strict
no-cover result on a non-losing position does not isolate constraint 2 as the
cause; this milestone establishes no sound constraint-2 false-negative case.
There is no justification to relax it.

## Remaining work and readiness

The solver is ready for broader **research benchmarking**, with reproducible
games and honest decision labels. Production integration should follow further
audits, rather than importing the exploratory policy into the public agent now.

Required for perfect play: a reviewed global response/parity invariant for all
compatible composites (or another sound proof-search route), a guaranteed White
reserved-column policy and composition argument, stronger targeted Highinverse
and Baseclaim adversarial exposure, and substantially broader search covering
positions beyond practical endgames. The evaluated wins against weak agents,
and the absence of counterexamples in small samples, do not resolve these
mathematical gaps or establish the empty-board value with this implementation.
