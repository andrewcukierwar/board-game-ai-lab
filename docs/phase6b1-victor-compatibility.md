# Phase 6B.1: Victor compatibility and covering-set witnesses

This phase adds a deterministic, bounded **compatible covering-set search** for
Claimeven, Baseinverse and Vertical. The independent verifier checks only finite
local geometry, conditional coverage and compatibility. Its successful status is
`coverage_verified_outcome_uncertified`. **Neither search nor verification awards
a strategic proof, a safe position, a non-loss bound, a game value or a move.**
Failure to cover is not evidence of a losing position.

The baseline is Phase 6A commit `e2e18a2`, branched directly into
`phase6b-victor-compatibility`. Existing local candidate semantics remain intact.
No agent, AlphaZero, canonical benchmark, API, UI, deployment or frozen experiment
artifact is changed. There is no public application integration.

## Primary-source review and adversarial audit

Source: Victor Allis (1988), *A Knowledge-Based Approach of Connect-Four: The Game
Is Solved: White Wins*, locally available at
[`research/references/allis-1988-connect4.pdf`](../research/references/allis-1988-connect4.pdf).
The local edition has 91 PDF pages; the page references below are 1-based PDF
pages, coinciding with the thesis contents. This phase read the complete Phase 6A
Victor package and [foundation document](phase6a-victor-foundation.md) before
implementation. The relevant local PDF pages were extracted with PDFKit; diagrams
6.2 and 6.3 and the §7.4 table were also rendered and visually inspected. Extraction
and rendering tools stayed in `/tmp`; no dependency or PDF change was required.

| Source | Definition used or boundary retained |
| --- | --- |
| Chapter 5, p.32 | A potential opponent group is a four-square winning line containing no defender stone. |
| §5.3, pp.33-34 | Shared response squares can demand conflicting replies; a Claimeven's lower trigger matters. |
| §5.4, pp.34-35 | Every potential group must be covered; failure gives no conclusion about the game. |
| Chapter 6 introduction, p.36 | Rules are inspected with the opponent to move; Black can attempt the whole board. |
| §§6.1-6.3, pp.36-39 | Existing local prerequisites and conditional group predicates. |
| §§7.1-7.3, pp.47-49 | Response obligations and Zugzwang interactions; no generic overlap policy for future rules. |
| §7.4, p.50 | Exactly the six CL/BI/VE entries, all Constraint 1. |
| §8.1, p.51 | Black may attempt coverage without first separately establishing Zugzwang control. Its strategic conclusion is deliberately not accepted here. |

The focused adversarial audit found **no confirmed mathematical error in Phase
6A's three local definitions**:

- Claimeven requires two empty adjacent vertical squares, even upper, and covers
  potential groups containing the upper square alone. The lower need not be
  directly playable but is an affected triggering square.
- Baseinverse requires two distinct directly playable empty squares and covers
  groups containing both. Neither parity nor equal height is required.
- Standalone Vertical requires two empty adjacent vertical squares, odd upper,
  and covers groups containing both. Its lower need not be directly playable.

**Diagram 6.2 discrepancy confirmed:** the seven-pair printed useful Baseinverse
list omits `b1-d3`. In the rendered board, b1 and d3 are directly playable, c2 is
White, and e4 is empty. Thus `b1-c2-d3-e4` contains both endpoints and no Black
stone. Under the formal §6.2 predicate it is covered, yielding eight useful pairs.
This is a source-list omission, not a detector error. Phase 6A's existing exact
regression remains authoritative; no fixture-specific exclusion was introduced.
The recorded §8.2 overlapping-example concern remains outside this Black-only
phase and has not been independently resolved here.

## Compatibility fragment

`compatibility.py` dispatches on unordered pairs of `RuleName` identifiers.
`required_constraints` returns the verified constraint tuple, and `compatible`
reconstructs intrinsic candidate geometry before checking the dispatched
constraints. It checks shape identities, not prerequisites on a particular board.

| Pair | Constraint |
| --- | --- |
| Claimeven / Claimeven | 1 |
| Claimeven / Baseinverse | 1 |
| Claimeven / Vertical | 1 |
| Baseinverse / Baseinverse | 1 |
| Baseinverse / Vertical | 1 |
| Vertical / Vertical | 1 |

Constraint 1 is `A.affected_squares ∩ B.affected_squares = ∅`. Both squares of
every instance are affected, including Claimeven's lower trigger. Duplicates
therefore conflict. Order is irrelevant. Disjoint coverage squares alone are
insufficient: the thesis §5.3 example Baseinverse a1-b1 / Claimeven b1-b2 has
separate coverage squares but conflicting responses to b1.

The dispatcher retains constraint-code vocabulary for eventual nine-rule
extension. Unsupported rule pairs and untyped identifiers raise `ValueError`;
missing matrix entries never mean compatibility. Constraints 2-4, composite
component equality and Specialbefore exceptions are not implemented.

> **Superseded (2026-10-08):** the complete matrix and constraints 2–4,
> including N.B.(ii), are now implemented; see
> [victor-nine-rule-implementation.md](victor-nine-rule-implementation.md) §4.
> The CL/BI/VE entries and their behaviour are unchanged.

## Black target inventory

`black_evaluation_context(position, defender=1)` revalidates and snapshots the
board. It requires Black as defender, White to move, and a nonterminal position.
Its context records the full immutable position, defender 1, opponent 0,
`black_opponent_to_move` mode and canonical target groups. Targets are recomputed
from the existing 69 potential winning lines by excluding every line with a Black
stone. White stones are allowed; entirely empty lines remain targets.

No caller-supplied targets, region exclusions, White odd threats or White threat
combinations can change the search domain. The search accepts a position and
optional defender/budget, rather than an untrusted candidate report or proposed
context. Manually constructed context objects are witness declarations only;
the verifier recomputes their meaning. This is a narrow Chapter 8.1-inspired
**inventory for attempted coverage**, not acceptance of Chapter 8's reasoning.

## Search and bounded completeness

`search_covering_set(position, *, defender=1, node_budget=10_000)` enumerates only
actual Phase 6A candidates and recomputes their conditional coverage. It filters
zero-coverage candidates, preserving the order of all useful candidates. Dropping
a candidate that covers no target cannot remove a solution.

The deterministic algorithm:

1. Compute target-index bitmasks for coverage and candidate-index bitmasks for
   conflicts, including a candidate's conflict with itself.
2. At each state, choose an uncovered group with the fewest currently compatible
   covering candidates. Tie by canonical group order.
3. Try that group's candidates in Phase 6A enumeration order: CL, BI, VE; vertical
   pairs by column and ascending upper thesis row; Baseinverses by combinations
   of matrix-ordered landing squares.
4. Select a candidate, remove its covered groups and block it and all conflicts.
   Backtrack after dead ends. No cutoff branch is called exhaustively impossible.
5. Emit selected evidence in original enumeration order. For each target, assign
   the first selected candidate covering it, using canonical target order.

Completeness within the supported universe follows from branching on every viable
candidate covering a chosen uncovered group: any extending cover must contain
at least one such candidate. Each branch covers that group and strictly grows the
selected set. Every selected pair is compatible by the conflict mask. The search
need not find a minimum-cardinality cover and may retain redundant instances.

The budget counts **entered recursive states**, including the root, failed states
and complete-cover leaves. It never exceeds the given nonnegative integer. A zero
budget returns unknown even when a root contradiction could be found cheaply.
Invalid budgets raise `ValueError`. Unsupported/invalid contexts return a distinct
status before search. Preprocessing is outside the node budget but bounded by a
standard board: at most 69 groups, 56 geometric candidates and 1,540 distinct
pairs. A selected set has at most 21 instances because each uses two disjoint
squares out of 42; recursion depth is correspondingly bounded. Preprocessing is
O(nm + n²) for n candidates and m groups. Each visited state scans at most O(mn)
incidences. Worst-case search is exponential, with repeated subsets possible;
there is no unbounded full-game search or experiment runner.

| SearchStatus | Meaning |
| --- | --- |
| `FOUND` / `compatible_covering_set_found` | A proposed coverage witness exists. |
| `EXHAUSTIVE_NO_COVER` / `no_cover_in_supported_universe` | Every relevant branch was exhausted in CL/BI/VE. This says nothing about other rules or the game value. |
| `BUDGET_EXHAUSTED` / `unknown_budget_exhausted` | Search stopped before completeness was established. |
| `UNSUPPORTED_CONTEXT` / `unsupported_or_invalid_context` | Wrong defender/turn, terminal or malformed position; no coverage conclusion. |

All search results expose `outcome_certification='uncertified'`. Only `FOUND`
carries a witness. An exhaustive fragment failure is a combinatorial conclusion,
not a strategic losing-position claim.

## Witness and independent verification

`CoverageWitness` is an untrusted, versioned (`coverage-6b1-v1`) record containing:

- Exact position/context identities and canonical target group tuple.
- Selected `CandidateEvidence`: rule identity, role-ordered squares and a complete
  claimed conditional coverage tuple for each candidate.
- Explicit `CoverageAssignment(group, candidate)` for each target, referring to a
  selected candidate.
- An `uncertified` outcome marker.

`verify_coverage_witness(position, witness)` returns either `REJECTED` with its
first detected reason, or `VERIFIED_UNCERTIFIED`. It reconstructs both snapshots
through the basic board validator, checks exact board/turn/player/mode/schema
identity and rejects terminal or unsupported contexts. It independently scans all
four line directions to reconstruct the 69-group geometry and filter Black
stones. It checks raw coordinate types and bounds, emptiness, Baseinverse landing
support and columns, vertical role/adjacency/parity predicates, and the complete
conditional coverage relation. It rejects duplicate candidates and checks raw
response-square disjointness for every selected pair. It requires exactly one
valid assignment per recomputed target to a selected, actually covering candidate.
Extra, duplicate, omitted or blocked target declarations are rejected.

The verifier calls **neither** candidate enumeration, `analyze_candidates`, the
search, its context factory, the compatibility dispatcher nor candidate coverage,
affected-square or prerequisite metadata. Tests replace those interfaces with
functions that raise to establish separation. It shares immutable data vocabulary,
coordinate/group constructors and the established basic board validator; it is
not an implementation in a separate language or a machine-checked formal proof.
Duplicated raw predicates must be separately reviewed when extending rules.
Selection and assignment ordering are irrelevant for checking; target and
per-candidate coverage tuples must be canonical, exact and duplicate-free.

Example:

```python
from games.connect4.victor import (
    Position, SearchStatus, VerificationStatus,
    search_covering_set, verify_coverage_witness,
)

position = Position.from_board(board, player_to_move=0)
result = search_covering_set(position, node_budget=1_000)
if result.status == SearchStatus.FOUND:
    checked = verify_coverage_witness(position, result.witness)
    assert checked.status == VerificationStatus.VERIFIED_UNCERTIFIED
    assert checked.outcome_certification == 'uncertified'
    # No strategic bound, safety verdict or game outcome follows from this API.
```

## Tests, examples and verification

The three new focused modules add **118 tests**:

- `test_victor_compatibility.py`: every supported pair in both orders, compatible
  and conflicting instances, duplicates, shared lower triggers, all 39 unsupported
  unordered matrix entries, malformed identifiers, and all pairs in the empty
  board's candidate universe against a raw-disjointness oracle and mirrors.
- `test_victor_coverage.py`: exact whole-board targets, original diagram 6.1
  coverage, constructed satisfiable/unsatisfiable endgames, root contradictions,
  individual coverage with incompatible joint obligations, exact node-budget
  boundaries, repeated determinism, mirrors and invalid contexts. An independent
  exhaustive subset enumerator cross-checks both endgames and their mirrors;
  each has only eight geometric candidates (256 subsets).
- `test_victor_verification.py`: 33 witness mutation cases, exact board binding,
  wrong players/turns/rules/squares/roles, coverage and assignment corruption,
  duplicates, conflicts, schema/outcome changes and terminal rejection. A manually
  assembled witness using diagram 6.1's 16 printed Claimevens verifies without
  search. Interface sabotage tests establish independent predicate recomputation.
  Two constructed basic-valid snapshots exercise successful nonempty Baseinverse
  and Vertical coverage; they are not attributed to a thesis diagram or legal replay.

Only labelled source fixtures are thesis examples. The endgames, replay sequences,
mutation cases and algorithm oracles are constructed here. Tests replay endgame
moves legally before evaluating the snapshots. The initial verification run found
two incorrect test assumptions that mutated data must compare unequal: Python
compares `True` equal to `1` and a string enum equal to its value string. The tests
now explicitly demand rejection of these type changes despite equality. No
production algorithm was changed to accommodate an incorrect expected result.

Verification from the existing `.venv`, with no Torch/training/benchmark workload:

```sh
.venv/bin/python -m pytest -q tests/test_victor_verification.py tests/test_victor_compatibility.py tests/test_victor_coverage.py tests/test_victor_rules.py tests/test_connect4_grounding.py tests/test_connect4_engine.py
git diff --check
```

Result: **258 passed**, including the 118 new tests and all 140 existing
Victor/grounding/engine cases. `git diff --check` passed. Work is confined to this
document, the Victor research package and the three new test modules.

## Limits, formal status and Phase 6B.2

The implemented predicates are source-grounded and cross-checked by focused
regressions and bounded independent reference algorithms. Verified witnesses
establish only that the submitted finite data satisfy the implemented fragment's
local predicates, complete conditional coverage and six compatibility entries.
There is **no formal position-bound acceptance**, proof-assistant theorem,
validated global strategy, exact solving, optimal play or claimed perfect player.

Outstanding correctness questions before Black strategic certificate acceptance:

- Does the exact Chapter 8.1 evaluation contract justify a joint strategy for
  every compatible fully covering set in this fragment, including odd-threat
  positions and all Zugzwang transitions? Reconstruct and independently review
  that argument rather than treating §8.1's prose as a checked certificate rule.
- What reachability evidence is required? Existing board validation checks
  dimensions, counts, gravity and terminal consistency, but not every history.
- What precise strategic bound would a future certificate assert, and what
  additional obligations establish it? Conditional prevention is not an exact
  draw or a Black win. Coverage failure still yields no strategic bound.
- Can an independently derived strategy execution model expose timing or
  obligation failures missed by local compatibility? Small exact endgames can
  falsify a proposed bound but cannot alone prove its general soundness.
- How should independent verifier implementations/versioning prevent semantic
  drift as composite rules and compatibility codes 2-4 are introduced?

Recommended **Phase 6B.2**: commission an independent mathematical review of the
Black Chapter 8 contract and certificate semantics, specify the exact proposed
non-loss bound and reachability prerequisites, and validate that specification
against tiny legally replayed endgames and adversarial strategy schedules. Keep
outcome acceptance disabled until those obligations are justified. Then add a
separate reviewed certificate boundary, with independent checking and versioning;
retain coverage witnesses as distinct untrusted research artifacts. Expand rules
and White contexts only in later source-reviewed increments, before exact search
or any public integration.
