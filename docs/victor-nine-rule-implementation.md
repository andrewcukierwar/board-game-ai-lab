# Victor: complete nine-rule Allis research system

Date: 2026-10-08. Branch `victor-complete-nine-rules`, based on
`phase6b5b-victor-hardening-review` at `c458fb8c7d41774564d252a76df21517406b4b14`.

**Summary.** All nine of Allis's strategic rules (§§6.1–6.9) are implemented,
together with the complete §7.4 compatibility matrix (constraints 1–4 and
N.B.(ii)). They are integrated into a bounded compatible-cover search with an
independent witness verifier. The output is **coverage evidence, never a game
value**. The `cl-bi-ve-black-nonloss-v1` certificate and executable strategy are
unchanged and accept none of the new rules.

Validation highlights:

- Producer, verifier and test reference agree on candidates, coverage and
  compatibility.
- The thesis diagrams 6.1 and 6.4–6.10 reproduce their stated claims.
- 7,224 nine-rule covers were checked against exact solutions with **no
  counterexample**. Loosening a source condition produces false covers that
  exact play refutes.

| Status | Rules |
| --- | --- |
| Fully operational: enumeration, prerequisites, coverage, §7.4, search, independent verification | all nine |
| Not provided (out of scope) | executable composite-rule responses, a nine-rule soundness proof, White (odd-threat / threat-combination) evaluation contexts |

## 1. Architecture

| Module | Responsibility |
| --- | --- |
| `rules.py` | CL/BI/VE two-square `RuleCandidate` (unchanged semantics). Composite rules are rejected here. |
| `composite.py` **(new)** | `Component`, `CompositeCandidate`, `SolutionClause`; intrinsic shape validation, board `prerequisite_failures`, `solution_clauses`, and enumeration of the six composite rules. |
| `compatibility.py` | Full 45-entry §7.4 matrix, constraint checks 1–4, `Footprint`, `failed_constraints`, `compatible`. |
| `coverage.py` | Black context, the CL/BI/VE search (unchanged API and results), and the shared bounded DFS `backtrack_cover`. |
| `nine_rules.py` **(new)** | `analyze_nine_rules`, `search_nine_rule_cover`, fast `conflict_masks`, `NineRuleWitness`, `NINE_RULE_OBLIGATIONS`. |
| `nine_rule_verification.py` **(new)** | `verify_nine_rule_witness`: independent raw-cell recomputation of every finite predicate. |
| `verification.py`, `certificate*.py`, `strategy.py` | Unchanged. They are still CL/BI/VE-only and reject nine-rule witnesses (tested). |

Pipeline: `Position` → `black_evaluation_context` (White to move, nonterminal,
Black defender) → enumerate the requested rules → attach clause coverage of
White's potential groups → drop zero-coverage candidates → build conflict
bitmasks → bounded MRV DFS (Allis §9.2 `FindChosenSet`) →
`NineRuleSearchResult` with an inspectable witness.

## 2. Rule definitions as implemented

Rows are counted from the bottom (1–6); "playable" means directly playable now.
Every composite also inherits the solutions of its constituent Claimevens
(groups containing the upper square) and Verticals (groups containing both
squares), as the thesis states for each rule.

| Rule | Required (implemented exactly) | Solutions (clauses) |
| --- | --- | --- |
| Claimeven §6.1 | empty (lower, upper) pair; upper even | groups ∋ upper |
| Baseinverse §6.2 | two playable squares, different columns | groups ∋ both |
| Vertical §6.3 | empty (lower, upper) pair; upper odd | groups ∋ both |
| Aftereven §6.4 | group with no White stone; ≥1 empty square; **every** empty square even with an empty square directly below; Claimeven (below, square) for each | `aftereven-columns`: the group has, **in every Aftereven column**, a square above that column's empty group square; plus component Claimevens |
| Lowinverse §6.5 | two different columns, each an empty pair with odd upper; heights may differ; lowers need not be playable | `upper-pair`; both component Verticals |
| Highinverse §6.6 | two different columns, each three empty consecutive squares (lower, middle, upper) with even upper | `upper-pair`; `middle-pair`; vertical (middle, upper) per column; **conditional** `lower-first+upper-second` only if the first lower is playable now, and symmetrically for the second |
| Baseclaim §6.7 | ordered roles (first, second, third), all playable, distinct columns; second on an odd row, so the square above it is even | `first+above-second`; `second+third` |
| Before §6.8 | group with no White stone; ≥1 empty square; no empty square on row 6. Each empty square *s* gets either Vertical (*s*, *s*+1), with any parity, or Claimeven (*s*−1, *s*) if *s* is even and *s*−1 is empty. Parts are pairwise disjoint; **not all Claimevens** | `successors`: the group contains the successor of **every** empty square; plus components |
| Specialbefore §6.9 | Before-type group; one empty group square *p* is playable and gets **no** component. The extra square *x* is playable, outside the group, in another column. Components cover the other empty squares; all parts are disjoint; all-Claimeven is allowed | `successors+extra`: all successors (including *p*'s) **and** *x*; `playable-pair`: *p* and *x*; plus components |

Identity and determinism:

- Components are normalized to ascending (column, row), and Highinverse
  columns to ascending column.
- Baseclaim and Specialbefore role order is significant and never normalized.
  (b1, c1, e1) and (e1, c1, b1) are different rules with different coverage
  (tested on diagram 6.7).
- Enumeration is deterministic: CL, BI, VE, then the six composites in
  `RuleName` order. Rules are generated in `ALL_GROUPS` or column order.
  Duplicates cannot arise because identity is canonical.
- `reflected()` maps every candidate to its mirror image. Mirror invariance of
  enumeration and coverage is tested.

## 3. Composite representation

`CompositeCandidate(rule, group, components, roles)`:

| Rule | `group` | `components` | `roles` |
| --- | --- | --- | --- |
| AE | the Aftereven group | Claimevens, one per empty square | `()` |
| LI | `None` | two odd-upper Verticals | `()` |
| HI | `None` | `()` | `(l1, m1, u1, l2, m2, u2)` |
| BC | `None` | `()` | `(first, second, third)`; `baseclaim_square` is derived |
| BE | the Before group | CL/VE per empty square | `()` |
| SB | the Specialbefore group | CL/VE per *other* empty square | `(playable group square, extra square)` |

A `Component(rule, lower, upper)` is a Claimeven (even upper) or a Vertical
(adjacency only). That is deliberately weaker than standalone `RuleCandidate(VE)`,
which still rejects even uppers: §6.8 uses Vertical e1-e2. `before_square`
is the group square a component handles: the upper square for a Claimeven,
the lower for a Vertical.

The constructor checks intrinsic geometry only. `prerequisite_failures(c, p)`
lists every violated board condition: occupied squares, non-playable role
squares, White stones in the group, and components that do not handle exactly
the group's empty squares. Every enumerated candidate has no failures (tested).

Coverage is a list of labelled `SolutionClause`s. A group is solved when it
meets every requirement set of some clause. The labels appear in witness
assignments, so each group's coverage is explained (for example
`aftereven-columns`, `claimeven:f1-f2`, `playable-pair`).

## 4. Compatibility (§7.4)

The matrix is transcribed from the visually checked p.50 table. It is
symmetric, defines all 45 unordered pairs, and is checked at import time. A
test compares it against a separate string transcription.

| Code | Implementation |
| --- | --- |
| 1 | complete square sets are disjoint |
| 2 | every Claimeven part in an inverse column lies **entirely above** the inverse's squares in that column. Claimeven parts are CL rules, the CL components of AE/BE/SB, and the Baseclaim's (second, above-second) pair |
| 3 | in every column the two square sets are disjoint or equal. A column part containing a Specialbefore special square never counts as equal (N.B.(ii)) |
| 4 | disjoint squares, and the inverses' column sets are equal or disjoint |

"Set of squares":

| Rule type | Squares |
| --- | --- |
| CL/BI/VE | both squares |
| AE/BE | component squares |
| SB | component squares + *p* + *x* |
| LI / HI | 4 / 6 squares |
| BC | first, second, third + the square above the second |

Identical instances always conflict.

**Pairwise is not global.** `compatible` is Allis's pairwise criterion only.
Joint executability of a whole collection is listed as an unproven obligation
(§10). It is not inferred.

Search conflicts are built by `conflict_masks`. Pairs that share no square
satisfy constraints 1 and 3 trivially, so only overlapping pairs and pairs
involving an inverse are evaluated. Tests check that the result is identical to
the O(n²) `pairwise_conflicts` reference.

## 5. Covering-set search

`search_nine_rule_cover(position, defender=1, node_budget=100_000, rules=ALL_RULES)`:

| Status | Meaning |
| --- | --- |
| `compatible_covering_set_found` | a §7.4-compatible collection solves every White potential group; the witness is still untrusted |
| `no_cover_in_supported_universe` | the DFS finished over the **complete** enumeration of the requested rules. This says nothing about the game value (§5.4) |
| `unknown_budget_exhausted` | the cutoff was reached; nothing is concluded |
| `unsupported_or_invalid_context` | not White to move, terminal, invalid, or White defender |

- The DFS branches on the uncovered group with the fewest compatible
  candidates (ties go to the lowest group index), trying candidates in
  enumeration order.
- Fully explored failing states `(uncovered, blocked)` are memoized. This is
  sound because the subproblem depends only on that pair, and memo hits are
  reported separately.
- The budget counts visited states, including the root and solution leaves.
- The established `search_covering_set` uses the same DFS with memoization
  disabled. Its statuses, node counts and witnesses are unchanged, and the
  existing exact-boundary tests still pass.
- `rules=` restricts the universe for ablation. With `rules=(CL, BI, VE)`, the
  nine-rule search agrees with the established search (tested on fixtures and
  mirrors).
- Nothing is truncated. The universe is finite and small: at most
  69 × 16 Befores and 69 × 4 × 6 × 8 Specialbefores in theory, and 866
  candidates on the empty board.

## 6. Independent verification

`verify_nine_rule_witness(position, witness)` rejects the first defect it finds.
It recomputes the following without calling producer enumeration, clauses,
prerequisites, compatibility, footprints, the search, or any
`CompositeCandidate`/`Component` property (tested by monkeypatching them all to
raise):

- position identity and turn;
- witness schema, `uncertified` status, the exact obligations tuple and the
  declared rule set;
- the target universe, from its own derivation of the 69 windows;
- every candidate's shape, canonical order, parity, playability and
  group-hole correspondence, from raw cells in bottom-based `(column, row)`
  coordinates;
- every clause (label and requirement sets), and the exact solved-group tuple;
- all selected pairs against its own transcription of the matrix and
  constraints, with the failing constraint codes reported;
- the clause in every assignment, and complete coverage of all targets.

Success returns `coverage_verified_outcome_uncertified` together with
`NINE_RULE_OBLIGATIONS`. Twenty-three corruption kinds are rejected (tested),
including:

- a component of the wrong kind, swapped Specialbefore roles, or a non-playable
  extra square;
- a forged Highinverse conditional clause;
- a Claimeven crossing a Lowinverse, which violates constraint 2 alone;
- an undeclared rule family, or stripped obligations.

The old `verify_coverage_witness` and `certificate_from_witness` reject
nine-rule witnesses, and the nine-rule verifier rejects old witnesses.

## 7. Source discrepancies and resolutions

| Issue | Resolution |
| --- | --- |
| Constraint 2 literally says "no Claimeven *below*". Read literally, it lets a Claimeven **overlap** an inverse for CL–LI/HI (and for BE/SB–LI, where constraint 3 still applies). §7.1 only says a Claimeven may be used *above* | A Claimeven must lie **entirely above** the inverse. Empirically necessary: the literal reading produced 48 exact-play counterexamples in 2,071 covers (§8) |
| N.B.(ii) says Specialbefore combines with "another Before, **Claimeven** or Aftereven" if they share Verticals/Claimevens, but the matrix entry CL–SB is constraint 1 (disjoint) | Follow the matrix (CL–SB disjoint). Apply N.B.(ii) to the constraint-3 partners (AE, BE, SB) and to BE/SB–LI |
| §6.8: an all-Claimeven Before "would be equal to an Aftereven" | Excluded from enumeration. This is provably lossless: the same group is an Aftereven, and the Aftereven's solutions and compatibilities are a superset |
| §6.9 does not say whether the extra square may be a group square, or how *p*'s component is replaced | *x* ∉ group, a different column from *p*, and *p* gets no component. The diagram 6.10 narrative frees e3 for Claimeven e3-e4 |
| Constraint 3 compares square sets, so a Before's even-upper Vertical f1-f2 "equals" an Aftereven's Claimeven f1-f2 | Allowed as written. Black's demands are consistent: never play f1, and answer f1 with f2 |
| Vertical (stacked) Before groups | Never representable with disjoint components, except with a single empty square. The generic disjointness check handles this; no special case |
| §6.2 printed list omits b1-d3 (pre-existing finding) | Unchanged: follow the formal definition |

The thesis examples reproduce as stated:

| Diagram | Reproduced claim |
| --- | --- |
| 6.4 | the two printed Afterevens plus Claimevens solve all groups; CL/BI/VE alone cannot; exact result: Black wins |
| 6.5 | the Aftereven d2-g2 does not solve c3-f3, and **no** candidate of any rule does; exact result: White wins |
| 6.6 | the printed Lowinverses and Highinverses exist, and the Highinverse conditional pairs are "no use" there |
| 6.7 | the Baseclaim (b1, c1, e1) solves b1-e4 and c1-f1; Claimeven c1-c2 conflicts with Baseinverse c1-e1 |
| 6.8 | d3-g3 is solved **only** by the Before d2-g2 (Vertical f2-f3) |
| 6.9 | b5-e2 is the only group containing b5 and e2; both component choices exist |
| 6.10 | the found cover contains Allis's exact Specialbefore (CL f1-f2, CL g1-g2, e2/d3) and Baseinverse a1-b1 |

## 8. Exact-solver evidence

All exact checks use White-to-move positions from random legal play. Up to 10
empty squares, the repository's `exact_oracle` was used. For 12–22 empty
squares, a scratch alpha-beta solver was used; it agreed with the oracle on 300
positions and is not committed.

| Empty squares | Positions | Nine-rule covers | Composite-dependent covers | White wins despite a cover |
| --- | ---: | ---: | ---: | ---: |
| ≤ 10 | 24,600 | 4,424 | 1,752 (CL/BI/VE search finds none) | **0** |
| 12–16 | 14,989 | 1,600 | 1,079 (cover uses a composite) | **0** |
| 18–22 | 16,754 | 1,200 | 1,034 (cover uses a composite) | **0** |

All six composite rules occur in these covers. Highinverse is rare (38 uses);
by comparison there were 2,057 Aftereven, 1,601 Before, 400 Lowinverse,
368 Specialbefore and 120 Baseclaim uses.

Every cover containing an Aftereven was an exact **Black win** (2,057 cases).
This is consistent with §9.2's "an Aftereven means Black wins", which is
recorded but **not inferred**.

**Mutation sensitivity** (same samplers; covers produced by a deliberately
loosened rule, and how many of them exact play refutes):

| Loosened condition | Covers | Refuted |
| --- | ---: | ---: |
| none (faithful rules) | 1,271 | 0 |
| Aftereven timing in **any** column | 4,960 | 31 |
| Before: any single successor | 5,259 | 162 |
| Specialbefore without the extra square | 5,454 | 200 |
| Lowinverse claims both uppers | 5,179 | 151 |
| constraint 2 removed | 5,011 | 34 |
| constraint 2 read literally (overlap allowed) | 2,071 | 48 |
| constraint 3 removed | 6,015 | 322 |
| constraint 1 removed | 2,164 | 261 |
| Highinverse conditional pairs unconditional | 6,011 | 0 |
| Baseclaim roles swapped | 6,045 | 0 |

One small fixture (≤ 10 empty squares) per refuted mutant is a regression test:
the faithful search reports no cover, the oracle says White wins, and the
mutant claims a cover.

## 9. Performance

Measured on a laptop with Python 3.11; times are for enumeration plus search.

| Position | Candidates | Useful | Conflict pairs | DFS nodes | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| empty board | 866 | 808 | 192,946 | 891 (no cover) | 0.74 s |
| diagram 6.4 | 431 | 349 | 40,296 | 9 (found) | 0.20 s |
| diagram 6.10 | 437 | 316 | 27,715 | 12 (found) | 0.15 s |
| 8 plies (median of 40) | 603 | 518 | 87,780 | 14 (max 397) | 0.45 s (max 0.56) |
| 16 plies | 361 | 280 | 27,518 | 3 (max 187) | 0.14 s (max 0.29) |
| 24 plies | 158 | 100 | 3,357 | 1 (max 17) | 0.02 s |
| 32 plies | 41 | 21 | 138 | 1 (max 41) | < 0.01 s |

Building the conflict graph dominates, as Allis also reports (§9.3). No
sampled position exhausted the default budget. The budget remains explicit,
and an exhausted search reports `unknown`.

## 10. Remaining proof obligations

Carried in every result and witness as `NINE_RULE_OBLIGATIONS`:

1. **Zugzwang control.** §8.1 argues Black need not check it. No reviewed proof
   covers the composite rules.
2. **Local soundness** of each composite rule: Aftereven/Before timing,
   Lowinverse/Highinverse parity, and the Baseclaim and Specialbefore variant
   analyses. These rest on Allis's informal arguments.
3. **Global composition.** Pairwise §7.4 compatibility must imply simultaneous
   executability of the whole collection. The natural invariant is §7.1's: each
   filled rule releases an even number of squares. It is stated, not proved.
4. **Executable responses.** No Black response function exists for the
   composite rules, so no strategy can be replayed or audited.
5. The §9.2 Aftereven "Black wins" strengthening is not inferred.
6. Historical reachability is not checked.

## 11. Limitations

- **Evaluation context.** Only Black to defend with White to move is
  supported. White odd-threat and threat-combination contexts (§§8.2–8.4) and
  their restricted regions are not implemented.
- **Weak empirical exposure for HI and BC.** Highinverse and Baseclaim are
  rarely selected in random samples. Their source conditions are implemented
  and tested on thesis and constructed boards, but the exact-oracle mutation
  check could not show that the Highinverse playability condition and the
  Baseclaim role assignment are load-bearing. In a dedicated run (about 56,000
  positions with 8–18 empty squares), both mutants produced about as many
  covers as the faithful rules (3,035 covers, again 0 counterexamples) and none
  was refuted. These rules almost never decide whether a cover exists in random
  play. A targeted position generator is needed.
- **Explicit interpretations.** Constraint 2, N.B.(ii) and the Specialbefore
  extra square follow the interpretations in §7.
- **No game values.** Exhaustive failure is not a White win, and a found cover
  is not a proved Black non-loss.

## 12. Tests

New files (189 tests):

- `test_victor_nine_rules.py` (68): every rule's thesis diagram;
  constructed positive and negative shapes (parity, adjacency, roles, columns,
  overlap, all-Claimeven Before, upper row, extra-square placement);
  prerequisites (occupied, not playable, White stone in group, unhandled
  holes); conditional Highinverse coverage; mirroring; canonical order; and
  agreement with the independent test reference
  (`victor_validation/nine_rule_reference.py`) on random positions.
- `test_victor_nine_rule_compatibility.py` (20): §5.3, §6.7, §6.9 and §7.1–7.3
  cases for every constraint; the interpretations; duplicates; corrupted
  inputs; agreement with the reference matrix, symmetry and mirroring on random
  universes; fast versus pairwise conflicts.
- `test_victor_nine_rule_search.py` (101): thesis covers; budget boundaries;
  every status; argument validation; the CL/BI/VE subset versus the established
  search; agreement with an **independent exhaustive non-MRV search over the
  reference predicates** on 40 endgames; 23 verifier corruptions; verifier
  independence; certificate-boundary isolation; an exact-oracle falsification
  sample; seven mutant fixtures.

Changed existing test: `test_all_unimplemented_matrix_entries_fail_closed`
asserted the 39 then-missing entries raised. It now asserts all 45 entries
against an independent transcription of the thesis table. No other existing
test changed.

Results:

- Full backend: **1327 passed, 14 skipped**. All skips are `torch`-unavailable
  module skips in neural/DQN suites.
- Victor certificate, strategy, oracle and audit suites pass unchanged.
- `git diff --check` is clean.

## 13. Recommended next step

Implement and audit an **executable nine-rule Black response function**, the
composite analogue of `sigma_R`. Validate it with the existing strategy harness
against every White continuation on exact-solvable endgames. That turns
obligations 2–4 from informal arguments into falsifiable, replayable claims.

Only after that should the White evaluation contexts (§§8.2–8.4) be added.
Those are required for a solver that claims White wins, and so for optimal play.

**GO** for building the nine-rule executable strategy and White contexts on this
foundation. **NO-GO** for treating nine-rule covers as proofs, exposing them
publicly, or folding composite rules into the CL/BI/VE certificate.
