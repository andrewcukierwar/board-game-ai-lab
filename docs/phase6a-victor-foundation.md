# Phase 6A: Victor formal solver foundation

This phase adds an independent research package at `games/connect4/victor/`.
It enumerates **Claimeven, Baseinverse and Vertical candidates** and their
**conditional opponent-group coverage**. It does not certify strategic
applicability, control of Zugzwang, compatibility, a safe position, a win/draw/loss,
or an optimal move. It is not exposed through an agent, API, UI, or LLM schema.

## Primary-source audit

Source: Victor Allis (1988), *A Knowledge-Based Approach of Connect-Four: The Game
Is Solved: White Wins*, local ignored
[`research/references/allis-1988-connect4.pdf`](../research/references/allis-1988-connect4.pdf).
The [public thesis](https://tromp.github.io/c4/connect4_thesis.pdf) identifies the
same work; the implementation was checked against the **local PDF**, not against
an external implementation or the existing knowledge catalog.

The 91-page local edition's 1-based PDF pages coincide with the thesis contents'
page references. Body pages have no visible folios. Thus §6.1 is PDF pages 36–37
(zero-based indices 35–36), and §7.4 is PDF page 50 (index 49). Text was extracted
with macOS PDFKit; pages 36–38 and 50 were rendered and visually inspected,
including diagrams 6.1–6.3 and the compatibility table. Extraction, renderings and
tooling stayed in `/tmp`, with no project dependencies added.

Consulted Chapters 3–8:

| Reference | Relevance |
| --- | --- |
| §§3.1–3.4, pp.16–24 | Potential versus playable threats, parity, timing and tactical counterexamples. |
| §§4.1–4.5, pp.25–31 | Definition and limitations of Zugzwang control; simple follow-up can lose from the initial position. |
| Chapter 5, p.32 | Potential opponent group: four consecutive squares containing no stone of the defending player. |
| §§5.3–5.4, pp.33–35 | Conflicting responses, complete coverage, and failure to cover implying no conclusion. |
| Chapter 6 introduction, p.36 | Opponent-to-move evaluation and White's restricted board region. |
| §§6.1–6.3, pp.36–39 | Implemented formal requirements and solutions. |
| §§6.4–6.9, pp.39–46 | Remaining six rule definitions, read for architecture only. |
| §§7.1–7.3, pp.47–49 | Zugzwang interactions; Baseinverse and Vertical are locally Zugzwang-independent. |
| §7.4, p.50 | Pair-specific compatibility constraints, conjunctive entries and Specialbefore exception. |
| §8.1, p.51 | Black may attempt coverage without separately establishing Zugzwang control first. |
| §8.2, pp.51–53 | White's odd-threat setup, reserved column and direct-playability exception. |
| §§8.3–8.4, pp.53–57 | Threat combinations, their distinct variants and region-specific conclusions. |

### Existing code and reuse decisions

Reviewed `docs/allis-grounding.md`, `grounding/knowledge.py`,
`grounding/analysis.py`, `agents/victor_agent.py`, the engine/Board classes and
`tests/test_connect4_grounding.py`.

The experimental `VictorAgent` is not a foundation for mathematical evidence:
its A1/A2/B/C/D patterns are locally invented; its scan demands a five-square
window even when looking for a four-square pattern; its initial proof-number
node selection can return without expanding the root; its cache key omits the
side to move; and cached `None` can flow into Boolean proof/disproof handling.
It has no Allis compatibility/coverage system. None of its implementation or
behavior was changed or imported.

Reuse is deliberately narrow:

- Adapt `grounding.analysis.GROUPS` into typed immutable groups. Tests independently
  enumerate endpoint geometry, check the 24 horizontal + 21 vertical + 24 diagonal
  total, and cross-check both grounding `GROUPS` and engine `_WINDOWS` (69 each).
- Reuse `validate_position` and `outcome` for established basic board legality and
  terminal detection. This avoids silently accepting a different position domain.
- Introduce new typed coordinates and rule evidence rather than altering the
  public grounding dictionaries or reference-only catalog. No rule claims are
  inserted into `supported_allis_rule_applications`.
- No new dependencies. Importing the existing grounding package also initializes
  its existing context/engine modules and NumPy dependency; no Torch, training,
  prototype search, PDF access, network access or LLM calls are used at runtime.

## Correctness milestones

1. **Locally correct candidate:** the empty/playable squares have the specified
   shape, and the conditional solved-group relation matches its formal rule.
2. **Compatible collection:** every selected pair satisfies all of §7.4's required
   constraints; component roles and special squares matter.
3. **Position-level proof:** the correct opponent-to-move Chapter 8 evaluation
   context and permitted region have been established, and a compatible set plus
   justified region-specific evidence covers every relevant opponent group. An
   independent verifier must check this entire chain.
4. **Complete game-theoretic solution:** sound strategic bounds are combined with
   complete exact search to establish an exact value and optimal move, including
   all unresolved continuations and correct terminal handling.

Only milestone 1 is implemented, with evidence explicitly labelled conditional.
A collection can cover every group yet demand incompatible replies. A compatible
collection can leave a dangerous group uncovered. Neither is a position proof.
Even a verified non-loss bound for Black is not an exact draw or a proven win.
Failure of the rule system to find a proof is **unknown**, never a loss.

## Implemented local definitions

All boards are 7 columns × 6 rows. `board[row_index][column]` is top-first;
`Square(5, 3)` is d1. Thesis rows are `6 - row_index`, numbered upward from 1.
Parity always refers to the thesis row. Player `0 = X = White` moves first;
`1 = O = Black` moves second. The `defender` is the player for whom candidate
coverage is being inspected, independently of the actual side to move.

A **potential opponent group** contains no defender stone. Opponent stones are
allowed, and the group need not be an immediate threat. For each candidate, the
report includes exactly the potential opponent groups satisfying the following
membership predicates:

| Rule | Geometric prerequisites | Conditional coverage | Assumptions still outside detection |
| --- | --- | --- | --- |
| Claimeven (§6.1, pp.36–37) | Two empty vertically adjacent squares; upper row even (2, 4 or 6). | Group contains the upper square; containing the lower is unnecessary. | Zugzwang-dependent strategy; evaluation region, compatible full coverage and proof verification. |
| Baseinverse (§6.2, pp.37–38) | Two distinct directly playable empty squares, necessarily in different columns. No parity or equal-height requirement. | Group contains both squares. | Locally Zugzwang-independent (§7.2), but simultaneous response obligations and the position framework remain unverified. |
| Vertical (§6.3, pp.38–39) | Two empty vertically adjacent squares; upper row odd (3 or 5). | Group contains both squares, hence only vertical groups. | Locally Zugzwang-independent (§7.2), with the same unverified framework. |

Claimeven and Vertical do **not** require their lower square to be directly
playable. Occupying it first would destroy the empty-pair prerequisite. An even
upper square is not a standalone Vertical; Before's parity exception is a future
component-specific representation, not a relaxation of this detector.

Baseinverse enumeration retains all geometrically valid pairs, including those
covering no group. Section 6.2 explicitly distinguishes such possible but useless
pairs. Consumers may filter on nonempty `conditional_solved_groups`; the detector
does not conflate usefulness with geometry.

No detector restricts White's candidates to a purported threat region, assumes
that White has a usable odd threat, or decides that Black controls Zugzwang.
Both defenders can be inspected on either turn. These choices are deliberate:
this is candidate inventory, not Chapter 8 evaluation.

## Types and public interfaces

| Module | Structures and responsibilities |
| --- | --- |
| `geometry.py` | Frozen, ordered `Square(row_index, column)` with `row`, `name`, `is_even`, `from_name`, `reflected`; canonical `Group(squares)` validates four distinct consecutive collinear squares; `ALL_GROUPS`. |
| `position.py` | Frozen `Position`, `Player` type, `Position.from_board(board, player_to_move)`, basic validation, `is_empty`, physical `is_playable`/`playable_squares`, `potential_groups(player)`, actual terminal detection. |
| `rules.py` | All nine `RuleName` identifiers; implemented `RuleCandidate` shapes; typed `Prerequisites`, affected/coverage squares, `ThesisReference`, Zugzwang-dependency metadata and three enumerators. |
| `evidence.py` | `CandidateEvidence(candidate, conditional_solved_groups)` and board-bound `CandidateReport`; `analyze_candidates(position, defender)` computes conditional local evidence. |
| `contracts.py` | Design-only `CompatibilityConstraint`, `CompatibilityObligation`, `EvaluationContextDraft`, `CoverageAssignment`, `CoveragePlan`, `CertificateDraft` and a `CertificateVerifier` protocol. No checking algorithms or accepted-proof type. |
| `__init__.py` | Explicit exports for implemented research interfaces; future contracts are accessed through their own module. |

Example:

```python
from games.connect4.victor import Position, analyze_candidates, enumerate_candidates

position = Position.from_board(board, player_to_move=0)
candidates = enumerate_candidates(position)
report = analyze_candidates(position, defender=1)
assert report.status == 'candidates_only'
# report.evidence[i].conditional_solved_groups is NOT an established prevention claim.
```

A candidate holds `(lower, upper)` roles for Claimeven/Vertical. Baseinverse
pairs are unordered and normalized by matrix coordinate. Both involved squares
are retained as `affected_squares`, including Claimeven's lower trigger.
`Prerequisites` states emptiness, playability, adjacency and upper parity.
Construction validates intrinsic geometry but does not check a board; only the
enumerators check the snapshot. Manually constructed candidates/reports/drafts
are untrusted data and must never become accepted proofs through a type cast.

Deterministic ordering is explicit: rule order CL, BI, VE; vertical pairs by
column left-to-right and upper thesis row ascending; Baseinverses by combinations
of landing squares sorted by `(row_index, column)`; groups by canonical square
tuples in that same coordinate ordering. Tuples and frozen values ensure detached
results. Symmetric positions give mirrored **sets**, not necessarily the same
list indices. Distinct candidates covering the same groups are retained because
their eventual compatibility may differ; duplicate rule instances are not emitted.

Malformed dimensions/cells, gravity violations, inconsistent counts/turn,
simultaneous winners and invalid terminal winners fail with `ValueError` through
the existing validator. A valid terminal win or draw produces no candidates.
Physical landing-square queries and geometric group queries remain meaningful on
terminal snapshots, but empty evidence must never be read as a vacuous proof.
Basic validation is not full historical reachability verification; a certificate
boundary will require legal replay or a separate reachability argument.

Reports carry the complete immutable board and side to move, defender, method
version, source identity, rule-specific page references, framework references,
and explicit unverified obligations. They have no strategic outcome, safety,
move, proof-success or Zugzwang-control field.

## Tests and source examples

`tests/test_victor_rules.py` distinguishes actual diagrams from constructed
fixtures. The diagram boards were transcribed from rendered PDF pages; their
short engine replays were constructed here and are **not** attributed to Allis.
Each replay is checked against the transcribed full matrix.

- **Diagram 6.1**, §6.1, pp.36–37: exact 16-pair Claimeven list, full d-column,
  skipped occupied lower squares at c1/e1 and conditional group membership.
  The thesis's whole-position draw argument is not asserted by these tests.
- **Diagram 6.2**, §6.2, pp.37–38: all seven listed useful Baseinverses, the printed
  coverage for a1–b1, c4–d3 and d3–f1, and the two explicitly useless examples
  a1–c4 (no common group) and f1–g1 (Black e1 blocks d1–g1).
- **Source-list omission:** the same diagram also admits **b1–d3**, solving
  **b1–c2–d3–e4**. Both endpoints are directly playable; c2 is White and the other
  squares are empty. This pair is absent from the printed seven-pair list.
  The test expects **eight** useful pairs under the formal definition, recording
  the discrepancy rather than changing the rule or pretending the list is exact.
- **Diagram 6.3**, §6.3, pp.38–39: e4–e5 conditionally solves exactly e2–e5 and
  e3–e6 for Black, while occupied e3 prevents a Claimeven for e4.

Constructed fixtures cover all 42 coordinate mappings, invalid coordinates and
groups, board boundaries, all column heights 0–6, unsupported versus landing
squares, wrong parity/adjacency/column/role cases, occupied pairs, distinct-pair
requirements, both defender colors, both terminal winners and a full draw,
invalid boards, deterministic ordering, duplicate elimination and immutability.

An independent endpoint-based group oracle cross-checks the 69 geometric groups.
On deterministic sampled legal prefixes, a separate bottom-based predicate
oracle exhausts **all unordered empty-square pairs**, checking candidate
completeness as well as absence of false positives. Coverage is independently
calculated from raw board cells and group membership; it does not call the
production prerequisite, coverage or playability properties. Mirroring checks
both candidate sets and every conditional coverage set for both defenders.
These are bounded CPU tests, not exhaustive position solving.

## All nine rules and planned extensions

The following remaining definitions describe intended work, not implemented
capabilities. The enum rejects attempts to construct candidates for these six
rules today.

| Rule | Status and intended representation/coverage |
| --- | --- |
| Claimeven | Implemented local candidates and conditional coverage as above. |
| Baseinverse | Implemented local candidates and conditional coverage as above. |
| Vertical | Implemented standalone odd-upper candidates and conditional coverage as above. |
| Aftereven (§6.4, pp.39–40) | Planned: own unblocked group completable with Claimevens; retain missing-square columns and component roles. Timing coverage must require a higher square in **every** missing-square column, plus all component Claimeven solutions. |
| Lowinverse (§6.5, pp.40–41) | Planned: two empty vertical pairs in different columns, upper squares odd, heights may differ and lowers need not be playable. Cover groups through both uppers plus both constituent Vertical solutions. |
| Highinverse (§6.6, pp.41–42) | Planned: two empty three-square column segments, tops even. Cover both tops, both middles, each segment's top two squares, and bottom/opposite-top pairs only when that bottom is directly playable. |
| Baseclaim (§6.7, pp.42–43) | Planned: three distinct landing squares with ordered roles; an even empty square above the second. Cover first/above-second and second/third groups. Preserve the coordinated response structure instead of combining conflicting standalone rules. |
| Before (§6.8, pp.43–45) | Planned: own unfinished unblocked group, no empty top-row square; explicit component alternatives per missing square. Cover all successors collectively plus retained component solutions. Allow even-upper Vertical components only in this representation; prefer the stronger Aftereven for all-Claimeven cases. |
| Specialbefore (§6.9, pp.45–46) | Planned: Before-like structure with a playable internal square and external playable square in another column. Model replacement of the internal response explicitly; cover successors plus external square, both special playable squares and retained components. Do not retain the replaced Vertical's coverage. |

## Compatibility, coverage and certificate design

`contracts.py` is vocabulary for future work, not a skeleton that returns
successful checks. `CertificateDraft.status` is fixed to `unverified`.
There is no concrete `CertificateVerifier`, no accepted certificate constructor,
no compatibility function and no covering-set search in this phase.

The future compatibility dispatcher must be symmetric and derive the required
constraints from this §7.4 table (p.50), visually checked against the PDF. Empty
cells reflect the lower-triangular presentation, not permissive combinations.

| | CL | BI | VE | AE | LI | HI | BC | BE | SB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CL | 1 | | | | | | | | |
| BI | 1 | 1 | | | | | | | |
| VE | 1 | 1 | 1 | | | | | | |
| AE | 1 | 1 | 1 | 3 | | | | | |
| LI | 2 | 1 | 1 | 1 & 2 | 4 | | | | |
| HI | 2 | 1 | 1 | 1 & 2 | 4 | 4 | | | |
| BC | 1 | 1 | 1 | 1 | 1 & 2 | 1 & 2 | 1 | | |
| BE | 1 | 1 | 1 | 3 | 2 & 3 | 1 & 2 | 1 | 3 | |
| SB | 1 | 1 | 1 | 3 | 2 & 3 | 1 & 2 | 1 | 3 | 3 |

Constraint 1: disjoint affected squares. Constraint 2: no Claimeven below the
inverse. Constraint 3: square sets in each column are disjoint or equal.
Constraint 4: disjoint square sets and inverse column sets either disjoint or
identical. Multiple codes are conjunctive. The two Specialbefore special squares
cannot be treated as equal shared components of another rule (the table's
additional note). Component squares, Claimeven lower roles and inverse columns
must remain available for these checks; a single generic overlap test is unsound.
Exact code for the ordering and Specialbefore exceptions needs separately
reviewed examples from §§7.1–7.4 before implementation.

Proposed pipeline:

1. Validate/snapshot the position, bind the defender and require opponent-to-move
   evaluation. For Black, attempt the Chapter 8.1 framework without an independent
   up-front control claim. For White, derive a specific §8.2 odd-threat setup or
   §8.4 threat-combination variant, including playable-square exceptions and
   reserved column(s). A user-supplied column exclusion is not evidence.
2. Derive the complete target group universe and every context-specific eliminated
   group from this setup. Preserve explicit reasons for those eliminations.
   Enumerate rule candidates only on the permitted region.
3. Compute local conditional coverage and an incompatibility graph using the
   table and all special constraints. Search for a compatible covering set with
   deterministic backtracking over the least-covered remaining group. A budget
   cutoff or an unsuccessful search returns unknown.
4. Emit a versioned certificate draft containing the full board/turn/defender,
   evaluation setup and source references, role-labelled rule instances, required
   compatibility obligations and explicit group-to-rule assignments. Exact square
   identities are canonical; no process-dependent hashes are authoritative.
5. An **independent verifier** recomputes reachability/terminal validity, evaluation
   context, region, each rule's local predicates, all covered/excluded groups,
   the full required pair matrix and coverage completeness. It must ignore any
   caller-supplied claim that an obligation is unnecessary. Reusing search's
   cached coverage or accepting its list of targets is insufficient. Corruptions
   of a square, turn, defender, region, role, target, matrix code or assignment
   must cause rejection. Verification must be versioned and bound to the exact
   snapshot. Only a future verifier can create an accepted **position bound**.
6. Add exact search around verified bounds and terminal values. Distinguish
   non-loss, win and exact draw; key transpositions by board, side and evaluation
   perspective. Validate optimal moves against small exhaustive endgames before
   any public integration. Complete search, not incomplete strategic coverage,
   is responsible for closing unresolved branches.

An independent certificate checker is a separate trust boundary, not a second
call to `analyze_candidates`. The present protocol's diagnostics do not award a
proof or an outcome, even if a future implementation returns no rejection reasons.

## Limitations, open questions and next phases

No unresolved ambiguity remains in the implemented three formal definitions.
The diagram 6.2 list discrepancy is documented and resolved in favor of the
formal definition; it must not be erased by a fixture-specific exception.

Outstanding work includes proof of historical reachability at the verifier
boundary, precise component semantics for the six remaining rules, every
compatibility constraint, all Chapter 8 context and threat-region derivations,
compatible-set search and independent certificate acceptance. Chapter 8.4 explicitly
leaves its general conclusions to variant analysis; an eventual implementation
needs those variants checked, not merely a paraphrase of the listed claims.
The §8.2 example also lists Claimeven d5–d6 in a later coverage explanation after
listing Baseinverse d5–e5; these overlap. Future source-example validation must
resolve that example discrepancy before claiming its displayed covering set is
a valid certificate. It does not affect the three local definitions here.

Recommended **Phase 6B** is compatibility and proof-boundary correctness for the
implemented subset: implement/test the CL/BI/VE matrix entries, derive full-board
Black opponent-to-move coverage targets, add a bounded compatible-set search and
an independent verifier for that limited fragment. Validate every accepted bound
against tiny exact endgames and include adversarial certificate mutation tests.
Keep uncovered positions unknown and preserve the public integration boundary.

Next, add the remaining rules incrementally, with original-source examples and
independent local coverage tests, followed by the full §7.4 matrix and White's
Chapter 8 contexts. Only after independent verification should exact solving,
optimal move selection and carefully versioned LLM evidence integration proceed.
Perfect play remains a longer-term objective, not a Phase 6A result.

## Verification and scope preservation

Executed from the repository root with the existing Python 3.11 `.venv`:

```sh
.venv/bin/python -m pytest -q tests/test_victor_rules.py tests/test_connect4_grounding.py tests/test_connect4_engine.py
.venv/bin/python -m pytest -q tests/test_connect4_evidence.py tests/test_connect4_provenance.py tests/test_connect4_history_api.py tests/test_connect4_explanations.py
git diff --check
```

Results: **140 passed** for Victor/grounding/engine (96 new Victor cases), and
**179 passed** for evidence/provenance/history/explanations; **319 passed** total.
`git diff --check` passed. The initial new-test run exposed
one incorrect test expectation about a blocked group and the diagram 6.2 printed
list omission; both expectations were corrected against the source and geometry.
No algorithm was changed to match an erroneous fixture.

Only this document, `games/connect4/victor/` and `tests/test_victor_rules.py` are
Phase 6A changes. AlphaZero training/research/provenance/artifacts, frozen canonical
benchmark data, all existing agents and their public behavior, API contracts,
React UI, LLM schemas, deployment and README benchmark claims are untouched.
The thesis PDF remains excluded by the existing `/research/references/*.pdf`
ignore rule. No training, large benchmark, exhaustive game search, push, merge or
deployment is part of this phase.
