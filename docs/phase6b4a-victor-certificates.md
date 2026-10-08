# Phase 6B.4A: Independent strategic certificate verifier

**Result.** This phase adds a separate, versioned certificate boundary,
`games/connect4/victor/certificate.py`. It checks the finite hypotheses H1-H4 of
the Claimeven/Baseinverse/Vertical (CL/BI/VE) Black non-loss theorem directly
from raw cells. It uses its own geometry, its own board validation and its own
replay. It calls no engine, grounding or production Victor code, and tests
enforce this.

A successful verdict is `theorem_hypotheses_verified`. It means only this: every
finite hypothesis of `cl-bi-ve-black-nonloss-v1` holds for the exact board that
was bound. The implied bound `Black value >= 0` is reported as **provisional
research evidence** (`provisional_bound`). Final game-theoretic certification
stays gated: `game_theoretic_certification` is always
`not_certified_pending_independent_review`. No public agent, API, UI or LLM
contract reads the new module.

**Recommendation: GO for Phase 6B.4B** (an executable σ_R move selector), under
the conditions in §12. **NO-GO** for exact values, draws, Black wins, solved
positions, optimal moves, White or Black-to-move contexts, any rule beyond
CL/BI/VE, or wiring a verdict into public outcome fields.

Baseline: `1956aa6a0eeb4f4c1b0227744992ffdff91526a7` (`phase6b3-victor-theorem-audit`),
verified as HEAD with a clean tree. Branch `phase6b4a-victor-certificates` was
created from that commit. Source: Allis (1988), local
`research/references/allis-1988-connect4.pdf`, SHA-256
`bca0a6bf53262e214dbff21ea4f878a1ba5298d15a11d93ce7eda3a1a326be86` (same file as
6B.2/6B.3). I re-read §§6.1-6.3, §7.1-7.4 and §8.1 (PDF pp. 36-39, 47-51) as
PDFKit-extracted text. Diagrams were not re-rendered.

## 1. Accepted finite predicates

Notation: the standard 7x6 board, White = `X` = player 0, Black = `O` = player 1.
Squares are named a1..g6, with row 1 at the bottom. A *group* is one of the 69
geometric fours. A *target* is a group with no Black stone.

| ID | Predicate recomputed by the verifier |
| --- | --- |
| H1 | Exact 6x7 matrix of exact `str` cells `' '`, `'X'`, `'O'`; gravity in every column; #X = #O (so White is to move, and `player_to_move` must be exactly `0`); no four for either player; board not full. |
| H2 | Each instance is one of the following. **CL** `(lower, upper)`: both empty, same column, upper = lower + 1, upper row even. **VE**: the same, with upper row odd. **BI**: two distinct current landing squares (the lowest empty square of their columns), listed by ascending column. Both BI squares must be empty and playable *now*; a CL/VE lower need not be playable. |
| H3 | The complete two-square sets of all instances are pairwise disjoint (§7.4 constraint 1 for all six CL/BI/VE pairs). Disjointness is pairwise by definition, so pairwise implies joint. |
| H4 | Every one of the derived targets contains the upper square of some CL, or both squares of some BI or VE. Targets are derived from the 69 groups generated inside the module, never from caller input. |

Nothing else is a hypothesis. Historical reachability, Zugzwang control, the
absence of odd threats, initial column parities and strategy traces are **not
required** (6B.3 §2). A covering set may contain redundant instances. The
MIXED3 fixture has two (a test pins this), and the theorem does not need
minimality.

## 2. Independent validation architecture

```
producer side (untrusted)                  | verifier (trusted boundary)
-------------------------------------------+-------------------------------------------
search_covering_set -> CoverageWitness     | certificate.py  (standard library only)
  -> certificate_producer.certificate_from_witness  -> StrategicCertificate
draft_certificate / mapping JSON           |   verify_certificate(...) -> CertificateVerification
                                           |   own: square parser, 69 groups, landing,
                                           |   gravity/counts/winner scan, H2-H4, replay
```

- `certificate.py` imports only `hashlib`, `json`, `dataclasses`, `enum`, `types` and
  `typing`. It does **not** call `grounding.validate_position`/`outcome`,
  `Position.from_board`, `verify_coverage_witness`, enumeration, compatibility
  or search. It also does not reuse their data types: rules are plain
  `RuleInstance(kind: str, squares: (str, str))` records.
- `certificate_producer.py` is on the producer side. It formats a production
  witness as a certificate and copies only the board, the rules and the
  assignments. It ignores the witness's coverage claims, its
  `VerificationStatus` and its `outcome_certification`. The verifier never
  imports it.
- The verifier reads every caller field once into a private snapshot (tuples
  of exact `str`/`int`). Mutating the caller's lists afterwards cannot change a
  verdict. Re-verifying the mutated certificate fails the digest binding.
- Exact-type checks (`type(x) is ...`) reject `bool`-for-`int`, `str`
  subclasses (for example a `RuleName` enum value used as a kind), forged
  subclasses and stray containers.

Enforcement in tests:
- An AST check limits imports to the standard library.
- A subprocess loads the module by file path while every `games.*`, `api`,
  `numpy`, `torch` and `victor_validation` import raises.
- A sabotage test replaces the following with functions that raise, then
  checks that every verdict in a 15-certificate battery is byte-for-byte
  unchanged:
  - grounding's `validate_position`, `outcome`, `_lines` and `GROUPS`;
  - `Position.from_board` and `Position.__post_init__`;
  - rule enumeration, compatibility, search and `analyze_candidates`;
  - the witness verifier and its helpers;
  - `ALL_GROUPS`.

## 3. Certificate schema and versioning

`StrategicCertificate` is a frozen, `slots` dataclass, so no forged attribute
can be attached. It is **untrusted input** and has no status, outcome or
acceptance field.

| Field | Meaning / check |
| --- | --- |
| `board` | 6x7 top-first matrix (H1). |
| `player_to_move`, `defender` | Exactly `0` and `1`. `defender=0` is unsupported context. A Black-to-move position that is consistent with the counts is unsupported context. A side that contradicts the counts is rejected. |
| `board_digest` | `sha256:` over `connect4-standard-7x6-v1;to_move=<p>;board=<rows>`. It binds the exact cells and side to move and is checked right after the shape. |
| `rules` | Tuple of `RuleInstance`, at most 21 (pigeonhole: 22 disjoint pairs need 44 squares). |
| `assignments` | Optional `(group name, rule index)` pairs. If present, they must match the recomputed coverage *exactly*: every target once, correct covering rule, no blocked or bogus group. Any correct alternative is accepted. |
| `replay` | Optional column history (provenance only, §5). |
| `schema`, `theorem`, `ruleset`, `rule_model`, `compatibility_model` | Must equal the supported identifiers exactly. A different string is UNSUPPORTED. A non-string is REJECTED. |

| Identifier | Value |
| --- | --- |
| Schema | `victor-strategic-certificate-v1` |
| Theorem | `cl-bi-ve-black-nonloss-v1` (H1-H4 + σ_R as clarified in §4) |
| Ruleset | `connect4-standard-7x6-v1` |
| Rule model | `allis1988-cl-bi-ve-v1` |
| Compatibility model | `allis1988-s7.4-cl-bi-ve-disjoint-v1` |
| Strategy formulation | `sigma-r-permissive-spare-v1` |
| Verifier | `victor-independent-certificate-verifier-6b4a-v1` |

`THEOREM_SPEC` records the statement, H1-H4, the strategy, what is not
assumed, the bound and the review status under the single theorem key. The
verdict `CertificateVerification` stamps every identifier itself (callers
cannot set them). It also carries:
- the recomputed evidence: counts, landing squares, the 69-group count, derived
  targets, the blocked-group count, per-rule coverage and a canonical assignment;
- a `certificate_digest` over the canonical certificate content.

A verdict is not transferable authority: consumers must re-run the verifier
rather than deserialise a result. JSON input goes through
`certificate_from_mapping`, which rejects any unknown key (`status`, `accepted`,
`outcome`, `value`, `proven`, `safe`, `zugzwang_controlled`, …) and never
defaults a version or context field.

### Statuses and stage order

Stages run in a fixed order:
1. type
2. versions
3. context
4. board shape and digest binding
5. remaining H1 predicates
6. rule parsing
7. H2
8. H3
9. H4 (and any assignments)
10. replay

The first stage with findings decides the verdict and reports all of that
stage's findings with stable codes (for example `H3.overlap`).
`hypotheses` records each H as `verified`, `failed` or `not_evaluated`.

| Status | When |
| --- | --- |
| `theorem_hypotheses_verified` | H1-H4 hold, and any supplied replay is verified. |
| `rejected` | Malformed or false content: shape, types, H1-H4, digest, duplicates, unknown rule names, bad coordinates, wrong assignments, bad replay. |
| `unsupported_version_or_context` | Well-formed but outside this theorem: a different version string, `defender=0`, a consistent Black-to-move board, or a real Allis rule outside CL/BI/VE (`aftereven`, `lowinverse`, `highinverse`, `baseclaim`, `before`, `specialbefore`). |
| `unknown_resource_cutoff` | The caller's `work_budget` was exhausted. The default (100,000 units) exceeds the work of every schema-bounded input, about 3,000 units at most, so this arises only when a caller lowers the budget. A cutoff never yields acceptance (tested at budgets 0-399 in steps of 7). |

No status means draw, win or solved, and the old
`coverage_verified_outcome_uncertified` status is untouched and not reused.

## 4. Strategy proof wording clarification

**Issue.** 6B.3 states σ_R's spare move as "any landing square that is not the
lower square of an active CL". Its Lemma L4, however, argues only that "a spare
move fills an even-row square". That leaves odd-row spares (for example a VE
lower or an unclaimed row-1/3/5 square) formally unargued.

**Resolution: every permitted spare preserves the invariant. No even-row
restriction is needed.** Corrected L4′ for a spare `t` follows (invariant I:
every active rule has both squares empty, and Black has never played a CL
lower):

- `t` is a landing square and not the lower square of an active CL by
  definition.
- `t` is not the upper square of an active CL. Under I that CL's lower is
  empty, so its upper is not a landing square.
- So `t` lies in no active CL. If `t` lies in an active BI or VE, that rule is
  retired: Black now holds one of its two squares, which is all its coverage
  needs. §6.3 (p.38) and §7.2 (p.49) explicitly allow Black to take a VE lower
  himself. Otherwise `t` touches no active rule.
- Every still-active rule keeps both squares empty. A retired CL's lower is
  occupied, because its upper is Black and gravity holds. So Black still has
  never played any CL lower.

The row parity of `t` is never used. Parity enters **only** through L3, which
shows that a permitted spare *exists*: an even-row landing square always does.
Forced replies are unchanged (L2). The theorem identifier therefore binds the
**set-valued** formulation `sigma-r-permissive-spare-v1`. The proof
quantifies over every permitted choice, so any refinement that always picks
from the permitted set inherits the guarantee. That includes "always pick an
even-row spare", a natural deterministic choice for 6B.4B. §7.1's
Claimeven restriction is exactly "do not play the CL lower", which agrees.

**Evidence that the distinction is real** (`tests/victor_validation/spare_policies.py`):
- The explorer quantifies over every White move and every Black reply that a
  policy permits.
- On pinned fixtures and on 39 sampled covers, `permissive` and `even_row` both
  held with zero invariant breaks. In 26 of the 39 covers, odd-row spares were
  actually explored.
- The deliberately unsound `unrestricted` policy also allows an active CL
  lower. It fails on a real position. The fixture is history
  `550066550544410241221101261344663650` with CL c5-c6 and CL d3-d4. H1-H4
  hold, and the audit solver confirms that White cannot force a win.
  - White plays d3; Black's forced reply is d4.
  - White plays d5; Black plays the CL lower c5 as a "spare".
  - White plays c6 and wins.

So the CL-lower exclusion is load-bearing, while the even-row wording was just
a sufficient special case. The strategy itself is unchanged.

## 5. Reachability versus theorem applicability

Reachability is **provenance, not a hypothesis**. Without a replay, a verified
certificate reports `position_provenance = mathematical_position_only` and
`history_backed = False`. The 6B.3 `UNREACHABLE` board (every column top is
White with White to move) verifies this way, and no history is invented.

With a replay, the verifier does the following:
1. Replays from an empty board with strict `int` columns 0..6.
2. Alternates White and Black, applies gravity, and rejects full columns.
3. Rejects any move after a completed four.
4. Requires the final board to equal the bound board exactly, nonterminal and
   with White to move.

Success gives `replay_status = verified_legal_history` and `history_backed = True`.
A wrong replay rejects the whole certificate. `hypotheses` still shows H1-H4 as
verified, so the two properties are reported separately. Any legal
transposition reaching the same board is accepted, because a replay is
existential evidence.

**Live-play policy (for later integration):** a proof-backed claim about an
*actual* game may be shown only if `history_backed` is true for that session's
immutable move history. A synthetic or analysis board may at most be described
as a mathematical position.

## 6. Exact scope of the bound

From H1-H4 and the reviewed paper proof (6B.3 §2, with L4′ above), Black has a
strategy under which White never completes a four. So V_B ≥ 0 for the game
played from that board. Nothing more is claimed:
- Not a draw. Of the 387 6B.3 witnesses with exact values, 282 were Black wins.
- Not a Black win, a solved position, an optimal or recommended move, or a
  complete player.
- Not a statement about any other board, about White, or about a Black-to-move
  board.
- No certified "no cover" conclusion. A failed certificate says nothing about
  the game value.

## 7. Concrete examples

MIXED3 is history `511322464155553320540623416612` (12 empties, White to move;
`X` = White, `O` = Black):

```
.....X.     rules: baseinverse b6-c6, claimeven d5-d6,
.XO..O.            claimeven e5-e6, vertical a4-a5
.OXOXXO
.OXOOOX     targets: 7 of 69 (62 already contain a Black stone)
XXOXXXO     landing squares: a3 b6 c6 d5 e5 g5
OOXOXXO
```

Verdict with replay: `theorem_hypotheses_verified`, H1-H4 all `verified`,
`replay_status = verified_legal_history`, `provisional_bound = black_value_at_least_draw`,
`game_theoretic_certification = not_certified_pending_independent_review`.
Per-rule target coverage is 2/5/3/2. The board digest is
`sha256:9c2c8461…0bd5`.

| Input | Verdict | Finding |
| --- | --- | --- |
| 6B.3 overlap counterexample (CL d5-d6, BI d3-e3, CL e3-e4) | rejected | `H3.overlap`: `baseinverse:d3-e3 and claimeven:e3-e4 share ['e3']` (H1, H2 verified) |
| 6B.3 "Claimodd" f2-f3, f4-f5 as Claimevens | rejected | `H2.parity` ×2 (`upper square f3 must be even`) |
| The same squares as Verticals | rejected | `H4.uncovered` (Verticals cover only groups containing both squares) |
| 6B.3 floating BI d6-e5 | rejected | `H2.not_playable`: `['e5'] not a current landing square` |
| MIXED3 with theorem `cl-bi-ve-black-nonloss-v2` | unsupported | `version.theorem` |
| MIXED3 with a replay missing its last two plies | rejected | `replay.board_mismatch`, while hypotheses still read H1-H4 verified |
| Diagram 6.1 reconstruction (34 empties), production witness + replay | verified | history-backed, 40 targets, CL-only |

The three counterexamples are real exact White wins under the relaxed
hypotheses (audit solver, plus the 6B.2 oracle for the first two).

## 8. Tests and adversarial results

`tests/test_victor_certificates.py`: **145 tests, about 1.2 s.**

- **Valid:** CL-only, BI-only, VE-only, the 6B.3 mixed column, a CL/BI/VE mix,
  the empty rule set on a board where every group is blocked, and the diagram
  6.1 production witness (34 empties).
- **H1:**
  - wrong turn (both directions), Black-to-move (unsupported), bad counts and
    gravity;
  - White, Black and contradictory fours, and a full 21/21 board;
  - twelve malformed boards: wrong dimensions, non-sequence input, `str` rows,
    and `0`, `True`, `'x'`, `'XX'` and `str`-subclass cells.
- **Versions and context:**
  - theorem, ruleset, rule-model, compatibility-model and schema strings;
  - non-string versions;
  - `defender` 0/True/'1' and `player_to_move` True/2.
- **H2:**
  - reversed roles, a gap, a column change, VE with an even upper, Claimodd;
  - occupied squares, an unplayable BI, non-canonical BI order, a repeated
    square;
  - `claimodd`, `lowinverse`, `aftereven`, a `RuleName` enum kind, and eight
    malformed coordinates.
- **H3:** the 6B.3 overlap, a §5.3-style BI on a CL lower, CL on VE, duplicates,
  and more than 21 instances.
- **H4:**
  - removing each rule matches the audit model (redundant removals stay valid,
    the rest fail only H4, on groups needing the removed rule);
  - an empty board with all 21 Claimevens leaves exactly the 12 odd-row
    horizontals uncovered;
  - eight assignment forgeries, plus an accepted alternative assignment.
- **Forgery:**
  - eleven forged mapping keys and missing identity keys;
  - `object.__setattr__` blocked;
  - certificate subclasses, a raw mapping, or a production witness passed as a
    certificate;
  - witness coverage and `outcome_certification` lies not carried over.
- **Binding:** a recoloured board under the old digest, a digest for the wrong
  side to move, and caller mutation after verification.
- **Replay:** truncation, odd length, illegal transposition, column 7, `True`,
  `None`, full column, move after a four, more than 42 moves, string type;
  every legal transposition of MIXED3's history; the unreachable board.
- **Symmetry:** all six valid fixtures mirror (with mirrored replays) to valid
  certificates with distinct digests. Unmirrored rules on a mirrored board are
  rejected.
- **Differential vs the 6B.3 audit checker** (`check_hypotheses`, untouched):
  - 741 rule sets on sampled positions (seed 6404, 4-14 empties): valid covers,
    each single removal, a random addition and a random replacement from a pool
    that includes Claimodd and floating BIs;
  - 219 accepted and 522 rejected, with no disagreement;
  - every rejection's first stage (H2/H3/H4) was among the audit's reasons.
- **Production pipeline:** on 44 sampled positions, production found a cover
  on 10, and each became a verified, history-backed certificate. The audit DFS
  agreed on cover existence for all 44.
- **Strategy wording:** §4 (pinned failure of the unrestricted policy, both
  sound policies on four fixtures, 39 sampled covers).

**Mutation testing of the verifier.** In scratch, I applied 26 single-point
bugs to `certificate.py` and ran the suite against each. All 26 were killed:
- disabled parity, landing, occupancy, adjacency, overlap and assignment checks;
- CL coverage moved to the lower square; BI/VE covered by one square; targets
  restricted to empty groups;
- relaxed counts and disabled turn, winner, gravity and full-board checks;
- disabled digest binding, versions, defender check, Black-to-move gate,
  duplicate detection and unsupported-kind detection;
- broken replay alternation, terminal stop and board binding;
- `isinstance` cell typing, `bool` side to move, and an ignored budget.

The first run left one survivor (`isinstance` cells, equivalent for `bool`).
That led to the `str`-subclass cell tests. The H4 mutants were also killed by
the H4 tests alone, not only by the isolation test.

## 9. Machine-checked hypotheses vs a verified theorem

| Layer | Status |
| --- | --- |
| H1-H4 for a given board and rule set | **Machine-checked** by an independent finite verifier (this phase). |
| L3 combinatorial core (an even-row landing square exists with odd empties) | Exhaustively checked over all 7^7 height vectors (6B.3). |
| L1, L2, L4′ and termination | **Paper proof**, reviewed by one auditor (6B.3) with the L4′ wording fixed here. Bounded model checks agree (6B.3: 2,412 covers; here: 39 covers × 2 policies). **Not** proof-assistant verified. |
| "H1-H4 ⇒ V_B ≥ 0" for *all* boards | Not mechanised. The verifier's acceptance is only as sound as that paper proof. |

## 10. Source and theorem ambiguities found

1. **L4 wording** (§4). The strategy allowed odd-row spares while L4 argued
   only even-row ones. Resolved: L4′ covers every permitted spare, and the
   identifier binds the permissive formulation.
2. **"Directly playable"** (§6.2) is a *current* property. A BI is valid only
   if both squares are landing squares on the certified board. That later
   moves keep them landing until they are filled is a lemma (L2), not a check.
3. **Black to move.** The proof also goes through with Black to move (6B.3
   remark), but the theorem identifier does not include it. Such boards are
   reported unsupported, not rejected.
4. **The full board** is terminal (a draw) and outside "nonterminal". It is
   rejected under H1 even though no White win is possible.
5. Thesis errata already pinned by 6B.3: diagram 6.2 omits BI b1-d3, and the
   §8.2 "Claimeven d5-d6" attribution. Neither affects this verifier.
6. **The thesis presupposes Zugzwang control** (Ch. 6 intro). For this fragment
   it is subsumed by H4 (6B.3 §3), so it is not a field or a hypothesis.

## 11. Remaining assurance obligations

1. An independent second review of L1, L2, L4′ and termination, or their
   formalisation (Lean/Coq, or an exhaustive abstract column model).
2. A code review of `certificate.py` by someone other than its author. It is
   about 720 lines, intentionally self-contained.
3. A decision on a policy owner for when, if ever,
   `game_theoretic_certification` may change. That needs a new verifier
   version and an explicit review record. It must never be flipped in place.
4. Any rule beyond CL/BI/VE needs a new spare-move lemma for forbidden regions
   (§7.1), new pairwise-to-joint arguments for constraints 2-4, and a **new
   theorem identifier**. `UNSUPPORTED_RULES` keeps them fail-closed until then.

## 12. Proposed Phase 6B.4B scope and GO/NO-GO

**GO** for implementing σ_R as an executable research move selector, provided
that:
- (a) it accepts only a `theorem_hypotheses_verified` certificate, re-verified
  on entry;
- (b) it tracks the active/retired state from the actual move sequence and
  plays the forced reply, else an even-row spare. Any choice from the permitted
  set is sound; even-row gives a simple deterministic rule;
- (c) it is property-tested against both exact solvers and the all-White-moves
  explorer, including a check that it never plays an active CL lower;
- (d) it stays research-only, with no public agent registration, no API field
  and no outcome display.

**NO-GO** in 6B.4B for:
- exact values, win-seeking or "optimal" play;
- White or Black-to-move contexts and rules beyond CL/BI/VE;
- enabling public outcome fields, until obligation 1 (§11) is met.

## 13. Verification and changed files

```sh
.venv/bin/python -m pytest -q tests/test_victor_certificates.py tests/test_victor_independent_audit.py \
  tests/test_victor_endgame_oracle.py tests/test_victor_strategy_validation.py tests/test_victor_rules.py \
  tests/test_victor_compatibility.py tests/test_victor_coverage.py tests/test_victor_verification.py \
  tests/test_connect4_grounding.py tests/test_connect4_engine.py tests/test_connect4_evidence.py \
  tests/test_connect4_provenance.py tests/test_connect4_history_api.py tests/test_connect4_explanations.py
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests -q
git diff --check
```

Results:
- The Victor and relevant backend list gives **648 passed** (503 from 6B.3
  plus 145 new).
- The full backend suite gives **926 passed, 14 skipped**. The skips are the
  existing optional PyTorch tests.
- `git diff --check` is clean.
- No training, neural evaluation, tournament or large search was run.

Files:
- `games/connect4/victor/certificate.py` (new): the independent verifier.
- `games/connect4/victor/certificate_producer.py` (new): producer-side
  formatting.
- `tests/test_victor_certificates.py` (new).
- `tests/victor_validation/spare_policies.py` (new; test-only, reuses the
  6B.3 audit `Board` without modifying it).
- `docs/phase6b4a-victor-certificates.md` (this report).

Untouched: the existing Victor modules and their exports, the 6B.3 audit model,
AlphaZero, frozen benchmarks, public agents (including `VictorAgent`), the API,
the UI, deployment and the LLM explanation contracts.
