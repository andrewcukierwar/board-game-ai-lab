# Phase 6B.3: Independent Victor theorem audit

**Verdict: the CL/BI/VE Black non-loss theorem is SOUND under the hypotheses
H1-H4 below.** I found no counterexample and no implementation defect in the
fragment. One specification error matters for the next phase: Phase 6B.2 listed
historical reachability and "Black actually maintains the invariant" as
*hypotheses*. They are not hypotheses. The theorem holds on every basic-valid
board, reachable or not. The strategy is part of the proof, not something a
certificate must evidence.

**Recommendation: GO for Phase 6B.4** to build a *separate*, bound-only
`Black value >= 0` certificate verifier for this fragment, under the conditions
in §9. **NO-GO for accepting certificates in this phase** (none is enabled
here), and NO-GO for exact values, optimal moves, White contexts or the six
remaining rules. Production status stays `coverage_verified_outcome_uncertified`.

Baseline: `1257078963afda75e37a8b52db9ea8e018fda9f5` (`phase6b2-victor-strategy-validation`),
verified as HEAD with a clean tree. Branch `phase6b3-victor-theorem-audit` was created from
that commit. No production Victor, AlphaZero, frozen artifact, benchmark, agent,
API, UI or deployment file is changed.

## 1. Primary-source review

Source: Allis (1988), local `research/references/allis-1988-connect4.pdf`, SHA-256
`bca0a6bf53262e214dbff21ea4f878a1ba5298d15a11d93ce7eda3a1a326be86` (same as 6B.2).

Text was extracted with `pypdf` from a throwaway scratchpad virtualenv (not a project
dependency). That text loses every diagram (stones extract as bare `●`). I rendered
pages with macOS PDFKit but visually inspected only the three that this audit
depends on: p.37 (diagram 6.2), p.50 (the §7.4 table) and p.51 (diagram 8.1).
The other diagrams were **not** inspected. Statements about them rest on the
thesis text and on the repository's existing fixtures. I read Chapters 4-8
(PDF pp. 25-57) in full. The thesis page numbers match the
1-based PDF pages.

| Section | What the source actually establishes |
| --- | --- |
| §4.1-4.2 | Follow-up gives the controller every even square. Pure follow-up from the empty board loses (diagram 4.4). Parity alone prevents nothing. |
| §4.4 | Follow-up still works when some columns have an odd number of empties, by pairing the playable even squares. This is the informal ancestor of the spare-move lemma. |
| §5.3 | BI a1-b1 and CL b1-b2 require different immediate replies to White b1, so rules sharing *any* square (including a CL lower) are incompatible. |
| §5.4 | "House of cards": the conclusion holds only if *every* potential White group is refuted. The thesis says the argument is not circular but proves nothing. |
| Ch. 6 intro | Rules are applied with the opponent to move. Black may try the whole board even when White has an odd threat. |
| §6.1-6.3 | CL: two empty vertically adjacent squares, even upper; solves groups containing the upper. BI: two directly playable squares; solves groups containing both. VE: two empty vertically adjacent squares, odd upper; solves groups containing both. "If the opponent never plays the lower square, the player can take that square himself" (p.38). |
| §7.1 | Zugzwang-dependent rules forbid the controller some squares. Rules combine if each rule, once filled, releases an even number of squares. |
| §7.2 | BI and VE are Zugzwang-independent and combine with anything if disjoint. |
| §7.4 (rendered p.50) | All six CL/BI/VE entries are constraint **1** (disjoint square sets). LI/HI use constraint 2 against CL; composite rules use 3/4. |
| §8.1 | Black need not check Zugzwang control first: if control fails, no rule set covers everything, so trying does no harm. This is a narrative claim, never proved. |
| §8.2 | White's rules apply off the odd-threat column. This context is outside the Black fragment. |

What the thesis does **not** contain: any explicit statement of the spare-move
lemma, any invariant, or a proof that pairwise compatibility implies joint
executability. Those steps are reconstructed in §2 below.

### Documented source discrepancies, re-examined

- **Diagram 6.2 (rendered p.37).** The position is White c1, d1, c2 and Black e1, d2,
  c3. The landing squares are a1, b1, c4, d3, e2, f1, g1. Group b1-c2-d3-e4 has no
  Black stone, so BI b1-d3 is useful, but the printed list omits it. **Confirmed as
  a thesis omission.** The code (enumerate all landing pairs) and the existing test
  `test_thesis_diagram_6_2_useful_and_useless_baseinverses` (same board, includes
  b1-d3) follow the formal definition. No change needed.
- **§8.2 "Claimeven d5-d6" (rendered p.51).** Diagram 8.1 reads: row 1
  `X O X X O . .`, row 2 `. X O O O`, row 3 `. X X X O`, row 4 `. . . O X`
  (X = White), Black to move, White odd threat a3. The *declared* rule set is
  CL b5-b6, c5-c6, f1-f2, f3-f4, f5-f6, g3-g4, g5-g6 plus BI d5-e5. That set is
  pairwise disjoint and covers all 19 Black groups that need no a-column square
  above a2. The three groups the text credits to "Claimeven d5-d6" (b6-e6, c6-f6,
  d6-g6) are already covered by the declared CLs on b6, c6, f6 and g6. **Resolution:
  this is a per-group attribution erratum, not an incompatible rule set.** CL d5-d6
  would indeed overlap BI d5-e5, but it is never declared. This is machine-checked in
  `test_thesis_diagram_8_1_declared_rules_are_disjoint_despite_d5_d6_attribution`.
  It is a source check only; no White context is accepted.

## 2. Independent theorem and proof

I derived the argument below from §§6.1-6.3 and the game rules *before* reading
the 6B.2 reconstruction. The two agree on strategy and parity step. They differ on
which hypotheses are needed (§4).

### Setting and hypotheses

Standard 7×6 board. Squares are named a1..g6, with rows counted from the bottom.
"Even" means an even row. One stone per turn, no passes. The game ends at the first
completed four (a win) or when the board is full (a draw). A *group* is one of the
69 geometric fours. For a board B, a *potential White group* is a group containing
no Black stone.

| ID | Hypothesis (all finitely checkable from B and R) |
| --- | --- |
| H1 | B obeys gravity; #White stones = #Black stones (so White is to move); neither side has a four. |
| H2 | Each r ∈ R is a CL (two empty vertically adjacent squares, upper even), a VE (same, upper odd), or a BI (two distinct current landing squares). |
| H3 | The instances in R have pairwise disjoint square sets (§7.4 constraint 1 for all six CL/BI/VE pairs). |
| H4 | Every potential White group G of B is *covered*: it contains the upper square of some CL, or both squares of some BI or VE in R. |

**Theorem.** Under H1-H4, Black has a strategy σ_R, computable from R alone,
under which White never completes a four. So for the game played from B,
Black's minimax value is V_B ≥ 0 (White cannot force a win).

**Not assumed:** historical reachability, the absence of White odd threats,
initial even column heights, a "Black controls Zugzwang" fact, a pairing of all
cells, or any trace of strategy execution.

### Strategy σ_R

Call r ∈ R *active* if a CL's upper square, or a BI/VE's both squares, are not yet
Black. Otherwise r is *retired*. After White's move m:

1. **Forced reply.** If m is the lower square of an active CL or VE, Black plays the
   square directly above m. If m is one square of an active BI, Black plays the other
   square.
2. **Spare move.** Otherwise Black plays any landing square that is not the lower
   square of an active CL. The proof shows a landing square on an even row always
   qualifies. Taking a BI square or a VE lower square this way ("proactive
   occupation") retires that rule.

### Lemmas

**Invariant I** (at each White turn): every active rule has both squares empty, and
no CL lower square has ever been played by Black.

**L1 (White cannot win while I holds).** Suppose White's move m completes G.
G had no Black stone at B, so it was covered by some r.
- If r is retired, a Black stone lies in G: for a CL it is the upper square, which
  G contains; for a BI/VE it is one of two squares, both in G.
- If r is an active CL, its upper square u ∈ G is empty. It is not a landing square
  because the lower square under it is empty, so m ≠ u and G is incomplete.
- If r is an active BI/VE, both of its squares are empty and in G. m fills at most
  one of them.

So G cannot be complete, a contradiction. This also covers White's move being the
very trigger of an obligation: White cannot win *before* Black's mandatory reply.

**L2 (forced replies are unique and legal).** By H3 each square lies in at most one
rule, so m triggers at most one obligation.
- A CL/VE reply is the square directly above m, which becomes a landing square
  when m is played.
- A BI partner was a landing square at B. A landing square stays one until it is
  filled, because stones are never removed. Under I it is still empty.

The reply is never another rule's square (H3). It is never a CL lower square either:
a CL upper is on an even row, and a VE upper (row 3/5) or a BI partner could equal
a CL lower only by sharing a square, which H3 forbids. **Pairwise disjointness implies
joint disjointness, because disjointness is a pairwise property.** That is why
pairwise compatibility suffices *for these three rule types only*.

**L3 (spare-move parity lemma).** At any Black turn the number of empty cells is
odd. White moved last, so the stone count is odd, and 42 minus an odd number is
odd. So some column has an odd number of empties, and its landing square is on an
even row.
- An even-row square is never a CL lower, since CL lowers are on rows 1, 3 and 5.
- It is never the upper square of an *untouched* CL, because that square is not a
  landing square while its lower is empty.
- It is never the upper square of a CL whose lower is filled. Under σ_R only White
  fills a CL lower, and Black then immediately takes the upper.

So a permitted spare move always exists. It may be a BI square or VE lower square.
Taking either retires that rule, and the rule's guarantee needs Black on only one
of its two squares.

I checked this lemma's combinatorial core exhaustively. All 411,771 height vectors
in {0..6}^7 with an odd number of empties have an even-row landing square
(`test_spare_move_parity_lemma_over_every_height_vector`).

**L4 (I is preserved).**
- A forced reply retires exactly the rule White touched and leaves the other active
  rules untouched (H3).
- A spare move fills an even-row square. That square is not a CL lower and not in
  any active CL, so active CLs stay empty. If it lies in a BI/VE, that rule is
  retired.
- White's move outside every active rule touches no active square.
- Black never plays a CL lower: replies are excluded by L2, spares by L3.

**Termination.** The game is finite with no passes. White plays the odd-numbered
moves, so White never fills the board and Black always has a legal move (L3). By
L1, every play under σ_R ends in a Black four or a full-board draw. A Black win
along the way only ends the game early. ∎

### Answers to the specific questions

- **Legal reachability:** not used. The parity in L3 needs only H1's stone counts.
  On unreachable but basic-valid boards, the theorem still bounds the game played
  from that board. Reachability matters only if one wants to say something about
  real game histories. That is an interpretation choice, not a soundness condition.
- **Turn parity:** H1 is load-bearing for L3. The same proof also goes through with
  Black to move: Black's first move is a spare. I observed this on 269 covered
  Black-to-move positions with no White win. It is a remark only and is not
  proposed as a supported context.
- **Pairwise vs joint, active vs retired, proactive occupation:** see L2 and L4.
  Proactive occupation is always safe for BI/VE (§6.3 p.38, §7.2 p.49), and never needed or
  permitted for a CL lower.
- **Arbitrary combinations:** nothing in the proof uses diagram geometry. It holds
  for any R satisfying H2-H4, including CL lowers with empty squares beneath them,
  CL and VE stacked in one column, and BI squares on even rows.
- **Exact meaning of the bound:** V_B ≥ 0 only. Of the 387 verified witnesses in §6,
  282 were exact Black wins and 103 were exact draws. Coverage does **not** imply
  "draw", and the strategy σ_R is not optimal.

### Hypotheses vs lemmas

H1-H4 are finite predicates of (B, R) that a verifier can recompute. L1-L4 are
universal mathematical lemmas, proved above. Only L3's combinatorial core is
exhaustively machine-checked. L1, L2 and L4 are paper proofs, supported by bounded
model checks (§6) but not formalised in a proof assistant.

## 3. Comparison with the thesis

The proof formalises what §§5.4, 7.1-7.2 and 8.1 argue informally:

- §7.1's principle ("filled rules release an even number of squares") reduces, for
  CL/BI/VE, to L3. CL barriers split columns into even-sized regions, and BI/VE
  forbid Black nothing.
- §8.1's claim that control need not be checked is *true* for this fragment. An
  uncovered odd threat makes H4 fail; coverage subsumes Zugzwang control.

This reduction is specific to the fragment:
- **LI/HI forbid Black their lower squares.** §7.1 diagram 7.2 shows a CL plus LI
  leaving Black no legal non-forbidden move. So L3 as stated is **false** for
  fragments with inverse rules.
- **Constraints 2-4 are not pairwise-disjointness properties.** The step "pairwise
  implies joint" in L2 must be re-proved for them.

The CL/BI/VE theorem and its version identifier must not be reused for any
extended rule set.

## 4. Assessment of the Phase 6B.2 reconstruction

The 6B.2 invariant, response rules, parity argument and "White cannot win before the
reply" argument are **correct and match mine**. Its execution model (all White
moves × all permitted Black spares) is a faithful bounded check of σ_R. Two points
of disagreement:

1. Its prerequisite 2 requires a *historically reachable* position. That is
   over-strict: the proof never uses history. I demonstrate this with 94 unreachable
   covered boards, all safe (§6), and a pinned regression test.
2. Its prerequisite 6 ("Black actually maintains the invariant") is a property of
   the proof's strategy, not a hypothesis on the position. The 6B.2 certificate
   table's demand for "execution evidence quantifying every adversarial White
   continuation" is therefore unnecessary. The theorem does that quantification
   once, for all positions.

Consequence: **a coverage witness that passes the current verifier already discharges
every theorem hypothesis.** The remaining gap is assurance (trust boundary,
versioning, proof review), not missing mathematical evidence.

## 5. Implementation audit (ranked by severity)

No correctness defect was found, so no production code was changed.

| # | Severity | Finding |
| --- | --- | --- |
| A1 | Medium (trust boundary) | `verify_coverage_witness` and the producer share `validate_position`/`outcome` (grounding) and the `Position`/`Square`/`Group`/`RuleCandidate`/`CoverageWitness` types. H1 rests entirely on the shared validator: the count check `player == x - o` that powers L3, and nonterminal detection. One bug there would invalidate L3 for producer and verifier simultaneously, and they would still agree. A certificate verifier must re-derive counts, gravity and winners itself. My audit model and the 6B.2 oracle both do this. |
| A2 | Medium (specification) | 6B.2's reachability and "maintains the invariant" prerequisites (§4). If carried into 6B.4 unchanged, they would add a required replay field and imply that strategy traces are evidence. Recommendation: reachability as an optional, verified provenance field; no strategy traces. |
| A3 | Medium (validation quality, now addressed) | 6B.2's survey had at most 8 empties. Its 39 witnesses were 24 CL-only, 6 BI+CL, 3 BI, 3 empty, 2 CL+VE and 1 VE, with none using all three types. Its only counterexample-path test used a monkeypatched oracle, so nothing showed that a real false implication would be caught. §6 extends this to 22 empties, 2,025 diverse covers, and real mutation counterexamples. |
| A4 | Low (source) | Diagram 6.2 omission and §8.2 attribution erratum (§1). The code is correct; both are now pinned. |
| A5 | Low (future hazard) | `RuleCandidate.depends_on_zugzwang` marks only CL. That is correct here, but L3's reliance on "Black is never forbidden an even-row square" is undocumented in code. Adding LI/HI/BE/SB without a new spare-move lemma would silently break soundness. |
| A6 | Info | **Search completeness.** Suppose a compatible cover S extends the current selection. The most-constrained uncovered group is covered by some j ∈ S. j is not blocked, because it is compatible with the selection and not yet chosen (otherwise the group would already be covered). Recursing on j preserves the property, so if any cover exists the DFS finds one. Removing zero-coverage candidates is safe. Budget status semantics are correct (6B.1 tests). Empirically, production and my independent DFS agree on cover existence in all 12,542 positions. Production never needed more than 10 nodes in 2,609 positions with 8-28 empties, so the default budget of 10,000 is not a practical completeness risk. |
| A7 | Info | **Rule enumeration** matches §§6.1-6.3 exactly: CL with upper on rows 2/4/6, VE with upper on rows 3/5, BI over all landing-square pairs. Terminal boards yield nothing. False positives would need an occupied or non-landing square, which both the enumerator and the verifier reject. |
| A8 | Info | **§7.4 matrix:** six entries, all constraint 1, with unknown pairs failing closed. This matches the rendered table. Disjointness compares *all* squares, including CL lower triggers (§5.3). |
| A9 | Info | **Exact oracles.** My audit solver (line masks, boolean AND/OR search with immediate-threat shortcuts) and the 6B.2 bitboard negamax agree on all 153 + 39 cross-checked positions (a scratch run plus the pinned test sample). They share no code. |

**How the modules could share a wrong assumption.** The enumerator, verifier and
6B.2 execution model all encode the *same* reading of the CL/BI/VE semantics:
- coverage squares;
- full-square disjointness;
- landing squares;
- "Black never plays a CL lower".

If that reading were wrong, all three would agree on the error. Only two checks
are independent of the reading: the exact game value (two unrelated solvers) and
the paper proof. The mutation experiments below show that the exact comparison is
not vacuous. Each plausible misreading I tried (overlap allowed, odd-upper
Claimeven, BI with a non-landing square) produces real, legally reachable positions
with a "cover" and an exact White win.

## 6. Counterexample search and independent validation

All tooling is test-only: `tests/victor_validation/independent_audit.py`. It is
standard library only and imports no `games`, engine, grounding, 6B.2 oracle or
harness code (enforced in a blocked-import subprocess test). It has its own 42-bit
geometry and 69 line masks, its own rule enumerator and cover DFS (first uncovered
line, unlike production's most-constrained group), its own hypothesis checker H1-H4,
a separate exact boolean solver, an all-spares strategy model checker, and a
backward reachability search.

**Generators.**
- Legal random play to a target number of empties, skipping moves that end the
  game. With probability p, Black instead answers in White's column (follow-up,
  §4.1). That bias yields Claimeven-rich covers at many empties, which a uniform
  sampler almost never reaches. Not uniform.
- Unreachable boards: each Black column top is swapped with a non-top White stone.
  With White to move, all tops being White means no legal history exists; backward
  search confirms it.

| Experiment (fixed seeds) | Result |
| --- | --- |
| Main campaign, seed 6203 (p ∈ {0, .5, .9}) and seed 1989 (p ∈ {.25, .75, 1}), 8-22 empties | **12,542** distinct legal White-to-move positions. **387** production covers, every one accepted by the production verifier *and* by my H1-H4 checker. **0** existence disagreements. **0** exact White wins (exact search completed for all 387). The all-spares strategy check **held 387/387**, 0 unknown. Witness kinds: CL 300, BI+CL 70, CL+VE 12, BI+VE 1, VE 1, BI+CL+VE 1, empty 2. Exact values: 282 Black wins, 103 draws, 2 unknown (only the Black-win sub-search hit its budget; White's inability to win was established). Witnesses reached 22 empties. |
| Cover diversity, seeds 7403 (2-16 empties) and 7404 (10-20 empties) | Up to 40-60 **distinct** valid covers per position, preferring BI/VE branches. **2,025** covers on 476 positions: 583 with BI, 980 with VE, **278 using all three types**. The all-spares check **held 2,025/2,025**, 0 unknown, 0 exact White wins. |
| Unreachable boards (within both main campaigns) | 3,751 confirmed unreachable; 94 covered; **0** violations (exact and strategy). Production verified coverage on such a board (pinned test). |
| Hypothesis mutations (positions with no sound cover) | A mutated "cover" with an exact White win: **overlap allowed 30**, **Claim-odd 161**, **floating BI 3**. Overlap and Claim-odd examples at ≤10 empties were re-confirmed by the 6B.2 oracle. |
| Black to move (remark) | 4,536 positions, 269 covered, 0 White wins. |
| Parity lemma | Exhaustive over all 7^7 height vectors. |

Total CPU time was a few minutes. No unrestricted or whole-game search was run, and
every search has a node budget that returns "unknown" when exceeded rather than
guessing.

### Preserved counterexamples to *relaxed* hypotheses

These are counterexamples to mutated hypotheses, not to the theorem. Columns are
0-6 (a-g), White moves first, and boards are listed top row first.

1. **Overlap (§5.3 shape), H3 dropped.** History `0610216610160066212202555525154343`,
   8 empties, White to move. Board:
   `XXX  OO / OOO  OX / XXO  OO / OXX  XO / OOXOXOX / XXXOXXO`.
   Rules CL d5-d6, BI d3-e3, CL e3-e4 cover every target, but BI and CL share e3.
   White e3 obliges both Black d3 (BI) and Black e4 (CL).
   - After Black d3, White e4 wins.
   - After Black e4, White d3 wins.

   Both exact solvers give White value +1, and production search reports no cover.
   The production verifier rejects this witness with exactly
   `selected candidates conflict`.
2. **Claim-odd, CL parity dropped.** History `2024243302333341506611441164162266`,
   8 empties. "Claims" on f3 and f5 (odd upper squares) cover all targets. White
   value +1 (both solvers). One line: White f2, Black a4, White f3 wins. Production
   cannot represent the rule: `RuleCandidate` raises a parity `ValueError`.
3. **Floating BI, landing requirement dropped.** For example, history
   `5550660011542250004435113333` (14 empties) with BI d6-e5, where e5 is not a
   landing square (column e lands on e4). The audit solver gives a White win. These positions are above the
   6B.2 oracle's 10-empty cap, so this one rests on the audit solver alone. Its
   agreement with the 6B.2 oracle is established on smaller boards.

### Limitations

- Sampled, not exhaustive. Boards with more than about 22 empties are rarely
  coverable and were barely sampled. Early-game covers such as the diagram 6.1
  reconstruction (34 empties) remain out of exact range.
- The all-spares checker is a second implementation of σ_R by the same reviewer.
  It shares the *reading* of the rules with production (see §5).
- L1, L2 and L4 are not formalised in a proof assistant.

## 7. Verifier-to-theorem correspondence

| Category | Status after this audit |
| --- | --- |
| 1. Finite verified covering-set evidence | Exists (6B.1). It checks H2, H3, H4, and H1 via the shared validator. |
| 2. Mathematically sound strategic theorem | Established on paper (§2) for CL/BI/VE. Independently corroborated; not machine-proved. |
| 3. A certificate satisfying all hypotheses | Today's verified witnesses already do this mathematically. There is no certificate *type* yet. |
| 4. A formally justified Black non-loss bound | Follows from 2 + 3 **once** a verifier with an independent trust boundary binds the theorem version (6B.4). Not enabled now. |
| 5. Exact game value | Not implied. 282 of 387 witnesses were exact Black wins. |
| 6. Optimal move policy | Not implied. σ_R guarantees non-loss only and can miss wins. |

### Obligations a future certificate verifier must check

The verifier trusts no producer field.

1. **Board (H1):** parse a strict 6×7 matrix itself and recount stones. Require
   White stones = Black stones. Check gravity, and detect fours with independent
   geometry. Do not reuse `grounding.validate_position` (A1).
2. **Context:** fixed whole-board Black context. Reject any reserved columns,
   excluded groups, White context or Black-to-move input.
3. **Targets (H4):** derive the 69 lines and keep those without a Black stone.
   Ignore or reject any producer-supplied target list unless it equals the derived one.
4. **Rules (H2):** for each claimed instance, recompute emptiness, adjacency, parity
   and landing status directly from cells. Accept only CL/BI/VE; any other rule id
   or rule-model version is rejected or unknown.
5. **Compatibility (H3):** check full-square pairwise disjointness itself. Ignore any
   claimed `compatible`/matrix entries.
6. **Coverage (H4):** recompute each target's covering rule. Reject an uncovered
   target; ignore producer assignments, or require them to match exactly.
7. **Theorem binding:** a fixed identifier such as `cl-bi-ve-nonloss-v1` naming H1-H4.
   Unknown versions fail closed.
8. **Ignore, and reject if present as authority:** `zugzwang_controlled`, strategy
   traces, oracle values, `safe`, `proven`, `status` and acceptance flags. None of
   them enters the decision.
9. **Reachability:** optional. If a replay is supplied, verify it fully from empty
   and bind the exact board. Its absence does not affect the bound. If project
   policy requires it, label it as policy.
10. **Output:** only `black_value_at_least_draw` (bound), with the theorem/verifier
    versions and a board digest. Never output "draw", "win", "solved" or a move
    recommendation.

## 8. Changes in this phase

- `docs/phase6b3-victor-independent-audit.md` (this report).
- `tests/victor_validation/independent_audit.py`: standalone audit model,
  generators, mutation knobs, `run_campaign` and `run_cover_diversity`.
- `tests/test_victor_independent_audit.py`: 9 bounded tests (about 0.7 s).
  - Import isolation.
  - Exhaustive parity lemma.
  - Solver agreement with the 6B.2 oracle.
  - The overlap and Claim-odd counterexamples, with production rejection.
  - A mixed CL/BI/VE column (f: CL f1-f2, free f3, VE f4-f5) surviving every
    spare schedule.
  - An unreachable covered board, still safe.
  - Diagram 8.1 rule-set check.
  - A campaign smoke test.

To reproduce the campaigns (stdout only, about 4 minutes in total):

```sh
PYTHONPATH=tests .venv/bin/python -c "from victor_validation import independent_audit as A; print(A.run_campaign(seed=6203, per_setting=300, empties=(8,10,12,14,16,18,20,22)))"
PYTHONPATH=tests .venv/bin/python -c "from victor_validation import independent_audit as A; print(A.run_campaign(seed=1989, per_setting=300, empties=(8,10,12,14,16,18,20,22), follow_ups=(0.25,0.75,1.0)))"
PYTHONPATH=tests .venv/bin/python -c "from victor_validation import independent_audit as A; print(A.run_cover_diversity())"
PYTHONPATH=tests .venv/bin/python -c "from victor_validation import independent_audit as A; print(A.run_cover_diversity(seed=7404, per_setting=500, empties=(10,12,14,16,18,20), follow_ups=(0.6,0.95), covers_per_position=60))"
```

Verification:
- `.venv/bin/python -m pytest -q` over the 6B.2 command's test list plus the new
  module gives **503 passed** (494 existing + 9 new).
- `git diff --check` is clean.
- No AlphaZero, training, benchmark or large exhaustive search was run.

## 9. GO/NO-GO and Phase 6B.4 plan

**GO** for implementing accepted Black non-loss certificates for CL/BI/VE in 6B.4,
provided that:
- (a) the verifier meets every obligation in §7, with its own board validation;
- (b) the only accepted claim is `V_B ≥ 0`, under a versioned theorem identifier;
- (c) mutation tests show rejection when each of H1-H4 is broken, including the
  overlap and Claim-odd boards above;
- (d) this document's proof is reviewed by a second person, or formalised.

**NO-GO** for exact values, optimal play, White contexts, Black-to-move contexts or
any rule outside CL/BI/VE in that phase.

Suggested 6B.4 steps:

1. Add a separate `StrategicCertificate`/verifier module, independent of
   grounding's validator, with the obligations of §7 and fail-closed versioning.
2. Implement σ_R as an explicit move selector, so a certified bound comes with an
   executable non-losing policy. Property-test it against the audit model and both
   exact solvers.
3. Mutation-test the verifier per hypothesis, and add negative tests that inject
   `zugzwang_controlled`, oracle values or `proven` flags to show they are inert.
4. Decide the reachability *policy* explicitly (optional provenance is recommended).
5. Optionally, mechanise L1-L4 in a proof assistant, or as an exhaustive check over
   an abstract column model.
6. Before AE/LI/HI/BC/BE/SB, state and prove a generalised spare-move lemma for
   forbidden regions (§7.1). Re-derive the pairwise-to-joint step for constraints
   2-4, and give that theorem a new version identifier.
