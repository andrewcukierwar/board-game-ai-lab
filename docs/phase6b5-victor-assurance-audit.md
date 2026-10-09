# Phase 6B.5: Independent end-to-end Victor assurance audit

Date: 2026-10-08. Baseline: `f254db41a689cb1266fa3b6421ef8ff1138ec48d`,
verified on `phase6b4b-victor-strategy` with a clean working tree. The audit branch
`phase6b5-victor-assurance-audit` was created directly from that commit.

**Mathematical verdict: the stated CL/BI/VE Black non-loss theorem is sound under
H1-H4 and the standard game rules.** The permissive strategy works for every
legal White continuation and every permitted Black spare choice. The argument
below is an independent reconstruction and code correspondence review, not a
proof-assistant verification. Earlier reviews were read as claims and compared
with the source and implementation; this was not a blinded review.

**Software verdict: H1-H4 checking and valid-input strategy execution withstand
this review, but the certificate boundary does not meet its complete snapshot,
bounded-input-processing and malformed-input-status claims.** Three confirmed
findings have seven reproducible failing probes. No immutable, well-formed
certificate violating H1-H4 was accepted in this audit; no selected legal move
was found to violate the theorem. Those statements do not erase the boundary
defects.

**Scope expansion: NO-GO for integrating the remaining six rules into the
certificate/strategy system now.** First harden the existing boundary and close
the recorded regressions. GO for separately scoped paper analysis of those
rules; none inherits this theorem. **Authoritative/public strategic claims:
NO-GO.** This review does not authorize changing the certification gate.

Production code, theorem/version identifiers, the public factory, API, UI,
explanations, deployment, AlphaZero campaign and frozen benchmarks are unchanged.
Only this report and two test files are added. Known defects are deliberately
preserved with strict `xfail` tests; they are not reported as passing checks.

## 1. Primary source and review method

Primary source: [Allis (1988), local thesis](../research/references/allis-1988-connect4.pdf),
SHA-256 `bca0a6bf53262e214dbff21ea4f878a1ba5298d15a11d93ce7eda3a1a326be86`.
Chapters 4-8, PDF/thesis pp. 25-57, were freshly extracted using macOS PDFKit
and read. Pages 34-38 and 48-51 were also visually inspected as rendered images,
including the compatibility matrix. Other diagrams were not visually audited.
Scratch extraction/rendering stayed outside the repository. Poppler and PDF
Python packages were unavailable; a temporary dependency-install attempt failed,
so native PDFKit was used without changing project dependencies.

| Source | Independent assessment |
| --- | --- |
| §§4.1-4.5 | Follow-up explains parity, but following up from the empty board loses. An even-row supply alone is not a non-loss proof. |
| §5.3, p.34 | BI a1-b1 and CL b1-b2 conflict at the lower trigger b1. H3 must include both CL squares, not just coverage squares. |
| §5.4, pp.34-35 | Complete coverage is essential. The narrative assertion that there is no circularity needs a simultaneous invariant argument. |
| §§6.1-6.3, pp.36-39 | CL covers its even upper; BI needs two current landing squares and covers groups containing both; VE covers both adjacent empty squares, with odd upper for this standalone rule. Proactive VE lower occupation is explicitly contemplated. |
| §§7.1-7.3, pp.47-49 | Parity restrictions and BI/VE independence explain why these rules compose. They do not establish the same composition theorem for every future rule. |
| §7.4, p.50 | All six CL/BI/VE pair types require constraint 1: disjoint full square sets. The rendered table confirms this, including CL-CL and VE-VE. |
| §8.1, p.51 | Black may try full-board coverage without a separate Zugzwang-control test. For this fragment, the constructive proof below justifies that conclusion. |
| §§6.4-6.9, 8.2-8.4 | Composite rules, early winning groups and White threat regions introduce different obligations. Their correctness is not certified here. |

The formal definitions, rather than example lists, determine the implemented
fragment. The useful BI b1-d3 omission in diagram 6.2's list is consistent with
the visible board: b1-c2-d3-e4 has no Black stone. The §8.2 attribution to CL
d5-d6 is not in its declared rule list; this remains a source erratum, not a
reason to broaden the current Black-only verifier.

## 2. Reconstructed theorem and hypotheses

Use a 7-column, 6-row board with rows 1-6 counted from the bottom. White is X,
player 0, and Black is O, player 1. Stones persist; each move places exactly one
stone at a column's lowest empty square. Players alternate without passes. Play
ends immediately on a four or a full board. These are part of the **standard
ruleset**, not additional facts supplied by a producer.

| ID | Precise hypothesis | Verdict |
| --- | --- | --- |
| H1 | Gravity holds, counts are equal, White is to move, neither color has a four, and the board is not full. | Sufficient. Counts and turn matter to the parity argument. Historical reachability is a distinct question. |
| H2 | CL: two empty adjacent vertical squares, upper even. VE: two empty adjacent vertical squares, upper odd. BI: two distinct current landing squares. | Matches §§6.1-6.3. CL/VE need not initially be playable. |
| H3 | Complete affected-square sets are pairwise disjoint. | Sufficient for unique obligations and noninterference here. Minimality is unnecessary; redundant disjoint rules are permitted. |
| H4 | Every one of the 69 fours with no initial Black stone contains a CL upper or both endpoints of a BI/VE instance. | Covers all possible future White wins, including currently empty groups. No threat-only filtering is sound. |

**Theorem.** If H1-H4 hold, Black can guarantee at least a draw from this board.
More strongly, every play conforming to the set-valued strategy below prevents
a White win, regardless of White's legal choices and Black's permitted spare
choices. This is a lower bound, not an exact draw, exact win, or optimality claim.

For a fixed original rule collection R, a CL is retired when Black occupies its
upper; a BI/VE is retired when Black occupies either endpoint. Otherwise it is
active. Retirement is permanent. After White touches an active pair, Black must
take its partner (for CL/VE White can only have played the lower). If no active
pair was touched, Black may take **any landing square except an active CL lower**.

At White-turn boundaries maintain invariant I:

1. Both squares of every active rule are empty.
2. Black has never occupied any original CL lower.
3. Every group originally covered by a retired rule contains a Black stone.

Item 3 follows from the definition of coverage and retirement; it is useful as
an explicit executable assertion. H2 initializes I because all pairs are empty.

### L1: White cannot win before Black responds

Any newly completed White group G would have had no initial Black stone, since
stones cannot disappear. H4 supplies a covering rule. If retired, it puts Black
inside G. If an active CL, its upper lies in G, is empty, and cannot be played
while its lower remains empty. If active BI/VE, two distinct cells of G are
empty; one White move cannot fill both. Every case contradicts completion.
This explicitly includes White's triggering move. There is no deferred-reply
loophole in which White wins first and Black responds after the game is over.

### L2: Mandatory replies are unique and playable

H3 puts a White square in at most one rule. A vertical upper becomes playable
immediately after its lower is filled. An untouched BI endpoint remains a
landing square: it was initially one, and filling a different column cannot
change that; playing in its own column would fill the endpoint first. The
invariant says it is still empty. The response cannot hit another active rule,
or any other CL lower, because all original pairs are disjoint. A rule's own
CL upper is not its lower. Thus the mandatory response restores emptiness for
all remaining active rules and retires the triggered rule.

Stacked pairs introduce no extra requirement. Gravity may delay access to a
pair but cannot make its upper available ahead of its empty lower. No rule
needs stones below it to retain a particular color.

### L3: A permitted spare always exists

After White moves there are an odd number of occupied squares, hence an odd
number of empty squares because the board has 42 cells. At least one column
has an odd number e of empties. Its landing row is 7-e, which is even. Such a
square exists, since e is positive, and cannot be a CL lower (always odd).
This proves existence without assuming Black controls Zugzwang, all initial
column heights are even, or White has no odd threats.

The even **height of the board** and turn parity are load-bearing. The argument
must not be silently transferred to other dimensions or turn contexts.

### L4': Every permitted spare preserves I

In the spare case White has touched no active pair, so every active pair still
has two empty cells. A legal spare t cannot be a CL upper because that upper's
lower is empty; it cannot be an active CL lower by the policy. Therefore it
touches no active CL. If it touches a BI or VE, Black's occupation retires that
rule and blocks every group it covers. All other active rules remain empty.
A retired CL lower is already occupied by White: gravity was required to put
Black on its upper, and item 2 rules out Black below it. It cannot subsequently
be played as a spare. Retired groups stay blocked by stone persistence.

No step uses the parity of t. Odd-row spares are sound when permitted; even-row
spares are a sufficient refinement, not the definition of the theorem. BI/VE
proactive retirement is crucial: demanding that Black always wait for White
to touch those pairs would unnecessarily forbid safe spares.

### Termination and conclusion

H1 gives an even number of remaining moves. White cannot fill the board on its
turn, and L1 prevents its winning. Black always has a legal response by L2/L3.
Each complete pair of moves restores I, unless Black has already won. A Black
win ends play immediately and requires no subsequent obligations to be carried
out. In at most 42 minus the initial stone count moves, play ends in a Black
win or full-board draw. This proves the universal non-loss claim.

No additional board predicate beyond H1-H4 was found necessary. No replay is
needed for a theorem about the game *from* a basic-valid board. A statement
about an actual played game separately needs provenance. Likewise, faithful
execution is the construction in the proof, not an extra certificate hypothesis.
Trusted code/runtime execution and stable input snapshots are software assurance
assumptions; a theorem about abstract boards does not supply them.

## 3. Finite verifier-to-theorem correspondence

The following is a complete block-by-block review of the baseline 723-line
`certificate.py`; line numbers refer to the unchanged audited commit.

| Lines | Review result |
| --- | --- |
| 1-77 | Statement, fixed identities, scope and provisional gate agree with the theorem. The registry is shallowly and internally mapping-protected. Its review-status text remains historically pending; this report does not change acceptance policy. |
| 79-119 | Square parsing requires exact strings, valid column and row. Group generation produces consecutive fours in four directions. Independently classified all four-subsets confirms precisely the same 69 groups, not merely the same count: 24 horizontal, 21 vertical, 12 in each diagonal direction. Canonical group helper is a producer aid, not an acceptance predicate. |
| 122-236 | Exact-type/slotted input records prevent added authority fields and subtype substitution at verification. Frozen records can still contain lists. Verdicts are forgeable Python data, explicitly not authorities; properties on a fabricated verdict do not constitute verification. Accepted status alone does not mean history-backed or public-certified. |
| 238-281 | Board hash unambiguously binds fixed ruleset, turn, top-first cells. Producer helpers can raise and do not prove legality. `draft_certificate` freezes the board and outer rules sequence but not a rule's nested list. Serialization assumes well-formed input. |
| 284-331 | Mapping parser rejects missing identity fields, extra keys and wrong rule entry shapes; replay/assignments alone are optional. It is a Python JSON-like adapter, not a byte-level JSON parser or duplicate-key detector. **F1:** conversion precedes cardinality/budget checks. |
| 334-393 | Budget configuration rejects bool/negative/non-int. Board validation uses exact lists/tuples and exact cells, independent landing/winner scans, exact version identifiers. Unknown supported-domain versions fail unsupported, never accepted. |
| 394-432 | Rules are reparsed, max 21, exact rule records/kinds, square syntax and duplicates checked. Known other Allis rules are unsupported. Parsed immutable coordinate tuples protect H2-H4 from later caller rule-list changes. Malformed diagnostics may invoke `repr` (**F3**). |
| 435-468 | H2 checks distinctness, emptiness, BI current landings and distinct columns, canonical BI ordering, vertical adjacency/roles and upper parity. Canonical ordering is an extra safe serialization restriction. New tests compare every distinct unordered pair and all three kinds on three boards: 7,749 cases. |
| 471-484 | H3 compares both endpoints; H4's needed set is CL upper only, otherwise both. Disjointness of coverage sets alone is not substituted. |
| 486-530 | Targets are reconstructed from all 69 groups without Black. Every target must have at least one covering rule. Optional assignments must exactly enumerate targets once and name an actually covering rule index; bool indices fail. Adding a malformed group's index to `assigned` cannot erase the recorded rejection. |
| 533-559 | Optional replay alternates from empty, rejects full columns, booleans, invalid columns and play after a win, binds the final board and White turn. Full-board continuation cannot evade the 42-ply cap/H1 check. Replay is existential provenance, not the actual session identity. |
| 562-577 | Digest includes rule ordering, versions, board, turn, defender, optional provenance and assignments. Board digest is redundant with these validated inputs and is not itself an authentication token. **F1/F3:** recursive serialization precedes optional validation and invokes arbitrary `repr`. **F2:** optional fields are read separately from their later checks. |
| 580-655 | Stage order is schema/version/context, frozen board/digest and H1. Gravity catches an occupied cell above any gap through an adjacent empty/occupied transition. Counts and stated turn must agree; Black-to-move is unsupported; both single and dual winners and full boards reject. |
| 657-695 | H2/H3/H4 success precedes optional replay; failures cannot fall through to acceptance. Cutoffs yield UNKNOWN with no bound. Bad replay rejects even with recorded H1-H4 success. Exception normalization is incomplete (**F3**). The comment about never consulting caller containers again is too broad (**F2**). |
| 697-723 | Recomputed evidence is immutable and derives from checked board/rules/coverage; assignments are rebuilt, not copied as authority. Mapping verification normalizes TypeError/ValueError parse failures then uses the same verifier. It is not a separate independent verifier. |

For stable, schema-valid data the acceptance path implies exactly H1-H4.
The new finite comparisons and code review support that correspondence; they
are not an exhaustive enumeration of all colored boards and rule collections.
The return digest can misidentify optional data under F2, even though its frozen
board/rule hypotheses still hold. Consequently an unqualified claim that **every
accepted whole certificate is correctly bound and validated** is false.

Unknown cutoffs are epistemic, not negative results. The work budget counts
selected abstract operations, not Python instructions, elapsed time, allocation
or input bytes. `DEFAULT_WORK_BUDGET` comfortably covers the fixed valid schema;
it does not protect against oversized invalid inputs.

## 4. Confirmed findings and reproducible failures

All findings refer to the unchanged baseline. No critical mathematical flaw or
false non-loss bound was established. Severity is calibrated to the current
research-only, in-process exposure, not a hypothetical public service.

### F1 — Medium: validation and resource accounting occur after unbounded work

`certificate_from_mapping` (294-331) constructs every supplied rule and converts
containers before checking board/rule/replay/assignment cardinality. Even
`verify_certificate_mapping(..., work_budget=0)` constructs 22 rules in the
small repro before returning UNKNOWN. This scales with attacker-supplied size.

`verify_certificate` computes `_certificate_digest` at line 663 before
`_assignments`/`_replay` validate their shapes and limits. With the valid early
board/rules below, `replay=[0]*1000` and `work_budget=69`, `_plain` visits every
entry before the next charge stops verification. Larger or recursively nested
data can consume work/memory unrelated to the budget. No destructive allocation
or denial-of-service stress test was run; the finite instrumentation establishes
the order-of-operations problem.

Two strict expected failures preserve this: mapping cardinality before rule
construction, and oversized optional data before digest traversal. Strategy's
bounded outer copying protects against these oversized outer arrays, but its
scalar leaves still reach F3; mapping parsing before strategy is also outside
that protection.

Recommendation: bounded structural/type preflight before conversions, hashing
and diagnostics; reject illegal optional shapes without walking them. Define
separate ingress size/depth limits and abstract verification-work semantics.
Any future byte parser also needs an explicit duplicate-key/size policy.

### F2 — Medium: optional fields are not part of the verified private snapshot

The board and parsed rules are snapshotted; `replay` and `assignments` are not.
Both are digested and later reread from the original certificate. A caller can
edit an ordinary list between those operations. This contradicts 6B.4A §2's
claim that every caller field is read once into a private snapshot.

The two deterministic interleaving reproducers wrap the real digest function,
let it finish, then mutate **the caller's list**, without changing any checking
predicate, theorem, status, module constant or certificate field via
`object.__setattr__`. This models a concurrent list edit; it is not a claim that
a remote JSON payload can schedule Python execution.

Using the early fixture, initially invalid replay `[6]` is replaced with the
valid `23333334` history after hashing. The result is `theorem_hypotheses_verified`
and history-backed, but carries:

```text
returned: sha256:0e313aca670e9d043c42fab908c35cc7a1e508c1e006671ceea08e055337ae57
correct:  sha256:36fcfb38a09f6fd02dcf07176830cb6c072ca81d6d275f74ba7b0a591946f1a1
```

Likewise initially incomplete assignments `[]` are replaced with the valid
recomputed assignment list after hashing:

```text
returned: sha256:9575f79acb8f3e0de6e90df8f836f8406adb23e2caf7f8ea03f5beca0fecab87
correct:  sha256:29ad1364b761c6cc57533862726e950408d4b8a41f12b5da3a338b6c1fe4b0ba
```

The immutable initial bad certificate would reject. The verified final replay
is legal. The defect is mixed-time identity, not a SHA-256 collision, forged
authentication, or a proof that an H1-H4-invalid board can pass. The strategy
already copies these containers before verifying; a new passing test mutates
both original optional containers at its verification boundary and confirms
that execution uses the intact copy.

Recommendation: independently snapshot **all** bounded data at the verifier
entry, validate and digest only that snapshot, or explicitly restrict the
interface to deeply immutable input. Retain the strategy's own copying.

### F3 — Low: malformed Python objects can escape rejection through `repr`

A replay tuple containing an object whose `__repr__` raises RuntimeError reaches
`_plain` before replay validation. That exception escapes `verify_certificate`,
`verify_certificate_mapping` and `select_black_move`. Three strict expected
failures reproduce this exact exception. The strategy's structural copy retains
the invalid scalar object, then invokes the verifier outside its snapshot catch.

This violates the promise of returning an explicit invalid-input status. It
does **not** return a move or accepted certificate. Such a leaf cannot come from
ordinary JSON decoding; it concerns the advertised Python-object boundary and
buggy/local callers. Code already executing arbitrary Python is not treated as
an isolated adversary here. The finding is avoidable callback execution and
exception handling, not a claimed remote-code-execution exploit.

Recommendation: reject unsupported leaf types before formatting or hashing;
diagnostics should name their type or location without invoking caller hooks.
Catching more exceptions around `repr` alone would leave the resource problem.

### F4 — Informational: assurance and identity are not authentication

Verdict/trace dataclasses can be constructed by arbitrary Python code; a forged
verification verdict can expose `provisional_bound` without running checks.
The implementation correctly refuses saved verdicts as strategy input and
accepts no execution trace as input. SHA-256 identifies
data; it neither authenticates a producer nor independently proves the theorem.
Caller code must not turn a saved result, matching hash, or coverage witness
into an authoritative game value. Runtime/module integrity is trusted.

## 5. Executable strategy review

Complete review of the unchanged `strategy.py`:

| Lines | Review result |
| --- | --- |
| 1-84 | Research scope and separate deterministic strategy identifier are explicit. Frozen result records contain no exact value, optimum or public outcome. Supported certificate identities are pinned literal strings. |
| 87-113 | Bounded copying handles mutable board rows, rule square lists, replay and nested assignments. Re-verification is unconditional. Scalar preflight still inherits F3. No stored verifier verdict or trace is consumed. |
| 116-133 | Landing uses bottom-up rows; terminal scan shares the verifier's 69 groups; retirement is reconstructed from Black occupation, not claimed flags. This sharing is a common-mode geometry risk, now also checked against the new four-subset model. |
| 136-151 | I is checked at White turns and after Black moves. An active pair must be empty; every retired rule blocks all original groups it covered; original CL lowers cannot contain Black. Coverage tuple alignment is supplied by the re-run trusted verifier. |
| 154-174 | Mandatory response has precedence over every spare and any immediate alternative Black win. H3 bounds touched rules to one. Upper-entry and unavailable-partner contradictions return invariant failures. Otherwise even-row landings are scanned in physical column order. |
| 177-211 | Exact request type, budget configuration, private snapshot, verification result and pinned verifier/formulation IDs control acceptance. UNKNOWN supplies no move. The verifier's uncaught F3 exception is inherited at line 199. |
| 213-254 | Continuation is a bounded tuple of exact ints, replayed from the certified board with White first. Terminal checks precede moves; full columns fail. Each prior Black move must match the forced response or exact spare tie-break, even if a different move was mathematically safe. Trace and retirement fields are recomputed from actual drops. |
| 255-268 | White terminal is an invariant failure, correctly catching an impossible theorem violation instead of proposing a post-win reply. Black terminal is allowed; invariant checks still hold immediately after it. No subsequent move can pass. |
| 269-284 | The returned board/trace ends **before** the selected future Black move. Black-turn active pairs may contain the triggering White stone; the empty-pair invariant belongs at White turns. Terminal and White-turn endpoints supply no move. State digest binds current turn/cells; original certificate digest and provenance are separate. |

The valid-input implementation is a faithful refinement of the theorem. In
particular, a forced VE upper can be odd, and the even-row spare filter does
not override it. A mandatory move may forgo an immediate Black win without
losing the non-loss guarantee. No guarantee of optimal play is made.

An arbitrary permissive Black spare may be rejected by this deterministic
selector because its lowest-column tie-break is stricter. That is intentional,
not a theorem counterexample. Reordered redundant rules change certificate
identity but not the unique response. The original collection remains fixed;
re-enumerating rules after moves would be unsound bookkeeping and is not done.

A stale or fabricated output trace cannot bypass checks: there is no trace
input. A fabricated result object is not a StrategyRequest. A supplied valid
continuation is checked from the initial board every time. Binding to an actual
external live game's current board/session remains a future integration duty;
this stateless research function is given only a certificate and continuation.

## 6. Cross-layer trust analysis

| Transition | Trusted/recomputed vs claimed | Failure and common-mode risk |
| --- | --- | --- |
| Raw board -> local candidates | Production `Position` uses grounding legality; geometry comes from grounding groups. Enumeration derives empty CL/VE pairs and all BI landing pairs. | Local candidates say nothing about a global bound. Shared production legality is not independent assurance. |
| Candidates -> compatible cover search | Search rebuilds context and all targets, coverage masks and full-square conflicts; DFS branches on a least-option uncovered group under an explicit node cap. | No cover is only about this rule universe. Cutoff is UNKNOWN. Removing zero-coverage rules cannot remove a necessary cover. A compatible solution extension always supplies a branch for the chosen uncovered group, so completed DFS is complete within the enumerated universe. |
| Search -> witness verification | Old witness verifier reconstructs targets, prerequisites, disjointness and assignments; it shares `Position` and vocabulary with production. | Its status is `coverage_verified_outcome_uncertified`, never exact solved. Shared H1 validation is a residual reason not to treat it as the final boundary. |
| Witness -> certificate producer | Adapter copies board, rule identities, optional assignments/history; BI ordering and group names are normalized. Evidence/status claims are ignored. | Adapter may raise on malformed producer objects; it is not an acceptance API. Omitting assignments does not omit independent H4 checking. A malicious producer must still pass the new boundary. |
| Certificate -> independent verifier | Own cells, geometry, counts, gravity, winners, rules, compatibility, target coverage and optional replay are recomputed. | No producer validity flags are trusted. Stable well-formed inputs have the intended finite semantics; F1-F3 limit the stronger hostile-input claims. |
| Verified certificate -> executable strategy | Strategy accepts original claim, copies and re-verifies, pins identities, replays complete continuation and every prior policy choice. | No cached verdict/trace authority. Strategy and verifier share group geometry, color/row conventions and the same mathematical interpretation; independent source reasoning and reference geometry are needed to challenge these. |
| Selected move -> terminal result | Caller must actually apply the selected legal move. Next call checks the entire claimed prefix. Winner/fullness are recomputed; post-terminal play rejects. | No public outcome field is updated. A caller executing a different move loses this strategy's guarantee; an externally fabricated history is not proof it actually happened. |

All implementations necessarily encode the same intended rule meanings; merely
agreeing with one another is not proof those meanings are sound. The inductive
argument above and exact game oracles supply different kinds of evidence. The
independent verifier's standard-library-only source can run in isolation, though
a normal package import also executes the parent package's imports; this is a
packaging dependency, not reuse of their predicates inside certificate checks.

## 7. Adversarial validation actually executed

New test-only model: `tests/victor_validation/assurance_model.py`. It imports only
`itertools`; it uses bottom-first column strings and classifies all 111,930
four-square subsets geometrically. It imports neither production nor previous
audit/strategy/oracle helpers. Its exploration quantifies over all White moves
and **all** permitted Black replies, including odd spares. Memoization merges
identical board states because the fixed original rules determine all future
obligations. The first encountered path is retained for assertion diagnostics.
Separate executable checks replay every path without that memoization.

This new model itself is ordinary reviewed Python, not a trusted formal kernel.
Its completion claim is conditional on its correctness. It has explicit state,
depth and path caps; a frontier yields UNKNOWN, never a completed claim.

### New early stacked-rule fixture

Zero-based history: `23333334`. Board top to bottom (`.` = empty):

```text
...X...
...O...
...X...
...O...
...X...
..XOO..
```

Sixteen disjoint instances:

```text
BI a1-c2; VE a2-a3;
CL a5-a6;
CL b1-b2, b3-b4, b5-b6;
CL c3-c4, c5-c6;
CL e3-e4, e5-e6;
CL f1-f2, f3-f4, f5-f6;
CL g1-g2, g3-g4, g5-g6.
```

The board history is the existing diagram-6.1 reconstruction; **this stacked
mixed-rule collection is newly constructed for this audit**. All three kinds
occupy column a at different heights, separated by free a4. The new checker and
independent certificate verifier both validate H1-H4. The deterministic
continuation `(0,2,0,0,0,4,0,0)` executes White a1 -> Black c2 (BI), White a2 ->
Black a3 (VE), White a4 -> Black e2 (spare), White a5 -> Black a6 (CL).
Corrupting each prior Black response is rejected.

### Finite results and bounds

| Experiment | Exact observed result | Scope |
| --- | --- | --- |
| Four-subset geometry and H2 differential | Same 69 groups; 7,749 distinct-pair/kind/board checks agree. For every locally valid single rule, uncovered H4 groups agree exactly. | Empty, early and 10-empty boards only; all coordinate pairs on those boards. |
| New early all-spares traversal | 120 White-boundary states, 226 two-ply transitions; 10 odd spares, 20 proactive BI and 21 proactive VE retirements; 84 frontier states; no contradiction. | 3 rounds, cap 2,000 states; **UNKNOWN beyond depth frontier**. A separate cap-1 run returns UNKNOWN. |
| 10-empty mixed board, history `33003316106600034431441142662221` | Complete: 38 states, 88 transitions, 18 draw-ending edges, 0 Black-win edges; 8 proactive BI and 10 proactive VE retirements. | All White moves and all permitted Black spares; cap 2,000 not reached. |
| 12-empty mixed board, history `511322464155553320540623416612` | Complete: 111 states, 481 transitions, 182 Black-win and 15 draw-ending edges; 36 odd spares, 60 proactive BI and 50 proactive VE retirements. | All White moves and all permitted Black spares; cap 2,000 not reached. |
| Actual selector vs new model, early fixture | 258 selections, no terminal paths within 3 rounds. | Prefix check only, 3,000-selection cap; no exact full-game value inferred. |
| Actual selector vs new model, 10-empty fixture | 102 selections, 35 terminal paths. | Complete deterministic strategy tree; every terminal and extra post-terminal ply checked. |
| Actual selector vs new model, 12-empty fixture | 79 selections, 45 terminal paths. | Complete deterministic strategy tree; same stopping checks. |
| Secondary exact oracle | 175 Black-turn checks, all exact; every selected move has nonnegative Black move value. | Only endpoints with <=10 empties; cap 50,000 positions per call. Earlier 12-empty states were not solved. |
| Malformed field matrix | 100 malformed field replacements rejected/unsupported, no bound; four additional forged mapping keys rejected. | Fixed finite plain-data inputs, not unbounded fuzzing. |
| Optional strategy snapshots | Mutating original replay and nested assignments before verification does not affect selection on its private copy. | New passing control for F2's strategy mitigation. |
| Boundary defect reproducers | Seven fail with `--runxfail`: two F1, two F2, three F3. | Small deterministic inputs and documented instrumentation. |

The two complete endgame boards were already in earlier suites, but the
reference model, geometric construction, early stacked cover and boundary
probes are new. This is not a claim to have newly discovered those endgames.
Exploration counts are states/edges after memoization, not unique terminal
boards or counts of all possible game histories.

Two exploratory, explicitly bounded fixture searches with seed 6505 were also
run: 30 continuations of the early existing cover, and up to 400 random
non-winning histories of length 32 with cover-search budget 2,000. Neither
found a new all-three-kind endgame. These were fixture-discovery attempts, not
exhaustive nonexistence results. The hand construction above provided the
intended mixed-column stress case. A separate fresh-model traversal of the old
CL-lower fixture (`550066550544410241221101261344663650`) completed with 5
states, 7 permitted transitions and 2 draw-ending edges. Its forbidden-spare
counterexample remains covered by existing tests, not counted as a new defect.

Existing relevant suites were rerun to cover historical unreachability,
full/nearly full boards, reflected/reordered/redundant rules, mandatory replies
that forgo wins, invalid digests and unknown versions, missed replies, malformed
continuations, post-terminal moves, replay provenance and resource cutoffs.
Their passing historical campaigns are regression evidence, not independent
proof. Existing independent exact oracles were inspected as secondary checks;
they too are software and share the standard game's interpretation.

## 8. Verification record

Commands used the existing project virtualenv and disabled live explanations:

```sh
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest -q -s tests/test_victor_assurance_audit.py
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest -q tests/test_victor_assurance_audit.py
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest -q tests/test_victor_assurance_audit.py --runxfail
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest -q tests/test_victor_*.py
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests -q
git diff --check
```

The first development run, before the final two added cases, was 10 passed and
6 expected failures (1.49s). The deliberate final defect-reproduction run was
**11 passed, 7 failed** (1.49s), with all seven failures at the documented
assertions/RuntimeError. Production is unchanged; these failures are retained
as strict xfails so an eventual fix causes XPASS and requires removing the marker.

| Final run | Passed | Failed | Skipped | Strict expected failures | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| New assurance tests | 11 | 0 | 0 | 7 | 1.37s |
| All Victor suites | 497 | 0 | 0 | 7 | 5.62s |
| Full backend | 998 | 0 | 14 | 7 | 25.21s |

`git diff --check` passed. The 14 existing optional PyTorch-dependent skips
are not passes. The seven xfails are unresolved confirmed defects, not skipped
experiments or successful defenses. The deliberate `--runxfail` run above
exposes them as seven ordinary failures.

No training, neural evaluation campaign, tournament or unrestricted solving was
run. Backend tests exercise existing lightweight test paths; the separate
campaign and artifacts were not changed or run.

## 9. Limitations, decisions and next milestones

**What is established:** a reviewed paper proof of the specific universal
CL/BI/VE strategy, a line-by-line implementation correspondence assessment,
independent finite geometry/local-predicate checks, new bounded early-game
exploration, complete exploration of two particular endgames, and concrete
counterexamples to stronger software boundary promises.

**What is not established:** mechanical formalization of the universal theorem,
exhaustive validation over all legal boards/covers, complete exploration of the
early fixture, absence of every Python/concurrency defect, authenticated game
provenance, exact values/optimal moves in general, or soundness of any additional
Allis rule. No proof assistant was used. A finite hypothesis checker applies
a mathematical theorem; it does not mechanically prove that theorem itself.

| Priority | Recommendation / assurance obligation |
| --- | --- |
| 1 — before certificate/strategy expansion | Fix F1/F2/F3 with bounded, exact-type snapshots of all certificate fields before hashing/use; avoid user callbacks in error formatting. Keep the reproducers and replace their xfails with passing regressions. Review the revised boundary independently. |
| 2 — before any authoritative claim | Establish explicit review/policy ownership; define stable immutable input/JSON limits, provenance and live-board binding, fail-closed statuses and the distinction between bound and exact value. Preserve the current gate until that work is approved. |
| 3 — each new rule | Write a separate theorem/version, exact prerequisites/coverage, combination semantics and executable obligations. Prove a new spare/executability invariant where required; do not append rules under `cl-bi-ve-black-nonloss-v1`. |
| 4 — stronger mathematical assurance | Formalize the finite board model, monotone groups, L1-L4' and termination in a proof assistant, or commission another genuinely independent proof review. This audit increases paper-proof assurance; it does not become a formal kernel. |
| 5 — maintenance | Update prior snapshot/resource documentation when hardening lands; retain the early mixed-column fixture, independent model and UNKNOWN frontier semantics. Ensure no future public consumer accepts witness/verdict/trace claims without re-verification and context binding. |

Recommended sequence: **6B.5A boundary hardening**, followed by a separately
reviewed **6C rule specification/proof milestone** (one new rule at a time), then
candidate/coverage implementation and independent finite checks, then executable
composition/terminal audits. Public certification should be its own later
milestone, not a side effect of adding rules or passing tests.

The present NO-GO for integration is a bounded engineering decision about
unfinished assurance obligations, not a rejection of the reconstructed theorem.
No theorem semantics or certification gate is changed by this audit.
