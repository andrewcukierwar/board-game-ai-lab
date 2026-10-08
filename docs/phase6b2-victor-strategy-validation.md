# Phase 6B.2: Victor strategic soundness validation

**Recommendation: NO-GO for accepted strategic certificates in this phase.**
The CL/BI/VE fragment admits the constructive Black non-loss argument below,
subject to explicit prerequisites. Bounded falsification found no contradiction.
Neither that observation nor the thesis's §8.1 narrative certifies our code.
The production status remains `coverage_verified_outcome_uncertified`.

Baseline: `bdc2193ecf7bc788b8d996182af8e98da7f0d34b` on
`phase6b-victor-compatibility`, verified present at HEAD with a clean working tree.
The new branch is `phase6b2-victor-strategy-validation`, created from that commit,
not main. Changes are confined to this document and test-only research tools.
No production Victor code, AlphaZero code/oracle/labels, canonical evidence,
agents, API, UI or deployment configuration is changed. No outcome acceptance,
training, model inference, tournament, paid call, push, merge or deployment occurs.

## Primary-source audit

Source: Victor Allis (1988), *A Knowledge-Based Approach of Connect-Four: The Game
Is Solved: White Wins*, local
[`research/references/allis-1988-connect4.pdf`](../research/references/allis-1988-connect4.pdf).
SHA-256: `bca0a6bf53262e214dbff21ea4f878a1ba5298d15a11d93ce7eda3a1a326be86`.
The 91-page edition uses the contents' page numbers as 1-based PDF pages.
Chapters 4-8 (PDF pages 25-57) were extracted with macOS PDFKit, and pages
34, 36, 38, 48 and 50 were rendered and visually inspected: the response conflict,
CL/VE diagrams, Zugzwang interaction and compatibility matrix. Research tooling
and extracted/rendered files stayed in `/tmp`. No PDF/runtime dependency was added.

Read the complete Phase 6A and 6B.1 Victor package, their documentation and tests,
and the grounding/engine interfaces they use. In particular, the independent
coverage verifier still shares the basic snapshot validator with production;
it does not establish historical reachability or execute a strategy. The new
oracle does not share that validator, winner detector, group geometry or search.

| Source | Mathematical content and implication |
| --- | --- |
| §§4.1-4.2, pp.25-27 | Follow-up assigns odd/even cells under particular conditions. Pure follow-up from empty loses; parity control alone does not prevent a White four. |
| §§4.3-4.5, pp.27-31 | Odd threats can change usable-board parity; control is useful only with prevention of the opponent's wins. An at-least-draw argument is explicitly distinguished from determining the exact value. |
| §5.3, pp.33-34 | BI a1-b1 and CL b1-b2 demand different immediate responses to b1. Lower trigger squares count in compatibility. |
| §5.4, pp.34-35 | Every potential White group must have a mutually executable refutation. One unresolved group invalidates this method's conclusion, without implying a White win. The claimed noncircularity needs an invariant, not merely individual claims. |
| Chapter 6 introduction, p.36 | Apply rules with the opponent to move. Black attempts the whole board, even when White has an odd threat; White needs a separately justified restricted context. |
| §§6.1-6.3, pp.36-39 | CL claims the even upper cell; BI/VE prevent possession of both pair cells. Empty pairs and BI landing support are essential. |
| §§6.4-6.10, pp.39-46 | Composite rules introduce additional timing, terminal and overlapping-component reasoning. None is needed or enabled in the argument here. |
| §§7.1-7.3, pp.47-49 | CL lower cells are forbidden to the defender while active; even released regions preserve move order. BI/VE can be occupied proactively. Inverse/composite interactions need more than disjointness. |
| §7.4, p.50 | All six supported CL/BI/VE combinations require disjoint full affected-square sets. Constraints 2-4 and composite exceptions remain outside this fragment. |
| §8.1, p.51 | Black need not separately prove Zugzwang control before attempting full compatible coverage. This is a proposed strategic conclusion, not verification of this implementation. |
| §§8.2-8.4, pp.51-57 | White's reserved threat column and threat-combination variants have different claims and timing exceptions. They cannot be obtained by changing the defender bit in the Black argument. |

No defect was found in the implemented local CL/BI/VE predicates or the six
compatibility entries. This is a source/code audit conclusion, not a formal
software correctness theorem. Retain the earlier source discrepancies: diagram
6.2's useful-BI list omits formally valid b1-d3; §8.2's displayed coverage discussion
uses overlapping d5-e5 BI and d5-d6 CL. Neither was patched away or used to relax
Black compatibility. The White example remains outside this acceptance proposal.

## Precisely what full compatible coverage could establish

For Black, write the exact value as `V_B ∈ {-1, 0, +1}` (loss/draw/win with best
play). The proposed bound is **`V_B >= 0`**: there exists a Black strategy against
every legal White continuation that prevents a White win. It does not assert
`V_B = 0`, `V_B = +1`, an optimal move, or anything about other positions.

The narrow fragment requires all of the following:

1. The standard 7-by-6 gravity game, White first, alternating one stone per turn,
   no passes, and immediate termination on the first four or a full-board draw.
2. A historically reachable, nonterminal **White-to-move** snapshot. With 42 cells
   and equal stone counts, it has an even number of empty cells. The argument
   uses the full board and Black as defender; no reserved/excluded column.
3. Every selected CL has two empty adjacent cells, even upper/odd lower. Every
   VE has two empty adjacent cells, odd upper. Every BI has two distinct empty
   landing cells in different columns. Intrinsic shapes alone are insufficient.
4. The selected instances have pairwise disjoint **both-cell** sets. Their response
   obligations must therefore never require two different replies to one move.
5. Every one of the 69 possible White winning lines containing no initial Black
   stone is covered: CL upper belongs to the line, or both BI/VE cells belong.
   Entirely empty lines count. Caller-supplied exclusions are forbidden.
6. Black actually maintains the response and spare-move invariant below. A static
   coverage witness cannot authorize arbitrary Black moves.

These are sufficient for the reconstructed fragment argument. An independently
checked initial claim that Black controls Zugzwang, the absence of an odd White
threat, all columns initially having even heights, pairing every empty cell, or
implementation of the six remaining rules is **not an additional prerequisite**
for this argument. Its move-order reasoning must still be justified. This
conclusion is restricted to CL/BI/VE on this board and evaluation context.

### Constructive strategy and inductive invariant

At every White turn, an active CL has both cells empty; a retired CL has its upper
cell occupied by Black. An active BI/VE has both cells empty; a retired BI/VE has
at least one cell occupied by Black. Black never plays an active CL lower.

After a White move in an active pair:

- CL/VE lower: Black takes its upper immediately. Gravity makes it playable.
  White cannot reach the upper first while the lower remains empty.
- BI endpoint: Black takes the other endpoint immediately. It was initially a
  landing cell and stays a landing cell until occupied; play in another column
  cannot remove its support. White cannot take two endpoints in one turn.
- Disjointness makes the demanded reply unique and ensures that it cannot consume
  another active CL's lower or another pair's cell.

After a White move outside all active pairs, Black may make **any legal spare move
which is not an active CL lower**. Occupying a BI endpoint or VE lower proactively
retires it while still blocking every line it covers (§7.2). An untouched CL's
upper cannot be a landing cell with its lower empty. A resolved pair's remaining
cell imposes no further obligation.

Why must a spare move exist? At a nonterminal Black turn the number of empty
cells is odd. If every available column landed on an active CL lower, every such
landing row would be odd. The number of empty cells from odd row `k` through row
6 is `7-k`, even. Full columns contribute zero. Summing would give an even total,
a contradiction. Therefore at least one permitted spare move exists. This is the
explicit parity step missing from a bare collection of local predicates. In
§7.1's region language, successive CL barriers delimit even-sized regions, while
BI/VE add no forbidden region.

Why can White not win **before** the response? A line with an initial Black cell
is permanently blocked. For each other line, retain its assigned rule. A retired
rule already blocks it. An active CL's upper stays inaccessible until White
fills the lower, after which Black immediately takes the upper. An active BI/VE
line still needs both empty pair cells: the first White occupation cannot complete
the line, and Black takes the other before a second White turn. Moves outside
active pairs cannot complete a covered line. Thus White cannot produce a first
win; legal Black moves preserve the invariant or end with a Black win. The finite,
no-pass game must end in a Black win or draw.

This supplies a paper reconstruction of why the fragment can justify a lower
bound under the listed assumptions, rather than appealing to §8.1's narrative.
It remains to independently review/formalize the invariant and establish the
faithfulness of any future strategic verifier. Tests of finitely many positions
cannot prove the universal theorem or the absence of verifier bugs.

## Independent small-endgame oracle

`tests/victor_validation/exact_oracle.py` is test-only, standard-library Python.
It imports no engine, grounding, Victor, agents, AlphaZero, NumPy or Torch. It uses
seven-bit columns with a sentinel row and shift-based four detection (1, 7, 6, 8),
distinct from the engine's and Victor's four-square windows.

Exact negamax values the **side to move**. A terminal winner gets +1 from the
winner's perspective, -1 from the loser's; a full draw gets 0. Parent moves negate
child values. `value_for(0/1)` binds White/Black explicitly, and every root move
value is from the root mover's perspective. All legal children are enumerated in
ascending column order, including after a winning child is found. There is no
heuristic, depth estimate, alpha-beta shortcut, strategic lookup or sealed label.

Memoization stores only completed exact values, keyed by both bitboards, column
heights and side to move. Each invocation has a fresh cache. The default cap is
8 remaining cells and 100,000 unique entered positions, including terminals.
Configurable hard limits are 0-10 remaining cells and 0-1,000,000 positions;
Boolean/noninteger/negative budgets are rejected. Terminal evaluation needs one
position but may occur before the endgame cap. Cap/budget cutoffs return distinct
unknown statuses, no partial value and no move values. No traversal exceeds its
position budget; depth is at most the remaining-cell cap. Matrix validation and
replay are bounded by the board's 42 cells/moves.

Correctness checks:

- Independent legal terminal replays for both winners and a horizontal win;
  full draw and both sides' directly constructed small-endgame matrices.
- All 69 lines, both colors, reflections, removal of each constituent cell and
  sentinel-boundary false positives; all 4,096 occupancies of two six-cell columns
  against a separate contiguous-run matrix winner detector.
- A directly constructed two-cell tactical matrix with `a6` winning and `f6`
  drawing; replayed immediate wins, losing alternatives and a forced drawing reply.
- A slow, uncached matrix minimax restricted to four empty cells. Across all legal
  descendants of the 13 original curated roots, **856 distinct positions** at
  four or fewer remaining cells (500 White turns, 356 Black turns) are checked,
  including every root move value. Traversal also checks independent move
  mechanics and winners above that frontier. There are 984 distinct visited
  positions across the roots, not an unrestricted whole-game enumeration.
- Determinism, real cache hits, exact budget boundaries, missing-value cutoffs,
  illegal/full/post-terminal moves, corrupted snapshots and exact replay binding.
- A fresh isolated Python process actively blocks imports of `games`, `torch`,
  `numpy` and `api` while importing/running the oracle.

Basic matrix validation intentionally does **not** claim reachability. For
example, bottom `OO` with `XX` immediately above has consistent counts and
gravity, but White cannot have made a first move on that board. The oracle may
evaluate constructed basic-valid boards for unit tests; the implication harness
only accepts independently replayed histories.

## Bounded adversarial validation and actual results

`strategic_harness.py` checks replay, invokes the independent oracle, searches
coverage, independently verifies any witness, and runs the bounded strategy
reference model when feasible. A verified witness with exact White value +1 is
retained as `potential_counterexample`, with board, move history, exact root move
values and selected rule/group identities. It is never relabelled to make the
implication pass. A synthetic contrary-oracle test checks that reporting path;
it is not an observed game counterexample.

The deterministic sampler enumerates legal moves, excludes moves that end the
game, selects with `random.Random(seed)`, and replays each selected prefix.
It is deliberately biased toward surviving late positions, **not uniform**.
The discovery/survey budget is fixed to 576 attempted histories: 64 at seed 1988
to 34 plies, 256 at seed 1988 to 36 plies, and 256 at seed 6202 to 34 plies.
63 attempts could not complete such a prefix. The remaining **513 distinct**
nonterminal White-to-move boards have 6 or 8 empty cells. Coverage uses 1,000
states; oracle/execution each use 100,000 positions, cap 8. The maximum observed
oracle count is **546**, and execution count **450**; no default budget was hit.

| Coverage result | White win | Draw | Black win | Total |
| --- | ---: | ---: | ---: | ---: |
| Witness found and independently verified | 0 | 10 | 29 | 39 |
| Exhaustive no cover in CL/BI/VE | 434 | 5 | 35 | 474 |
| Total generated positions | 434 | 15 | 64 | 513 |

Every generated witness completed the bounded execution checks. Those executions
examined 1,634 White edges, 427 forced Black reply edges and 1,397 spare Black reply
edges, including 87 proactive BI/VE occupations. Edges are counted at unique
execution states within each root; these are not counts of whole histories.

The 13 selected original fixtures and their legally replayed reflections give
**26 additional comparison records**, including early source-diagram extensions.
16 carry verified witnesses (6 draws, 10 Black wins). The remaining 10 split into
4 White wins, 2 draws and 4 Black wins. Across survey and curated comparisons there
are **539 records / 529 unique endgame boards**, and **55 verified-witness
comparisons / 49 unique witness boards**. Some originals were selected from the
survey, so records are not independent samples. No potential counterexample or
execution-obligation failure was observed. These are exploratory implication
checks; we do not yet certify all strategic assumptions.

| Original fixture ID in `positions.json` | Empty cells | Exact White value | Cover | Observation |
| --- | ---: | ---: | --- | --- |
| `6b1-cover-black-win` | 8 | -1 | CL | Replayed Phase 6B.1 cover; a non-loss witness coexists with an actual Black win. |
| `6b1-no-cover-black-win` | 6 | -1 | None | Individually coverable groups cannot be covered compatibly; Black still wins exactly. |
| `claim-draw` | 6 | 0 | CL | f5-f6 response with spare moves. |
| `claim-base-draw` | 6 | 0 | CL + BI | c5-c6 and b5-d5; wrong Black replies can lose. |
| `base-black-win` | 6 | -1 | BI | Odd landing cells c5/e5; symmetric replies and proactive occupation. |
| `vertical-black-win` | 6 | -1 | VE | c4-c5; White-lower response and proactive Black lower. |
| `claim-vertical-black-win` | 8 | -1 | CL + VE | Mixed parity, many spare moves and proactive VE occupation. |
| `zero-target-black-win` | 6 | -1 | Empty set | Every White group already blocked; this is nonterminal, not a terminal vacuous proof. |
| `immediate-white-win` | 6 | +1 | None | Column 6 wins now; alternatives in columns 2 and 5 lose. |
| `no-cover-forced-draw` | 6 | 0 | None | Only column 5 draws; alternatives 0/6 allow immediate Black wins. |
| `odd-threat-forced-loss` | 6 | -1 | None | Unsupported odd d5 threat; a unique immediate-survival reply still loses eventually. |
| `odd-threat-white-win` | 6 | +1 | None | Unsupported odd c3 threat and an immediate White win. |
| `source-6-1-extension` | 8 | 0 | CL | Constructed legal continuation of the diagram 6.1 reconstruction; not an Allis move list. |

The original diagram 6.1 reconstruction `[2,3,3,3,3,3,3,4]` and its reflection are
also checked: both have verified finite coverage, but 34 empty cells, so the exact
oracle and execution model return `unknown_remaining_cap`. Their game values
are not assigned by this experiment. The original diagram's matrix and printed
CL list remain covered by the preexisting source tests.

Four explicit resource probes preserve distinctions: positive coverage budget 1,
coverage budget 0, oracle budget 1, and execution budget 1. They respectively
produce coverage unknown, coverage unknown, exact-value unknown, and execution
unknown. Completed exact comparisons at those repeated roots are not interpreted
as filling in the missing strategic checks. Including these and the two early
source boards gives **545 analysis records / 531 unique boards**; 542 records have
exact values, one has an oracle-budget unknown, and two have cap unknowns.

### Reproducible evidence and a deliberately bad response

[`tests/victor_validation/positions.json`](../tests/victor_validation/positions.json)
contains all curated moves, top-first board strings, exact White/Black values,
all root move values, coverage/search/verification statuses, selected candidate
squares and covered groups, resource observations and reproducible survey specs.
Columns in move lists are **0-6**; White is 0/X and Black is 1/O. Schema is
`exploratory-6b2-v1`. The full generated-record digest is
`502cc9f9c96a4379493d6ca50a879fd2bf1e18e2c299a50afe4e4b9273fcdbbb`.
The digest is for canonical JSON observations, not a strategic signature.

The following commands recompute the observations, without overwriting baselines:

```sh
PYTHONPATH=tests .venv/bin/python -m victor_validation.report > /tmp/phase6b2-report.json
PYTHONPATH=tests .venv/bin/python -m victor_validation.report --include-generated > /tmp/phase6b2-all.json
```

The second includes all 513 generated histories, boards and observations, including
any potential counterexamples. The stored digest tests full-record reproducibility.
If a future run disagrees, retain its output and investigate the mathematics,
reachability, oracle, local predicates and response assumptions; do not regenerate
the baseline merely to pass. The report writer only emits stdout.

`claim-base-draw` has a verified CL c5-c6 / BI b5-d5 witness and exact draw value.
White d5 (column 3) triggers a mandatory Black b5 (column 1). If Black instead
plays c5 (column 2), White c6 (column 2) completes c6-d5-e4-f3. The legal suffix
`[3,2,2]` wins for White. The root is **not** a White forced win: Black's correct
BI reply prevents it. This preserves a concrete warning about arbitrary policy
execution rather than a strategic counterexample. It is a regression test.

## Reachability and execution boundary

A legal replay from empty is sufficient **existential evidence of reachability**;
it need not identify the actual historical game or prove that the path was optimal.
A future verifier must independently start empty, alternate White/Black, reject
out-of-range/noninteger/full columns and post-terminal moves, apply gravity, check
every prefix for both winners/draw, and compare the resulting complete matrix and
side exactly. The input is at most 42 columns. For the Black strategic context the
final snapshot must be nonterminal and White to move. Counts/gravity/last-win
checks alone do not replace replay. A signed history or external game ID is
provenance, not mathematical reachability evidence.

The execution model uses the separate matrix reference game's contiguous-run
winner and moves, not oracle minimax or production move evaluation. It explores
**every White move and every permitted Black spare response**, with the forced
pair response when triggered. It retires satisfied pairs and checks response
playability, unresolved-pair emptiness, conflicting replies, no available spare
move, and White winning before a reply. Memoization includes board, turn and any
pending response square. It counts unique states and fails unknown on cutoff.
There is no randomly chosen or invented deterministic spare-response policy.

All permitted spare moves is a stronger bounded check than existence of a good
spare move. The particular set-valued completion is our reconstruction from
§§7.1-7.2, not a policy explicitly supplied by the thesis. Faithfulness of its
general invariant requires review. The tests establish finite executions only.
The current coverage verifier neither checks a replay nor encodes that invariant;
those are the principal gaps between a coverage witness and an executable
strategic certificate. Composite rules cannot inherit this execution model.

## Proposed formal certificate specification (design only)

The eventual boundary must create a **separate strategic artifact**. It must never
cast a verified coverage witness, a draft with no diagnostic errors, or an oracle
observation into accepted strategic evidence.

| Field/obligation | Required verifier behavior |
| --- | --- |
| Schema and position binding | Canonical full 42-cell matrix, dimensions, strict cell/coordinate types, side to move, White-first convention and ruleset ID. Exact equality, not a process hash or revision alone. |
| Replay | Complete bounded column sequence from empty; independently validate each prefix and first terminal event; bind final board/turn exactly. Reject missing or mismatched reachability evidence. |
| Perspective | Defender Black/1/O, adversary White/0/X; strict integer IDs. Unsupported White certificates are unknown, never color-swapped acceptance. |
| Permitted context | Standard board, nonterminal White to move, whole-board Black evaluation, no caller exclusions/reserved columns, ordinary gravity/no passes. Derive context independently. |
| Rule prerequisites | Recompute current emptiness, adjacency, parity, BI landing support and role-ordered shapes directly from cells. Initially whitelist CL/BI/VE only. Source/model IDs must match reviewed semantics. |
| Complete targets | Independently generate all 69 geometric winning lines and select exactly those without a Black stone. Validate every assignment, coverage predicate and claimed complete relation. Reject omissions, duplicates, extra/blocker lines and unselected assignments. |
| Compatibility | Independently derive all six required entries and verify full affected-cell disjointness for every pair, including CL lowers. Unsupported matrix entries cannot default to compatible. |
| Strategic applicability | Bind a reviewed fragment theorem/invariant version. Verify its complete hypotheses: reachability, context, geometry, compatibility, total coverage, response legality/uniqueness and parity-preserving spare availability. A bare `zugzwang_controlled=true` is insufficient. Any execution evidence must quantify every adversarial White continuation, rather than one favorable trace; bounded cutoff remains unknown. |
| Asserted bound | Only `Black value >= 0` / White cannot force a win. Do not label it exact draw, Black win, forced optimal move or full Connect 4 solution. Exact values require additional independently checked arguments/search. |
| Independent verification | Separate trust boundary recomputes replay, geometry, prerequisites, targets, coverage, compatibility and strategic hypotheses; never trusts producer metadata, caches, preselected regions or proof-success flags. Test isolation and mutation checks remain necessary; a second implementation is still not a proof assistant. |
| Rejection/unknown | Invalid identity, replay, prerequisites, assignments, compatibility or provenance: reject with reason. Unsupported context/rule/theorem or resource cutoff: unknown with reason. No cover: only fragment failure, no game-value conclusion. Contradictory exact research evidence: preserve and investigate; do not accept. |
| Versioning/provenance | Separate certificate-schema, rule-model, compatibility-model, strategy-theorem and independent-verifier versions; exact source identity/hash/sections, producer commit, verifier commit, bounds and replay digest. Unsupported semantic versions fail closed. Provenance cannot substitute for obligations. |

The existing successful coverage status continues to mean only finite implemented
predicate satisfaction. **No successful game-theoretic certificate status or bound
acceptance has been added.** Test-only `exact` values and
`bounded_execution_checked_outcome_uncertified` observations are neither public
certificate statuses nor accepted strategic bounds.

Before a subsequent acceptance phase, independently review the constructive
invariant, formalize its hypotheses at the verifier boundary, implement/check
reachability there, test strategic-certificate mutations and semantic version
rejection, and validate the verifier-to-theorem correspondence. Longer games and
the six composite rules/White evaluation contexts remain separate proof work.
GO for that narrow follow-up design/review work; **NO-GO for enabling accepted
strategic certificates from today's coverage API or promoting finite tests to a
general proof**. The fragment is incomplete for optimal Connect 4 play.

## Verification and changed files

Environment: existing `.venv`, Python 3.11.17, bounded CPU-only tests. Run:

```sh
.venv/bin/python -m pytest -q tests/test_victor_endgame_oracle.py tests/test_victor_strategy_validation.py tests/test_victor_rules.py tests/test_victor_compatibility.py tests/test_victor_coverage.py tests/test_victor_verification.py tests/test_connect4_grounding.py tests/test_connect4_engine.py tests/test_connect4_evidence.py tests/test_connect4_provenance.py tests/test_connect4_history_api.py tests/test_connect4_explanations.py
git diff --check
```

Final result: **494 passed**, including **57 new** oracle/harness tests and the
437 relevant existing Victor, grounding, engine and evidence/history/explanation
tests. `git diff --check` passed. Initial new-test verification found two test
expectation problems: wrong rejection-message specificity and comparing tuple
execution suffixes with serialized JSON lists. The expectations/serialization
comparison were corrected; no mathematical predicate or outcome fixture was
changed to suppress a contradiction.

Intended files:

- `docs/phase6b2-victor-strategy-validation.md`
- `tests/test_victor_endgame_oracle.py`
- `tests/test_victor_strategy_validation.py`
- `tests/victor_validation/__init__.py`
- `tests/victor_validation/exact_oracle.py`
- `tests/victor_validation/reference_game.py`
- `tests/victor_validation/strategic_harness.py`
- `tests/victor_validation/report.py`
- `tests/victor_validation/positions.json`

All tooling is test-only and outside the runtime package. The original thesis
remains ignored. No research data or frozen evaluation label was used as oracle
truth. Commit message: `test(victor): add strategic soundness validation`.
