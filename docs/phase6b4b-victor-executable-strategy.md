# Phase 6B.4B: Executable certificate-backed Black strategy

This phase adds `games/connect4/victor/strategy.py`, a deterministic,
research-only implementation of the constructive strategy for
`cl-bi-ve-black-nonloss-v1`. It chooses a mandatory rule response first;
otherwise it takes the lowest physical column with an even-row landing square.
It never evaluates moves or searches for an optimal move. No public agent,
factory, API, UI, explanation contract or strategic acceptance gate changes.

Baseline: `78d125b2b4a304838c7aa21b135bcd9ac8879499`, confirmed with a clean
tree on `phase6b4a-victor-certificates`. Branch `phase6b4b-victor-strategy`
was created directly from that commit, without involving `main`.

## 1. Focused adversarial boundary review

Before implementation, I reviewed the 6B.3 audit, the 6B.4A certificate report,
`certificate.py`, the producer adapter, coverage search, the standalone audit
model and the spare-policy explorer. The rule semantics and proof used here
are those recorded in the existing Allis §§6.1–6.3 and 7.1–7.4 reviews; this
phase does not claim a new independent thesis or proof review.

No confirmed defect requiring a change to H1–H4 or `certificate.py` was found.
The following consumer hazards were addressed before relying on its verdict:

| Hazard | Resolution |
| --- | --- |
| A saved verification object could outlive changed certificate contents. | The selector takes the original untrusted certificate, never a verdict, and invokes `verify_certificate` on each call. |
| A frozen dataclass can contain mutable lists; verification followed by use of caller containers would create a check/use gap. | Bounded structural copying freezes the board, rules, square pairs, assignments and optional initial replay **before** verification. Execution uses only that verified private copy. A test mutates the caller's board and squares immediately after verification and confirms that selection still uses the original snapshot; a subsequent call rejects the changed input. |
| A legal current board can conceal a missed reply, a forbidden CL lower, or a different earlier spare choice. | Replay every continuation ply, enforcing the deterministic policy at every previous Black turn. Do not infer compliance from the final board. |
| Fresh enumeration on the current board would lose original rule obligations and retirements. | Keep the original verified rule instances and reconstruct their state from the complete continuation. The certificate digest includes the rules and their ordering. |
| Even-row spares alone would ignore odd-row mandatory VE/BI replies. | Compute the unique mandatory reply first, regardless of its row, before considering spares. |
| A non-loss construction could be presented as a solved value or best move. | The API has no value, bound, outcome, best-move or solved field; public certification remains disabled. |

The snapshot is data-input protection, not isolation against arbitrary Python
code modifying module globals or using `object.__setattr__` concurrently.
Copying concurrently edited containers is not atomic. Phase 6B.5A uses fixed
indexed copy lengths, so list growth cannot extend a copy beyond its checked
limit; shrinkage rejects. The verifier also takes its own complete snapshot and
rejects unsupported scalar leaves before hashing or diagnostics.
Digests identify content; they are not signatures or transferable authorities.
Neither execution nor its tests are imported by the independent verifier.

## 2. Mathematical scope and mandatory responses

The initial certificate must satisfy exactly the existing finite hypotheses:

- **H1:** standard 7×6 gravity-valid nonterminal board, equal X/O counts and
  White to move.
- **H2:** original CL/VE pairs are empty, vertically adjacent with the specified
  upper parity; original BI endpoints are distinct current landing squares.
- **H3:** complete two-square sets are pairwise disjoint.
- **H4:** every initial group without a Black stone is covered by a CL upper,
  or by both endpoints of a BI/VE.

White is X/player 0; Black is O/player 1. Physical columns are 0–6, left to
right. Thesis square names are a1–g6, with row 1 at the bottom. A CL/VE's
ordered pair is `(lower, upper)`; a BI pair uses ascending column order.

After each White move, while the game is nonterminal:

| White touches an active rule | Required Black response |
| --- | --- |
| CL lower | That CL's upper |
| Either BI endpoint | The other endpoint |
| VE lower | That VE's upper |

H3 ensures at most one obligation. Each reply must be the actual landing square
of its physical column. An active CL/VE upper cannot legally be reached by
White while the invariant holds. A response conflict, unavailable mandatory
reply or unexpectedly reached upper fails explicitly. Black cannot choose a
spare instead of the mandatory response, even if another move wins immediately.
A pinned MIXED3 test demonstrates exactly this: after White d5, b6 wins at once,
but this strategy still requires d6.

Otherwise Black takes an even-row landing square, choosing the lowest physical
column index. Black never takes an active CL lower. The row-parity lemma
guarantees such a spare on every valid nonterminal Black turn; failure to find
one is an invariant failure, not an invitation to guess another move.

The theorem's formulation remains `sigma-r-permissive-spare-v1`, which allows
any landing square except an active CL lower. This implementation is its
refinement `sigma-r-even-row-lowest-column-v1`. Earlier Black spares must match
this refinement's exact tie-break; a different sound permissive spare is
rejected as a different strategy. Forced responses reflect with the board;
lowest-physical-column spare choices intentionally do not.

## 3. Active and retired rules

All original rules begin active. A CL retires when Black holds its upper.
A BI or VE retires when Black holds either endpoint. Proactive BI/VE occupation
by an even-row spare is permitted and retires that instance. A VE lower is
even; a VE upper is odd, so proactive lower occupation and forced upper replies
are both exercised.

State is recomputed from board contents, never from supplied flags. At every
White turn and after every Black response (including a terminal Black move),
the selector checks:

1. Every active instance has both squares empty.
2. Black holds no original CL lower, including after retirement.
3. Every retired instance still blocks **all** initial target groups that it
   covered, which includes every group validly assigned to it by the producer.

The replay also checks every historical mandatory response and spare selection.
At the final Black-turn snapshot, the rule just triggered by White is still
active and can have its triggering square occupied by White: the empty-pair
invariant applies at **White turns**, after Black's response. The selected
move is an obligation to execute next, not a move already in the returned
board or trace.

## 4. Stateless API and integrity

The implementation imports only the standard library and `certificate.py`.
It does not call enumeration, coverage search, the producer, the engine,
grounding, either test oracle, heuristics or neural models.

```python
from games.connect4.victor.strategy import (
    StrategyRequest, StrategyStatus, select_black_move,
)

# certificate is the original untrusted StrategicCertificate.
# continuation contains ALL columns played since its certified position.
result = select_black_move(StrategyRequest(certificate, (3,)))
if result.status is StrategyStatus.MOVE_SELECTED:
    column = result.column  # physical column 0..6
```

`StrategyRequest`, `StrategyResult`, `ContinuationState` and `ExecutedMove`
are frozen, slotted dataclasses. The continuation must be a tuple of exact
integers (booleans are rejected). The selector supports the certificate
verifier's list/tuple containers through bounded freezing; its output board,
rules and trace are deeply immutable tuples and frozen rule records. For JSON,
call the existing strict `certificate_from_mapping` parser first. A mapping,
saved certificate verdict, saved strategy result or forged subtype cannot
serve as a certificate/request authority.

Every call:

1. Takes a bounded private certificate snapshot, then runs `verify_certificate`.
2. Requires `HYPOTHESES_VERIFIED` and pins literal schema, theorem, ruleset,
   rule-model and compatibility-model identifiers to their exact 6B.4A
   versions. Also pins the supported verifier and strategy-formulation IDs.
3. Replays the continuation from the certificate's exact initial cells, starting
   with White. Turns alternate, gravity determines the played square, full
   columns fail and any move after the first terminal position fails.
4. Checks each previous Black move against the response dictated at that time,
   then checks the White-turn invariant and reconstructed retirements.
5. Requires a nonterminal Black-turn endpoint before returning a move.

There are at most 42 continuation moves and at most 21 rule instances.
`certificate_work_budget` defaults to the verifier's 100,000 units;
`replay_budget` defaults to 42 plies. Lower budgets can produce UNKNOWN without
a move. Invalid budget configuration raises `ValueError`, like the certificate
verifier; malformed game inputs produce explicit result statuses.

A selected result carries the legal column and square, forced/spare response
kind, triggering original rule (if any), active/retired instances, verified
current board and its digest, exact continuation and execution trace, initial
board digest, certificate digest, supported identities and strategy ID.
Each trace ply states player/column/square; Black plies also identify the
response kind, trigger and newly retired instances. The certificate digest
changes if rules, initial provenance or other certificate content changes.

`starting_history_backed` comes only from the re-run verifier's optional
initial replay check. A synthetic initial board can satisfy H1–H4 and support
this strategy without a historical replay. Legal continuation from such a
board does **not** make the starting board history-backed. The pinned
unreachable board (all column tops White, with White to move) remains accepted
as a mathematical position, with this flag false. A supplied incorrect initial
replay rejects the entire certificate. Real-session integration would need to
retain its original certificate and exact history; no such integration is
provided here.

## 5. Statuses and failure handling

| Status | Meaning |
| --- | --- |
| `legal_move_selected` | Certificate and complete continuation verified; legal next Black action selected. |
| `invalid_certificate` | Malformed input, failed binding/hypothesis, or invalid supplied initial replay. |
| `unsupported_version_or_context` | Unsupported certificate/request identity or a nonterminal continuation ending with White to move. |
| `illegal_continuation` | Invalid columns/types/length, full column, or play after terminal. |
| `previously_violated_strategy` | A prior legal Black move missed a mandatory reply or the deterministic spare policy. |
| `terminal_position` | Complete compliant replay ends at the first terminal Black win or full board; no next move. |
| `unknown_resource_cutoff` | Verification/replay budget exhausted; no next move. |
| `unexpected_invariant_failure` | Internal mechanical obligation/invariant failed despite a verified prefix; no fallback. |

Failures have stable finding codes and details. Historical policy failures
identify the earliest violating ply, played square, required square and rule
where applicable. Terminal and White-turn endpoints retain their fully checked
state; invalid or incomplete replay supplies no purported verified final state.
If a White terminal were reached following a checked prefix, it would be
reported as an invariant failure with the continuation, not as a successful
selection. Tests preserve the full initial board/rules/provenance and move path
in contradiction assertions. No contradictory execution was observed.

## 6. Multi-turn example

The 6B.3 MIXED certificate binds history
`33003316106600034431441142662221` and original rules BI e6–g6, CL c5–c6,
CL f1–f2, VE f4–f5. At each Black selection, submit the same certificate and
the complete continuation through the latest White move:

| Continuation columns | White's latest square | Selected Black square | Action/transition when executed |
| --- | --- | --- | --- |
| `(2,)` | c5 | c6 (column 2) | Forced CL reply; retire c5–c6. |
| `(2,2,5)` | f1 | f2 (column 5) | Forced CL reply; retire f1–f2. |
| `(2,2,5,5,5)` | f3 | e6 (column 4) | Spare; lowest even-row landing; proactively retire BI e6–g6. |
| `(2,2,5,5,5,4,5)` | f4 | f5 (column 5) | Forced VE reply on an odd row; retire f4–f5. |

After the fourth response, all original instances are retired and the board
is nonterminal with White to move. Passing the resulting eight-ply continuation
to selection returns unsupported context with a verified state, because no
Black move is due yet. Taking f4 instead of e6 on the earlier spare turn would
obey the permissive theorem but fail this deterministic refinement.

VE_ONLY separately exercises two successive forced replies d2→d3 and d4→d5,
then the f6 spare after White d6, ending at the full board. MIXED3 exercises
proactive VE a4 and proactive BI b6 retirement; both corresponding Black moves
are observed terminal wins, so any later ply is rejected.

## 7. Tests and independent comparisons

`tests/test_victor_executable_strategy.py` covers CL, both BI directions, VE,
mandatory priority over an immediate win, even-row spares and deterministic
ties, simultaneous active instances, mixed rules, repeated activations,
proactive retirement, partial columns, terminal/full-board stopping, immutable
snapshots, versions/context/digest/rule/replay/assignment tampering, saved-verdict
rejection, malformed certificates and continuations, missed mandatory replies,
wrong earlier spare choices, resources, reflection and provenance distinctions.

The complete-execution battery checks **22 certificate/board cases**: seven
pinned cases (including a synthetic unreachable board and an empty rule set)
plus 15 sampled covers, seed 6442, with 4/6/8 empties. All initial cases have
at most 10 empties. It explores **every legal White continuation** with the
deterministic Black strategy, without memo pruning of paths, under a 10,000
selection cap per case. Results: **299 selections, 34 terminal Black-win
paths, 96 full-board paths, no White-win path and no invariant failure**.

The tests do not use the production strategy as their oracle:

- Mandatory/spare actions are recomputed from the unchanged 6B.3 audit
  bitboards and raw original rule tuples, using the pre-White board.
- Legal columns and reconstructed boards are checked against the Phase 6B.2
  seven-bit-column oracle. Every initial board and each distinct encountered
  Black-turn board is solved exactly within its endgame cap. Both exact
  implementations agree on values; every selected move has a nonnegative
  Black oracle value. These values exist only in tests, not the research API.
- The untouched 6B.3 audit oracle also checks that White cannot force a win.
  Retirements and blocked groups are checked independently on its bitboards.
- The existing all-spares strategy checker and both permissive/even-row policy
  explorers hold on every battery case, with no invariant breaks.

The real unsound relaxations remain pinned: H3 overlap, H2 Claimodd parity and
H2 floating BI all have exact White wins under the relaxed predicates, and
selection rejects their certificates. The CL-lower failure binds history
`550066550544410241221101261344663650`: continuation `(3,3,3,2,2)` is
d3,d4,d5,c5,c6, where the forbidden Black spare c5 enables White's c6 win.
The unrestricted explorer reproduces this full path; the executable selector
rejects the earliest bad Black move at ply 3. These are existing negative
examples, not contradictions to the verified theorem or selector.

## 8. Verification record

Commands (project `.venv`, no new dependencies):

```sh
.venv/bin/python -m pytest -q tests/test_victor_executable_strategy.py tests/test_victor_certificates.py
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest -q \
  tests/test_victor_executable_strategy.py tests/test_victor_certificates.py \
  tests/test_victor_independent_audit.py tests/test_victor_endgame_oracle.py \
  tests/test_victor_strategy_validation.py tests/test_victor_rules.py \
  tests/test_victor_compatibility.py tests/test_victor_coverage.py tests/test_victor_verification.py \
  tests/test_connect4_grounding.py tests/test_connect4_engine.py tests/test_connect4_evidence.py \
  tests/test_connect4_provenance.py tests/test_connect4_history_api.py tests/test_connect4_explanations.py
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests -q
git diff --check
```

Results:

| Run | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Strategy alone (included in the runs below) | 61 | 0 | 0 |
| Focused strategy + certificate tests | 206 | 0 | 0 |
| All Victor research + listed grounding/engine regressions | 709 | 0 | 0 |
| Complete backend suite | 987 | 0 | 14 |

The full backend run took about 24 seconds. The 14 skips are existing optional
PyTorch-dependent tests; they did not execute and are not counted as passes.
`git diff --check` is clean. No training, full-game exhaustive search,
tournament, heavy dependency install or large compute run was performed.

## 9. What this establishes and what remains

| Concept | Scope in this phase |
| --- | --- |
| Finite certificate-hypothesis verification | Recomputed H1–H4 and optional provenance, by the unchanged independent verifier. |
| Constructive Black non-loss strategy | Deterministic executable refinement of the paper theorem; research-only assurance. |
| Strategy execution trace | Checked finite replay of mechanical obligations, bound to original certificate and continuation. |
| Exact game value | Independent bounded test evidence only; never a selector output or claim. |
| Optimal move selection | Neither implemented nor implied. Mandatory moves can pass up immediate wins. |

Public certification remains
`not_certified_pending_independent_review`. This phase adds execution evidence;
it does not discharge the universal theorem's remaining assurance obligations:

1. Independent second proof review or formalisation of L1, L2, L4′ and termination.
2. Independent code review of `certificate.py` **and** the new selector,
   including snapshot binding, turn boundaries and terminal handling.
3. Explicit review record and policy ownership before any new verifier version
   could enable authoritative certification or public integration.
4. Separate theorem identifiers, spare-move proofs and joint-executability
   arguments before extending beyond CL/BI/VE. This implementation must not
   acquire other rules under the existing theorem identity.

**GO for independent assurance review. NO-GO for beginning a major expansion
of the Victor rule/strategy scope or enabling public strategic certification
until that proof and code review is complete.** The current fragment is ready
for that review; finite test agreement is not a substitute for it.

Changed files are only the new selector, its dedicated tests and this report.
Existing Victor exports, verifier, producer, coverage, test oracles and explorers
are untouched, as are AlphaZero/neural training, frozen benchmark artifacts,
public `VictorAgent`, agent factory/API, React UI, LLM explanations and deployment.
