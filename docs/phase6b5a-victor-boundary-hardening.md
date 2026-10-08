# Phase 6B.5A: Victor certificate boundary hardening

Date: 2026-10-08. Started with a clean working tree on
`phase6b5-victor-assurance-audit` at
`d178974ce63ee7384031b0b85c57cd3796d83a6c`. The implementation branch
`phase6b5a-victor-boundary-hardening` was created directly from that HEAD.

The three confirmed software findings F1/F2/F3 in the
[historical independent audit](phase6b5-victor-assurance-audit.md) are closed by
bounded ingress processing and complete private snapshots. All seven strict
xfail reproducers now pass as ordinary tests; their assertions are unchanged.
The audit report remains unchanged and still describes the defective baseline.

The CL/BI/VE theorem, H1-H4, original Allis rule meanings, compatibility,
covering sets, version identifiers, deterministic even-row Black strategy,
forced-response priority and provisional/public certification gates are unchanged.
There are no public application, engine, grounding, AlphaZero, producer,
independent model or exact-oracle changes. No training, large campaign or
full-game exhaustive search was run.

## 1. Findings, fixes and trust boundaries

| Finding | Root cause | Correction |
| --- | --- | --- |
| F1: work before resource checks | Mapping conversion built every supplied rule and converted arrays before cardinality checks. Digest serialization recursively visited optional data before checking its shape and bounds. Abstract work charges did not bound these operations. | Exact built-in container and size checks precede each copy. Fixed-count indexed copies cannot grow with a caller's list. Mapping keys are bounded and checked before lookup/hash. Every leaf is type/size/range checked before constructing certificate/rule records or hashing. |
| F2: digest/check inconsistency | Only board and parsed rules were privately frozen. Replay and assignments were separately read for digest construction and verification. Frozen dataclasses could still contain mutable lists. | Read all certificate fields into a private fixed-field record, copy every permitted container at fixed depth into tuples, validate every scalar, then construct a private frozen certificate. H1-H4, assignments, replay and digest construction use only this snapshot. |
| F3: hostile diagnostics | `_plain` used `repr` for unknown leaves; malformed rule/assignment/replay messages also formatted unchecked objects. RuntimeError from hostile `__repr__` escaped verification and strategy. Mapping-key diagnostics could hash or represent arbitrary keys. | Unsupported types reject before serialization, hashing or formatting. Diagnostics use stable codes and bounded locations, such as rule index or replay ply. `_plain` has no representation fallback. Mapping key validation uses exact dict iteration, not lookup against unchecked keys. Strategy snapshot errors use a constant safe description. |

`certificate.py` still imports only the standard library and independently
reconstructs legality, geometry, H1-H4 and replay. It imports no enumeration,
engine, grounding, old coverage verifier, producer, strategy or audit model.
Existing isolated-import and production-sabotage tests continue to pass.

`strategy.py` retains its separate bounded snapshot and unconditional certificate
re-verification on every selection. Its copies now also capture a single length
and use indexed copying. Execution uses the strategy's private copy after the
verifier has independently checked it. Saved verdicts and traces remain data,
never input authorities.

## 2. Supported schema and ingress bounds

The direct entrypoint requires an exact `StrategicCertificate`; every direct
rule requires an exact `RuleInstance`. Subclasses are rejected. Exact-type comparisons use identity, so even hostile
metaclass equality hooks cannot run during container checks. Arrays accept
ordinary exact Python `list` or `tuple`, including lists nested inside frozen
records. The mapping entrypoint requires exact `dict` objects, including each
rule entry, with exact string keys.

| Component | Structural and scalar limits |
| --- | --- |
| Certificate fields | Exactly the defined schema: `schema`, `theorem`, `ruleset`, `rule_model`, `compatibility_model`, `board`, `player_to_move`, `defender`, `board_digest`, `rules`; only `replay` and `assignments` may be omitted. No extra fields. At most 12 keys; keys must match the fixed names, with length checked before comparison/hash. |
| Version identifiers | Exact `str`, at most 128 characters. Supported values must still match the existing identifiers exactly; bounded other versions remain UNSUPPORTED. |
| Board | Exactly 6 rows, each exactly 7 cells. Cells are exact one-character `str` values ` `, `X`, `O`. |
| Turn and defender | Exact `int` in 0..1; bool/int subclasses reject. White-to-move and Black-defender theorem restrictions are unchanged. |
| Board digest | Exact `str`, at most 71 characters; must equal the recomputed `sha256:` identity. |
| Rules | At most 21. Mapping entries contain exactly `kind` and `squares`. Kind is exact `str`, at most 32 characters. The three supported and six explicitly unsupported Allis kinds retain their existing meaning/status. Unknown kinds reject. |
| Rule squares | Exactly 2 exact two-character strings; the existing parser still enforces a1..g6. |
| Optional replay | `None` or at most 42 exact integer columns 0..6; game legality and board binding are independently replayed later. |
| Optional assignments | `None` or at most 69 entries, each exactly 2 fields: exact group string of at most 11 characters and exact integer rule index 0..20. Existing H4 checks still require canonical groups, actual covering rule indices, exactly one assignment per target and no non-targets. |
| Nesting | Fixed schema only: board -> row -> scalar; rules -> rule record/dict -> square pair -> scalar; replay -> scalar; assignments -> pair -> scalar. There is no general recursive traversal or deepcopy. Cycles/nested containers in scalar slots reject. |

Scalar limits prevent a bounded outer array from carrying an unbounded string
or integer into hashing/diagnostics. They do not exclude any previously accepted
valid certificate. Negative/out-of-domain assignment indices reject before
hashing; bounded indices absent from the actual rule set still fail the existing
assignment check. Input allocations already performed by a caller are outside
this boundary; rejection/copying does not scale with oversized payload length.

`certificate_from_mapping` raises a `ValueError` containing a locally generated
finding code and safe detail. `verify_certificate_mapping` turns it into a
structured REJECTED result. It first validates budget configuration, preserving
the documented programming-error behavior even when the mapping is malformed.

`certificate_to_mapping` is a bounded producer shape adapter, not an acceptance
API or JSON encoder. It checks/copies containers and record types without invoking
leaf callbacks. Unsupported scalar leaves may remain inert data in its output
so the verifier can reject them, including the original F3 mapping regression.
Its output is JSON-ready only for well-formed scalar claims. Invalid shapes raise
safe `ValueError`; consumers must verify before encoding untrusted claims.
`draft_certificate` and other producer conveniences are not hostile-input
verification APIs and do not acquire verification authority from this change.

## 3. Snapshot and digest guarantee

The private certificate contains only frozen records, tuples, exact bounded
strings, exact bounded integers and optional `None`. All source fields and
container elements needed for verification are captured before any mathematical
predicate is evaluated. Validation examines these copied values; digest
serialization examines the same values; evidence and provenance are recomputed
from them. No caller-owned mutable container is read after snapshot creation.

This guarantees **consistency between the verified private snapshot and its
returned certificate digest**, including nested lists in otherwise frozen
records. A caller edit immediately after digest construction cannot repair an
invalid snapshot or corrupt a valid one. The board digest binding, H1-H4,
optional assignment validity and optional replay validity apply to that same
private snapshot. Rule ordering still contributes to identity.

Copying arbitrary concurrently edited input is **not atomic**. A snapshot may
combine values observed at different instants while copying; it is nevertheless
bounded, privately immutable and completely checked before use. Growth after a
length check cannot extend the copy; shrinkage causing a missing index rejects
with `schema.input_changed`. Detected dict resizing rejects with that code too.
Concurrent replacement with the same size need not be detected. No claim is made
that the snapshot corresponds to a single historical instant of the caller's
graph. Arbitrary module tampering or concurrent `object.__setattr__` attacks by
Python code inside the verifier are outside the data-input boundary.

Digest payload layout and encoding are unchanged for valid inputs. A before/after
comparison captured **396 full verdict/strategy records** on nine fixtures: six
existing valid endgames plus the assurance early/mixed/odd fixtures, each with
history, without history, with assignments and with deeply mutable list forms.
Four work budgets (0, 69, 100, default) and all seven one-ply White columns were
checked. Records match byte-for-byte, including digests, verdict identities,
coverage evidence, provenance, selected moves and strategy states. No accepted
valid result or digest changed in that comparison.

## 4. Resource semantics and malformed results

Ingress type/size checks are separate from the abstract verification work budget.
Malformed oversized/recursive/unsupported-type data rejects before work charges
or hashes, including at work budget 0. This is a known schema failure, not a
conversion of an UNKNOWN mathematical result into rejection.

For valid bounded input the existing charges are unchanged: 69 H1 units, rule
count for H2, squared rule count for H3, target-count times rule-count (minimum
one) for H4, assignment count, and 16 units per initial replay move. The default
100,000 budget comfortably covers the schema. Genuine exhaustion still returns
UNKNOWN, without a bound or strategy move. No cutoff is reported as acceptance,
rejection or completed search. Budgets are abstract operations, not elapsed time,
Python instruction counts or allocation quotas.

Structural validation now precedes mathematical stages. Thus malformed replay
shape/type/column errors reject with H1-H4 `not_evaluated` and no certificate
digest; the direct verifier marks replay REJECTED. A mapping parse rejection
retains its existing `not_evaluated` replay metadata. Well-formed replay errors
(full column, post-terminal move, board mismatch) still occur after H1-H4 and
report those hypotheses verified. The original replay test now explicitly checks
both stage outcomes; all its cases and rejection/provenance assertions remain.
Malformed rule/context combinations may likewise reject at ingress earlier than
at the old version/H1 stage. Mathematical predicates have not changed.

No arbitrary iteration, representation, formatting, equality or hashing hooks
are used to reject unsupported input types. Remaining formatting of claim values
is limited to previously checked bounded exact strings/integers; continuation
formatting is limited to a validated bounded tuple of exact columns. Exception
text used by the mapping wrapper originates from local safe parse errors; strategy
response errors originate from operations on verified data. The correction does
not rely on a broad `except Exception` catch.

## 5. Regression and independent evidence

Baseline reproduction: `test_victor_assurance_audit.py --runxfail -q` produced
**11 passed, 7 failed, 0 skipped, 0 xfailed**. Normal baseline execution had
11 passed and 7 strict xfails. All seven corresponding markers are now removed,
without weakening/deleting their assertions or adding skips/xfails.

The seven cases close both early-work paths, both replay/assignment digest
interleavings, and all three exploding-repr entrypoints (direct verifier,
mapping verifier, strategy). A separate focused selection now gives
**7 passed, 11 deselected, 0 failed, 0 skipped, 0 xfailed**.

`test_victor_boundary_hardening.py` adds **118 passing cases**, covering:

- Complete tuple snapshots and exact verdict/strategy equality after mutation of
  original board rows, rule squares/outer rule list, replay and nested assignments.
- Exact rejection/digest equality when initially invalid replay or assignments
  are repaired by caller edits immediately after digest construction.
- Oversized boards, rows, rules, square pairs, replay, assignments and assignment
  entries through all three entrypoints at budgets 0 and 69, before either hash.
- Recursive data, unexpected objects, exploding callbacks, bool/int and str
  subclasses, hostile list/tuple/dict subclasses and metaclass equality/hash
  hooks, oversized scalar strings/ints, and hostile mapping keys whose
  hash/equality/representation must not run.
- List growth/shrinkage between length capture and copying, and dict resizing
  during mapping iteration, through deterministic scheduling hooks that edit
  only caller-owned data.
- Incomplete exact dataclass records, tuple/list mapping support, optional-field
  omission, shape-adapter behavior and invalid-budget programming errors.

The existing bounded independent assurance model and exact game oracles are
unchanged. Their existing geometry/H2 comparison still covers 7,749 cases;
certificate differential checking still records 741 cases (219 accepted,
522 rejected). Existing complete deterministic executions still cover 22 cases,
299 selections, 34 Black-win paths and 96 full-board paths. The assurance early
prefix (258 selections) and complete mixed/odd strategy trees (102/79 selections,
35/45 terminal paths) retain their pinned outcomes, including 175 exact
nonnegative selected-move checks. The all-spares early frontier remains UNKNOWN;
the two bounded complete endgame explorations retain their 38/111 states and
88/481 transitions. These are finite regression results, not universal exhaustive
search or a new proof of the theorem.

All original real H2 parity/playability and H3 full-square-overlap counterexamples,
CL-lower spare counterexample, empty/full/nearly-full positions, unreachable
mathematical positions, legal-history provenance, trust isolation, mutations,
forced priorities and certificate-to-strategy re-verification remain covered.

## 6. Verification record

Commands use the existing virtualenv, explanations disabled and no API key:

```sh
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests/test_victor_certificates.py tests/test_victor_executable_strategy.py tests/test_victor_assurance_audit.py tests/test_victor_boundary_hardening.py -q
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests/test_victor_independent_audit.py tests/test_victor_endgame_oracle.py tests/test_victor_strategy_validation.py -q
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests/test_victor_*.py -q
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests -q
git diff --check
```

| Final run | Passed | Failed | Skipped | Xfailed | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Focused certificate, executable strategy, assurance and boundary tests | 342 | 0 | 0 | 0 | 3.06s |
| Independent audit, endgame oracle and strategy validation suites | 66 | 0 | 0 | 0 | 2.59s |
| All Victor tests | 622 | 0 | 0 | 0 | 6.02s |
| Complete backend | 1123 | 0 | 14 | 0 | 25.60s |

All seven former xfails and all 118 additional hardening cases pass normally.
The 14 existing optional PyTorch-dependent skips remain skips, not passes.
`git diff --check` passes. The 396-record baseline comparison also passes.
Only intended certificate, strategy, tests and documentation files are included
in the commit; the historical audit, producer and independent models remain
unchanged.

## 7. Remaining limitations, review and Phase 6C decision

Passing tests establish closure of these reproduced software boundary defects,
not absence of all software/concurrency defects. A separate independent review
must inspect the revised exact-type/container/key preflight, bounded indexed
copying, scalar limits, no-callback diagnostics and snapshot-only digest/check
paths. It should rerun the seven original probes, hostile-key/leaf and resizing
cases, isolated imports and valid-data differential comparisons. This
implementation session is not that independent reviewer; no new independent
human/model review is claimed.

SHA-256 remains identity, not authentication. History-backed provenance is an
existential legal replay, not a binding to an external session. The theorem is
still a reviewed paper proof rather than a proof-assistant result; finite game
oracles do not prove arbitrary new-rule compositions. A future byte-level JSON
parser needs its own byte/depth/duplicate-key policy. The ingress boundary starts
with Python objects, after caller allocation/decoding. Runtime/module integrity
remains trusted. Producer conveniences and external JSON encoding are outside
acceptance authority. Public claims remain provisional and gated.

**GO for beginning Phase 6C as separately scoped rule specification/proof
research, one additional rule at a time.** The confirmed F1/F2/F3 regressions no
longer block that research. **NO-GO for integrating new rules into this
certificate/strategy contract until independent boundary review and the new
rule's theorem, prerequisites, compatibility, coverage and executable-composition
obligations are satisfied.** New rules must not be appended under the existing
CL/BI/VE theorem identifier. **NO-GO for authoritative public strategic claims**;
this corrective phase does not authorize changing the certification gate.
