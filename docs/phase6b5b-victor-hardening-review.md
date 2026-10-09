# Phase 6B.5B: Independent review of Victor boundary hardening

Date: 2026-10-08. Started with a clean working tree on
`phase6b5a-victor-boundary-hardening` at
`93d60560e9d52fbc85460702a104ba6b9d6484ee`. The review branch
`phase6b5b-victor-hardening-review` was created directly from that HEAD.

Scope: an adversarial software review of the Phase 6B.5A diff
(`d178974..93d6056`) against findings F1/F2/F3 of the
[historical audit](phase6b5-victor-assurance-audit.md), as described in the
[6B.5A report](phase6b5a-victor-boundary-hardening.md). It does not re-audit
the CL/BI/VE theorem, add rules or touch any public surface.

**Summary.** F1, F2 and F3 are closed. One residual F3-class defect (R1, low
severity) was found. It predates 6B.5A and is fixed narrowly in `strategy.py` with a
regression test. Valid-input behavior is unchanged across 3,594 differential
records against the pre-hardening baseline. **GO** for Phase 6C as scoped below;
**NO-GO** for public strategic certification.

## 1. Independent verdicts

### F1: bounded ingress, CLOSED

Traced `verify_certificate`, `verify_certificate_mapping`,
`certificate_from_mapping` and `select_black_move` from entry to the first
work charge or hash.

- **Containers are size-checked before copying.** `_sequence` checks the exact
  container type (`type(x) is list or type(x) is tuple`, identity only, so no
  metaclass `__eq__`), takes one `len`, rejects if it is over the limit, then
  copies by index over `range(size)`. List growth after the length check cannot
  extend the copy. Shrinkage raises `IndexError`, which becomes
  `schema.input_changed`.
- **Mapping parsing is bounded before any lookup.** `_mapping_fields` requires an
  exact `dict` and `len <= len(allowed)` (12 top-level, 2 per rule). It iterates at
  most `size` items through the C dict iterator, which calls no key hooks. Each key
  must be an exact `str` whose length is at most the longest allowed name before the
  tuple membership test. Only those bounded exact strings are hashed by
  `dict(items)`. The old per-rule construction before the cardinality check is gone.
- **Nesting is fixed-depth.** Board -> row -> cell, rules -> record/dict -> pair ->
  scalar, replay -> scalar, assignments -> pair -> scalar. There is no
  recursion or `deepcopy`. A cycle in a scalar slot is rejected by type.
- **Scalars are bounded before hashing or formatting.** Probes timed at
  roughly 0.01–0.1 ms at budget 0, all rejected:
  - a 50 MB board cell, board digest or mapping key;
  - a 4-million-bit int in `player_to_move`, every replay ply and an assignment index;
  - 21 rule dicts with 100,000 keys each.
- **Resource exhaustion stays distinct.** Ingress precedes every
  `_Budget.charge`. Malformed input is REJECTED at budget 0. Well-formed input
  at budget 0 is UNKNOWN (`test_mapping_tuple_arrays_and_optional_omissions_still_work`).
  Charge formulas are unchanged.

Residual: allocation done by the caller (for example JSON decoding) happens before
this boundary, as 6B.5A states.

### F2: snapshot consistency, CLOSED

Every field was traced through the direct path (`_snapshot_certificate`) and the
mapping path (`_mapping_fields` -> `_snapshot_data`):

1. Each caller attribute or key is read exactly once into a private `dict`. The
   direct path reads all 12 slots with `getattr`. The mapping path copies at most
   12 items.
2. `_copy_containers` replaces every permitted container with a new tuple.
   `_snapshot_data` then type- and range-checks every leaf in those tuples.
   Accepted leaves are exact `str`/`int`/`None`, so they are immutable.
3. A new `StrategicCertificate` is built only from the private dict and tuples.
   `verify_certificate` rebinds `certificate` to it. The caller object is
   unreachable from that point on.
4. Versions, context, board digest, H1–H4, `_snapshot_rules`, the
   certificate digest (`_plain` of snapshot tuples), assignments, replay and
   `RecomputedEvidence` all read only the snapshot.

So the verifier validates exactly the data its returned digest identifies.
There is no remaining nested-mutable gap.

- **Strategy.** `strategy._snapshot` makes its own bounded tuple copy. Its scalar
  slots come from an exact frozen record, and `dataclasses.replace` runs no value
  hooks. It then calls the verifier on that copy. The verifier's second
  snapshot is equal element-for-element: tuples of immutable leaves that the
  verifier has type-checked. Selection uses only the strategy copy plus
  `verification.evidence`. Rule order is preserved and duplicates are rejected,
  so `rule_coverage[j]` matches `cert.rules[j]`.
- **Real-thread check (new).** A second thread flips one replay ply or one
  assignment rule index between a valid and an invalid value, with a 1 µs switch
  interval. Over 4,000 verifications per field, about half saw each state, and every
  result equalled the static valid or invalid verdict in full, digest
  included. Strategy selections carried only the valid digest. Copying is
  **not atomic**: a snapshot may combine values observed at different moments.
  The guarantee is that whatever was copied is exactly what was checked and
  digested.
- **Same-size dict replacement.** If a key is deleted and re-inserted
  concurrently, `_mapping_fields` may copy a mixture or, after a resize, miss an
  optional key. 6B.5A documents this. The result is still a fully checked private
  snapshot whose digest describes it. No incorrect acceptance or misidentification
  follows.

### F3: safe invalid-input handling, CLOSED for data; one record-level residual (R1)

- **No callbacks on caller leaves or containers.** Every check uses `type(x) is`
  identity. `len`, indexing and dict iteration run only on exact built-ins.
  Equality, hashing and containment run only after a value is confirmed to be an
  exact `str`/`int`.
- **`_plain` no longer calls `repr`.** It is reached only after full validation.
- **Diagnostics are bounded.** All `!r`/f-string interpolation of claim data is on
  values already checked as exact, bounded strings/ints: version ≤128, kind ≤32,
  group ≤11, continuation ≤42 ints. Lone surrogates in those strings neither
  escape nor break the digest, because `json.dumps` uses `ensure_ascii`.
- **Exception text is local.** `verify_certificate_mapping` only stringifies
  `ValueError`s built locally from `_Stop`. The verifier's broad `except` remains
  a fail-closed fallback (`schema.malformed`), reachable only through records
  built without `__init__`.
- **Malicious objects vs ordinary data.** Ordinary JSON-decoded data (dict, list,
  str, int, bool, None, float) is handled entirely by exact-type rejection; bool
  and float are rejected wherever an int is required. Malicious Python objects
  (subclasses, metaclass hooks, exploding dunders, incomplete records) are
  rejected by the same identity checks. The historical audit's original
  exploding-`repr` probes now pass at all three entrypoints.

**R1, found and fixed:** `select_black_move` read `request.continuation`
outside any handler. An exact `StrategyRequest` built with
`object.__new__` and missing its `continuation` slot raised an uncaught
`AttributeError`. 6B.5A tested incomplete certificate and rule records, but
not incomplete request records. Details are in section 2.

## 2. Newly discovered defects (by severity)

| ID | Severity | Status | Description |
| --- | --- | --- | --- |
| R1 | Low | Fixed here | `select_black_move(object.__new__(StrategyRequest))` with `certificate` set but `continuation` unset raised `AttributeError: 'StrategyRequest' object has no attribute 'continuation'`. It requires bypassing `__init__`, and no hostile callback runs. It also reproduces on baseline `d178974`, so it predates 6B.5A and was not introduced by it. |

**Fix (`strategy.py`).** The function now reads `request.certificate` and
`request.continuation` once, together, immediately after the exact-type check.
A missing slot returns `UNSUPPORTED` with code `request.type`, matching the
existing wrong-request-type result. Both values are then used only from those
locals. Side effect: a request missing only its `certificate` slot used to return
`INVALID_CERTIFICATE`. It now returns `UNSUPPORTED`/`request.type`, consistent
with other malformed request records. Requests built with the constructor
are unaffected.

**Regression.** `tests/test_victor_hardening_review.py::test_incomplete_strategy_request_is_a_structured_result`
covers three cases: missing certificate, missing continuation and both missing.
All three fail without the fix and pass with it.

**Informational observations (no change):**

- **Dead defensive branches.** `_snapshot_rules`, `_assignments` and
  `_replay` keep type/length branches that ingress now makes unreachable.
  They are harmless, fail closed and keep the stage functions self-contained.
- **Producer aid.** `certificate_to_mapping` passes malformed leaves through as
  inert data. This is documented. A consumer that JSON-encodes before verifying
  will hit `TypeError` in its own encoder, outside acceptance authority.

## 3. Test results

Run with `EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest ...`
(Python 3.11.17).

| Run | Passed | Failed | Skipped | Xfailed |
| --- | ---: | ---: | ---: | ---: |
| `test_victor_assurance_audit.py --runxfail` (no markers remain; proves no suppression) | 18 | 0 | 0 | 0 |
| Original seven F1/F2/F3 probes, selected by name | 7 | 0 | 0 | 0 |
| Focused: certificates, executable strategy, assurance, boundary hardening, this review | 351 | 0 | 0 | 0 |
| All Victor tests | 631 | 0 | 0 | 0 |
| Complete backend | 1132 | 0 | 14 | 0 |

- **No suppression.** The 14 skips are the existing optional PyTorch skips.
  `grep` finds no `xfail`/`skip` markers in `tests/test_victor_*.py`.
- **Original seven unchanged.** The 6B.5A diff of
  `test_victor_assurance_audit.py` only removes the seven `xfail` decorators
  and updates the docstring. All assertions are unchanged.
- **`test_victor_certificates.py` change.** The single edit relaxes the
  hypothesis states expected for ingress-level replay errors
  (`replay.column`/`too_long`/`type`) from `verified` to `not_evaluated`.
  Status, codes, replay status and `history_backed` assertions are retained. This
  follows directly from ingress preceding mathematics, and does not weaken a safety
  guarantee.
- **New tests** in `test_victor_hardening_review.py`:
  - 3 cases for R1;
  - 4 cases for surrogate/non-ASCII strings in the bounded fields;
  - 2 threaded-mutation invariant tests (600 iterations each). Interleaving coverage
    is probabilistic, but the asserted invariant holds for every observed result.
- `git diff --check` passes.

## 4. Valid-input compatibility

An independent differential loaded the `d178974` `certificate.py`/`strategy.py`
from git as a separate package and compared full result records with HEAD,
including the R1 fix. Records covered every field: status, findings, hypotheses,
replay status, evidence, digests and strategy state/trace.

- **Fixtures (9).** The 6 valid fixtures in `test_victor_certificates.py`, plus
  the assurance early, mixed and odd certificates.
- **Variants.** Each fixture was tested with history, without history and with
  assignments. Each variant was tested in tuple form and in fully mutable list form.
- **Budgets.** 0, 69, 100 and the default.
- **Valid verifier records: 216, all identical.** This includes every
  certificate digest and every UNKNOWN cutoff.
- **Strategy records: 1,722, all identical** (726 of them move selections). These
  cover all 7 White openings and all 7 White replies after each selected Black
  reply.
- **Well-formed invalid verifier records: 1,656.** The invalid variants were:
  - wrong digest, unsupported version, unsupported kind, unknown kind, bad square
    and duplicate rule;
  - defender 0 and Black to move;
  - replay with a wrong last move, and odd-length replay;
  - wrong, extra, missing, non-canonical and unused-index assignments;
  - every single-rule drop and every square swap.

  All 1,656 have identical status, codes and hypotheses. The only difference is the
  intended F3 change in 54 `rules.square` details, which no longer echo the claimed
  squares.

H1–H4 predicates, `GROUPS` (69), the coverage rule, compatibility checks,
work-charge formulas and the digest payload layout are textually unchanged in
the diff, apart from type-check idiom and diagnostic text. Documented
stage-order changes affect only malformed inputs, which are now REJECTED at ingress
before versions/H1:
- a version string over 128 characters is now REJECTED instead of UNSUPPORTED;
- ingress-level replay or assignment structure errors now report
  `not_evaluated` hypotheses.

## 5. Architecture checks

- **Isolation.** `certificate.py` imports only `hashlib`, `json`, `dataclasses`,
  `enum`, `types` and `typing`. The existing isolated-import and
  production-sabotage tests pass.
- **Public surface.** No public module (`battle.py`, `agent_factory.py`,
  grounding, API) imports `certificate` or `strategy`. 6B.5A changed only the
  Victor research package, its tests and docs.
- **Certification gate.** Both result types still pin
  `game_theoretic_certification = 'not_certified_pending_independent_review'`
  (`init=False`). This review does not change it.

## 6. Remaining limitations

- Snapshotting a concurrently edited object graph is not atomic. Same-size dict
  replacement and same-length element replacement are not detected. Any snapshot
  is still fully checked and correctly identified.
- `object.__setattr__` on frozen records by code running concurrently in the same
  process, and module or runtime tampering, are outside the data boundary.
- The boundary starts at Python objects. Byte-level JSON parsing (size, depth,
  duplicate keys) needs its own policy.
- SHA-256 provides identity, not authentication. A replay proves only that some
  legal history exists.
- The theorem remains a reviewed paper proof. Finite oracles and differentials
  are regression evidence, not universal proof.
- Threaded tests sample interleavings and do not enumerate them. The structural
  argument in section 1 is the primary evidence for F2.

## 7. Decisions

**GO for beginning Phase 6C implementation**, scoped as in 6B.5A: rule
specification and proof research, one additional rule at a time, under a
new theorem identifier, with its own prerequisites, compatibility, coverage
and executable-composition obligations. The independent boundary review that
6B.5A required before integration is now complete. F1/F2/F3 do not block it, and
R1 is fixed.

**NO-GO for public strategic certification.** The gate stays disabled. The
theorem still lacks an independent second review or formalisation, and this
review establishes software boundary correctness only.
