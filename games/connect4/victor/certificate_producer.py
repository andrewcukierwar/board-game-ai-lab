"""Producer-side formatting of a coverage witness as an UNTRUSTED certificate.

This module sits on the producer side of the trust boundary. It may use the
production Victor types, but its output is only a claim. ``certificate.py``
never imports it, and a witness's verification status, coverage claims and
``outcome_certification`` are not carried across. Only the board, the rules
and (optionally) the assignments are, and the verifier recomputes everything.
"""
from .certificate import (
    RuleInstance, StrategicCertificate, canonical_group_name, draft_certificate,
)
from .coverage import CoverageWitness
from .rules import RuleCandidate


def rule_instance(candidate: RuleCandidate) -> RuleInstance:
    """CL/VE keep (lower, upper); BI is re-ordered by ascending column."""
    squares = candidate.squares
    if candidate.rule.value == 'baseinverse':
        squares = tuple(sorted(squares, key=lambda s: s.column))
    return RuleInstance(candidate.rule.value, tuple(s.name for s in squares))


def certificate_from_witness(witness: CoverageWitness, *, replay=None,
                             include_assignments: bool = True) -> StrategicCertificate:
    """Format a search witness; the result proves nothing until independently verified."""
    candidates = tuple(e.candidate for e in witness.evidence)
    index = {c: i for i, c in enumerate(candidates)}
    assignments = None
    if include_assignments:
        assignments = tuple(
            (canonical_group_name(s.name for s in a.group.squares), index[a.candidate])
            for a in witness.assignments)
    return draft_certificate(
        [list(row) for row in witness.context.position.board],
        tuple(rule_instance(c) for c in candidates), replay=replay, assignments=assignments)
