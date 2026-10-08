"""Future proof-system vocabulary; no game-theoretic certificate acceptance.

Every draft below is untrusted input. None confers a proof status or computes an
outcome. These internal interfaces may evolve with the six remaining rules.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal, Protocol

from .geometry import Group, Square
from .position import Player, Position
from .rules import RuleCandidate, ThesisReference


class CompatibilityConstraint(IntEnum):
    """Codes from Allis §7.4, p.50; compatibility.py implements only CL/BI/VE."""

    DISJOINT_SQUARES = 1
    NO_CLAIMEVEN_BELOW_INVERSE = 2
    COLUMNWISE_DISJOINT_OR_EQUAL = 3
    DISJOINT_SQUARES_AND_DISJOINT_OR_EQUAL_COLUMN_SETS = 4


@dataclass(frozen=True)
class CompatibilityObligation:
    first: RuleCandidate
    second: RuleCandidate
    required_constraints: tuple[CompatibilityConstraint, ...]
    # A future verifier derives these from the matrix, never trusts caller input.


@dataclass(frozen=True)
class EvaluationContextDraft:
    """Proposed Chapter 8 setup; reserved squares/groups require separate evidence."""

    position: Position
    defender: Player
    mode: Literal['black', 'white_odd_threat', 'white_threat_combination']
    reserved_columns: tuple[int, ...]
    threat_squares: tuple[Square, ...]
    references: tuple[ThesisReference, ...]


@dataclass(frozen=True)
class CoverageAssignment:
    group: Group
    candidate: RuleCandidate


@dataclass(frozen=True)
class CoveragePlan:
    """A proposed target/assignment list. Completeness is explicitly unchecked."""

    target_groups: tuple[Group, ...]
    assignments: tuple[CoverageAssignment, ...]


@dataclass(frozen=True)
class CertificateDraft:
    """Future search output, never an accepted proof. No trusted excluded-group list.

    A verifier must derive the evaluation region and target groups from the exact
    position/context, rebuild all rule semantics, and check every obligation.
    """

    context: EvaluationContextDraft
    candidates: tuple[RuleCandidate, ...]
    compatibility: tuple[CompatibilityObligation, ...]
    coverage: CoveragePlan
    schema_version: Literal['draft-6a'] = field(default='draft-6a', init=False)
    status: Literal['unverified'] = field(default='unverified', init=False)


class CertificateVerifier(Protocol):
    """Future boundary; Phase 6A supplies NO implementation or accepted-proof type."""

    def rejection_reasons(self, draft: CertificateDraft) -> tuple[str, ...]:
        """Design hook. Even an empty list is not a game-theoretic outcome."""
        ...
