"""Independent coverage-witness checking, with no position-bound acceptance.

Shares immutable vocabulary and basic board legality with the enumerator. It does
NOT call candidate enumeration, analyze_candidates, compatibility, the search,
its context factory, or its caches. Raw predicates below intentionally duplicate
this small reviewed mathematical fragment to expose implementation disagreement.
"""
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Literal

from .contracts import CoverageAssignment
from .coverage import BlackEvaluationContext, CoverageWitness
from .evidence import CandidateEvidence
from .geometry import Group, Square
from .position import Position
from .rules import RuleCandidate, RuleName


class VerificationStatus(str, Enum):
    VERIFIED_UNCERTIFIED = 'coverage_verified_outcome_uncertified'
    REJECTED = 'rejected'


@dataclass(frozen=True)
class WitnessVerification:
    status: VerificationStatus
    rejection_reasons: tuple[str, ...] = ()
    outcome_certification: Literal['uncertified'] = field(default='uncertified', init=False)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _square(square: Square) -> None:
    _require(type(square) is Square, 'invalid square type')
    Square(square.row_index, square.column)  # Validate even bypassed frozen constructors.


def _group(group: Group) -> None:
    _require(type(group) is Group and type(group.squares) is tuple, 'invalid group type')
    for square in group.squares:
        _square(square)
    _require(Group(group.squares) == group, 'noncanonical or invalid group')


def _groups(groups: tuple[Group, ...]) -> None:
    _require(type(groups) is tuple, 'group claims must be tuples')
    for group in groups:
        _group(group)


def _opponent_groups(board: tuple[tuple[str, ...], ...]) -> tuple[Group, ...]:
    # Independently construct the entire 69-window geometry, then filter Black.
    groups = []
    for r in range(6):
        for c in range(7):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if 0 <= r + 3 * dr < 6 and 0 <= c + 3 * dc < 7:
                    squares = tuple(Square(r + i * dr, c + i * dc) for i in range(4))
                    if all(board[s.row_index][s.column] != 'O' for s in squares):
                        groups.append(Group(squares))
    return tuple(sorted(groups))


def _candidate(candidate: RuleCandidate, board: tuple[tuple[str, ...], ...]) -> frozenset[Square]:
    _require(type(candidate) is RuleCandidate and type(candidate.squares) is tuple,
             'invalid candidate type')
    _require(type(candidate.rule) is RuleName and candidate.rule in (
        RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL), 'unsupported candidate rule')
    _require(len(candidate.squares) == 2, 'candidate must contain two squares')
    for square in candidate.squares:
        _square(square)
        _require(board[square.row_index][square.column] == ' ', 'candidate square is occupied')
    a, b = candidate.squares
    _require(a != b, 'candidate squares must be distinct')
    if candidate.rule == RuleName.BASEINVERSE:
        _require(a.column != b.column and a < b, 'invalid Baseinverse columns/order')
        for square in (a, b):
            _require(square.row_index == 5 or board[square.row_index + 1][square.column] != ' ',
                     'Baseinverse square is not directly playable')
        return frozenset((a, b))
    parity = 0 if candidate.rule == RuleName.CLAIMEVEN else 1
    _require(a.column == b.column and a.row_index == b.row_index + 1
             and (6 - b.row_index) % 2 == parity, 'invalid vertical roles/adjacency/parity')
    return frozenset((b,)) if candidate.rule == RuleName.CLAIMEVEN else frozenset((a, b))


def verify_coverage_witness(position: Position, witness: CoverageWitness) -> WitnessVerification:
    """Treat both inputs as untrusted; only certify finite coverage predicates.

    Rejection reports the first detected defect. Even success does not establish
    historical reachability, joint strategic soundness, safety, or a game value.
    """
    try:
        _require(type(position) is Position, 'invalid position type')
        checked = Position.from_board(position.board, position.player_to_move)
        _require(checked == position, 'noncanonical position snapshot')
        _require(checked.player_to_move == 0 and not checked.terminal,
                 'requires nonterminal White-to-move position')
        _require(type(witness) is CoverageWitness, 'invalid witness type')
        _require(witness.schema_version == 'coverage-6b1-v1'
                 and witness.outcome_certification == 'uncertified', 'invalid witness schema/status')
        context = witness.context
        _require(type(context) is BlackEvaluationContext, 'invalid context type')
        _require(type(context.position) is Position, 'invalid context position type')
        rebound = Position.from_board(context.position.board, context.position.player_to_move)
        _require(rebound == context.position and rebound == checked, 'witness position differs')
        _require(type(context.defender) is int and context.defender == 1
                 and type(context.opponent) is int and context.opponent == 0
                 and context.mode == 'black_opponent_to_move', 'unsupported context identity')
        targets = _opponent_groups(checked.board)
        _groups(context.target_groups)
        _require(context.target_groups == targets, 'target groups differ from recomputed universe')
        _require(type(witness.evidence) is tuple, 'invalid evidence container')
        coverage = {}
        for evidence in witness.evidence:
            _require(type(evidence) is CandidateEvidence, 'invalid evidence type')
            candidate = evidence.candidate
            required = _candidate(candidate, checked.board)
            _require(candidate not in coverage, 'duplicate selected candidate')
            covered = tuple(g for g in targets if required <= set(g.squares))
            _groups(evidence.conditional_solved_groups)
            _require(evidence.conditional_solved_groups == covered, 'conditional coverage claim differs')
            coverage[candidate] = covered
        for first, second in combinations(coverage, 2):
            # Exact six supported §7.4 entries all require Constraint 1.
            _require(set(first.squares).isdisjoint(second.squares), 'selected candidates conflict')
        _require(type(witness.assignments) is tuple, 'invalid assignment container')
        assigned = set()
        for assignment in witness.assignments:
            _require(type(assignment) is CoverageAssignment, 'invalid assignment type')
            _group(assignment.group)
            _candidate(assignment.candidate, checked.board)
            _require(assignment.group in targets and assignment.group not in assigned,
                     'extraneous or duplicate group assignment')
            _require(assignment.candidate in coverage, 'assignment candidate is not selected')
            _require(assignment.group in coverage[assignment.candidate], 'assignment does not cover group')
            assigned.add(assignment.group)
        _require(assigned == set(targets), 'full target coverage is missing')
    except (ValueError, TypeError, AttributeError, IndexError) as exc:
        return WitnessVerification(VerificationStatus.REJECTED, (str(exc),))
    return WitnessVerification(VerificationStatus.VERIFIED_UNCERTIFIED)
