"""Independent verification of nine-rule coverage witnesses (research only).

Shares only immutable vocabulary types and basic board legality with the
producer. It does NOT call candidate enumeration, ``solution_clauses``,
prerequisite checks, compatibility, footprints, the search or any
``CompositeCandidate``/``Component`` property: every rule is re-derived from
raw fields and raw board cells, in bottom-based ``(column, row)`` coordinates,
with its own transcription of the §7.4 matrix. Disagreement with the producer
is a rejection.

Success means only that the finite coverage and §7.4 predicates hold for the
bound board. It is NOT a non-loss proof: see ``NINE_RULE_OBLIGATIONS``.
"""
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

from .composite import Component, CompositeCandidate
from .coverage import BlackEvaluationContext
from .geometry import Group, Square
from .nine_rules import (
    ALL_RULES, NINE_RULE_OBLIGATIONS, SCHEMA_VERSION, NineRuleAssignment, NineRuleWitness,
    RuleEvidence,
)
from .position import Position
from .rules import RuleCandidate, RuleName
from .verification import VerificationStatus

Cell = tuple[int, int]  # (column 0..6, row 1..6), row 1 at the bottom.
EMPTY = ' '

# Independent transcription of Allis §7.4 (p.50), row rule then column rules.
_MATRIX_TEXT = """
claimeven 1
baseinverse 1 1
vertical 1 1 1
aftereven 1 1 1 3
lowinverse 2 1 1 12 4
highinverse 2 1 1 12 4 4
baseclaim 1 1 1 1 12 12 1
before 1 1 1 3 23 12 1 3
specialbefore 1 1 1 3 23 12 1 3 3
"""


def _matrix() -> dict[frozenset, frozenset[int]]:
    rows = [line.split() for line in _MATRIX_TEXT.strip().splitlines()]
    return {frozenset((row[0], rows[j][0])): frozenset(int(d) for d in code)
            for row in rows for j, code in enumerate(row[1:])}


_PAIR_CODES = _matrix()


@dataclass(frozen=True)
class NineRuleVerification:
    status: VerificationStatus
    rejection_reasons: tuple[str, ...] = ()
    outcome_certification: Literal['uncertified'] = field(default='uncertified', init=False)
    unproven_obligations: tuple[str, ...] = field(default=NINE_RULE_OBLIGATIONS, init=False)


@dataclass
class _Shape:
    rule: str
    cells: frozenset
    clauses: list  # [(label, [frozenset[Cell], ...])]
    claimevens: list  # [(lower Cell, upper Cell)]
    inverse: dict  # column -> frozenset[Cell]
    special: frozenset


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _windows() -> frozenset[frozenset[Cell]]:
    lines = set()
    for c in range(7):
        for r in range(1, 7):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(c + i * dc, r + i * dr) for i in range(4)]
                if all(0 <= x < 7 and 1 <= y <= 6 for x, y in cells):
                    lines.add(frozenset(cells))
    return frozenset(lines)


_WINDOWS = _windows()


def _cell(square) -> Cell:
    _require(type(square) is Square, 'invalid square type')
    _require(type(square.row_index) is int and type(square.column) is int
             and 0 <= square.row_index < 6 and 0 <= square.column < 7, 'square out of range')
    return square.column, 6 - square.row_index


def _name(cell: Cell) -> str:
    return f'{"abcdefg"[cell[0]]}{cell[1]}'


def _group_cells(group) -> frozenset[Cell]:
    _require(type(group) is Group and type(group.squares) is tuple and len(group.squares) == 4,
             'invalid group type')
    cells = [_cell(s) for s in group.squares]
    _require(frozenset(cells) in _WINDOWS, 'squares do not form a group')
    _require(list(group.squares) == sorted(group.squares), 'noncanonical group order')
    return frozenset(cells)


class _Board:
    def __init__(self, board):
        self.board = board

    def at(self, cell: Cell) -> str:
        return self.board[6 - cell[1]][cell[0]]

    def empty(self, cell: Cell) -> bool:
        return self.at(cell) == EMPTY

    def playable(self, cell: Cell) -> bool:
        return self.empty(cell) and (cell[1] == 1 or not self.empty((cell[0], cell[1] - 1)))


def _single(label: str, *cells: Cell):
    return label, [frozenset((c,)) for c in cells]


def _basic(candidate, board: _Board) -> _Shape:
    _require(type(candidate.squares) is tuple and len(candidate.squares) == 2,
             'candidate must contain two squares')
    rule = candidate.rule
    _require(type(rule) is RuleName and rule in (RuleName.CLAIMEVEN, RuleName.BASEINVERSE,
                                                RuleName.VERTICAL), 'unsupported two-square rule')
    a, b = (_cell(s) for s in candidate.squares)
    _require(a != b and board.empty(a) and board.empty(b), 'candidate squares must be empty and distinct')
    if rule == RuleName.BASEINVERSE:
        _require(a[0] != b[0] and candidate.squares[0] < candidate.squares[1],
                 'invalid Baseinverse columns/order')
        _require(board.playable(a) and board.playable(b), 'Baseinverse square is not directly playable')
        return _Shape(rule.value, frozenset((a, b)), [_single('baseinverse', a, b)], [], {}, frozenset())
    _require(a[0] == b[0] and b[1] == a[1] + 1, 'invalid vertical roles/adjacency')
    even = rule == RuleName.CLAIMEVEN
    _require((b[1] % 2 == 0) == even, 'invalid vertical parity')
    clauses = [_single('claimeven', b)] if even else [_single('vertical', a, b)]
    return _Shape(rule.value, frozenset((a, b)), clauses, [(a, b)] if even else [], {}, frozenset())


def _component(component) -> tuple[str, Cell, Cell]:
    _require(type(component) is Component, 'invalid component type')
    _require(type(component.rule) is RuleName
             and component.rule in (RuleName.CLAIMEVEN, RuleName.VERTICAL), 'invalid component rule')
    lower, upper = _cell(component.lower), _cell(component.upper)
    _require(lower[0] == upper[0] and upper[1] == lower[1] + 1, 'component squares not adjacent')
    if component.rule == RuleName.CLAIMEVEN:
        _require(upper[1] % 2 == 0, 'Claimeven component upper square must be even')
    return component.rule.value, lower, upper


def _component_clause(kind: str, lower: Cell, upper: Cell):
    label = f'{kind}:{_name(lower)}-{_name(upper)}'
    return _single(label, upper) if kind == 'claimeven' else _single(label, lower, upper)


def _distinct(cells: list) -> bool:
    return len(cells) == len(set(cells))


def _composite(candidate, board: _Board) -> _Shape:
    rule = candidate.rule
    _require(type(rule) is RuleName and rule in ALL_RULES[3:], 'unsupported composite rule')
    _require(type(candidate.components) is tuple and type(candidate.roles) is tuple,
             'invalid composite containers')
    parts = [_component(k) for k in candidate.components]
    keys = [(lower[0], lower[1]) for _, lower, _ in parts]
    _require(keys == sorted(keys) and _distinct(keys), 'noncanonical component order')
    roles = [_cell(s) for s in candidate.roles]
    part_cells = [c for _, lower, upper in parts for c in (lower, upper)]
    opponent = 'X'  # Black is the only supported defender.
    group = None
    if candidate.group is not None:
        group = _group_cells(candidate.group)
        _require(all(board.at(c) != opponent for c in group), 'group contains an opponent stone')
    name = rule.value

    if name in ('lowinverse', 'highinverse', 'baseclaim'):
        _require(group is None, f'{name} has no group')
    else:
        _require(group is not None, f'{name} needs a group')

    if name == 'aftereven':
        _require(not roles and parts and all(k == 'claimeven' for k, _, _ in parts),
                 'Aftereven consists of Claimevens only')
        uppers = {upper for _, _, upper in parts}
        _require(_distinct(part_cells) and all(lower not in group for _, lower, _ in parts),
                 'Aftereven Claimevens overlap or claim lower group squares')
        _require(all(board.empty(c) for c in part_cells), 'Aftereven square occupied')
        _require({c for c in group if board.empty(c)} == uppers,
                 'Aftereven Claimevens must claim exactly the empty group squares')
        timing = ('aftereven-columns', [frozenset((u[0], r) for r in range(u[1] + 1, 7))
                                        for _, _, u in parts])
        clauses = [timing] + [_component_clause(*p) for p in parts]
        return _Shape(name, frozenset(part_cells), clauses,
                      [(lo, up) for _, lo, up in parts], {}, frozenset())

    if name == 'lowinverse':
        _require(not roles and len(parts) == 2 and all(k == 'vertical' for k, _, _ in parts),
                 'Lowinverse needs two Verticals')
        (_, l1, u1), (_, l2, u2) = parts
        _require(l1[0] != l2[0] and u1[1] % 2 == 1 and u2[1] % 2 == 1,
                 'Lowinverse needs two columns with odd upper squares')
        _require(all(board.empty(c) for c in part_cells), 'Lowinverse square occupied')
        clauses = [_single('upper-pair', u1, u2), _component_clause(*parts[0]),
                   _component_clause(*parts[1])]
        return _Shape(name, frozenset(part_cells), clauses, [],
                      {l1[0]: frozenset((l1, u1)), l2[0]: frozenset((l2, u2))}, frozenset())

    if name == 'highinverse':
        _require(not parts and len(roles) == 6, 'Highinverse needs six role squares')
        columns = (roles[:3], roles[3:])
        for low, mid, up in columns:
            _require(low[0] == mid[0] == up[0] and mid[1] == low[1] + 1 and up[1] == low[1] + 2
                     and up[1] % 2 == 0, 'invalid Highinverse column')
        _require(columns[0][0][0] < columns[1][0][0], 'Highinverse columns must ascend and differ')
        _require(all(board.empty(c) for c in roles), 'Highinverse square occupied')
        (l1, m1, u1), (l2, m2, u2) = columns
        clauses = [_single('upper-pair', u1, u2), _single('middle-pair', m1, m2),
                   _single(f'vertical:{_name(m1)}-{_name(u1)}', m1, u1),
                   _single(f'vertical:{_name(m2)}-{_name(u2)}', m2, u2)]
        if board.playable(l1):
            clauses.append(_single('lower-first+upper-second', l1, u2))
        if board.playable(l2):
            clauses.append(_single('lower-second+upper-first', l2, u1))
        return _Shape(name, frozenset(roles), clauses, [],
                      {l1[0]: frozenset(columns[0]), l2[0]: frozenset(columns[1])}, frozenset())

    if name == 'baseclaim':
        _require(not parts and len(roles) == 3 and len({c for c, _ in roles}) == 3,
                 'Baseclaim needs three squares in distinct columns')
        first, second, third = roles
        _require(all(board.playable(c) for c in roles), 'Baseclaim square is not directly playable')
        q = (second[0], second[1] + 1)
        _require(q[1] <= 6 and q[1] % 2 == 0, 'square above the second Baseclaim square must be even')
        clauses = [_single('first+above-second', first, q), _single('second+third', second, third)]
        return _Shape(name, frozenset(roles + [q]), clauses, [(second, q)], {}, frozenset())

    # Before / Specialbefore.
    handled = [upper if kind == 'claimeven' else lower for kind, lower, upper in parts]
    special = []
    if name == 'before':
        _require(not roles and parts, 'Before needs components and no roles')
        _require(any(k == 'vertical' for k, _, _ in parts),
                 'all-Claimeven Before is excluded in favour of Aftereven')
    else:
        _require(len(roles) == 2, 'Specialbefore needs a playable group square and an extra square')
        playable, extra = roles
        _require(playable in group and board.playable(playable),
                 'Specialbefore playable square must be a directly playable group square')
        _require(extra not in group and extra[0] != playable[0] and board.playable(extra),
                 'Specialbefore extra square must be directly playable outside the group column')
        special = [playable, extra]
    all_cells = part_cells + special
    _require(_distinct(all_cells), 'Before parts overlap')
    _require(all(board.empty(c) for c in all_cells), 'Before square occupied')
    covered = handled + special[:1]
    _require(_distinct(covered) and {c for c in group if board.empty(c)} == set(covered),
             'components must handle exactly the empty group squares')
    _require(all(c[1] < 6 for c in covered), 'Before group square in the upper row')
    _require(all(c not in group for c in part_cells if c not in handled),
             'component squares other than handled squares must lie outside the group')
    successors = sorted((c, r + 1) for c, r in covered)
    if name == 'before':
        clauses = [('successors', [frozenset((s,)) for s in successors])]
    else:
        clauses = [_single('successors+extra', *successors, special[1]),
                   _single('playable-pair', *special)]
    clauses += [_component_clause(*p) for p in parts]
    claimevens = [(lo, up) for kind, lo, up in parts if kind == 'claimeven']
    return _Shape(name, frozenset(all_cells), clauses, claimevens, {}, frozenset(special))


def _shape(candidate, board: _Board) -> _Shape:
    if type(candidate) is RuleCandidate:
        return _basic(candidate, board)
    _require(type(candidate) is CompositeCandidate, 'invalid candidate type')
    return _composite(candidate, board)


def _solves(shape: _Shape, group: frozenset[Cell]) -> list[str]:
    return [label for label, requirements in shape.clauses
            if all(group & required for required in requirements)]


def _violations(a: _Shape, b: _Shape) -> list[int]:
    codes = _PAIR_CODES[frozenset((a.rule, b.rule))]
    failed = []
    if 1 in codes and a.cells & b.cells:
        failed.append(1)
    if 2 in codes:
        inverse, other = (a, b) if a.inverse else (b, a)
        _require(bool(inverse.inverse) and not other.inverse, 'constraint 2 needs one inverse')
        if any(lo[0] in inverse.inverse and lo[1] <= max(r for _, r in inverse.inverse[lo[0]])
               for lo, _ in other.claimevens):
            failed.append(2)
    if 3 in codes:
        for column in {c for c, _ in a.cells & b.cells}:
            part_a = {x for x in a.cells if x[0] == column}
            part_b = {x for x in b.cells if x[0] == column}
            if part_a != part_b or part_a & (a.special | b.special):
                failed.append(3)
                break
    if 4 in codes:
        ca, cb = set(a.inverse), set(b.inverse)
        if a.cells & b.cells or not (ca == cb or not ca & cb):
            failed.append(4)
    return failed


def _clause_claims(clauses) -> list:
    _require(type(clauses) is tuple, 'clause claims must be a tuple')
    result = []
    for clause in clauses:
        label, requirements = clause.label, clause.requirements
        _require(type(label) is str and type(requirements) is tuple, 'invalid clause claim')
        sets = []
        for required in requirements:
            _require(type(required) is frozenset, 'invalid clause requirement')
            sets.append(frozenset(_cell(s) for s in required))
        result.append((label, frozenset(sets)))
    return result


def verify_nine_rule_witness(position: Position, witness: NineRuleWitness) -> NineRuleVerification:
    """Treat both inputs as untrusted; recompute every finite predicate.

    Rejection reports the first detected defect. Success never establishes
    reachability, joint strategic soundness, safety or a game value.
    """
    try:
        _require(type(position) is Position, 'invalid position type')
        checked = Position.from_board(position.board, position.player_to_move)
        _require(checked == position, 'noncanonical position snapshot')
        _require(checked.player_to_move == 0 and not checked.terminal,
                 'requires nonterminal White-to-move position')
        _require(type(witness) is NineRuleWitness, 'invalid witness type')
        _require(witness.schema_version == SCHEMA_VERSION
                 and witness.outcome_certification == 'uncertified'
                 and witness.unproven_obligations == NINE_RULE_OBLIGATIONS,
                 'invalid witness schema/status/obligations')
        rules = witness.rules
        _require(type(rules) is tuple and rules and all(type(r) is RuleName for r in rules)
                 and list(rules) == [r for r in ALL_RULES if r in rules], 'invalid rule set')
        context = witness.context
        _require(type(context) is BlackEvaluationContext and type(context.position) is Position,
                 'invalid context type')
        rebound = Position.from_board(context.position.board, context.position.player_to_move)
        _require(rebound == context.position and rebound == checked, 'witness position differs')
        _require(type(context.defender) is int and context.defender == 1
                 and type(context.opponent) is int and context.opponent == 0
                 and context.mode == 'black_opponent_to_move', 'unsupported context identity')
        board = _Board(checked.board)
        targets = {w for w in _WINDOWS if all(board.at(c) != 'O' for c in w)}
        _require(type(context.target_groups) is tuple, 'target claims must be a tuple')
        claimed_targets = [_group_cells(g) for g in context.target_groups]
        _require(len(claimed_targets) == len(set(claimed_targets))
                 and set(claimed_targets) == targets
                 and list(context.target_groups) == sorted(context.target_groups),
                 'target groups differ from recomputed universe')
        target_cells = claimed_targets  # Same set, in the producer's canonical order.

        _require(type(witness.evidence) is tuple, 'invalid evidence container')
        shapes = {}
        for evidence in witness.evidence:
            _require(type(evidence) is RuleEvidence, 'invalid evidence type')
            candidate = evidence.candidate
            shape = _shape(candidate, board)
            _require(candidate.rule in rules, 'candidate rule outside the declared rule set')
            _require(candidate not in shapes, 'duplicate selected candidate')
            claims = _clause_claims(evidence.clauses)
            _require([label for label, _ in claims] == [label for label, _ in shape.clauses]
                     and all(req == frozenset(mine) for (_, req), (_, mine)
                             in zip(claims, shape.clauses)), 'clause claim differs')
            solved = tuple(g for g, cells in zip(context.target_groups, target_cells)
                           if _solves(shape, cells))
            _require(type(evidence.conditional_solved_groups) is tuple, 'invalid coverage claim')
            for group in evidence.conditional_solved_groups:
                _group_cells(group)
            _require(evidence.conditional_solved_groups == solved, 'conditional coverage claim differs')
            shapes[candidate] = shape
        for first, second in combinations(shapes, 2):
            failed = _violations(shapes[first], shapes[second])
            _require(not failed, 'selected candidates conflict: constraint '
                     + '&'.join(str(code) for code in failed))
        _require(type(witness.assignments) is tuple, 'invalid assignment container')
        assigned = set()
        for assignment in witness.assignments:
            _require(type(assignment) is NineRuleAssignment, 'invalid assignment type')
            cells = _group_cells(assignment.group)
            _require(cells in targets and cells not in assigned, 'extraneous or duplicate group assignment')
            _require(assignment.candidate in shapes, 'assignment candidate is not selected')
            _require(type(assignment.clause) is str
                     and assignment.clause in _solves(shapes[assignment.candidate], cells),
                     'assignment clause does not solve group')
            assigned.add(cells)
        _require(assigned == targets, 'full target coverage is missing')
    except (ValueError, TypeError, AttributeError, IndexError, KeyError) as exc:
        return NineRuleVerification(VerificationStatus.REJECTED, (str(exc),))
    return NineRuleVerification(VerificationStatus.VERIFIED_UNCERTIFIED)
