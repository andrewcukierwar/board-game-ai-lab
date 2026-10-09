"""Allis §§6.4–6.9 composite rules: shapes, board prerequisites and coverage.

Research semantics only. A ``CompositeCandidate`` is a local rule shape whose
constructor checks intrinsic geometry; ``prerequisite_failures`` binds it to a
board; ``solution_clauses`` states the source-defined groups it conditionally
solves. Nothing here asserts compatibility, joint executability, Zugzwang
control, an executable response strategy or a game value.

Coverage is a disjunction of labelled clauses. A group is solved by a clause
when it meets EVERY requirement set of that clause; almost all requirement sets
are single squares ("contains square s"). Aftereven's timing clause is the one
place a requirement is a range ("some square above the group square").
"""
from dataclasses import dataclass
from itertools import permutations, product

from .geometry import ALL_GROUPS, COLUMNS, ROWS, Group, Square
from .position import Player, Position, validate_player
from .rules import RuleCandidate, RuleName, ThesisReference, enumerate_candidates

CL, BI, VE = RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL
AE, LI, HI = RuleName.AFTEREVEN, RuleName.LOWINVERSE, RuleName.HIGHINVERSE
BC, BE, SB = RuleName.BASECLAIM, RuleName.BEFORE, RuleName.SPECIALBEFORE
COMPOSITE_RULES = (AE, LI, HI, BC, BE, SB)

_REFERENCES = {
    AE: ThesisReference('6.4', (39, 40)), LI: ThesisReference('6.5', (40, 41)),
    HI: ThesisReference('6.6', (41, 42)), BC: ThesisReference('6.7', (42, 43)),
    BE: ThesisReference('6.8', (43, 45)), SB: ThesisReference('6.9', (45, 46)),
}


def above(square: Square) -> Square | None:
    return Square(square.row_index - 1, square.column) if square.row < ROWS else None


def below(square: Square) -> Square | None:
    return Square(square.row_index + 1, square.column) if square.row > 1 else None


def _squares_above(square: Square) -> frozenset[Square]:
    return frozenset(Square(r, square.column) for r in range(square.row_index))


def _pair_name(lower: Square, upper: Square) -> str:
    return f'{lower.name}-{upper.name}'


@dataclass(frozen=True)
class Component:
    """A constituent vertical pair (lower, upper) of a composite rule.

    Claimeven components need an even upper square. Vertical components need
    only adjacency: a Before may use an even-upper Vertical (§6.8, e1-e2),
    unlike the standalone §6.3 rule. Lowinverse imposes its own odd parity.
    """

    rule: RuleName
    lower: Square
    upper: Square

    def __post_init__(self) -> None:
        if type(self.rule) is not RuleName or self.rule not in (CL, VE):
            raise ValueError('a component is a Claimeven or Vertical')
        for square in (self.lower, self.upper):
            if type(square) is not Square:
                raise ValueError('component squares must be Squares')
            Square(square.row_index, square.column)
        if self.lower.column != self.upper.column or self.upper.row != self.lower.row + 1:
            raise ValueError('component squares must be vertically adjacent, lower first')
        if self.rule == CL and not self.upper.is_even:
            raise ValueError('a Claimeven component needs an even upper square')

    @property
    def squares(self) -> tuple[Square, Square]:
        return self.lower, self.upper

    @property
    def name(self) -> str:
        return f'{self.rule.value}:{_pair_name(self.lower, self.upper)}'

    @property
    def before_square(self) -> Square:
        """The (Special)Before group square this component handles (§6.8).

        Claimeven: the controller takes the even upper square itself. Vertical:
        the controller takes the lower square or, if the opponent does, its successor.
        """
        return self.upper if self.rule == CL else self.lower

    def clause(self) -> 'SolutionClause':
        if self.rule == CL:
            return SolutionClause(self.name, (frozenset((self.upper,)),))
        return SolutionClause(self.name, (frozenset((self.lower,)), frozenset((self.upper,))))

    def reflected(self) -> 'Component':
        return Component(self.rule, self.lower.reflected(), self.upper.reflected())


def _component_key(component: Component) -> tuple[int, int]:
    return component.lower.column, component.lower.row


@dataclass(frozen=True)
class SolutionClause:
    """Solves a group iff the group meets every requirement set."""

    label: str
    requirements: tuple[frozenset[Square], ...]

    def solves(self, group: Group) -> bool:
        squares = set(group.squares)
        return all(not squares.isdisjoint(required) for required in self.requirements)


@dataclass(frozen=True)
class CompositeCandidate:
    """One application of a composite rule; construction checks no board.

    Field layout per rule (``roles`` are explicitly role-ordered):

    =============  =====  ==========================================  ==========================
    rule           group  components                                  roles
    =============  =====  ==========================================  ==========================
    Aftereven      yes    Claimevens, one per empty group square      ()
    Lowinverse     no     two odd-upper Verticals, ascending column   ()
    Highinverse    no     ()                                          (l1, m1, u1, l2, m2, u2)
    Baseclaim      no     ()                                          (first, second, third)
    Before         yes    CL/VE per empty group square, not all CL    ()
    Specialbefore  yes    CL/VE per OTHER empty group square          (playable group sq, extra)
    =============  =====  ==========================================  ==========================

    Components are normalized to ascending (column, row); Highinverse columns to
    ascending column. Baseclaim/Specialbefore role order is significant.
    """

    rule: RuleName
    group: Group | None = None
    components: tuple[Component, ...] = ()
    roles: tuple[Square, ...] = ()

    def __post_init__(self) -> None:
        if type(self.rule) is not RuleName or self.rule not in COMPOSITE_RULES:
            raise ValueError('rule is not a composite Allis rule')
        if type(self.components) is not tuple or type(self.roles) is not tuple:
            raise ValueError('components and roles must be tuples')
        for component in self.components:
            if type(component) is not Component:
                raise ValueError('components must be Component instances')
            Component(component.rule, component.lower, component.upper)
        for square in self.roles:
            if type(square) is not Square:
                raise ValueError('role squares must be Squares')
            Square(square.row_index, square.column)
        if self.group is not None:
            if type(self.group) is not Group or type(self.group.squares) is not tuple:
                raise ValueError('group must be a Group')
            for square in self.group.squares:
                if type(square) is not Square:
                    raise ValueError('group squares must be Squares')
            Group(self.group.squares)
        object.__setattr__(self, 'components', tuple(sorted(self.components, key=_component_key)))
        if self.rule == HI and len(self.roles) == 6:
            columns = sorted((self.roles[:3], self.roles[3:]), key=lambda t: t[0].column)
            object.__setattr__(self, 'roles', columns[0] + columns[1])
        _VALIDATORS[self.rule](self)

    # ----------------------------------------------------------- structure

    @property
    def reference(self) -> ThesisReference:
        return _REFERENCES[self.rule]

    @property
    def baseclaim_square(self) -> Square:
        """Baseclaim's non-playable even square, directly above the second square."""
        if self.rule != BC:
            raise ValueError('only a Baseclaim has this square')
        return above(self.roles[1])

    @property
    def affected_squares(self) -> frozenset[Square]:
        """The rule's complete "set of squares" for §7.4 constraints 1, 3 and 4."""
        if self.rule == BC:
            return frozenset(self.roles + (self.baseclaim_square,))
        return frozenset(s for c in self.components for s in c.squares) | frozenset(self.roles)

    @property
    def claimeven_parts(self) -> tuple[tuple[Square, Square], ...]:
        """Every Claimeven (lower, upper) inside the rule, for §7.4 constraint 2."""
        if self.rule == BC:
            return ((self.roles[1], self.baseclaim_square),)
        return tuple(c.squares for c in self.components if c.rule == CL)

    @property
    def inverse_columns(self) -> dict[int, frozenset[Square]]:
        """Lowinverse/Highinverse squares per inverse column; empty for other rules."""
        if self.rule == LI:
            return {c.lower.column: frozenset(c.squares) for c in self.components}
        if self.rule == HI:
            return {t[0].column: frozenset(t) for t in (self.roles[:3], self.roles[3:])}
        return {}

    @property
    def special_squares(self) -> frozenset[Square]:
        """§7.4 N.B.(ii): Specialbefore squares never 'equal' another rule's squares."""
        return frozenset(self.roles) if self.rule == SB else frozenset()

    @property
    def depends_on_zugzwang(self) -> bool:
        """Informational (§7.1–7.3); Verticals and the BI-like pairs do not."""
        if self.rule in (BE, SB):
            return any(c.rule == CL for c in self.components)
        return True

    def solution_clauses(self, position: Position) -> tuple[SolutionClause, ...]:
        """Source-defined coverage. Only Highinverse consults the board (§6.6).

        The position is NOT checked against the prerequisites here.
        """
        if type(position) is not Position:
            raise ValueError('position must be a Position snapshot')
        return _CLAUSES[self.rule](self, position)

    def reflected(self) -> 'CompositeCandidate':
        return CompositeCandidate(
            self.rule, None if self.group is None else self.group.reflected(),
            tuple(c.reflected() for c in self.components),
            tuple(s.reflected() for s in self.roles))


# ------------------------------------------------------- intrinsic validation

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _disjoint(squares: list[Square]) -> bool:
    return len(squares) == len(set(squares))


def _validate_aftereven(c: CompositeCandidate) -> None:
    _require(c.group is not None and not c.roles and c.components,
             'Aftereven needs a group and at least one Claimeven')
    _require(all(k.rule == CL for k in c.components), 'Aftereven components are Claimevens')
    _require(all(k.upper in c.group.squares and k.lower not in c.group.squares
                 for k in c.components), 'Aftereven Claimevens must claim group squares')
    _require(_disjoint([s for k in c.components for s in k.squares]),
             'Aftereven Claimevens must be disjoint')


def _validate_lowinverse(c: CompositeCandidate) -> None:
    _require(c.group is None and not c.roles and len(c.components) == 2,
             'Lowinverse needs exactly two Verticals')
    _require(all(k.rule == VE and not k.upper.is_even for k in c.components),
             'Lowinverse pairs are Verticals with odd upper squares')
    _require(c.components[0].lower.column != c.components[1].lower.column,
             'Lowinverse columns must differ')


def _validate_highinverse(c: CompositeCandidate) -> None:
    _require(c.group is None and not c.components and len(c.roles) == 6,
             'Highinverse needs two three-square columns')
    for lower, middle, upper in (c.roles[:3], c.roles[3:]):
        _require(lower.column == middle.column == upper.column
                 and middle.row == lower.row + 1 and upper.row == middle.row + 1,
                 'Highinverse column squares must be consecutive, lowest first')
        _require(upper.is_even, 'Highinverse upper squares must be even')
    _require(c.roles[0].column != c.roles[3].column, 'Highinverse columns must differ')


def _validate_baseclaim(c: CompositeCandidate) -> None:
    _require(c.group is None and not c.components and len(c.roles) == 3,
             'Baseclaim needs three role-ordered squares')
    _require(len({s.column for s in c.roles}) == 3, 'Baseclaim squares need distinct columns')
    _require(not c.roles[1].is_even, 'the square above the second Baseclaim square must be even')


def _validate_before_parts(c: CompositeCandidate, extra: tuple[Square, ...]) -> None:
    targets = [k.before_square for k in c.components]
    _require(_disjoint(targets) and all(t in c.group.squares for t in targets),
             'each component must handle a distinct group square')
    _require(all(s == k.before_square or s not in c.group.squares
                 for k in c.components for s in k.squares),
             'component squares other than the handled square lie outside the group')
    _require(all(t.row < ROWS for t in targets + list(extra[:1])),
             'Before group squares may not lie in the upper row')
    _require(_disjoint([s for k in c.components for s in k.squares] + list(extra)),
             'Before parts must be pairwise disjoint')


def _validate_before(c: CompositeCandidate) -> None:
    _require(c.group is not None and not c.roles and c.components,
             'Before needs a group and at least one component')
    _require(any(k.rule == VE for k in c.components),
             'an all-Claimeven Before is the stronger Aftereven (§6.8)')
    _validate_before_parts(c, ())


def _validate_specialbefore(c: CompositeCandidate) -> None:
    _require(c.group is not None and len(c.roles) == 2,
             'Specialbefore needs a group, a playable group square and an extra square')
    playable, extra = c.roles
    _require(playable in c.group.squares, 'the playable square must be a group square')
    _require(extra not in c.group.squares and extra.column != playable.column,
             'the extra square must lie outside the group, in another column')
    _validate_before_parts(c, (playable, extra))


_VALIDATORS = {AE: _validate_aftereven, LI: _validate_lowinverse, HI: _validate_highinverse,
               BC: _validate_baseclaim, BE: _validate_before, SB: _validate_specialbefore}


# ------------------------------------------------------------------ coverage

def _clause(label: str, *squares: Square) -> SolutionClause:
    return SolutionClause(label, tuple(frozenset((s,)) for s in squares))


def _aftereven_clauses(c, position):
    timing = SolutionClause('aftereven-columns',
                            tuple(_squares_above(k.upper) for k in c.components))
    return (timing,) + tuple(k.clause() for k in c.components)


def _lowinverse_clauses(c, position):
    first, second = c.components
    return (_clause('upper-pair', first.upper, second.upper), first.clause(), second.clause())


def _highinverse_clauses(c, position):
    (l1, m1, u1), (l2, m2, u2) = c.roles[:3], c.roles[3:]
    clauses = [_clause('upper-pair', u1, u2), _clause('middle-pair', m1, m2),
               _clause(f'vertical:{_pair_name(m1, u1)}', m1, u1),
               _clause(f'vertical:{_pair_name(m2, u2)}', m2, u2)]
    # Conditional (§6.6): only when that lower square is directly playable NOW.
    if position.is_playable(l1):
        clauses.append(_clause('lower-first+upper-second', l1, u2))
    if position.is_playable(l2):
        clauses.append(_clause('lower-second+upper-first', l2, u1))
    return tuple(clauses)


def _baseclaim_clauses(c, position):
    first, second, third = c.roles
    return (_clause('first+above-second', first, c.baseclaim_square),
            _clause('second+third', second, third))


def _before_clauses(c, position):
    successors = SolutionClause('successors', tuple(
        frozenset((above(k.before_square),)) for k in c.components))
    return (successors,) + tuple(k.clause() for k in c.components)


def _specialbefore_clauses(c, position):
    playable, extra = c.roles
    successors = [above(k.before_square) for k in c.components] + [above(playable)]
    return ((_clause('successors+extra', *sorted(successors), extra),
             _clause('playable-pair', playable, extra))
            + tuple(k.clause() for k in c.components))


_CLAUSES = {AE: _aftereven_clauses, LI: _lowinverse_clauses, HI: _highinverse_clauses,
            BC: _baseclaim_clauses, BE: _before_clauses, SB: _specialbefore_clauses}


def solution_clauses(candidate: 'RuleCandidate | CompositeCandidate',
                     position: Position) -> tuple[SolutionClause, ...]:
    """Uniform coverage for all nine rules (§§6.1–6.9)."""
    if type(candidate) is RuleCandidate:
        if candidate.rule == CL:
            return (_clause('claimeven', candidate.squares[1]),)
        return (_clause(candidate.rule.value, *candidate.squares),)
    if type(candidate) is CompositeCandidate:
        return candidate.solution_clauses(position)
    raise ValueError('candidate must be a RuleCandidate or CompositeCandidate')


# ------------------------------------------------------ board prerequisites

def _opponent_mark(controller: Player) -> str:
    validate_player(controller)
    return 'X' if controller == 1 else 'O'


def _cell(position: Position, square: Square) -> str:
    return position.board[square.row_index][square.column]


def prerequisite_failures(candidate: CompositeCandidate, position: Position,
                          controller: Player = 1) -> tuple[str, ...]:
    """Every violated §6 'Required' condition on this board; empty means all hold."""
    if type(candidate) is not CompositeCandidate or type(position) is not Position:
        raise ValueError('requires a CompositeCandidate and a Position')
    opponent = _opponent_mark(controller)
    failures = []
    if position.terminal:
        failures.append('position is terminal')
    occupied = sorted(s for s in candidate.affected_squares if not position.is_empty(s))
    if occupied:
        failures.append('occupied rule squares: ' + ' '.join(s.name for s in occupied))
    if candidate.rule in (BC, SB):  # Every role square must be a landing square.
        failures += [f'{s.name} is not directly playable' for s in candidate.roles
                     if not position.is_playable(s)]
    if candidate.group is not None:
        group = candidate.group.squares
        if any(_cell(position, s) == opponent for s in group):
            failures.append('group contains an opponent stone')
        handled = {k.before_square for k in candidate.components} | (
            {candidate.roles[0]} if candidate.rule == SB else set())
        if {s for s in group if position.is_empty(s)} != handled:
            failures.append('components do not handle exactly the empty group squares')
    return tuple(failures)


# --------------------------------------------------------------- enumeration

def _vertical_shapes(position: Position, upper_rows: tuple[int, ...]) -> dict[int, list]:
    shapes = {c: [] for c in range(COLUMNS)}
    for column in range(COLUMNS):
        for upper_row in upper_rows:
            lower = Square(ROWS + 1 - upper_row, column)
            upper = Square(ROWS - upper_row, column)
            if position.is_empty(lower) and position.is_empty(upper):
                shapes[column].append(Component(VE, lower, upper))
    return shapes


def enumerate_afterevens(position: Position, controller: Player = 1) -> tuple[CompositeCandidate, ...]:
    """§6.4: groups completable using only even Claimeven squares; ALL_GROUPS order."""
    opponent = _opponent_mark(controller)
    if position.terminal:
        return ()
    result = []
    for group in ALL_GROUPS:
        if any(_cell(position, s) == opponent for s in group.squares):
            continue
        empties = [s for s in group.squares if position.is_empty(s)]
        if empties and all(s.is_even and position.is_empty(below(s)) for s in empties):
            try:
                result.append(CompositeCandidate(
                    AE, group, tuple(Component(CL, below(s), s) for s in empties)))
            except ValueError:  # Overlapping Claimevens (vertical group): not a rule.
                pass
    return tuple(result)


def enumerate_lowinverses(position: Position) -> tuple[CompositeCandidate, ...]:
    """§6.5: two empty odd-upper vertical pairs in different columns."""
    if position.terminal:
        return ()
    shapes = _vertical_shapes(position, (3, 5))
    return tuple(CompositeCandidate(LI, None, (a, b))
                 for c1 in range(COLUMNS) for c2 in range(c1 + 1, COLUMNS)
                 for a in shapes[c1] for b in shapes[c2])


def enumerate_highinverses(position: Position) -> tuple[CompositeCandidate, ...]:
    """§6.6: two empty three-square columns with even upper squares."""
    if position.terminal:
        return ()
    triples = {c: [] for c in range(COLUMNS)}
    for column in range(COLUMNS):
        for lower_row in (2, 4):
            triple = tuple(Square(ROWS - lower_row - i, column) for i in range(3))
            if all(position.is_empty(s) for s in triple):
                triples[column].append(triple)
    return tuple(CompositeCandidate(HI, roles=a + b)
                 for c1 in range(COLUMNS) for c2 in range(c1 + 1, COLUMNS)
                 for a in triples[c1] for b in triples[c2])


def enumerate_baseclaims(position: Position) -> tuple[CompositeCandidate, ...]:
    """§6.7: ordered (first, second, third) playable squares; even square above second."""
    if position.terminal:
        return ()
    return tuple(CompositeCandidate(BC, roles=roles)
                 for roles in permutations(position.playable_squares, 3)
                 if not roles[1].is_even)


def _before_options(position: Position, square: Square) -> list[Component]:
    options = [Component(VE, square, above(square))]
    lower = below(square)
    if square.is_even and lower is not None and position.is_empty(lower):
        options.append(Component(CL, lower, square))
    return options


def _before_groups(position: Position, controller: Player):
    opponent = _opponent_mark(controller)
    for group in ALL_GROUPS:
        empties = [s for s in group.squares if position.is_empty(s)]
        if (empties and all(s.row < ROWS for s in empties)
                and all(_cell(position, s) != opponent for s in group.squares)):
            yield group, empties


def enumerate_befores(position: Position, controller: Player = 1) -> tuple[CompositeCandidate, ...]:
    """§6.8: every disjoint CL/VE choice per empty group square, except all-Claimeven."""
    if position.terminal:
        return ()
    result = []
    for group, empties in _before_groups(position, controller):
        for choice in product(*(_before_options(position, s) for s in empties)):
            try:
                result.append(CompositeCandidate(BE, group, choice))
            except ValueError:  # Overlapping parts or all-Claimeven: not a Before.
                pass
    return tuple(result)


def enumerate_specialbefores(position: Position,
                             controller: Player = 1) -> tuple[CompositeCandidate, ...]:
    """§6.9: a Before with one playable group square paired with an extra playable square."""
    if position.terminal:
        return ()
    result = []
    for group, empties in _before_groups(position, controller):
        for playable in empties:
            if not position.is_playable(playable):
                continue
            others = [s for s in empties if s != playable]
            for extra in position.playable_squares:
                if extra in group.squares or extra.column == playable.column:
                    continue
                for choice in product(*(_before_options(position, s) for s in others)):
                    try:
                        result.append(CompositeCandidate(SB, group, choice, (playable, extra)))
                    except ValueError:  # Overlapping parts: not a Specialbefore.
                        pass
    return tuple(result)


def enumerate_composites(position: Position, controller: Player = 1) -> tuple[CompositeCandidate, ...]:
    """All six composite rules, in RuleName order. Complete within the stated shapes."""
    validate_player(controller)
    return (enumerate_afterevens(position, controller) + enumerate_lowinverses(position)
            + enumerate_highinverses(position) + enumerate_baseclaims(position)
            + enumerate_befores(position, controller)
            + enumerate_specialbefores(position, controller))


def enumerate_all_candidates(position: Position, controller: Player = 1):
    """All nine rules: Phase 6A CL/BI/VE order, then the six composites."""
    return enumerate_candidates(position) + enumerate_composites(position, controller)
