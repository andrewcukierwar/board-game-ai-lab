"""The complete nine-rule compatibility matrix of Allis §7.4 (PDF p.50).

This checks intrinsic rule shapes and the four §7.4 constraints. It does NOT
check position validity or joint strategic executability: pairwise §7.4
compatibility is Allis's stated combination criterion, not a proved theorem
that any pairwise-compatible collection can be executed simultaneously.

Interpretations (docs/victor-nine-rule-implementation.md §4):
- Constraint 2, "no Claimeven is below the inverse": every Claimeven part (a
  Claimeven rule, an Aftereven/(Special)Before Claimeven component or the
  Baseclaim's second-square/even-square pair) lying in an inverse column must
  lie ENTIRELY above the inverse's squares in that column (§7.1: a Claimeven may
  be used above a Lowinverse, not below or across it).
- Constraint 3, "column-wise disjoint or equal": in every column the two
  rules' squares are disjoint or identical sets; by N.B.(ii) a column part
  holding a Specialbefore special square is never 'equal', so it must be disjoint.
- Identical instances always conflict (they are never two distinct rules).
"""
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

from .composite import CompositeCandidate, Component
from .contracts import CompatibilityConstraint
from .geometry import Group, Square
from .rules import RuleCandidate, RuleName

CL, BI, VE = RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL
AE, LI, HI = RuleName.AFTEREVEN, RuleName.LOWINVERSE, RuleName.HIGHINVERSE
BC, BE, SB = RuleName.BASECLAIM, RuleName.BEFORE, RuleName.SPECIALBEFORE
C1 = CompatibilityConstraint.DISJOINT_SQUARES
C2 = CompatibilityConstraint.NO_CLAIMEVEN_BELOW_INVERSE
C3 = CompatibilityConstraint.COLUMNWISE_DISJOINT_OR_EQUAL
C4 = CompatibilityConstraint.DISJOINT_SQUARES_AND_DISJOINT_OR_EQUAL_COLUMN_SETS

# Transcribed row by row from the visually checked lower-triangular table.
_ORDER = (CL, BI, VE, AE, LI, HI, BC, BE, SB)
_TABLE = (
    ((C1,),),
    ((C1,), (C1,)),
    ((C1,), (C1,), (C1,)),
    ((C1,), (C1,), (C1,), (C3,)),
    ((C2,), (C1,), (C1,), (C1, C2), (C4,)),
    ((C2,), (C1,), (C1,), (C1, C2), (C4,), (C4,)),
    ((C1,), (C1,), (C1,), (C1,), (C1, C2), (C1, C2), (C1,)),
    ((C1,), (C1,), (C1,), (C3,), (C2, C3), (C1, C2), (C1,), (C3,)),
    ((C1,), (C1,), (C1,), (C3,), (C2, C3), (C1, C2), (C1,), (C3,), (C3,)),
)
_CONSTRAINTS = MappingProxyType({
    frozenset((row_rule, _ORDER[j])): codes
    for row_rule, row in zip(_ORDER, _TABLE) for j, codes in enumerate(row)
})
if len(_CONSTRAINTS) != 45:  # pragma: no cover - import-time transcription check
    raise RuntimeError('the §7.4 matrix must define all 45 unordered rule pairs')
INVERSES = frozenset((LI, HI))


def required_constraints(first: RuleName, second: RuleName) -> tuple[CompatibilityConstraint, ...]:
    if type(first) is not RuleName or type(second) is not RuleName:
        raise ValueError('compatibility requires RuleName identifiers')
    return _CONSTRAINTS[frozenset((first, second))]


def checked_candidate(candidate):
    """Reconstruct untrusted intrinsic geometry; no board or coverage assertions."""
    if type(candidate) is RuleCandidate and type(candidate.squares) is tuple:
        for square in candidate.squares:
            if type(square) is not Square:
                raise ValueError('candidate square must be a Square')
            Square(square.row_index, square.column)
        rebuilt = RuleCandidate(candidate.rule, candidate.squares)
    elif type(candidate) is CompositeCandidate:
        group = candidate.group
        if group is not None and (type(group) is not Group or type(group.squares) is not tuple):
            raise ValueError('candidate group must be a Group')
        if type(candidate.components) is not tuple or type(candidate.roles) is not tuple:
            raise ValueError('candidate parts must be tuples')
        rebuilt = CompositeCandidate(
            candidate.rule, None if group is None else Group(group.squares),
            tuple(Component(k.rule, k.lower, k.upper) if type(k) is Component else k
                  for k in candidate.components), candidate.roles)
    else:
        raise ValueError('candidate must be a canonical RuleCandidate or CompositeCandidate')
    if rebuilt != candidate:
        raise ValueError('candidate square roles/order are not canonical')
    return rebuilt


@dataclass(frozen=True)
class Footprint:
    """What §7.4 needs from one rule; derived from a trusted, validated candidate."""

    rule: RuleName
    squares: frozenset[Square]
    claimevens: tuple[tuple[Square, Square], ...]
    inverse_columns: MappingProxyType
    special: frozenset[Square]

    def column(self, column: int) -> frozenset[Square]:
        return frozenset(s for s in self.squares if s.column == column)


@lru_cache(maxsize=65_536)
def footprint(candidate) -> Footprint:
    if type(candidate) is RuleCandidate:
        claimevens = (candidate.squares,) if candidate.rule == CL else ()
        return Footprint(candidate.rule, candidate.affected_squares, claimevens,
                         MappingProxyType({}), frozenset())
    return Footprint(candidate.rule, candidate.affected_squares, candidate.claimeven_parts,
                     MappingProxyType(candidate.inverse_columns), candidate.special_squares)


def _disjoint(a: Footprint, b: Footprint) -> bool:
    return a.squares.isdisjoint(b.squares)


def _no_claimeven_below_inverse(a: Footprint, b: Footprint) -> bool:
    inverse, other = (a, b) if a.rule in INVERSES else (b, a)
    if inverse.rule not in INVERSES or other.rule in INVERSES:
        raise ValueError('constraint 2 needs exactly one inverse')
    for lower, upper in other.claimevens:
        squares = inverse.inverse_columns.get(lower.column)
        if squares is not None and lower.row <= max(s.row for s in squares):
            return False
    return True


def _columnwise_disjoint_or_equal(a: Footprint, b: Footprint) -> bool:
    for column in {s.column for s in a.squares & b.squares}:
        part_a, part_b = a.column(column), b.column(column)
        if part_a != part_b or part_a & (a.special | b.special):
            return False
    return True


def _inverse_columns_disjoint_or_equal(a: Footprint, b: Footprint) -> bool:
    columns_a, columns_b = set(a.inverse_columns), set(b.inverse_columns)
    return _disjoint(a, b) and (columns_a == columns_b or columns_a.isdisjoint(columns_b))


_CHECKS = MappingProxyType({C1: _disjoint, C2: _no_claimeven_below_inverse,
                            C3: _columnwise_disjoint_or_equal,
                            C4: _inverse_columns_disjoint_or_equal})


def failed_constraints(first, second) -> tuple[CompatibilityConstraint, ...]:
    """The §7.4 constraints this pair violates; trusts already-validated candidates."""
    a, b = footprint(first), footprint(second)
    return tuple(code for code in required_constraints(a.rule, b.rule) if not _CHECKS[code](a, b))


_COLUMN = tuple(63 << (6 * c) for c in range(7))


def _mask(squares) -> int:
    return sum(1 << (6 * s.column + s.row_index) for s in squares)


@dataclass(frozen=True)
class CompactFootprint:
    """Integer form of a ``Footprint`` for bulk pairwise checks (same semantics).

    ``inverse_top`` maps each inverse column to its highest inverse row (§7.4
    constraint 2); ``claimevens`` holds (column, lower row) of every Claimeven part.
    """

    rule: RuleName
    squares: int
    special: int
    claimevens: tuple[tuple[int, int], ...]
    inverse_top: MappingProxyType
    inverse_columns: int


def compact(fp: Footprint) -> CompactFootprint:
    tops = {c: max(s.row for s in squares) for c, squares in fp.inverse_columns.items()}
    return CompactFootprint(fp.rule, _mask(fp.squares), _mask(fp.special),
                            tuple((lower.column, lower.row) for lower, _ in fp.claimevens),
                            MappingProxyType(tops), sum(1 << c for c in tops))


def _c2(a: CompactFootprint, b: CompactFootprint) -> bool:
    inverse, other = (a, b) if a.rule in INVERSES else (b, a)
    if inverse.rule not in INVERSES or other.rule in INVERSES:
        raise ValueError('constraint 2 needs exactly one inverse')
    tops = inverse.inverse_top
    return not any(c in tops and row <= tops[c] for c, row in other.claimevens)


def _c3(a: CompactFootprint, b: CompactFootprint) -> bool:
    shared, special = a.squares & b.squares, a.special | b.special
    for column in _COLUMN:
        if shared & column:
            part = a.squares & column
            if part != b.squares & column or part & special:
                return False
    return True


def _c4(a: CompactFootprint, b: CompactFootprint) -> bool:
    return not a.squares & b.squares and (a.inverse_columns == b.inverse_columns
                                          or not a.inverse_columns & b.inverse_columns)


_COMPACT_CHECKS = MappingProxyType({C1: lambda a, b: not a.squares & b.squares,
                                    C2: _c2, C3: _c3, C4: _c4})


def compact_conflict(a: CompactFootprint, b: CompactFootprint) -> bool:
    """True iff ``failed_constraints`` is nonempty for the underlying candidates."""
    return any(not _COMPACT_CHECKS[code](a, b) for code in _CONSTRAINTS[frozenset((a.rule, b.rule))])


def compatible(first, second) -> bool:
    """Symmetric; duplicates conflict. Unsupported or malformed shapes raise ValueError."""
    first, second = checked_candidate(first), checked_candidate(second)
    return first != second and not failed_constraints(first, second)
