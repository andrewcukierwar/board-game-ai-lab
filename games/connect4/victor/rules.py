"""Allis §§6.1–6.3 geometric candidates, never position-level certification."""
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Literal

from .geometry import COLUMNS, ROWS, Square
from .position import Position


class RuleName(str, Enum):
    CLAIMEVEN = 'claimeven'
    BASEINVERSE = 'baseinverse'
    VERTICAL = 'vertical'
    AFTEREVEN = 'aftereven'
    LOWINVERSE = 'lowinverse'
    HIGHINVERSE = 'highinverse'
    BASECLAIM = 'baseclaim'
    BEFORE = 'before'
    SPECIALBEFORE = 'specialbefore'


@dataclass(frozen=True)
class ThesisReference:
    section: str
    pages: tuple[int, int]  # Inclusive thesis / 1-based PDF pages in the local edition.


@dataclass(frozen=True)
class Prerequisites:
    """Local requirements only. Excludes the still-unverified evaluation framework."""

    empty_squares: tuple[Square, ...]
    directly_playable_squares: tuple[Square, ...]
    upper_parity: Literal['even', 'odd'] | None
    vertically_adjacent: bool


@dataclass(frozen=True)
class RuleCandidate:
    """A local rule shape; construction alone does not check any board.

    Vertical pairs have explicit (lower, upper) roles. Baseinverse is unordered
    and normalized. Geometry alone is never authority to execute a response.
    """

    rule: RuleName
    squares: tuple[Square, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rule, RuleName):
            raise ValueError('rule must be a RuleName')
        if self.rule not in (RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL):
            raise ValueError('rule is not implemented in Phase 6A')
        squares = tuple(self.squares)
        if len(squares) != 2 or len(set(squares)) != 2:
            raise ValueError('these rules require two distinct squares')
        if self.rule == RuleName.BASEINVERSE:
            squares = tuple(sorted(squares))
            if squares[0].column == squares[1].column:
                raise ValueError('two landing squares cannot share a column')
        else:
            lower, upper = squares
            even = self.rule == RuleName.CLAIMEVEN
            if (lower.column != upper.column or upper.row != lower.row + 1
                    or upper.is_even != even):
                raise ValueError('invalid lower/upper adjacency or upper-square parity')
        object.__setattr__(self, 'squares', squares)

    @property
    def prerequisites(self) -> Prerequisites:
        if self.rule == RuleName.BASEINVERSE:
            return Prerequisites(self.squares, self.squares, None, False)
        parity: Literal['even', 'odd'] = 'even' if self.rule == RuleName.CLAIMEVEN else 'odd'
        return Prerequisites(self.squares, (), parity, True)

    @property
    def affected_squares(self) -> frozenset[Square]:
        """Both response squares, not just the square used by coverage."""
        return frozenset(self.squares)

    @property
    def coverage_squares(self) -> frozenset[Square]:
        """A group must contain EVERY one of these squares to be conditionally solved."""
        if self.rule == RuleName.CLAIMEVEN:
            return frozenset((self.squares[1],))
        return self.affected_squares

    @property
    def depends_on_zugzwang(self) -> bool:
        return self.rule == RuleName.CLAIMEVEN  # §7.2: BI and VE are independent.

    @property
    def reference(self) -> ThesisReference:
        return {
            RuleName.CLAIMEVEN: ThesisReference('6.1', (36, 37)),
            RuleName.BASEINVERSE: ThesisReference('6.2', (37, 38)),
            RuleName.VERTICAL: ThesisReference('6.3', (38, 39)),
        }[self.rule]

    def reflected(self) -> 'RuleCandidate':
        return RuleCandidate(self.rule, tuple(s.reflected() for s in self.squares))


def _vertical_pairs(position: Position, rule: RuleName) -> tuple[RuleCandidate, ...]:
    if position.terminal:
        return ()
    upper_rows = (2, 4, 6) if rule == RuleName.CLAIMEVEN else (3, 5)
    result = []
    for column in range(COLUMNS):
        for upper_row in upper_rows:
            lower = Square(ROWS + 1 - upper_row, column)
            upper = Square(ROWS - upper_row, column)
            if position.is_empty(lower) and position.is_empty(upper):
                result.append(RuleCandidate(rule, (lower, upper)))
    return tuple(result)


def enumerate_claimevens(position: Position) -> tuple[RuleCandidate, ...]:
    """§6.1: empty vertical pairs, even upper row; lower need not be playable."""
    return _vertical_pairs(position, RuleName.CLAIMEVEN)


def enumerate_baseinverses(position: Position) -> tuple[RuleCandidate, ...]:
    """§6.2: all pairs of distinct landing squares, INCLUDING zero-coverage pairs."""
    if position.terminal:
        return ()
    return tuple(RuleCandidate(RuleName.BASEINVERSE, pair)
                 for pair in combinations(position.playable_squares, 2))


def enumerate_verticals(position: Position) -> tuple[RuleCandidate, ...]:
    """§6.3 standalone rule: empty vertical pairs with an odd upper row."""
    return _vertical_pairs(position, RuleName.VERTICAL)


def enumerate_candidates(position: Position) -> tuple[RuleCandidate, ...]:
    """Stable CL, BI, VE order. No turn, threat-region, or compatibility inference."""
    return (enumerate_claimevens(position) + enumerate_baseinverses(position)
            + enumerate_verticals(position))
