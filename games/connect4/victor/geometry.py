"""Immutable standard-board geometry. No strategic conclusions."""
from dataclasses import dataclass

from games.connect4.grounding.analysis import GROUPS

ROWS, COLUMNS = 6, 7


@dataclass(frozen=True, order=True)
class Square:
    """Matrix coordinates, top first; parity always uses bottom-based ``row``."""

    row_index: int
    column: int

    def __post_init__(self) -> None:
        if (type(self.row_index) is not int or type(self.column) is not int
                or not 0 <= self.row_index < ROWS or not 0 <= self.column < COLUMNS):
            raise ValueError('square must have integer row_index 0..5 and column 0..6')

    @property
    def row(self) -> int:
        return ROWS - self.row_index

    @property
    def name(self) -> str:
        return f'{chr(97 + self.column)}{self.row}'

    @property
    def is_even(self) -> bool:
        return self.row % 2 == 0

    @classmethod
    def from_name(cls, name: str) -> 'Square':
        if (not isinstance(name, str) or len(name) != 2
                or name[0] not in 'abcdefg' or name[1] not in '123456'):
            raise ValueError('square name must be a1..g6')
        return cls(ROWS - int(name[1]), ord(name[0]) - 97)

    def reflected(self) -> 'Square':
        return Square(self.row_index, COLUMNS - 1 - self.column)


@dataclass(frozen=True, order=True)
class Group:
    """Four consecutive collinear squares, canonically ordered by matrix position."""

    squares: tuple[Square, ...]

    def __post_init__(self) -> None:
        squares = tuple(sorted(self.squares))
        if len(squares) != 4 or len(set(squares)) != 4:
            raise ValueError('a group must contain four distinct squares')
        dr = squares[1].row_index - squares[0].row_index
        dc = squares[1].column - squares[0].column
        if ((dr, dc) not in ((0, 1), (1, 0), (1, 1), (1, -1))
                or any((b.row_index - a.row_index, b.column - a.column) != (dr, dc)
                       for a, b in zip(squares, squares[1:]))):
            raise ValueError('group squares must form a consecutive straight line')
        object.__setattr__(self, 'squares', squares)

    def reflected(self) -> 'Group':
        return Group(tuple(s.reflected() for s in self.squares))


# Adapt the existing, tested 69 windows; do not introduce another production scan.
ALL_GROUPS = tuple(sorted(Group(tuple(Square(r, c) for r, c in g)) for g in GROUPS))
