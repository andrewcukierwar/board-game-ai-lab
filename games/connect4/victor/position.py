"""Detached position snapshots with the existing basic legality checks."""
from dataclasses import dataclass
from typing import Literal, Sequence

from games.connect4.grounding.analysis import outcome, validate_position
from .geometry import ALL_GROUPS, COLUMNS, ROWS, Group, Square

Player = Literal[0, 1]  # X/White, O/Black
Matrix = tuple[tuple[str, ...], ...]


def validate_player(player: int) -> None:
    if type(player) is not int or player not in (0, 1):
        raise ValueError('player must be 0 (X/White) or 1 (O/Black)')


@dataclass(frozen=True)
class Position:
    """Basic legality, NOT a proof of historical reachability. No board mutation.

    Lists/tuples (including the engine Board) are accepted and frozen on entry.
    Full replay validation belongs to the future proof certificate boundary.
    """

    board: Matrix
    player_to_move: Player

    def __post_init__(self) -> None:
        validate_position(self.board, self.player_to_move)
        object.__setattr__(self, 'board', tuple(tuple(row) for row in self.board))

    @classmethod
    def from_board(cls, board: Sequence[Sequence[str]], player_to_move: Player) -> 'Position':
        # Validate before conversion so strings/generators cannot bypass the contract.
        validate_position(board, player_to_move)
        return cls(tuple(tuple(row) for row in board), player_to_move)

    @property
    def terminal(self) -> bool:
        return outcome(self.board)['status'] != 'ongoing'

    def is_empty(self, square: Square) -> bool:
        return self.board[square.row_index][square.column] == ' '

    def is_playable(self, square: Square) -> bool:
        """Physical landing square; terminal status is checked by enumeration."""
        return (self.is_empty(square) and
                (square.row == 1 or self.board[square.row_index + 1][square.column] != ' '))

    @property
    def playable_squares(self) -> tuple[Square, ...]:
        """Physical landing squares ordered by matrix position, even on terminal boards."""
        return tuple(s for r in range(ROWS) for c in range(COLUMNS)
                     if self.is_playable(s := Square(r, c)))

    def potential_groups(self, player: Player) -> tuple[Group, ...]:
        """All groups without the other player's stones (Allis ch.5, p.32)."""
        validate_player(player)
        blocker = 'O' if player == 0 else 'X'
        return tuple(g for g in ALL_GROUPS
                     if all(self.board[s.row_index][s.column] != blocker for s in g.squares))
