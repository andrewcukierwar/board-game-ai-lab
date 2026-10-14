"""Immutable neural-search states adapted from main's validated MCTS bitboards.

Geometry and drop arithmetic match mcts_bitboard.py at 8704142702da. Each
column uses six bottom-first cells and one empty sentinel. Shifts 1/7/6/8
detect vertical/horizontal/diagonal fours without wrapping between columns.
Only standard gravity-respecting Connect4 inputs are supported.
"""
from dataclasses import dataclass
from numbers import Integral

CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)
BOTTOM = tuple(1 << (7 * col) for col in range(7))
TOP = tuple(1 << (7 * col + 5) for col in range(7))
COLUMN = tuple(63 << (7 * col) for col in range(7))
BOARD_MASK = sum(COLUMN)
# Flattened neural input order: top row first, physical columns ascending.
INPUT_MASKS = tuple(1 << (7 * col + 5 - row) for row in range(6) for col in range(7))


def has_four(bits):
    for shift in (1, 7, 6, 8):
        pairs = bits & (bits >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


@dataclass(frozen=True, slots=True)
class BitboardState:
    """Detached state with cached winner; -1 also represents a draw.

    Like the engine, legal columns describe physical availability even after
    a win, but drop rejects terminal states. The turn flips after every move,
    including a win, so the winning successor has terminal value -1.
    """
    pieces: tuple[int, int]
    occupied: int
    current_player: int
    winner: int = -1

    @classmethod
    def from_game(cls, game):
        if isinstance(game, cls):
            return game
        if type(game.current_player) is not int or game.current_player not in (0, 1):
            raise ValueError("current_player must be X=0 or O=1")
        if game.piece != ('X' if game.current_player == 0 else 'O'):
            raise ValueError("current_player and piece disagree")
        board = tuple(tuple(row) for row in game.board)
        if len(board) != 6 or any(len(row) != 7 for row in board):
            raise ValueError("Expected a 6x7 board")
        pieces = [0, 0]
        for col in range(7):
            empty_seen = False
            for row in range(6):
                cell = board[5 - row][col]
                if cell not in ('X', 'O', ' '):
                    raise ValueError("Expected only X, O and spaces")
                if cell == ' ':
                    empty_seen = True
                else:
                    if empty_seen:
                        raise ValueError("Bitboard input must respect gravity")
                    pieces[cell == 'O'] |= 1 << (7 * col + row)
        # Simultaneous winners are outside legal play; refuse ambiguous scans.
        wins = (has_four(pieces[0]), has_four(pieces[1]))
        if all(wins):
            raise ValueError("Bitboard input cannot have two winners")
        winner = 0 if wins[0] else 1 if wins[1] else -1
        return cls(tuple(pieces), pieces[0] | pieces[1], game.current_player, winner)

    @property
    def piece(self):
        return 'XO'[self.current_player]

    @property
    def board(self):
        """Read-only diagnostic view; search and inference never construct it."""
        return tuple(tuple('X' if self.pieces[0] & mask else
                           'O' if self.pieces[1] & mask else ' '
                           for mask in INPUT_MASKS[row * 7:(row + 1) * 7])
                     for row in range(6))

    def get_valid_moves(self):
        return [col for col in CENTER_ORDER if not self.occupied & TOP[col]]

    def is_valid_move(self, col):
        return (isinstance(col, Integral) and not isinstance(col, bool)
                and 0 <= col < 7 and not self.occupied & TOP[col])

    def check_winner(self):
        return self.winner

    def is_game_over(self):
        return self.winner != -1 or self.occupied == BOARD_MASK

    def drop(self, col):
        if self.is_game_over() or not self.is_valid_move(col):
            raise ValueError("Invalid or terminal move")
        move = (self.occupied + BOTTOM[col]) & COLUMN[col]
        mover = self.current_player
        own = self.pieces[mover] | move
        pieces = (own, self.pieces[1]) if mover == 0 else (self.pieces[0], own)
        return BitboardState(pieces, self.occupied | move, 1 - mover,
                             mover if has_four(own) else -1)
