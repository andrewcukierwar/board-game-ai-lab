"""Detached MCTS state and uniform random rollouts, using only the stdlib.

Seven bits per column: six playable cells, bottom first, then an empty sentinel.
Shifts 1/7/6/8 match Negamax and Victor geometry. No search/pruning code is
imported from either agent. Inputs are legal, gravity-respecting engine boards.
"""
from dataclasses import dataclass
from numbers import Integral

CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)
BOTTOM = tuple(1 << (7 * col) for col in range(7))
TOP = tuple(1 << (7 * col + 5) for col in range(7))
COLUMN = tuple(63 << (7 * col) for col in range(7))
BOARD_MASK = sum(COLUMN)


def has_four(bits):
    for shift in (1, 7, 6, 8):
        pairs = bits & (bits >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


@dataclass(frozen=True, slots=True)
class BitboardState:
    """Immutable tree snapshot; winner=-1 means no winner (including draws).

    Cache the outcome once at conversion/drop, rather than rescanning at every
    selection step. Keep the engine's turn flip even after a winning move.
    """
    pieces: tuple[int, int]
    occupied: int
    current_player: int
    winner: int = -1

    @classmethod
    def from_game(cls, game):
        if isinstance(game, cls):
            return game
        pieces = [0, 0]
        for col in range(7):
            for row in range(6):
                cell = game.board[5 - row][col]
                if cell in ('X', 'O'):
                    pieces[cell == 'O'] |= 1 << (7 * col + row)
        winner = 0 if has_four(pieces[0]) else 1 if has_four(pieces[1]) else -1
        return cls(tuple(pieces), pieces[0] | pieces[1], game.current_player, winner)

    @property
    def board(self):
        """Read-only array view for diagnostics/legacy private-node inspection.

        Search never materializes this view.
        """
        return tuple(tuple('X' if self.pieces[0] & (1 << (7 * col + 5 - row)) else
                           'O' if self.pieces[1] & (1 << (7 * col + 5 - row)) else ' '
                           for col in range(7)) for row in range(6))

    def get_valid_moves(self):
        # Like the engine, physical availability includes post-win columns.
        return [col for col in CENTER_ORDER if not self.occupied & TOP[col]]

    def is_valid_move(self, col):
        return (isinstance(col, Integral) and not isinstance(col, bool)
                and 0 <= col < 7 and not self.occupied & TOP[col])

    def check_winner(self):
        return self.winner

    def is_game_over(self):
        return self.winner != -1 or self.occupied == BOARD_MASK

    def drop(self, col):
        """Return a new position; reject invalid and post-terminal moves."""
        if self.is_game_over() or not self.is_valid_move(col):
            raise ValueError('Invalid or terminal move')
        move = (self.occupied + BOTTOM[col]) & COLUMN[col]
        mover = self.current_player
        own = self.pieces[mover] | move
        pieces = (own, self.pieces[1]) if mover == 0 else (self.pieces[0], own)
        return BitboardState(pieces, self.occupied | move, 1 - mover,
                             mover if has_four(own) else -1)


def random_rollout(state, rng):
    """Same uniform legal-column policy/order as the engine rollout.

    Work in local integers; allocate no state objects or boards per simulated
    move. Generate legal columns once and remove a column only when it fills.
    """
    if state.is_game_over():
        return state.winner
    occupied = state.occupied
    pieces = list(state.pieces)
    mover = state.current_player
    legal = state.get_valid_moves()
    choice = rng.choice
    remaining = 42 - occupied.bit_count()
    while legal:
        col = choice(legal)
        move = (occupied + BOTTOM[col]) & COLUMN[col]
        own = pieces[mover] | move
        if has_four(own):
            return mover
        pieces[mover] = own
        occupied |= move
        remaining -= 1
        if not remaining:
            return -1
        if move & TOP[col]:
            legal.remove(col)
        mover = 1 - mover
    return -1
