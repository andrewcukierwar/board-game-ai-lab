"""Independent, deliberately tiny Connect 4 oracle using only the standard library.

No engine, grounding, Victor, AlphaZero or neural imports. Top-first matrices are
converted to seven-bit columns (six cells plus an unused sentinel). Exact negamax
visits EVERY legal child, caches only completed values, and returns no partial
value after a cutoff. Replay is the historical reachability witness.
"""
from dataclasses import dataclass

Matrix = tuple[tuple[str, ...], ...]
MAX_REMAINING = 10
MAX_POSITIONS = 1_000_000


def has_four(bits: int) -> bool:
    """Bit arithmetic, independent of production four-cell window geometry."""
    for shift in (1, 7, 6, 8):
        adjacent = bits & (bits >> shift)
        if adjacent & (adjacent >> (2 * shift)):
            return True
    return False


@dataclass(frozen=True)
class EndgamePosition:
    white: int
    black: int
    heights: tuple[int, ...]
    turn: int

    @classmethod
    def from_board(cls, board, turn):
        if type(turn) is not int or turn not in (0, 1):
            raise ValueError('turn must be integer 0 or 1')
        if (type(board) not in (list, tuple) or len(board) != 6
                or any(type(row) not in (list, tuple) or len(row) != 7 for row in board)
                or any(type(cell) is not str or cell not in (' ', 'X', 'O')
                       for row in board for cell in row)):
            raise ValueError('requires a 6 by 7 matrix of spaces/X/O')
        bits = [0, 0]
        heights = []
        for c in range(7):
            height = 0
            gap = False
            for h in range(6):
                cell = board[5 - h][c]
                if cell == ' ':
                    gap = True
                else:
                    if gap:
                        raise ValueError('gravity violation')
                    bits['XO'.index(cell)] |= 1 << (7 * c + h)
                    height += 1
            heights.append(height)
        if bits[0].bit_count() - bits[1].bit_count() != turn:
            raise ValueError('counts/turn disagree')
        winners = [p for p in (0, 1) if has_four(bits[p])]
        if len(winners) > 1 or (winners and winners[0] != 1 - turn):
            raise ValueError('inconsistent terminal winner')
        if winners:
            p = winners[0]
            if not any(bits[p] & (bit := 1 << (7 * c + h - 1))
                       and not has_four(bits[p] ^ bit)
                       for c, h in enumerate(heights) if h):
                raise ValueError('win predates last move')
        return cls(*bits, tuple(heights), turn)

    @property
    def board(self) -> Matrix:
        return tuple(tuple('X' if self.white & (1 << (7 * c + 5 - r)) else
                           'O' if self.black & (1 << (7 * c + 5 - r)) else ' '
                           for c in range(7)) for r in range(6))

    @property
    def remaining(self):
        return 42 - sum(self.heights)

    @property
    def winner(self):
        if has_four(self.white):
            return 0
        if has_four(self.black):
            return 1
        return None

    @property
    def terminal(self):
        return self.winner is not None or self.remaining == 0

    @property
    def legal_columns(self):
        return () if self.terminal else tuple(c for c, h in enumerate(self.heights) if h < 6)

    def landing(self, column):
        if type(column) is not int or column not in self.legal_columns:
            raise ValueError('illegal or post-terminal move')
        return 5 - self.heights[column], column

    def drop(self, column):
        self.landing(column)
        bit = 1 << (7 * column + self.heights[column])
        heights = list(self.heights)
        heights[column] += 1
        return EndgamePosition(self.white | (bit if self.turn == 0 else 0),
                               self.black | (bit if self.turn == 1 else 0),
                               tuple(heights), 1 - self.turn)


def replay(moves) -> EndgamePosition:
    """Existential reachability proof: legal alternating play, stop at first win/draw."""
    if type(moves) not in (tuple, list) or len(moves) > 42:
        raise ValueError('replay must contain at most 42 columns')
    position = EndgamePosition(0, 0, (0,) * 7, 0)
    for column in moves:
        position = position.drop(column)
    return position


def verify_replay(board, turn, moves) -> EndgamePosition:
    declared = EndgamePosition.from_board(board, turn)
    played = replay(moves)
    if declared != played:
        raise ValueError('replay differs from exact board/turn')
    return played


@dataclass(frozen=True)
class OracleResult:
    status: str
    turn: int
    mover_value: int | None
    move_values: tuple[tuple[int, int], ...]
    visited_positions: int
    cache_hits: int
    max_remaining: int
    position_budget: int

    def value_for(self, player):
        if type(player) is not int or player not in (0, 1):
            raise ValueError('perspective must be integer 0 or 1')
        if self.mover_value is None:
            return None
        return self.mover_value if player == self.turn else -self.mover_value


class _Cutoff(Exception):
    pass


def solve(position: EndgamePosition, *, max_remaining=8, position_budget=100_000):
    """Exact {-1,0,1} for the mover; unknown for either strict resource cutoff.

    Hard ceilings prevent accidentally using this helper as a production solver.
    Terminal positions need no endgame cap but still consume one position. The
    budget counts unique entered positions (including terminals); cache storage
    cannot exceed it. All root moves are valued from the ROOT mover's perspective.
    """
    if type(max_remaining) is not int or not 0 <= max_remaining <= MAX_REMAINING:
        raise ValueError('max_remaining must be integer 0..10')
    if type(position_budget) is not int or not 0 <= position_budget <= MAX_POSITIONS:
        raise ValueError('position_budget must be integer 0..1000000')
    if type(position) is not EndgamePosition:
        raise ValueError('requires EndgamePosition')
    if (type(position.white) is not int or position.white < 0
            or type(position.black) is not int or position.black < 0
            or type(position.heights) is not tuple or len(position.heights) != 7
            or any(type(h) is not int or not 0 <= h <= 6 for h in position.heights)):
        raise ValueError('noncanonical bitboard fields')
    # Reject forged bitboards/heights as well as malformed direct construction.
    if EndgamePosition.from_board(position.board, position.turn) != position:
        raise ValueError('noncanonical bitboard position')
    visited = hits = 0
    cache = {}

    def result(status, value=None, moves=()):
        return OracleResult(status, position.turn, value, moves, visited, hits,
                            max_remaining, position_budget)

    if not position.terminal and position.remaining > max_remaining:
        return result('unknown_remaining_cap')

    def visit(p):
        nonlocal visited, hits
        if p in cache:
            hits += 1
            return cache[p]
        if visited >= position_budget:
            raise _Cutoff
        visited += 1
        winner = p.winner
        if winner is not None:
            value = 1 if winner == p.turn else -1
        elif p.remaining == 0:
            value = 0
        else:
            value = max(-visit(p.drop(c)) for c in p.legal_columns)
        cache[p] = value
        return value

    try:
        value = visit(position)
        moves = tuple((c, -visit(position.drop(c))) for c in position.legal_columns)
    except _Cutoff:
        return result('unknown_position_budget')
    return result('exact', value, moves)
