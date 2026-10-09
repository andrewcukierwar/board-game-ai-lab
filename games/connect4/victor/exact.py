"""Bounded terminal-only alpha-beta. No strategic or heuristic leaf values.

Seven-bit columns keep a sentinel between columns. TT entries carry bounds;
only a completed full-window root search exposes W/D/L and optimal moves.
"""
from dataclasses import dataclass
from math import isfinite
from time import monotonic

from .position import Position

CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)


def four(bits):
    return any((pairs := bits & (bits >> d)) & (pairs >> (2 * d))
               for d in (1, 7, 6, 8))


@dataclass(frozen=True)
class SearchBudget:
    nodes: int = 200_000
    seconds: float | None = 1.0
    max_remaining: int = 14
    table_entries: int = 200_000

    def __post_init__(self):
        for name in ('nodes', 'max_remaining', 'table_entries'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.max_remaining > 24:
            raise ValueError('max_remaining ceiling is 24; no opening solves')
        if (self.seconds is not None and
                (type(self.seconds) not in (int, float) or not isfinite(self.seconds)
                 or self.seconds < 0)):
            raise ValueError('seconds must be finite and nonnegative, or None')


@dataclass(frozen=True)
class Bits:
    pieces: tuple[int, int]
    heights: tuple[int, ...]
    turn: int

    @classmethod
    def from_position(cls, position):
        p = Position.from_board(position.board, position.player_to_move)
        pieces, heights = [0, 0], [0] * 7
        for c in range(7):
            for r in range(6):
                cell = p.board[5 - r][c]
                if cell != ' ':
                    pieces['XO'.index(cell)] |= 1 << (7 * c + r)
                    heights[c] += 1
        return cls(tuple(pieces), tuple(heights), p.player_to_move)

    @property
    def remaining(self):
        return 42 - sum(self.heights)

    @property
    def value(self):
        if four(self.pieces[1 - self.turn]):
            return -1
        if four(self.pieces[self.turn]):
            return 1
        return 0 if not self.remaining else None

    def legal(self):
        return tuple(c for c in CENTER_ORDER if self.heights[c] < 6)

    def drop(self, c):
        pieces, heights = list(self.pieces), list(self.heights)
        pieces[self.turn] |= 1 << (7 * c + heights[c])
        heights[c] += 1
        return Bits(tuple(pieces), tuple(heights), 1 - self.turn)

    def winning(self, player):
        return tuple(c for c in self.legal() if four(
            self.pieces[player] | (1 << (7 * c + self.heights[c]))))

    @property
    def board(self):
        return tuple(tuple('X' if self.pieces[0] & (1 << (7 * c + 5 - r)) else
                           'O' if self.pieces[1] & (1 << (7 * c + 5 - r)) else ' '
                           for c in range(7)) for r in range(6))


@dataclass(frozen=True)
class ExactResult:
    status: str
    value: int | None
    move_values: tuple[tuple[int, int], ...]
    nodes: int
    cache_hits: int
    table_entries: int
    elapsed: float

    @property
    def best_move(self):
        return (max(self.move_values, key=lambda cv: cv[1])[0]
                if self.status == 'exact' and self.move_values else None)


class Cutoff(Exception):
    pass


class Work:
    """Shared hard accounting for exact search and adversarial policy replay."""
    def __init__(self, budget):
        self.budget, self.nodes = budget, 0
        self.start = monotonic()

    def enter(self):
        if self.nodes >= self.budget.nodes:
            raise Cutoff('unknown_node_budget')
        if (self.budget.seconds is not None and
                monotonic() - self.start >= self.budget.seconds):
            raise Cutoff('unknown_time_budget')
        self.nodes += 1


# Integer bitboards for the search core: ``pos`` holds the mover's stones,
# ``mask`` all stones, same seven-bit column layout as ``Bits``.
BOTTOM = sum(1 << (7 * c) for c in range(7))
BOARD = BOTTOM * 63
COLUMN = tuple(63 << (7 * c) for c in range(7))
EXACT, LOWER, UPPER = 0, 1, 2


def winning_squares(pos, mask):
    """Empty board cells on which the stones in ``pos`` would complete a four."""
    r = (pos << 1) & (pos << 2) & (pos << 3)
    for d in (7, 6, 8):
        p = (pos << d) & (pos << 2 * d)
        r |= p & (pos << 3 * d)
        r |= p & (pos >> d)
        p = (pos >> d) & (pos >> 2 * d)
        r |= p & (pos << d)
        r |= p & (pos >> 3 * d)
    return r & (BOARD ^ mask)


def _mirror(b):
    return (((b & 127) << 42) | (((b >> 7) & 127) << 35) | (((b >> 14) & 127) << 28)
            | (b & (127 << 21)) | (((b >> 28) & 127) << 14) | (((b >> 35) & 127) << 7)
            | ((b >> 42) & 127))


class _Search:
    """Terminal-only WDL negamax; every pruning rule below is a game-rule fact.

    - A player who cannot stop two immediate opponent wins, or whose every move
      lets the opponent win at once, loses (value -1).
    - A move directly below an opponent winning square loses at once (the
      mover cannot win immediately), so it is skipped; with no other move the
      player loses.
    - With at most two empty cells, no immediate win for either side means draw.
    - A position and its mirror image have the same value (shared table key).
    Move ordering (most new own threats, then centre) affects only speed.
    """
    def __init__(self, work, budget):
        self.work, self.budget, self.table, self.hits = work, budget, {}, 0

    def visit(self, pos, mask, played, alpha, beta):
        """Precondition: the player to move cannot complete four immediately."""
        self.work.enter()
        possible = (mask + BOTTOM) & BOARD
        threats = winning_squares(pos ^ mask, mask)
        forced = possible & threats
        if forced:
            if forced & (forced - 1):
                return -1
            possible = forced
        moves = possible & ~(threats >> 1)
        if not moves:
            return -1
        if played >= 40:
            return 0
        key = min(pos + mask, _mirror(pos) + _mirror(mask))
        a0, b0 = alpha, beta
        entry = self.table.get(key)
        if entry is not None:
            self.hits += 1
            flag, score = entry
            if flag == EXACT:
                return score
            if flag == LOWER:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            if alpha >= beta:
                return score
        ordered = []
        for c in CENTER_ORDER:
            move = moves & COLUMN[c]
            if move:
                ordered.append((-winning_squares(pos | move, mask | move).bit_count(),
                                len(ordered), move))
        ordered.sort()
        best, opponent = -2, pos ^ mask
        for _, _, move in ordered:
            value = -self.visit(opponent, mask | move, played + 1, -beta, -alpha)
            if value > best:
                best = value
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        break
        flag = UPPER if best <= a0 else LOWER if best >= b0 else EXACT
        if key not in self.table and len(self.table) >= self.budget.table_entries:
            raise Cutoff('unknown_table_budget')
        self.table[key] = (flag, best)
        return best

    def child_value(self, pos, mask, played, column):
        """Exact value, for the root mover, of dropping in ``column``."""
        self.work.enter()
        move = (mask + (1 << (7 * column))) & COLUMN[column]
        if four(pos | move):
            return 1
        if played + 1 == 42:
            return 0
        opponent, mask = pos ^ mask, mask | move
        if winning_squares(opponent, mask) & (mask + BOTTOM) & BOARD:
            return -1
        # Null windows: does the opponent win? does the opponent avoid losing?
        if self.visit(opponent, mask, played + 1, 0, 1) >= 1:
            return -1
        return 1 if self.visit(opponent, mask, played + 1, -1, 0) <= -1 else 0


def _root(position):
    root = Bits.from_position(position)
    pos, mask = root.pieces[root.turn], root.pieces[0] | root.pieces[1]
    return root, pos, mask, 42 - root.remaining


def solve_exact(position: Position, budget: SearchBudget = SearchBudget()):
    """Exact mover-relative {-1,0,+1}; interruption discards ALL partial values.

    Node budgets and ordering are deterministic. A wall-clock deadline is an
    optional safety limit, whose precise stopping point is machine-dependent.
    Table exhaustion stops the search rather than silently growing memory.
    """
    if type(budget) is not SearchBudget:
        raise ValueError('budget must be SearchBudget')
    root, pos, mask, played = _root(position)
    work = Work(budget)
    search = _Search(work, budget)

    def result(status, value=None, moves=()):
        return ExactResult(status, value, moves, work.nodes, search.hits, len(search.table),
                           monotonic() - work.start)

    if root.value is None and root.remaining > budget.max_remaining:
        return result('unknown_remaining_cap')
    try:
        work.enter()  # root, each root child and every recursive entry count, including TT hits
        if root.value is not None:
            return result('exact', root.value)
        # Every root child is solved completely, so every returned move value is exact.
        moves = tuple((c, search.child_value(pos, mask, played, c)) for c in root.legal())
        return result('exact', max(v for _, v in moves), moves)
    except Cutoff as exc:
        return result(str(exc))
