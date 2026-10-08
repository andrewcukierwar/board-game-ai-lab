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


def solve_exact(position: Position, budget: SearchBudget = SearchBudget()):
    """Exact mover-relative {-1,0,+1}; interruption discards ALL partial values.

    Node budgets and ordering are deterministic. A wall-clock deadline is an
    optional safety limit, whose precise stopping point is machine-dependent.
    Table exhaustion stops the search rather than silently growing memory.
    """
    if type(budget) is not SearchBudget:
        raise ValueError('budget must be SearchBudget')
    root = Bits.from_position(position)
    work, table, hits = Work(budget), {}, 0

    def result(status, value=None, moves=()):
        return ExactResult(status, value, moves, work.nodes, hits, len(table),
                           monotonic() - work.start)

    if root.value is None and root.remaining > budget.max_remaining:
        return result('unknown_remaining_cap')

    def visit(p, alpha, beta):
        nonlocal hits
        work.enter()
        if p.value is not None:
            return p.value
        key = (*p.pieces, p.turn)
        a0, b0 = alpha, beta
        if key in table:
            flag, score = table[key]
            hits += 1
            if flag == 'exact':
                return score
            if flag == 'lower':
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            if alpha >= beta:
                return score
        wins = p.winning(p.turn)
        if wins:
            return 1  # existential terminal witness; maximal possible WDL
        threats = p.winning(1 - p.turn)
        order = sorted(p.legal(), key=lambda c: c not in threats)
        best = -2
        for c in order:
            best = max(best, -visit(p.drop(c), -beta, -alpha))
            alpha = max(alpha, best)
            if alpha >= beta or best == 1:
                break
        flag = 'upper' if best <= a0 else 'lower' if best >= b0 else 'exact'
        if key not in table and len(table) >= budget.table_entries:
            raise Cutoff('unknown_table_budget')
        table[key] = (flag, best)
        return best

    try:
        work.enter()  # root and every recursive entry count, including TT hits
        if root.value is not None:
            return result('exact', root.value)
        # Each child gets a full window, so every returned root move value is exact.
        moves = tuple((c, -visit(root.drop(c), -2, 2)) for c in root.legal())
        return result('exact', max(v for _, v in moves), moves)
    except Cutoff as exc:
        return result(str(exc))
