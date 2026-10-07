"""Production depth-limited Negamax with exact terminals and bound-typed caching.

Adapted from the read-only Phase 4 corrected reference, with compact bitboards
for synchronous public play. The legacy open-window heuristic is unchanged on
nonterminal leaves. No research module, checkpoint, or neural dependency is used.
"""
from math import inf

WIN_SCORE = 1_000_000
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)
EXACT, LOWER, UPPER = 'exact', 'lower', 'upper'
WEIGHTS = (0, 1, 3, 9, 81)
# Seven bits per column: six playable cells plus an always-empty sentinel.
WINDOWS = tuple(
    sum(1 << ((col + i * dc) * 7 + row + i * dr) for i in range(4))
    for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1))
    for col in range(7) for row in range(6)
    if 0 <= col + 3 * dc < 7 and 0 <= row + 3 * dr < 6)


def has_four(bits):
    for shift in (1, 7, 6, 8):
        pairs = bits & (bits >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


class SearchState:
    """Detached engine position; play/undo never touches the caller's board."""
    __slots__ = ('pieces', 'heights', 'mover', 'count')

    def __init__(self, game):
        self.pieces = [0, 0]
        self.heights = [0] * 7
        self.mover = game.current_player
        for col in range(7):
            for row in range(6):
                piece = game.board[5 - row][col]
                if piece != ' ':
                    self.pieces[0 if piece == 'X' else 1] |= 1 << (7 * col + row)
                    self.heights[col] += 1
        self.count = sum(self.heights)

    def legal(self):
        return [col for col in CENTER_ORDER if self.heights[col] < 6]

    def play(self, col):
        self.pieces[self.mover] |= 1 << (7 * col + self.heights[col])
        self.heights[col] += 1
        self.count += 1
        self.mover = 1 - self.mover

    def undo(self, col):
        self.mover = 1 - self.mover
        self.count -= 1
        self.heights[col] -= 1
        self.pieces[self.mover] ^= 1 << (7 * col + self.heights[col])

    def terminal_value(self, depth):
        if has_four(self.pieces[self.mover]):
            return WIN_SCORE + depth
        if has_four(self.pieces[1 - self.mover]):
            return -(WIN_SCORE + depth)
        if self.count == 42:
            return 0
        return None

    def heuristic(self):
        own, other = self.pieces[self.mover], self.pieces[1 - self.mover]
        score = 0
        for window in WINDOWS:
            mine, theirs = own & window, other & window
            if not theirs:
                score += WEIGHTS[mine.bit_count()]
            if not mine:
                score -= WEIGHTS[theirs.bit_count()]
        return score


class SearchTable:
    def __init__(self):
        self.entries = {}
        self.nodes = 0
        self.hits = 0
        self.cutoffs = 0


def negamax(state, depth, alpha=-inf, beta=inf, table=None):
    """Fail-soft alpha-beta value from the current mover's perspective."""
    if table is not None:
        table.nodes += 1
    terminal = state.terminal_value(depth)
    if terminal is not None:
        return terminal
    if depth == 0:
        return state.heuristic()
    key = (*state.pieces, state.mover, depth)
    alpha_original, beta_original = alpha, beta
    if table is not None and key in table.entries:
        flag, value = table.entries[key]
        table.hits += 1
        if flag == EXACT:
            return value
        if flag == LOWER:
            alpha = max(alpha, value)
        else:
            beta = min(beta, value)
        if alpha >= beta:
            return value
    best = -inf
    for col in state.legal():
        state.play(col)
        try:
            value = -negamax(state, depth - 1, -beta, -alpha, table)
        finally:
            state.undo(col)
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            if table is not None:
                table.cutoffs += 1
            break
    if table is not None:
        flag = UPPER if best <= alpha_original else LOWER if best >= beta_original else EXACT
        table.entries[key] = (flag, best)
    return best


class NegamaxAgent:
    def __init__(self, depth):
        if type(depth) is not int or depth < 1:
            raise ValueError('depth must be a positive integer')
        self.depth = depth
        self.last_scores = None
        self.last_stats = None

    def __str__(self):
        return f'Negamax Agent {self.depth}'

    __repr__ = __str__

    def score_moves(self, game):
        """Exact depth-limited values for ALL legal root moves; fresh table."""
        state = SearchState(game)
        if state.terminal_value(self.depth) is not None:
            raise ValueError('Cannot choose a move from a terminal position')
        table, scores = SearchTable(), {}
        for col in state.legal():
            state.play(col)
            try:
                scores[col] = -negamax(state, self.depth - 1, -inf, inf, table)
            finally:
                state.undo(col)
        self.last_scores = scores
        self.last_stats = dict(nodes=table.nodes, entries=len(table.entries),
                               hits=table.hits, cutoffs=table.cutoffs)
        return scores

    def choose_move(self, game):
        scores = self.score_moves(game)
        # Dict insertion follows CENTER_ORDER; exact ties stay deterministic.
        return max(scores, key=scores.get)
