"""Production depth-limited Negamax with exact terminals and bound-typed caching.

Adapted from the read-only Phase 4 corrected reference, with compact bitboards
for synchronous public play. The legacy open-window heuristic is maintained
incrementally and unchanged on nonterminal leaves. No research module,
checkpoint, or neural dependency is used.
"""
from math import inf
from games.connect4.agents.negamax_tt import pack_entry, unpack_entry

WIN_SCORE = 1_000_000
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)
EXACT, LOWER, UPPER = 'exact', 'lower', 'upper'
WEIGHTS = (0, 1, 3, 9, 81)
BOARD_MASK = sum(63 << (7 * col) for col in range(7))
BOTTOM_MASK = sum(1 << (7 * col) for col in range(7))
# Seven bits per column: six playable cells plus an always-empty sentinel.
WINDOWS = tuple(
    sum(1 << ((col + i * dc) * 7 + row + i * dr) for i in range(4))
    for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1))
    for col in range(7) for row in range(6)
    if 0 <= col + 3 * dc < 7 and 0 <= row + 3 * dr < 6)
# Index by the same seven-bit cell address used by the bitboards (sentinels
# have no memberships). A window code is X_count + 5 * O_count: counts 0..4
# are represented exactly, including blocked and both-empty windows.
CELL_WINDOWS = tuple(tuple(i for i, window in enumerate(WINDOWS)
                           if window & (1 << cell)) for cell in range(49))
WINDOW_SCORES = tuple(WEIGHTS[code % 5] if code // 5 == 0 else
                      -WEIGHTS[code // 5] if code % 5 == 0 else 0
                      for code in range(25))
PLAY_DELTAS = tuple(tuple(WINDOW_SCORES[code + step] - WINDOW_SCORES[code]
                         if code % 5 + code // 5 < 4 else 0
                         for code in range(25)) for step in (1, 5))


def has_four(bits):
    for shift in (1, 7, 6, 8):
        pairs = bits & (bits >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


def winning_squares(pieces, occupied):
    """Empty cells completing four, including cells not yet playable.

    Same bit geometry as Victor's search; used ONLY for ordering here. An
    unsupported threat is never a terminal value or a reason to skip a move.
    """
    wins = (pieces << 1) & (pieces << 2) & (pieces << 3)
    for shift in (7, 6, 8):
        pairs = (pieces << shift) & (pieces << (2 * shift))
        wins |= pairs & (pieces << (3 * shift))
        wins |= pairs & (pieces >> shift)
        pairs = (pieces >> shift) & (pieces >> (2 * shift))
        wins |= pairs & (pieces << shift)
        wins |= pairs & (pieces >> (3 * shift))
    return wins & (BOARD_MASK ^ occupied)


class SearchState:
    """Detached engine position; play/undo never touches the caller's board."""
    __slots__ = ('pieces', 'heights', 'mover', 'count', 'window_counts',
                 'score', 'score_history')

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
        self.window_counts = [(self.pieces[0] & window).bit_count() +
                              5 * (self.pieces[1] & window).bit_count()
                              for window in WINDOWS]
        self.score = sum(WINDOW_SCORES[code] for code in self.window_counts)
        self.score_history = []

    def legal(self):
        return [col for col in CENTER_ORDER if self.heights[col] < 6]

    def ordered_moves(self, hint=None, tactical='tactical'):
        """Wins, TT hint, then own threat count; stable center-first ties."""
        moves = self.legal()
        own = self.pieces[self.mover]
        if tactical == 'none' or own.bit_count() < (3 if tactical == 'wins' else 2):
            if hint in moves:
                moves.remove(hint)
                moves.insert(0, hint)
            return moves
        occupied = self.pieces[0] | self.pieces[1]
        wins = winning_squares(own, occupied) if tactical in ('wins', 'tactical') else 0
        if tactical == 'wins':
            if hint in moves:
                moves.remove(hint)
                moves.insert(0, hint)
            # Usually there is no playable win: avoid allocating ranking tuples.
            if wins & ((occupied + BOTTOM_MASK) & BOARD_MASK):
                moves.sort(key=lambda col: bool(wins & (1 << (7 * col + self.heights[col]))),
                           reverse=True)
            return moves
        ranked = []
        for col in moves:
            move = 1 << (7 * col + self.heights[col])
            threats = (winning_squares(own | move, occupied | move).bit_count()
                       if tactical in ('threats', 'tactical') else 0)
            ranked.append((bool(move & wins), col == hint, threats, col))
        # Python's stable sort preserves CENTER_ORDER when all priorities tie.
        ranked.sort(key=lambda item: item[:3], reverse=True)
        return [item[3] for item in ranked]

    def play(self, col):
        cell = 7 * col + self.heights[col]
        counts, deltas = self.window_counts, PLAY_DELTAS[self.mover]
        step = 1 if self.mover == 0 else 5
        score = self.score
        self.score_history.append(score)
        for window in CELL_WINDOWS[cell]:
            code = counts[window]
            score += deltas[code]
            counts[window] = code + step
        self.score = score
        self.pieces[self.mover] |= 1 << cell
        self.heights[col] += 1
        self.count += 1
        self.mover = 1 - self.mover

    def undo(self, col):
        self.mover = 1 - self.mover
        self.count -= 1
        self.heights[col] -= 1
        cell = 7 * col + self.heights[col]
        counts = self.window_counts
        step = 1 if self.mover == 0 else 5
        for window in CELL_WINDOWS[cell]:
            counts[window] -= step
        self.score = self.score_history.pop()
        self.pieces[self.mover] ^= 1 << cell

    def terminal_value(self, depth):
        if has_four(self.pieces[self.mover]):
            return WIN_SCORE + depth
        if has_four(self.pieces[1 - self.mover]):
            return -(WIN_SCORE + depth)
        if self.count == 42:
            return 0
        return None

    def heuristic(self):
        return self.score if self.mover == 0 else -self.score


class SearchTable:
    def __init__(self, *, tt_moves=True, tactical='wins'):
        # Internal switches for measured ablations; public agent settings stay fixed.
        self.tt_moves = tt_moves
        self.tactical = tactical
        self.entries = {}
        self.leaves = self.terminals = self.probes = self.canonicalizations = self.reflected_hits = 0
        self.nodes = 0
        self.hits = 0
        self.cutoffs = 0


def negamax(state, depth, alpha=-inf, beta=inf, table=None):
    """Fail-soft alpha-beta value from the current mover's perspective."""
    if table is not None:
        table.nodes += 1
    terminal = state.terminal_value(depth)
    if terminal is not None:
        if table is not None:
            table.terminals += 1
        return terminal
    if depth == 0:
        if table is not None:
            table.leaves += 1
        return state.heuristic()
    key = (state.pieces[0] | (state.pieces[1] << 49) |
           (state.mover << 98) | (depth << 99))
    if table is not None:
        table.probes += 1
    alpha_original, beta_original = alpha, beta
    hint = None
    entry = table.entries.get(key) if table is not None else None
    if entry is not None:
        signed, flag, hint_code = entry >> 6, entry & 3, (entry >> 2) & 15
        value = signed // 2 if signed & 1 == 0 else -(signed // 2) - 1
        move = None if hint_code == 15 else hint_code
        if flag == 3:
            raise ValueError("Invalid TT entry")
        if table.tt_moves:
            hint = move
        table.hits += 1
        if flag == 0:
            return value
        if flag == 1:
            alpha = max(alpha, value)
        else:
            beta = min(beta, value)
        if alpha >= beta:
            return value
    best, best_move = -inf, None
    # At depth one, threat scoring has no reply horizon and costs more than it
    # saves. TT hints remain useful; leaf evaluation and terminal checks stay exact.
    tactical = table.tactical if table is not None else 'wins'
    for col in state.ordered_moves(hint, tactical if depth > 1 else 'none'):
        state.play(col)
        try:
            value = -negamax(state, depth - 1, -beta, -alpha, table)
        finally:
            state.undo(col)
        if value > best:
            best, best_move = value, col
        alpha = max(alpha, value)
        if alpha >= beta:
            if table is not None:
                table.cutoffs += 1
            break
    if table is not None:
        flag = 2 if best <= alpha_original else 1 if best >= beta_original else 0
        # For bounds this is only a searched-move hint, never an exact value.
        # Nonterminal interior search yields integer score and legal column.
        signed = 2 * best if best >= 0 else -2 * best - 1
        table.entries[key] = (signed << 6) | (best_move << 2) | flag
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
