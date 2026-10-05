"""Corrected, separately versioned depth-limited Negamax reference opponent (torch-free).

Fixes the two defects the Phase 4D.3 review verified in the legacy
``agents.negamax_agent.NegamaxAgent`` (which stays unchanged as a labelled
historical opponent):

1. **Cache semantics.** The transposition table stores EXACT / LOWER / UPPER
   bound types keyed by (board, side to move, remaining depth). A fail-high or
   fail-low result is never reused as an exact value under another window.
2. **Terminal scoring.** A completed four is an exact dominant score
   ``±(WIN_SCORE + remaining depth)`` (faster wins and slower losses preferred);
   a full board without four is exactly 0. Every nonterminal heuristic is
   strictly smaller in magnitude than WIN_SCORE.

The nonterminal leaf heuristic is the legacy window heuristic (open windows
weighted 1/3/9 for 1/2/3 own pieces, minus the opponent's), from the mover's
perspective. Each root decision computes every legal move's exact depth-limited
score with a full window and a fresh table; exact ties are broken uniformly by
an injected RNG, or center-first when no RNG is given.
"""
import math

WIDTH, HEIGHT = 7, 6
VERSION = "connect4-reference-negamax-v1"
WIN_SCORE = 1_000_000
WEIGHTS = (0, 1, 3, 9)
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)
EXACT, LOWER, UPPER = "exact", "lower", "upper"


def _cell(col, row):
    return col * HEIGHT + row


WINDOWS = tuple(
    tuple(_cell(col + i * dc, row + i * dr) for i in range(4))
    for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1))
    for col in range(WIDTH) for row in range(HEIGHT)
    if 0 <= col + 3 * dc < WIDTH and 0 <= row + 3 * dr < HEIGHT)
WINDOWS_THROUGH = tuple(tuple(w for w in WINDOWS if cell in w) for cell in range(WIDTH * HEIGHT))


class State:
    """Flat cells (col-major, row 0 = bottom): 0 empty, 1 = X, 2 = O; ``mover`` is 1 or 2."""
    __slots__ = ("cells", "heights", "mover", "count")

    def __init__(self, cells, heights, mover, count):
        self.cells, self.heights, self.mover, self.count = cells, heights, mover, count

    @classmethod
    def from_moves(cls, moves):
        state = cls([0] * (WIDTH * HEIGHT), [0] * WIDTH, 1, 0)
        for move in moves:
            if state.has_four() or not state.play(move):
                raise ValueError("Illegal history or history continues after a win")
        return state

    @classmethod
    def from_game(cls, game):
        """From an engine Connect4 (top row first, 'X'/'O'/' ')."""
        cells, heights = [0] * (WIDTH * HEIGHT), [0] * WIDTH
        for col in range(WIDTH):
            for row in range(HEIGHT):
                piece = game.board[HEIGHT - 1 - row][col]
                if piece == " ":
                    break
                cells[_cell(col, row)] = 1 if piece == "X" else 2
                heights[col] += 1
            if any(game.board[HEIGHT - 1 - r][col] != " " for r in range(heights[col], HEIGHT)):
                raise ValueError("Floating piece in engine board")
        count = sum(heights)
        mover = 1 if game.current_player == 0 else 2
        if sum(c == 1 for c in cells) - sum(c == 2 for c in cells) != (0 if mover == 1 else 1):
            raise ValueError("Engine board piece counts disagree with the player to move")
        return cls(cells, heights, mover, count)

    def legal(self):
        return [c for c in CENTER_ORDER if self.heights[c] < HEIGHT]

    def play(self, col):
        if type(col) is not int or not 0 <= col < WIDTH or self.heights[col] >= HEIGHT:
            return False
        cell = _cell(col, self.heights[col])
        self.cells[cell] = self.mover
        self.heights[col] += 1
        self.count += 1
        self.mover = 3 - self.mover
        return True

    def undo(self, col):
        self.heights[col] -= 1
        self.cells[_cell(col, self.heights[col])] = 0
        self.count -= 1
        self.mover = 3 - self.mover

    def wins_at(self, col):
        """Whether the piece just placed in ``col`` completed four."""
        cell = _cell(col, self.heights[col] - 1)
        owner = self.cells[cell]
        return any(all(self.cells[c] == owner for c in window) for window in WINDOWS_THROUGH[cell])

    def has_four(self):
        return any(self.cells[w[0]] and all(self.cells[c] == self.cells[w[0]] for c in w) for w in WINDOWS)

    def key(self):
        return (self.mover, tuple(self.cells))

    def heuristic(self):
        """Open-window score for the mover; |score| < 69 * 9 < WIN_SCORE."""
        own, other, score = self.mover, 3 - self.mover, 0
        cells = self.cells
        for window in WINDOWS:
            mine = theirs = 0
            for c in window:
                value = cells[c]
                if value == own:
                    mine += 1
                elif value == other:
                    theirs += 1
            if not theirs:
                score += WEIGHTS[mine] if mine < 4 else 0
            if not mine:
                score -= WEIGHTS[theirs] if theirs < 4 else 0
        return score


class Table:
    """Bound-typed transposition table: key -> (flag, value)."""

    def __init__(self):
        self.entries = {}
        self.hits = 0

    def get(self, key):
        return self.entries.get(key)

    def put(self, key, flag, value):
        self.entries[key] = (flag, value)


def negamax(state, depth, alpha, beta, table=None, last_move=None):
    """Depth-limited negamax score for state.mover with correct bound-typed caching.

    ``last_move`` is the column just played into ``state`` (None at a root whose
    previous move is known not to have won).
    """
    if last_move is not None and state.wins_at(last_move):
        return -(WIN_SCORE + depth)
    if state.count == WIDTH * HEIGHT:
        return 0
    if depth == 0:
        return state.heuristic()
    key = (state.key(), depth)
    alpha_original = alpha
    if table is not None:
        entry = table.get(key)
        if entry is not None:
            flag, value = entry
            if flag == EXACT:
                table.hits += 1
                return value
            if flag == LOWER and value > alpha:
                alpha = value
            elif flag == UPPER and value < beta:
                beta = value
            if alpha >= beta:
                table.hits += 1
                return value
    best = -math.inf
    for col in state.legal():
        state.play(col)
        try:
            value = -negamax(state, depth - 1, -beta, -alpha, table, col)
        finally:
            state.undo(col)
        if value > best:
            best = value
        if value > alpha:
            alpha = value
        if alpha >= beta:
            break
    if table is not None:
        flag = UPPER if best <= alpha_original else LOWER if best >= beta else EXACT
        table.put(key, flag, best)
    return best


def minimax(state, depth, last_move=None):
    """Plain depth-limited negamax: no pruning, no table (verification oracle)."""
    if last_move is not None and state.wins_at(last_move):
        return -(WIN_SCORE + depth)
    if state.count == WIDTH * HEIGHT:
        return 0
    if depth == 0:
        return state.heuristic()
    best = -math.inf
    for col in state.legal():
        state.play(col)
        try:
            best = max(best, -minimax(state, depth - 1, col))
        finally:
            state.undo(col)
    return best


def root_scores(state, depth):
    """Exact depth-limited score of every legal root move (full window, fresh table)."""
    if depth < 1 or type(depth) is not int:
        raise ValueError("depth must be a positive integer")
    if state.has_four() or state.count == WIDTH * HEIGHT:
        raise ValueError("Cannot search a terminal position")
    table, scores = Table(), {}
    for col in state.legal():
        state.play(col)
        try:
            scores[col] = -negamax(state, depth - 1, -math.inf, math.inf, table, col)
        finally:
            state.undo(col)
    return scores


class ReferenceNegamaxAgent:
    """Corrected depth-limited Negamax opponent; see module docstring for the contract."""
    version = VERSION

    def __init__(self, depth, *, rng=None):
        if type(depth) is not int or depth < 1:
            raise ValueError("depth must be a positive integer")
        self.depth, self.rng = depth, rng
        self.last_scores = None

    def __str__(self):
        return f"Reference Negamax depth {self.depth} ({VERSION})"

    def choose_move(self, game):
        scores = root_scores(State.from_game(game), self.depth)
        best = max(scores.values())
        tied = [col for col in CENTER_ORDER if scores.get(col) == best]
        self.last_scores = scores
        return tied[0] if self.rng is None else self.rng.choice(tied)

    def describe(self):
        return dict(version=VERSION, depth=self.depth, win_score=WIN_SCORE, heuristic_weights=list(WEIGHTS),
                    ties="seeded uniform" if self.rng is not None else "center-first",
                    table="fresh per decision; EXACT/LOWER/UPPER bounds keyed by board, mover, depth")
