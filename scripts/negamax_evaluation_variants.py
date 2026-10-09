"""Evaluation-only ablations; all use the unchanged production search body."""
from games.connect4.agents import negamax_agent as engine


class InverseDeltaState(engine.SearchState):
    """Compact counts, computing the inverse score delta instead of a stack."""

    def play(self, col):
        cell = 7 * col + self.heights[col]
        counts, deltas = self.window_counts, engine.PLAY_DELTAS[self.mover]
        step = 1 if self.mover == 0 else 5
        score = self.score
        for window in engine.CELL_WINDOWS[cell]:
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
        counts, deltas = self.window_counts, engine.PLAY_DELTAS[self.mover]
        step = 1 if self.mover == 0 else 5
        score = self.score
        for window in engine.CELL_WINDOWS[cell]:
            code = counts[window] - step
            score -= deltas[code]
            counts[window] = code
        self.score = score
        self.pieces[self.mover] ^= 1 << cell


class ArrayCountsState(engine.SearchState):
    """Separate X/O arrays with branch-based direct score deltas on play."""

    def __init__(self, game):
        super().__init__(game)
        self.counts = [[code % 5 for code in self.window_counts],
                       [code // 5 for code in self.window_counts]]
        self.window_counts = None  # the ablation maintains only the two arrays

    def play(self, col):
        cell = 7 * col + self.heights[col]
        own, other = self.counts[self.mover], self.counts[1 - self.mover]
        score = self.score
        self.score_history.append(score)
        sign = 1 if self.mover == 0 else -1
        weights = engine.WEIGHTS
        for window in engine.CELL_WINDOWS[cell]:
            mine, theirs = own[window], other[window]
            if not theirs:
                score += sign * (weights[mine + 1] - weights[mine])
            elif not mine:
                score += sign * weights[theirs]
            own[window] = mine + 1
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
        own = self.counts[self.mover]
        for window in engine.CELL_WINDOWS[cell]:
            own[window] -= 1
        self.score = self.score_history.pop()
        self.pieces[self.mover] ^= 1 << cell


VARIANTS = {'compact-stack': engine.SearchState,
            'compact-inverse': InverseDeltaState, 'arrays-stack': ArrayCountsState}
