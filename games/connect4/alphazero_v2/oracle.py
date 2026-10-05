"""Torch-free exact Connect 4 oracle: immediate tactics and two independent solvers.

Values are game-theoretic outcomes for the player to move: +1 win, 0 draw,
-1 loss (no distance-to-win preference). Two implementations deliberately
share no code beyond the move-history convention:

* ``BitboardSolver`` (method A): bitboard negamax with alpha-beta over the
  three-valued outcome, a transposition table storing explicit lower/upper
  bounds, forced-block and "do not play under an opponent threat" pruning.
  Bounded by a node budget; exceeding it raises ``SolverBudgetExceeded``.
* ``exhaustive_value`` (method B): plain memoized minimax over a column-stack
  representation with no pruning, ordering or threat logic. Only feasible
  for late positions; used to cross-check method A on a subset.

Every history is replayed through the real engine before either solver sees
it (``engine_position``), so labels refer to legal engine states.
"""
from dataclasses import dataclass

from ..connect4 import Connect4

WIDTH, HEIGHT = 7, 6
H1 = HEIGHT + 1
SOLVER_VERSION = "connect4-bitboard-alphabeta-wdl-v1"
EXHAUSTIVE_VERSION = "connect4-column-stack-exhaustive-minimax-v1"
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)


class SolverBudgetExceeded(RuntimeError):
    pass


def engine_position(moves):
    """Replay a history through the engine; reject illegal or post-terminal moves."""
    game = Connect4()
    for move in moves:
        if type(move) is not int or game.is_game_over() or not game.make_move(move):
            raise ValueError(f"Illegal history: {list(moves)}")
    return game


# Shared immediate-tactics helpers (engine-independent column-height scan) ----------------

def _heights(moves):
    heights = [0] * WIDTH
    for move in moves:
        heights[move] += 1
    return heights


def _grid(moves):
    """grid[col][row] = 0 (X) / 1 (O) / None, row 0 = bottom; independent of the engine."""
    grid = [[None] * HEIGHT for _ in range(WIDTH)]
    heights = [0] * WIDTH
    for ply, move in enumerate(moves):
        if not 0 <= move < WIDTH or heights[move] >= HEIGHT:
            raise ValueError("Illegal history for grid scan")
        grid[move][heights[move]] = ply % 2
        heights[move] += 1
    return grid, heights


def _completes_four(grid, col, row, player):
    for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            c, r = col + sign * dc, row + sign * dr
            while 0 <= c < WIDTH and 0 <= r < HEIGHT and grid[c][r] == player:
                count += 1
                c, r = c + sign * dc, r + sign * dr
        if count >= 4:
            return True
    return False


def winning_moves_scan(moves):
    """Columns that immediately win for the player to move (grid scan, ascending)."""
    grid, heights = _grid(moves)
    player = len(moves) % 2
    return [c for c in range(WIDTH) if heights[c] < HEIGHT and _completes_four(grid, c, heights[c], player)]


def legal_moves_scan(moves):
    _, heights = _grid(moves)
    return [c for c in range(WIDTH) if heights[c] < HEIGHT]


def safe_moves_scan(moves):
    """Moves after which the opponent has no immediate win (a move that wins is safe)."""
    result = []
    for move in legal_moves_scan(moves):
        after = list(moves) + [move]
        if move in winning_moves_scan(moves) or len(after) == 42 or not winning_moves_scan(after):
            result.append(move)
    return result


def tactical_label(moves):
    """Model-blind one-reply tactical classification of a nonterminal history.

    ``unique_win``: exactly one immediately winning column. ``unique_safe``:
    no own immediate win, the opponent threatens an immediate win, and exactly
    one legal column leaves no opponent immediate win. Otherwise None.
    """
    wins = winning_moves_scan(moves)
    if len(wins) == 1:
        return "unique_win", wins[0]
    if wins:
        return None, None
    threats = _opponent_threats(moves)
    safe = safe_moves_scan(moves)
    if threats and len(safe) == 1:
        return "unique_safe", safe[0]
    return None, None


def _opponent_threats(moves):
    grid, heights = _grid(moves)
    opponent = 1 - len(moves) % 2
    return [c for c in range(WIDTH) if heights[c] < HEIGHT and _completes_four(grid, c, heights[c], opponent)]


def line_directions(moves, col):
    """Directions ('horizontal','vertical','diagonal_up','diagonal_down') completed by col for the mover."""
    grid, heights = _grid(moves)
    row, player = heights[col], len(moves) % 2
    found = []
    for name, (dc, dr) in (("horizontal", (1, 0)), ("vertical", (0, 1)),
                           ("diagonal_up", (1, 1)), ("diagonal_down", (1, -1))):
        count = 1
        for sign in (1, -1):
            c, r = col + sign * dc, row + sign * dr
            while 0 <= c < WIDTH and 0 <= r < HEIGHT and grid[c][r] == player:
                count += 1
                c, r = c + sign * dc, r + sign * dr
        if count >= 4:
            found.append(name)
    return found


# Method A: bitboard alpha-beta -----------------------------------------------------------

def _bottom_mask():
    mask = 0
    for col in range(WIDTH):
        mask |= 1 << (col * H1)
    return mask


BOTTOM = _bottom_mask()
BOARD_MASK = BOTTOM * ((1 << HEIGHT) - 1)
COLUMN_MASKS = tuple(((1 << HEIGHT) - 1) << (col * H1) for col in range(WIDTH))
TOP_MASKS = tuple(1 << (HEIGHT - 1 + col * H1) for col in range(WIDTH))
BOTTOM_MASKS = tuple(1 << (col * H1) for col in range(WIDTH))


def _alignment(p):
    for shift in (H1, H1 - 1, H1 + 1, 1):
        m = p & (p >> shift)
        if m & (m >> (2 * shift)):
            return True
    return False


def _winning_cells(position, mask):
    """Empty cells (playable or not) that would complete four for ``position``'s owner."""
    r = (position << 1) & (position << 2) & (position << 3)  # vertical
    for shift in (H1, H1 - 1, H1 + 1):
        p = (position << shift) & (position << (2 * shift))
        r |= p & (position << (3 * shift))
        r |= p & (position >> shift)
        p = (position >> shift) & (position >> (2 * shift))
        r |= p & (position << shift)
        r |= p & (position >> (3 * shift))
    return r & (BOARD_MASK ^ mask)


def _popcount(x):
    return bin(x).count("1")


@dataclass
class BitboardState:
    position: int  # stones of the player to move
    mask: int      # all stones
    moves: int     # stones played

    @classmethod
    def from_history(cls, moves):
        position = mask = 0
        for move in moves:
            if not 0 <= move < WIDTH or mask & TOP_MASKS[move]:
                raise ValueError("Illegal history for bitboard")
            position ^= mask
            mask |= mask + BOTTOM_MASKS[move]
        return cls(position, mask, len(moves))

    def can_play(self, col):
        return not self.mask & TOP_MASKS[col]

    def is_winning_move(self, col):
        position = self.position | ((self.mask + BOTTOM_MASKS[col]) & COLUMN_MASKS[col])
        return _alignment(position)

    def played(self, col):
        return BitboardState(self.position ^ self.mask, self.mask | (self.mask + BOTTOM_MASKS[col]), self.moves + 1)

    def key(self):
        return self.position + self.mask


class BitboardSolver:
    """Exact W/D/L for the player to move (method A). ``max_nodes`` bounds work per call."""

    def __init__(self, max_nodes=2_000_000):
        self.max_nodes = max_nodes
        self.table = {}
        self.nodes = 0

    def _negamax(self, state, alpha, beta):
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise SolverBudgetExceeded("Solver node budget exceeded")
        if state.moves == WIDTH * HEIGHT:
            return 0
        for col in range(WIDTH):
            if state.can_play(col) and state.is_winning_move(col):
                return 1
        if state.moves == WIDTH * HEIGHT - 1:
            return 0  # the last move does not win, so the board fills: draw
        possible = (state.mask + BOTTOM) & BOARD_MASK
        opponent_wins = _winning_cells(state.position ^ state.mask, state.mask)
        forced = possible & opponent_wins
        if forced:
            if forced & (forced - 1):
                return -1  # two immediate opponent wins cannot both be blocked
            possible = forced
        possible &= ~(opponent_wins >> 1)  # playing below an opponent threat loses
        if not possible:
            return -1
        key = state.key()
        alpha_original, beta_original = alpha, beta
        lower, upper = self.table.get(key, (-1, 1))
        if lower >= beta:
            return lower
        if upper <= alpha:
            return upper
        alpha, beta = max(alpha, lower), min(beta, upper)
        if alpha >= beta:
            return alpha
        best = -2
        # Order by threats created (more first), then center-first; ordering never changes values.
        order = sorted((col for col in CENTER_ORDER if possible & COLUMN_MASKS[col]),
                       key=lambda col: -_popcount(_winning_cells(
                           state.position | (possible & COLUMN_MASKS[col]), state.mask)))
        for col in order:
            value = -self._negamax(state.played(col), -beta, -alpha)
            if value > best:
                best = value
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        if best <= alpha_original:
            upper = min(upper, best)
        elif best >= beta_original:
            lower = max(lower, best)
        else:
            lower = upper = best
        self.table[key] = (lower, upper)
        return best

    def value(self, moves):
        state = BitboardState.from_history(moves)
        if _alignment(state.position ^ state.mask):
            raise ValueError("Position is already won")
        return self._negamax(state, -1, 1)

    def action_values(self, moves):
        """Exact outcome of every legal action for the player to move, keyed by column."""
        state = BitboardState.from_history(moves)
        result = {}
        for col in range(WIDTH):
            if state.can_play(col):
                result[col] = 1 if state.is_winning_move(col) else (
                    0 if state.moves + 1 == WIDTH * HEIGHT else -self._negamax(state.played(col), -1, 1))
        return result


# Method B: exhaustive memoized minimax (no pruning) ------------------------------------

def exhaustive_value(moves, max_states=3_000_000):
    """Exact W/D/L by full minimax over every reachable state (method B)."""
    grid, heights = _grid(moves)
    if any(_completes_four_existing(grid, c, r) for c in range(WIDTH) for r in range(heights[c])):
        raise ValueError("Position is already won")
    columns = tuple(tuple(grid[c][:heights[c]]) for c in range(WIDTH))
    memo = {}

    def solve(columns, player):
        if columns in memo:
            return memo[columns]
        if len(memo) >= max_states:
            raise SolverBudgetExceeded("Exhaustive state budget exceeded")
        best, any_move = -2, False
        for col in range(WIDTH):
            if len(columns[col]) < HEIGHT:
                any_move = True
                after = columns[:col] + (columns[col] + (player,),) + columns[col + 1:]
                if _stack_wins(after, col, player):
                    value = 1
                elif all(len(c) == HEIGHT for c in after):
                    value = 0
                else:
                    value = -solve(after, 1 - player)
                best = max(best, value)
        if not any_move:
            best = 0
        memo[columns] = best
        return best

    return solve(columns, len(moves) % 2)


def exhaustive_action_values(moves, max_states=3_000_000):
    result = {}
    for col in legal_moves_scan(moves):
        if col in winning_moves_scan(moves):
            result[col] = 1
        elif len(moves) + 1 == WIDTH * HEIGHT:
            result[col] = 0
        else:
            result[col] = -exhaustive_value(list(moves) + [col], max_states)
    return result


def _completes_four_existing(grid, col, row):
    player = grid[col][row]
    if player is None:
        return False
    for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
        if all(0 <= col + i * dc < WIDTH and 0 <= row + i * dr < HEIGHT and grid[col + i * dc][row + i * dr] == player
               for i in range(4)):
            return True
    return False


def _stack_wins(columns, col, player):
    row = len(columns[col]) - 1

    def cell(c, r):
        return columns[c][r] if 0 <= c < WIDTH and 0 <= r < len(columns[c]) else None
    for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            c, r = col + sign * dc, row + sign * dr
            while cell(c, r) == player:
                count += 1
                c, r = c + sign * dc, r + sign * dr
        if count >= 4:
            return True
    return False


def mirror_moves(moves):
    return [WIDTH - 1 - m for m in moves]


def board_key(moves):
    """Canonical board+actor key independent of move order: (actor, columns bottom-up)."""
    grid, heights = _grid(moves)
    return (len(moves) % 2, tuple(tuple(grid[c][:heights[c]]) for c in range(WIDTH)))


def family_key(moves):
    """Reflection family: the lexicographically smaller of a board key and its mirror."""
    key = board_key(moves)
    mirrored = (key[0], key[1][::-1])
    return min(repr(key), repr(mirrored))
