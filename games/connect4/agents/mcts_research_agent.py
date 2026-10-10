"""Research-only MCTS variants for the v3 strength study.

Not registered in the agent factory or any API. The production ``MCTSAgent``
is the baseline and is not modified; with a default ``ResearchConfig`` this
agent reproduces its moves, tree statistics and RNG consumption exactly.
See docs/search-mcts-v3/DESIGN.md for the declared components.
"""
import math
import random
from dataclasses import dataclass

from .mcts_agent import MCTSAgent, Node
from .mcts_bitboard import (
    BOARD_MASK, BOTTOM, COLUMN, BitboardState, random_rollout,
)

BOTTOM_MASK = sum(BOTTOM)
TOP_MASK = sum(1 << (7 * col + 5) for col in range(7))
ROLLOUTS = ('uniform', 'decisive', 'safe')
EXPANSIONS = ('random', 'center')


def winning_cells(bits):
    """Cells that would complete four for ``bits``, ignoring occupancy.

    Callers intersect the result with the playable cells, which removes
    occupied cells, sentinel bits and anything shifted off the board.
    """
    wins = (bits << 1) & (bits << 2) & (bits << 3)
    for shift in (7, 6, 8):
        up, down = bits << shift, bits >> shift
        wins |= up & (bits << 2 * shift) & ((bits << 3 * shift) | down)
        wins |= down & (bits >> 2 * shift) & ((bits >> 3 * shift) | up)
    return wins


def tactical_rollout(state, rng, avoid_gifts=False):
    """Random rollout that takes immediate wins and blocks single threats.

    Two simultaneous opponent threats cannot both be blocked, so the rollout
    ends as a loss for the mover. With ``avoid_gifts`` the uniform choice
    skips columns whose move lets the opponent win directly on top, unless
    every column does. Threat sets make a per-move four check unnecessary:
    a move wins exactly when it lands on one of the mover's winning cells.
    """
    if state.is_game_over():
        return state.winner
    mover = state.current_player
    occupied = state.occupied
    own, opp = state.pieces[mover], state.pieces[1 - mover]
    own_wins, opp_wins = winning_cells(own), winning_cells(opp)
    legal = state.get_valid_moves()
    remaining = 42 - occupied.bit_count()
    choice = rng.choice
    while True:
        playable = (occupied + BOTTOM_MASK) & BOARD_MASK
        if own_wins & playable:
            return mover
        move = opp_wins & playable
        if move:
            if move & (move - 1):
                return 1 - mover
            col = (move.bit_length() - 1) // 7
        else:
            if avoid_gifts and playable & ((opp_wins & BOARD_MASK) >> 1):
                gifts = (opp_wins & BOARD_MASK) >> 1
                safe = [c for c in legal
                        if not (occupied + BOTTOM[c]) & COLUMN[c] & gifts]
                col = choice(safe or legal)
            else:
                col = choice(legal)
            move = (occupied + BOTTOM[col]) & COLUMN[col]
        occupied |= move
        remaining -= 1
        if not remaining:
            return -1
        if move & TOP_MASK:
            legal.remove(col)
        own, opp = opp, own | move
        own_wins, opp_wins = opp_wins, winning_cells(opp)
        mover = 1 - mover


def safe_rollout(state, rng):
    return tactical_rollout(state, rng, True)


@dataclass(frozen=True)
class ResearchConfig:
    """Explicit switches; the defaults are the production algorithm."""
    exploration: float = 1.41
    rollout: str = 'uniform'
    solver: bool = False
    expansion: str = 'random'

    def __post_init__(self):
        if self.rollout not in ROLLOUTS or self.expansion not in EXPANSIONS:
            raise ValueError('Unknown rollout or expansion policy')
        if isinstance(self.exploration, bool) or not isinstance(self.exploration, (int, float)) \
                or not math.isfinite(self.exploration) or self.exploration < 0:
            raise ValueError('exploration must be a finite non-negative number')
        if not isinstance(self.solver, bool):
            raise ValueError('solver must be a boolean')


class ResearchNode(Node):
    """Node with an optional game-theoretic value for ``player_just_moved``.

    ``proven`` is 1.0 (that player wins), 0.5 (draw), 0.0 (loses) or None.
    With ``tactics`` the constructor also resolves one-move tactics for the
    player to move: an immediate win proves the node lost for the previous
    player, two opponent threats prove it won, and a single opponent threat
    leaves the block as the only move worth expanding (every other move loses
    at once, so restricting expansion keeps exhaustive proofs sound).
    """

    __slots__ = ('proven',)

    def __init__(self, game_state, parent=None, move=None, tactics=False):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        mover = game_state.current_player
        self.player_just_moved = 1 - mover
        self.children = {}
        self.wins = 0.0
        self.visits = 0
        self.proven = None
        self.untried_moves = []
        occupied = game_state.occupied
        if game_state.winner != -1:
            self.proven = 1.0
        elif occupied == BOARD_MASK:
            self.proven = 0.5
        elif not tactics:
            self.untried_moves = game_state.get_valid_moves()
        else:
            playable = (occupied + BOTTOM_MASK) & BOARD_MASK
            pieces = game_state.pieces
            if winning_cells(pieces[mover]) & playable:
                self.proven = 0.0
            else:
                threats = winning_cells(pieces[1 - mover]) & playable
                if not threats:
                    self.untried_moves = game_state.get_valid_moves()
                elif threats & (threats - 1):
                    self.proven = 1.0
                else:
                    self.untried_moves = [(threats.bit_length() - 1) // 7]


class ResearchMCTSAgent(MCTSAgent):
    """Configurable MCTS for A/B strength experiments against ``MCTSAgent``."""

    def __init__(self, simulation_limit=1000, *, rng=None, config=None):
        super().__init__(simulation_limit, rng=rng)
        self.config = ResearchConfig() if config is None else config
        if not isinstance(self.config, ResearchConfig):
            raise TypeError('config must be a ResearchConfig')
        self._rollout = dict(uniform=random_rollout, decisive=tactical_rollout,
                             safe=safe_rollout)[self.config.rollout]
        self.last_stats = None

    def __str__(self):
        return f'Research MCTS Agent ({self.simulation_limit} sims, {self.config})'

    __repr__ = __str__

    def choose_move(self, game):
        """Return a legal column without modifying game; reject finished games."""
        config = self.config
        solver = config.solver
        root = ResearchNode(BitboardState.from_game(game))
        self.last_stats = dict(simulations=0, root_proven=None)
        if root.is_terminal() or not root.untried_moves:
            raise ValueError('Cannot choose a move from a terminal position or without legal moves')
        if not isinstance(game, BitboardState) and not game.get_valid_moves():
            raise ValueError('Cannot choose a move from a terminal position or without legal moves')

        # Same root guards as production; unsafe root moves lose immediately,
        # so restricting the root to safe moves keeps solver proofs sound.
        winning_moves = self._winning_moves(root.game_state)
        if winning_moves:
            return self.rng.choice(winning_moves)
        root.untried_moves = self._safe_moves(root.game_state) or root.untried_moves

        rng = self.rng
        rollout = self._rollout
        select = self._select_child_solver if solver else self._select_child
        center_first = config.expansion == 'center'
        executed = 0
        for _ in range(self.simulation_limit):
            node = root
            if solver:
                while node.proven is None and not node.untried_moves and node.children:
                    node = select(node)
                expand = node.proven is None
            else:
                while node.is_fully_expanded() and node.children and not node.is_terminal():
                    node = select(node)
                expand = bool(node.untried_moves) and not node.is_terminal()

            if expand:
                untried = node.untried_moves
                if center_first:
                    move = untried.pop(0)
                else:
                    move = rng.choice(untried)
                    untried.remove(move)
                child = ResearchNode(node.game_state.drop(move), node, move, solver)
                node.children[move] = child
                node = child

            executed += 1
            if solver and node.proven is not None:
                value, mover = node.proven, node.player_just_moved
                self._backpropagate(node, mover if value == 1.0 else
                                    1 - mover if value == 0.0 else -1)
                self._propagate_proof(node)
                if root.proven is not None:
                    break
            else:
                self._backpropagate(node, rollout(node.game_state, rng))

        self.last_stats = dict(simulations=executed, root_proven=root.proven)
        children = root.children
        if solver:
            proven_wins = [move for move, child in children.items() if child.proven == 1.0]
            if proven_wins:
                return rng.choice(proven_wins)
            # Never prefer a proven loss while an alternative was searched.
            children = {move: child for move, child in children.items()
                        if child.proven != 0.0} or children
        most_visits = max(child.visits for child in children.values())
        return rng.choice([move for move, child in children.items()
                           if child.visits == most_visits])

    def _select_child(self, node):
        # Production arithmetic and tie order with a configurable constant.
        exploration = self.config.exploration
        log_visits = math.log(node.visits) if node.visits else 0.0
        best, ties = -math.inf, []
        for move, child in node.children.items():
            score = (child.wins / child.visits + exploration * math.sqrt(log_visits / child.visits)
                     if child.visits else math.inf)
            if score > best:
                best, ties = score, [move]
            elif score == best:
                ties.append(move)
        return node.children[self.rng.choice(ties)]

    def _select_child_solver(self, node):
        """UCB over children that are not proven lost for the mover.

        Called only on unproven, fully expanded nodes, which always have an
        unproven child and no proven-winning child. A proven draw competes
        with its exact value and no exploration bonus.
        """
        exploration = self.config.exploration
        log_visits = math.log(node.visits) if node.visits else 0.0
        best, ties = -math.inf, []
        for move, child in node.children.items():
            proven = child.proven
            if proven is not None:
                if proven == 0.0:
                    continue
                score = proven
            elif child.visits:
                score = child.wins / child.visits + exploration * math.sqrt(log_visits / child.visits)
            else:
                score = math.inf
            if score > best:
                best, ties = score, [move]
            elif score == best:
                ties.append(move)
        return node.children[self.rng.choice(ties)]

    @staticmethod
    def _propagate_proof(node):
        """Back up a proven value: any winning reply, or all replies proven."""
        parent = node.parent
        while parent is not None and parent.proven is None:
            if node.proven == 1.0:
                parent.proven = 0.0
            elif parent.untried_moves:
                return
            else:
                best = 0.0
                for child in parent.children.values():
                    value = child.proven
                    if value is None:
                        return
                    if value > best:
                        best = value
                parent.proven = 1.0 - best
            node, parent = parent, parent.parent
