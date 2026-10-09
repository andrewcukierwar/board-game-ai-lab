"""Standalone UCT search with random rollouts and bounded root tactical checks."""

import math
import random

from .mcts_bitboard import BitboardState, random_rollout


class Node:
    """Store reward for the player who made the incoming move.

    Connect4 advances current_player after every move, including a winning move.
    Thus children of a node all score outcomes for that node's player to move,
    and maximizing child UCB is correct at every depth. The root uses the same
    previous-player convention; only its visit count is used for selection.
    ``wins`` is total reward: win=1, draw=0.5, loss=0.
    """

    __slots__ = ('game_state', 'parent', 'move', 'player_just_moved', 'children',
                 'wins', 'visits', 'untried_moves')

    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.player_just_moved = 1 - game_state.current_player
        self.children = {}
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = [] if self.is_terminal() else game_state.get_valid_moves()

    def ucb1(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def is_fully_expanded(self):
        return not self.untried_moves

    def is_terminal(self):
        return self.game_state.is_game_over()


class MCTSAgent:
    """Bounded MCTS; inject ``random.Random(seed)`` as rng for repeatable play.

    Before search, take an immediate win or exclude moves allowing an immediate
    opponent win when an alternative exists. These root-only tactical guards
    are separate from UCT and do not prove safety beyond the next reply.
    """

    def __init__(self, simulation_limit=1000, *, rng=None):
        if isinstance(simulation_limit, bool) or not isinstance(simulation_limit, int):
            raise TypeError('simulation_limit must be a positive integer')
        if simulation_limit <= 0:
            raise ValueError('simulation_limit must be a positive integer')
        self.simulation_limit = simulation_limit
        self.rng = random.Random() if rng is None else rng

    def __str__(self):
        return f"MCTS Agent ({self.simulation_limit} sims)"

    def __repr__(self):
        return self.__str__()

    def choose_move(self, game):
        """Return a legal column without modifying game; reject finished games."""
        root = Node(BitboardState.from_game(game))
        if root.is_terminal() or not root.untried_moves:
            raise ValueError('Cannot choose a move from a terminal position or without legal moves')
        # Respect defensive engine adapters reporting legal-move exhaustion.
        if not isinstance(game, BitboardState) and not game.get_valid_moves():
            raise ValueError('Cannot choose a move from a terminal position or without legal moves')

        winning_moves = self._winning_moves(root.game_state)
        if winning_moves:
            return self.rng.choice(winning_moves)
        root.untried_moves = self._safe_moves(root.game_state) or root.untried_moves

        for _ in range(self.simulation_limit):
            node = root

            # Selection: each child's reward belongs to this node's mover.
            while node.is_fully_expanded() and node.children and not node.is_terminal():
                node = self._select_child(node)

            # Expansion: immutable compact state, with no engine copying.
            if not node.is_terminal() and node.untried_moves:
                move = self.rng.choice(node.untried_moves)
                game_copy = node.game_state.drop(move)
                node.untried_moves.remove(move)
                child = Node(game_copy, parent=node, move=move)
                node.children[move] = child
                node = child

            # Simulation and backpropagation.
            winner = self._simulate(node.game_state)
            self._backpropagate(node, winner)

        # Robust child: visits, not UCB or accumulated reward. With even one
        # simulation, expansion creates a visited legal child. Break ties via rng.
        most_visits = max(child.visits for child in root.children.values())
        return self.rng.choice([
            move for move, child in root.children.items() if child.visits == most_visits
        ])

    def _winning_moves(self, game):
        game = BitboardState.from_game(game)
        if game.is_game_over():
            return []
        moves = []
        for move in game.get_valid_moves():
            after = game.drop(move)
            if after.check_winner() == game.current_player:
                moves.append(move)
        return moves

    def _safe_moves(self, game):
        """Keep moves with no winning opponent reply, including terminal draws."""
        game = BitboardState.from_game(game)
        if game.is_game_over():
            return []
        moves = []
        for move in game.get_valid_moves():
            after = game.drop(move)
            if after.is_game_over() or not self._winning_moves(after):
                moves.append(move)
        return moves

    def _select_child(self, node):
        # All children share the same parent visits. Keep arithmetic and tie
        # order identical to ucb1(), including an RNG call for singleton ties.
        log_visits = math.log(node.visits) if node.visits else 0.0
        best, ties = -math.inf, []
        for move, child in node.children.items():
            score = (child.wins / child.visits + 1.41 * math.sqrt(log_visits / child.visits)
                     if child.visits else math.inf)
            if score > best:
                best, ties = score, [move]
            elif score == best:
                ties.append(move)
        return node.children[self.rng.choice(ties)]

    def _simulate(self, game):
        if not isinstance(game, BitboardState) and not game.get_valid_moves():
            return game.check_winner()
        return random_rollout(BitboardState.from_game(game), self.rng)

    def _backpropagate(self, node, winner):
        while node is not None:
            node.visits += 1
            if winner == -1:
                node.wins += 0.5
            elif winner == node.player_just_moved:
                node.wins += 1
            node = node.parent
