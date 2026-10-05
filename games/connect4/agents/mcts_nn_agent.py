"""Plain neural PUCT tree, with current-player values and opt-in root tactics."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import random

import numpy as np

from .mcts_agent import MCTSAgent
from ..neural_mcts import (
    Connect4Net, NeuralInference, encode_current_player, legal_policy,
    load_checkpoint, nonnegative_finite, probability_vector,
)


class Node:
    """One parent, one detached state; Q is for this node's player to move."""
    def __init__(self, game_state, parent=None, move=None, prior_p=0.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.prior_p = prior_p
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q_value(self):
        return self.value_sum / self.visits if self.visits else 0.0

    def puct(self, exploration):
        # max(1,N) supplies prior-sensitive selection on the first traversal.
        return (-self.q_value + exploration * self.prior_p
                * math.sqrt(max(1, self.parent.visits)) / (1 + self.visits))


def backup(node, value):
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("Leaf value must be finite and in [-1,1]")
    while node is not None:
        node.visits += 1
        node.value_sum += value
        value = -value
        node = node.parent


def terminal_value(game):
    if not game.is_game_over():
        raise ValueError("Terminal value requires a completed game")
    winner = game.check_winner()
    return 0.0 if winner == -1 else (1.0 if winner == game.current_player else -1.0)


def visit_policy(visits, temperature):
    """N(a) ** (1/tau), with zero support preserved and stable log scaling.

    At tau=0, equal probability among maximum-visit edges; action ties are
    sampled with the injected RNG. This same distribution is the policy target.
    """
    temperature = nonnegative_finite(temperature, "temperature")
    counts = np.asarray(visits, dtype=np.float64)
    if (counts.shape != (7,) or not np.isfinite(counts).all() or (counts < 0).any()
            or (counts != np.floor(counts)).any() or counts.max() <= 0):
        raise ValueError("Expected seven nonnegative integer visits with positive total")
    policy = np.zeros(7, dtype=np.float64)
    if temperature == 0:
        policy[counts == counts.max()] = 1
    else:
        positive = counts > 0
        logs = np.log(counts[positive])
        # Subtract before dividing: maximum stays exactly zero even for tiny tau.
        with np.errstate(over="ignore", under="ignore"):
            policy[positive] = np.exp((logs - logs.max()) / temperature)
    policy /= policy.sum()
    return probability_vector(policy)


def tactical_root_moves(game):
    """Phase 4A actual-successor/reply scans; never fake a changed player."""
    helper = MCTSAgent(1)
    wins = helper._winning_moves(game)
    if wins:
        return wins, "immediate_win"
    safe = helper._safe_moves(game)
    if safe and len(safe) < len(game.get_valid_moves()):
        return safe, "safe_responses"
    return game.get_valid_moves(), "none"


class RootDirichletNoise:
    """Private PCG64 stream, independent of search, training and global RNGs.

    One draw per enabled search, in permitted physical-column order. Initial and
    final bit-generator states plus per-search samples permit exact replay.
    Disabled noise returns the original priors without normalization or draws.
    """
    domain = "connect4-self-play-root-dirichlet-v1"

    def __init__(self, epsilon=0.0, alpha=0.30, *, seed=42):
        if (type(epsilon) not in (int, float) or not math.isfinite(epsilon)
                or not 0 <= epsilon <= 1):
            raise ValueError("root_noise_epsilon must be finite in [0,1]")
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("root_dirichlet_alpha must be positive and finite")
        if type(seed) is not int:
            raise ValueError("root noise seed must be an integer")
        self.epsilon, self.alpha = float(epsilon), float(alpha)
        self.seed = seed
        self.seed_digest = hashlib.sha256(f"{self.domain}:{seed}".encode()).hexdigest()
        self._rng = np.random.Generator(np.random.PCG64(int(self.seed_digest, 16)))
        self.draws = 0

    def mix(self, priors, moves):
        if self.epsilon == 0:
            return priors, ()
        noise = np.zeros(7, dtype=np.float64)
        noise[list(moves)] = self._rng.dirichlet(np.full(len(moves), self.alpha))
        mixed = (1 - self.epsilon) * np.asarray(priors) + self.epsilon * noise
        mixed /= mixed.sum()
        mixed = probability_vector(mixed)
        self.draws += 1
        return mixed, tuple(float(p) for p in noise)

    def rng_state(self):
        return deepcopy(self._rng.bit_generator.state)

    def record(self):
        return dict(domain=self.domain, seed=self.seed, seed_sha256=self.seed_digest,
                    seed_integer=str(int(self.seed_digest, 16)), bit_generator="PCG64",
                    numpy_version=np.__version__, epsilon=self.epsilon, alpha=self.alpha,
                    draws=self.draws, action_order="permitted physical columns, ascending")


@dataclass(frozen=True)
class SearchResult:
    move: int
    policy: tuple[float, ...]
    visits: tuple[int, ...]
    temperature: float
    tactical_guard: bool
    guard_applied: str
    root: Node  # Per-call diagnostic tree, never retained by the agent/model.
    root_noise_epsilon: float = 0.0
    root_dirichlet_alpha: float = 0.30
    root_prior_before_noise: tuple[float, ...] = ()
    root_prior: tuple[float, ...] = ()
    root_noise_sample: tuple[float, ...] = ()
    root_noise_draw_index: int | None = None


class MCTSNNAgent:
    """Raw neural search by default; no automatic example collection.

    A simulation is one root-to-leaf traversal, evaluation, and alternating
    backup. Root expansion is uncounted initialization; its value is discarded.
    Exactly simulation_limit root edges are visited, even with a budget of one.
    rng is a random.Random-compatible instance; concurrent calls should pass
    independent per-call rng objects. No tree survives a call on this agent.
    """
    def __init__(self, model, simulation_limit=1000, temperature=1.0, *,
                 exploration=1.41, rng=None, tactical_guard=False,
                 root_noise_epsilon=0.0, root_dirichlet_alpha=0.30, root_noise_seed=42):
        if isinstance(simulation_limit, bool) or not isinstance(simulation_limit, int):
            raise TypeError("simulation_limit must be a positive integer")
        if simulation_limit < 1:
            raise ValueError("simulation_limit must be a positive integer")
        if type(tactical_guard) is not bool:
            raise TypeError("tactical_guard must be boolean")
        self.simulation_limit = simulation_limit
        self.temperature = nonnegative_finite(temperature, "temperature")
        self.exploration = nonnegative_finite(exploration, "exploration")
        self.inference = model if isinstance(model, NeuralInference) else NeuralInference(model)
        self.rng = random.Random() if rng is None else rng
        self.tactical_guard = tactical_guard
        self.root_noise = RootDirichletNoise(root_noise_epsilon, root_dirichlet_alpha,
                                            seed=root_noise_seed)

    def _expand(self, node, allowed_moves=None):
        prediction = self.inference.predict_legal(node.game_state)
        moves = node.game_state.get_valid_moves() if allowed_moves is None else allowed_moves
        priors = legal_policy(prediction.policy, moves)
        for move in moves:
            after = deepcopy(node.game_state)
            if not after.make_move(move):
                raise RuntimeError("Engine rejected a legal search move")
            node.children[move] = Node(after, parent=node, move=move, prior_p=priors[move])
        return prediction.value

    def _select_child(self, node, rng):
        scores = {move: child.puct(self.exploration) for move, child in node.children.items()}
        best = max(scores.values())
        return node.children[rng.choice([move for move, score in scores.items() if score == best])]

    def search(self, game, *, rng=None):
        rng = self.rng if rng is None else rng
        root = Node(deepcopy(game))
        if root.game_state.is_game_over() or not root.game_state.get_valid_moves():
            raise ValueError("Cannot search a terminal position or without legal moves")
        encode_current_player(root.game_state)  # Validate even before root tactics.
        allowed, applied = (tactical_root_moves(root.game_state) if self.tactical_guard
                            else (root.game_state.get_valid_moves(), "disabled"))
        self._expand(root, allowed)
        before_noise = tuple(float(root.children[a].prior_p) if a in root.children else 0.0
                             for a in range(7))
        priors, noise_sample = self.root_noise.mix(before_noise, sorted(root.children))
        if self.root_noise.epsilon > 0:
            for action, child in root.children.items():
                child.prior_p = float(priors[action])
        for _ in range(self.simulation_limit):
            node = root
            while node.children:
                node = self._select_child(node, rng)
            value = (terminal_value(node.game_state) if node.game_state.is_game_over()
                     else self._expand(node))
            backup(node, value)
        visits = tuple(root.children[a].visits if a in root.children else 0 for a in range(7))
        if sum(visits) != self.simulation_limit:
            raise RuntimeError("Root-edge visit accounting failed")
        policy = visit_policy(visits, self.temperature)
        support = [a for a in range(7) if policy[a] > 0]
        move = rng.choices(support, weights=[policy[a] for a in support], k=1)[0]
        return SearchResult(move, tuple(policy), visits, self.temperature,
                            self.tactical_guard, applied, root,
                            self.root_noise.epsilon, self.root_noise.alpha, before_noise,
                            tuple(float(p) for p in priors), noise_sample,
                            self.root_noise.draws if noise_sample else None)

    def choose_move(self, game):
        return self.search(game).move


def load_pretrained_mcts_nn_agent(model_path=None, simulation_limit=1000, temperature=0.1, **kwargs):
    """Explicit versioned artifacts only; historical default loading is retired."""
    if model_path is None:
        raise ValueError("An explicit canonical checkpoint is required; historical weights are forensic only")
    return MCTSNNAgent(load_checkpoint(model_path), simulation_limit, temperature, **kwargs)
