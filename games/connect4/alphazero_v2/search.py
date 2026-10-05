"""V2 raw PUCT search: corrected v1 tree semantics, separate target and action.

Reused unchanged from v1: single-parent Node with Q for that node's player to
move, ``score = -Q(child) + 1.41 P sqrt(max(1, N(parent))) / (1 + N(child))``,
alternating undiscounted backup and engine terminal values. Root expansion is
uncounted; exactly ``simulations`` root-edge traversals occur. No tactical
guard, transposition reuse or tree reuse. Root noise exists only when a
self-play caller passes a V2RootNoise; evaluation has no way to enable it.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import random

import numpy as np

from ..agents.mcts_nn_agent import Node, backup, terminal_value, visit_policy
from .config import PUCT_EXPLORATION, V2Config, action_temperature
from .data import pre_move_ply, visit_target
from .network import V2Inference, legal_moves_tuple


class V2RootNoise:
    """Self-play root Dirichlet noise on legal actions: P' = (1-eps) P + eps eta.

    Private PCG64 stream, one draw per noisy search, independent of search,
    action, sampling, augmentation and global RNGs. State is saved/restored.
    """
    domain = "connect4-alphazero-v2-self-play-root-dirichlet"

    def __init__(self, epsilon=0.25, alpha=1.0, *, seed):
        V2Config(root_noise_epsilon=epsilon, root_dirichlet_alpha=alpha)  # shared validation
        if type(seed) is not int:
            raise ValueError("root noise seed must be an integer")
        self.epsilon, self.alpha, self.seed = float(epsilon), float(alpha), seed
        digest = hashlib.sha256(f"{self.domain}:{seed}".encode()).hexdigest()
        self._rng = np.random.Generator(np.random.PCG64(int(digest, 16)))
        self.draws = 0

    def mix(self, priors, legal_moves):
        """Return (searched priors, noise sample or None); illegal entries stay zero."""
        moves = sorted(legal_moves_tuple(legal_moves))
        priors = np.asarray(priors, dtype=np.float64)
        illegal = [a for a in range(7) if a not in moves]
        if priors.shape != (7,) or not np.isfinite(priors).all() or (priors < 0).any() or priors[illegal].any():
            raise ValueError("Root priors must be finite, nonnegative and legal-only")
        if self.epsilon == 0:
            return priors.copy(), None
        noise = np.zeros(7, dtype=np.float64)
        noise[moves] = self._rng.dirichlet(np.full(len(moves), self.alpha))
        mixed = (1 - self.epsilon) * priors + self.epsilon * noise
        self.draws += 1
        return mixed / mixed.sum(), noise

    def get_state(self):
        return {"bit_generator": deepcopy(self._rng.bit_generator.state), "draws": self.draws}

    def set_state(self, state):
        if state["bit_generator"]["bit_generator"] != "PCG64" or type(state["draws"]) is not int:
            raise ValueError("Invalid root-noise state")
        self._rng.bit_generator.state = deepcopy(state["bit_generator"])
        self.draws = state["draws"]


@dataclass(frozen=True)
class V2SearchResult:
    visits: tuple[int, ...]          # root-child visits, physical columns
    simulations: int
    root_logits: tuple[float, ...]   # raw network logits at the root
    root_value: float                # raw root network value (diagnostic; never backed up)
    root_prior: tuple[float, ...]    # legal-masked network policy
    search_prior: tuple[float, ...]  # root priors actually searched (after optional noise)
    root_noise: tuple[float, ...] | None
    root_noise_draw: int | None
    root: Node                       # per-call diagnostic tree

    @property
    def visit_target(self):
        return visit_target(self.visits)


@dataclass(frozen=True)
class ActionSelection:
    move: int
    temperature: float
    distribution: tuple[float, ...]  # execution distribution; never a training target


def select_action(visits, temperature, rng):
    """tau=1: visit-proportional; tau=0: uniform among maximum-visit actions via rng."""
    distribution = visit_policy(visits, temperature)
    support = [a for a in range(7) if distribution[a] > 0]
    move = rng.choices(support, weights=[distribution[a] for a in support], k=1)[0]
    return ActionSelection(move, float(temperature), tuple(float(p) for p in distribution))


class PUCTSearch:
    def __init__(self, inference, simulations):
        if isinstance(simulations, bool) or type(simulations) is not int or simulations < 1:
            raise ValueError("simulations must be a positive integer")
        self.inference = inference if isinstance(inference, V2Inference) else V2Inference(inference)
        self.simulations = simulations

    def _expand(self, node):
        prediction = self.inference.predict(node.game_state)
        for move in node.game_state.get_valid_moves():
            after = deepcopy(node.game_state)
            if not after.make_move(move):
                raise RuntimeError("Engine rejected a legal search move")
            node.children[move] = Node(after, parent=node, move=move, prior_p=prediction.policy[move])
        return prediction

    @staticmethod
    def _select_child(node, rng):
        scores = {move: child.puct(PUCT_EXPLORATION) for move, child in node.children.items()}
        best = max(scores.values())
        return node.children[rng.choice([move for move, score in scores.items() if score == best])]

    def run(self, game, *, rng, root_noise=None):
        root = Node(deepcopy(game))
        if root.game_state.is_game_over() or not root.game_state.get_valid_moves():
            raise ValueError("Cannot search a terminal position or without legal moves")
        prediction = self._expand(root)
        search_prior, noise = prediction.policy, None
        if root_noise is not None:
            mixed, noise = root_noise.mix(prediction.policy, root.children)
            search_prior = tuple(float(p) for p in mixed)
            for action, child in root.children.items():
                child.prior_p = search_prior[action]
        for _ in range(self.simulations):
            node = root
            while node.children:
                node = self._select_child(node, rng)
            backup(node, terminal_value(node.game_state) if node.game_state.is_game_over()
                   else self._expand(node).value)
        visits = tuple(root.children[a].visits if a in root.children else 0 for a in range(7))
        if sum(visits) != self.simulations or root.visits != self.simulations:
            raise RuntimeError("Root-edge visit accounting failed")
        return V2SearchResult(visits, self.simulations, prediction.logits, prediction.value,
                              prediction.policy, search_prior,
                              None if noise is None else tuple(float(x) for x in noise),
                              None if noise is None else root_noise.draws, root)


@dataclass(frozen=True)
class SelfPlayDecision:
    ply: int
    search: V2SearchResult
    action: ActionSelection


class SelfPlayer:
    """Self-play policy: root noise ON, temperature 1 for plies < exploratory_plies, else 0."""

    def __init__(self, inference, config, *, search_rng, action_rng, root_noise):
        if not isinstance(root_noise, V2RootNoise):
            raise ValueError("Self-play requires an owned V2RootNoise stream")
        self.search = PUCTSearch(inference, config.self_play_simulations)
        self.config, self.search_rng, self.action_rng, self.root_noise = config, search_rng, action_rng, root_noise

    def decide(self, game):
        ply = pre_move_ply(game)
        result = self.search.run(game, rng=self.search_rng, root_noise=self.root_noise)
        return SelfPlayDecision(ply, result, select_action(
            result.visits, action_temperature(ply, self.config.exploratory_plies), self.action_rng))


class EvaluationAgent:
    """Evaluation/inference policy: no root noise, temperature 0, no example capture.

    Default 512 simulations. Pass independent per-call rngs for concurrent use.
    """

    def __init__(self, inference, simulations=512, *, rng=None):
        self.search_engine = PUCTSearch(inference, simulations)
        self.rng = random.Random(0) if rng is None else rng

    def search(self, game, *, rng=None):
        rng = self.rng if rng is None else rng
        result = self.search_engine.run(game, rng=rng)
        return result, select_action(result.visits, 0.0, rng)

    def choose_move(self, game):
        return self.search(game)[1].move
