"""Evaluation of v2 inference artifacts: arena agents, tactical/solved/value runs and calibration.

Deployment-mode search everywhere: raw PUCT, root noise OFF, tactical guard
OFF, temperature 0 with seeded maximum-visit ties. Nothing here trains, and no
function touches a GenerationRunner's RNG streams: every call derives its own
domain-separated streams, so evaluation can run between generations without
changing the training trajectory.
"""
import hashlib
from pathlib import Path
import random

import numpy as np
import torch

from ..connect4 import Connect4
from .config import V2Config
from .data import seeded_rng
from .network import V2Inference, load_inference_checkpoint, weights_sha256
from .oracle import engine_position
from .search import EvaluationAgent, SelfPlayer, V2RootNoise
from .selfplay import play_game as self_play_game

EVALUATION_DOMAIN = "connect4-alphazero-v2-evaluation"
CALIBRATION_DOMAIN = "connect4-alphazero-v2-held-out-calibration"
PHASE4D2F_CHECKPOINT = dict(
    path="experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/candidate.pt",
    sha256="78276912bdef5f1b1b53ac6b2ec3ede2eed20a8ebb5796a9bdcbb57e4cb28c51")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class V2SearchArenaAgent:
    """Raw v2 PUCT at a fixed budget (default 512), noise OFF, tau=0, seeded ties."""

    def __init__(self, inference, simulations=512, *, name="v2", identity=None):
        self.inference = inference if isinstance(inference, V2Inference) else V2Inference(inference)
        self.simulations, self.name = simulations, name
        self.identity = identity or dict(weights_sha256=weights_sha256(self.inference.model))

    def describe(self):
        return dict(name=self.name, kind="alphazero-v2 raw PUCT", simulations=self.simulations,
                    root_noise=False, tactical_guard=False, temperature=0, ties="seeded uniform", **self.identity)

    def choose(self, game, rng):
        result, selection = EvaluationAgent(self.inference, self.simulations, rng=rng).search(game)
        return selection.move, dict(visits=list(result.visits), raw_value=result.root_value)


class V2NNOnlyAgent:
    """Argmax of the legal-masked network policy; exact ties broken by the per-game RNG."""

    def __init__(self, inference, *, name="v2_nn_only", identity=None):
        self.inference = inference if isinstance(inference, V2Inference) else V2Inference(inference)
        self.name = name
        self.identity = identity or dict(weights_sha256=weights_sha256(self.inference.model))

    def describe(self):
        return dict(name=self.name, kind="alphazero-v2 NN only", ties="seeded uniform", **self.identity)

    def choose(self, game, rng):
        prediction = self.inference.predict(game)
        best = max(prediction.policy)
        move = rng.choice([a for a in range(7) if prediction.policy[a] == best])
        return move, dict(raw_value=prediction.value)


class V1SearchArenaAgent:
    """Retained Phase 4D.2f W/D/L network through the unchanged v1 MCTSNNAgent at 512.

    Same deployment conventions: temperature 0 with seeded ties, tactical guard
    OFF, root noise OFF. The v1 legal-policy path (softmax then mask) is part
    of that historical agent and is not altered.
    """

    def __init__(self, path, expected_sha256, simulations=512, *, name="phase4d2f_512"):
        from ..neural_mcts import load_checkpoint
        if _sha256(path) != expected_sha256:
            raise ValueError("Phase 4D.2f checkpoint hash differs from the declared hash")
        self.model = load_checkpoint(path)
        self.simulations, self.name = simulations, name
        self.identity = dict(path=str(path), sha256=expected_sha256)

    def describe(self):
        return dict(name=self.name, kind="phase4d2f v1 W/D/L MCTSNNAgent", simulations=self.simulations,
                    root_noise=False, tactical_guard=False, temperature=0, ties="seeded uniform", **self.identity)

    def choose(self, game, rng):
        from ..agents.mcts_nn_agent import MCTSNNAgent
        result = MCTSNNAgent(self.model, self.simulations, 0.0, rng=rng, tactical_guard=False).search(game)
        return result.move, dict(visits=list(result.visits))


def load_v2_agent(path, simulations=512, *, expected_sha256=None, name="v2"):
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError("Inference artifact hash differs from the expected hash")
    inference = load_inference_checkpoint(path)
    return V2SearchArenaAgent(inference, simulations, name=name,
                              identity=dict(path=str(path), sha256=_sha256(path),
                                            weights_sha256=weights_sha256(inference.model)))


# Position packages ----------------------------------------------------------------------

def _row_rng(namespace, row_id, seed):
    return seeded_rng(f"{EVALUATION_DOMAIN}:{namespace}:{row_id}", seed)


def search_rows(inference, rows, *, simulations=512, seeds=(0, 1, 2, 3), namespace="positions", check=None):
    """Chosen actions and root visits for every row and declared search/tie seed."""
    choices, visits = {}, {}
    for row in rows:
        game = engine_position(row["moves"])
        choices[row["id"]], visits[row["id"]] = [], []
        for seed in seeds:
            if check is not None:
                check()
            result, selection = EvaluationAgent(inference, simulations, rng=_row_rng(namespace, row["id"], seed)).search(game)
            choices[row["id"]].append(selection.move)
            visits[row["id"]].append(tuple(result.visits))
    return choices, visits


def nn_only_rows(inference, rows, *, namespace="nn-only"):
    choices = {}
    for row in rows:
        prediction = inference.predict(engine_position(row["moves"]))
        best = max(prediction.policy)
        rng = _row_rng(namespace, row["id"], 0)
        choices[row["id"]] = [rng.choice([a for a in range(7) if prediction.policy[a] == best])]
    return choices


def raw_values(inference, rows):
    """Raw tanh value for the player to move at every row (no search)."""
    return {row["id"]: inference.predict(engine_position(row["moves"])).value for row in rows}


# Behavioral calibration -------------------------------------------------------------------

def calibration_games(model, config, *, games, seed, check=None):
    """Held-out self-play with the training schedule (noise ON, tau schedule), no updates.

    Uses fresh calibration-domain streams, never a runner's. Records the raw
    pre-search network value and the eventual actor-relative outcome.
    """
    if not isinstance(config, V2Config):
        raise ValueError("calibration requires a V2Config")
    inference = model if isinstance(model, V2Inference) else V2Inference(model)
    player = SelfPlayer(inference, config,
                        search_rng=seeded_rng(f"{CALIBRATION_DOMAIN}:search", seed),
                        action_rng=seeded_rng(f"{CALIBRATION_DOMAIN}:action", seed),
                        root_noise=V2RootNoise(config.root_noise_epsilon, config.root_dirichlet_alpha,
                                               seed=int(hashlib.sha256(f"{CALIBRATION_DOMAIN}:noise:{seed}".encode())
                                                        .hexdigest(), 16) % 2 ** 63))
    values = {}

    def observer(index, decision, seconds):
        values.setdefault(index, []).append(decision.search.root_value)
    records = []
    for index in range(games):
        completed = self_play_game(player, 0, index, check=check, observer=observer)  # generation 0: held out
        for example, value in zip(completed.examples, values[index]):
            records.append(dict(game=index, ply=example.ply, actor=example.actor, value=value,
                                outcome=example.outcome))
    return records


def untrained_model(seed):
    """The exact fresh initialization a GenerationRunner with this seed starts from."""
    from .generation import initial_model
    return initial_model(seed)


def inference_identity(inference):
    return dict(weights_sha256=weights_sha256(inference.model))


def process_rng_fingerprint():
    """Global RNG fingerprint, so callers can assert evaluation never consumed them."""
    return (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state().clone())
