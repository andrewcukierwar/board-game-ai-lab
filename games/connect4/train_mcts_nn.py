"""Training foundations only. No self-play runner, checkpoint writer, or CLI.

Capture pre-move examples explicitly, finalize only completed games, and reuse
one NeuralTrainer across intended updates/iterations. No historical weights.
"""
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.nn import functional as F

from .neural_mcts import (
    ENCODING, NeuralInference, encode_current_player, nonnegative_finite,
    probability_vector, validate_input, validate_logits,
)

POLICY_TARGET = "root-visits-temperature-v1"


@dataclass(frozen=True)
class TrainingExample:
    observation: tuple[tuple[int, ...], ...]
    acting_player: int
    policy: tuple[float, ...]
    temperature: float
    tactical_guard: bool = False
    outcome: int | None = None
    encoding: str = ENCODING
    policy_target: str = POLICY_TARGET

    def __post_init__(self):
        # Frozen dataclasses alone do not freeze nested arrays/lists. Own tuples.
        board = np.asarray(self.observation)
        if board.shape != (6, 7) or not np.isin(board, [-1, 0, 1]).all():
            raise ValueError("Observation must be a finite canonical 6x7 board")
        if type(self.acting_player) is not int or self.acting_player not in (0, 1):
            raise ValueError("Acting player must be X=0 or O=1")
        if self.encoding != ENCODING or self.policy_target != POLICY_TARGET:
            raise ValueError("Unsupported training representation/target convention")
        if self.outcome is not None and (type(self.outcome) is not int or self.outcome not in (-1, 0, 1)):
            raise ValueError("Outcome must be -1, 0, +1 or pending None")
        if type(self.tactical_guard) is not bool:
            raise TypeError("tactical_guard must be boolean")
        policy = probability_vector(self.policy)
        if (policy[board[0] != 0] != 0).any():
            raise ValueError("Policy target assigns mass to an illegal column")
        object.__setattr__(self, "observation", tuple(tuple(int(c) for c in row) for row in board))
        object.__setattr__(self, "policy", tuple(float(p) for p in policy))
        object.__setattr__(self, "temperature", nonnegative_finite(self.temperature, "temperature"))


def capture_example(search_result):
    """Capture the searched pre-move state, even if the caller has since moved.

    The target is exactly the search action distribution at its recorded tau,
    including uniform maximum-visit ties at zero. Guard usage is explicit.
    """
    game = search_result.root.game_state
    if game.is_game_over() or not game.get_valid_moves():
        raise ValueError("Training observation must be pre-move and nonterminal")
    return TrainingExample(encode_current_player(game)[0, 0].tolist(), game.current_player,
                           search_result.policy, search_result.temperature,
                           tactical_guard=search_result.tactical_guard)


def outcome_for_player(winner, acting_player):
    if type(winner) is not int or winner not in (-1, 0, 1):
        raise ValueError("Winner must be X=0, O=1 or draw=-1")
    if type(acting_player) is not int or acting_player not in (0, 1):
        raise ValueError("Acting player must be X=0 or O=1")
    return 0 if winner == -1 else (1 if winner == acting_player else -1)


def finalize_examples(examples, completed_game):
    """Return newly labeled examples; incomplete games cannot produce labels."""
    if not completed_game.is_game_over():
        raise ValueError("Outcomes require a completed game")
    winner = completed_game.check_winner()
    result = []
    for example in examples:
        if example.outcome is not None:
            raise ValueError("Example is already labeled")
        result.append(replace(example, outcome=outcome_for_player(winner, example.acting_player)))
    return tuple(result)


def reflect_completed_example(example):
    """Reflect physical columns, preserving the completed actor-relative contract.

    Reconstruct via dataclass validation; neither nested input nor labels change.
    """
    if not isinstance(example, TrainingExample) or example.outcome is None:
        raise ValueError("Reflection requires a completed TrainingExample")
    return replace(example, observation=tuple(row[::-1] for row in example.observation),
                   policy=example.policy[::-1])


def batch_tensors(examples):
    if not examples or any(example.outcome is None for example in examples):
        raise ValueError("A nonempty batch of completed examples is required")
    states = torch.tensor([e.observation for e in examples], dtype=torch.float32).unsqueeze(1)
    policies = torch.tensor([e.policy for e in examples], dtype=torch.float32)
    # Classes are [win, draw, loss], distinct from winner IDs [X=0, O=1].
    values = torch.tensor([{1: 0, 0: 1, -1: 2}[e.outcome] for e in examples], dtype=torch.long)
    return states, policies, values


def training_loss(policy_logits, value_logits, policies, outcomes, *, return_components=False):
    """L = mean(-sum_a pi_a log softmax(p)_a) + mean(CE(v, WDL class)).

    Equal head weights, no regularization in this foundation. Both inputs are
    raw logits; outcomes are int64 classes win=0/draw=1/loss=2.
    """
    if not isinstance(policies, torch.Tensor) or policies.ndim != 2 or policies.shape[0] < 1:
        raise ValueError("Policy targets require a nonempty (N,7) tensor")
    n = policies.shape[0]
    validate_logits(policy_logits, value_logits, n)
    if (policies.shape != (n, 7) or policies.dtype != torch.float32 or policies.device.type != "cpu"
            or not torch.isfinite(policies).all() or (policies < 0).any() or (policies > 1).any()
            or not torch.allclose(policies.sum(1), torch.ones(n), rtol=0, atol=1e-6)):
        raise ValueError("Policy targets must be finite normalized CPU float32 (N,7)")
    if (not isinstance(outcomes, torch.Tensor) or outcomes.shape != (n,)
            or outcomes.device.type != "cpu" or outcomes.dtype != torch.long
            or ((outcomes < 0) | (outcomes > 2)).any()):
        raise ValueError("Outcome classes must be CPU int64 (N,) in [0,2]")
    policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(1).mean()
    value_loss = F.cross_entropy(value_logits, outcomes)
    loss = policy_loss + value_loss
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    return (loss, policy_loss, value_loss) if return_components else loss


class NeuralTrainer:
    """Persistent Adam; sequential training/inference phases on a CPU model.

    Owns the mode boundary: eval between updates, train during update, eval on
    return (also on failure). No optimizer recreation at iteration boundaries.
    """
    def __init__(self, model, *, learning_rate=0.001):
        learning_rate = nonnegative_finite(learning_rate, "learning_rate")
        if learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if getattr(model, "representation_version", None) != ENCODING:
            raise ValueError("Training requires a fresh canonical model")
        model.eval()
        NeuralInference(model)  # Validate parameters/device before optimizer creation.
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.steps = 0
        self.last_metrics = None

    def step(self, examples):
        states, policies, outcomes = batch_tensors(examples)
        validate_input(states)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            loss, policy_loss, value_loss = training_loss(
                *self.model(states), policies, outcomes, return_components=True)
            loss.backward()
            gradients = [p.grad for p in self.model.parameters() if p.requires_grad]
            if not gradients or any(g is None or not torch.isfinite(g).all() for g in gradients):
                raise ValueError("Missing or nonfinite training gradients")
            gradient_norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(g) for g in gradients]))
            if not torch.isfinite(gradient_norm):
                raise ValueError("Nonfinite gradient norm")
            self.optimizer.step()
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError("Nonfinite parameters after update; discard this trainer")
            self.steps += 1
            self.last_metrics = dict(
                combined_loss=float(loss.detach()), policy_loss=float(policy_loss.detach()),
                value_loss=float(value_loss.detach()), gradient_norm=float(gradient_norm),
                policy_target_entropy=float(-(policies * policies.clamp_min(1e-30).log()).sum(1).mean()),
                finite_gradients=True, finite_parameters=True)
            return float(loss.detach())
        finally:
            self.model.eval()


if __name__ == "__main__":
    raise SystemExit("Foundation only: self-play training requires a separately authorized Phase 4D.2 runner.")
