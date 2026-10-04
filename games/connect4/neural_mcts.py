"""Optional CPU neural contracts for search and future policy-only inference.

No search, training, device selection, or model-mode mutation occurs here.
Historical weights require the explicitly separate forensic loader/encoder.
"""
from dataclasses import dataclass
from copy import deepcopy
import math
from numbers import Real

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ENCODING = "connect4-current-player-1x6x7-v1"
HISTORICAL_ENCODING = "connect4-absolute-x-o-v0"
ARCHITECTURE = "connect4-conv64-128-128-policy7-wdl3-v1"
CHECKPOINT_CONTRACT = {
    "format_version": 1,
    "architecture": ARCHITECTURE,
    "encoding": ENCODING,
    "action_order": list(range(7)),
    "value_order": ["win", "draw", "loss"],
    "value_perspective": "current_player",
    "outputs": "logits",
}


def nonnegative_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def encode_historical_absolute(game):
    """Forensic X=+1/O=-1 encoding; never used as canonical input without a flip."""
    board = np.asarray(game.board)
    if board.shape != (6, 7) or not np.isin(board, ["X", "O", " "]).all():
        raise ValueError("Expected a 6x7 board containing only X, O and spaces")
    return torch.tensor(np.where(board == "X", 1, np.where(board == "O", -1, 0)),
                        dtype=torch.float32).reshape(1, 1, 6, 7)


def encode_current_player(game):
    """Detached float32 (1,1,6,7), top row first; physical columns 0..6."""
    if type(game.current_player) is not int or game.current_player not in (0, 1):
        raise ValueError("current_player must be X=0 or O=1")
    if game.piece != ("X" if game.current_player == 0 else "O"):
        raise ValueError("current_player and piece disagree")
    return encode_historical_absolute(game) * (1 if game.current_player == 0 else -1)


def validate_input(states):
    if (not isinstance(states, torch.Tensor) or states.dtype != torch.float32
            or states.device.type != "cpu" or states.ndim != 4
            or states.shape[0] < 1 or tuple(states.shape[1:]) != (1, 6, 7)):
        raise ValueError("Input must be CPU float32 (N,1,6,7), N >= 1")
    if not torch.isfinite(states).all() or not ((states == -1) | (states == 0) | (states == 1)).all():
        raise ValueError("Input cells must be finite -1, 0 or +1")


def validate_logits(policy, value, batch_size):
    for name, tensor, width in (("policy", policy, 7), ("value", value, 3)):
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != (batch_size, width)
                or tensor.dtype != torch.float32 or tensor.device.type != "cpu"
                or not torch.isfinite(tensor).all()):
            raise ValueError(f"{name} logits must be finite CPU float32 ({batch_size},{width})")


def probability_vector(values, size=7):
    result = np.asarray(values, dtype=np.float64)
    if (result.shape != (size,) or not np.isfinite(result).all()
            or (result < 0).any() or (result > 1).any()
            or not np.isclose(result.sum(), 1, rtol=0, atol=1e-6)):
        raise ValueError(f"Expected {size} finite nonnegative probabilities summing to one")
    return result.copy()


def legal_policy(policy, legal_moves):
    """Mask normalized network policy; uniform if all legal mass is zero."""
    policy = probability_vector(policy)
    moves = tuple(legal_moves)
    if (not moves or len(set(moves)) != len(moves)
            or any(type(m) is not int or not 0 <= m < 7 for m in moves)):
        raise ValueError("Expected nonempty distinct legal action indices 0..6")
    masked = np.zeros(7, dtype=np.float64)
    # Scaling first also handles subnormal legal mass without division overflow.
    peak = max(policy[m] for m in moves)
    masked[list(moves)] = policy[list(moves)] / peak if peak > 0 else 1
    masked /= masked.sum()
    return probability_vector(masked)


class Connect4Net(nn.Module):
    """Historical architecture and parameter names; both heads return logits."""
    representation_version = ENCODING

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.policy_conv = nn.Conv2d(128, 32, 1)
        self.policy_fc = nn.Linear(32 * 6 * 7, 7)
        self.value_conv = nn.Conv2d(128, 32, 1)
        self.value_fc1 = nn.Linear(32 * 6 * 7, 64)
        self.value_fc2 = nn.Linear(64, 3)

    def forward(self, states):
        validate_input(states)
        shared = F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(states))))))
        policy = self.policy_fc(F.relu(self.policy_conv(shared)).flatten(1))
        value = self.value_fc2(F.relu(self.value_fc1(F.relu(self.value_conv(shared)).flatten(1))))
        validate_logits(policy, value, states.shape[0])
        return policy, value


@dataclass(frozen=True)
class Prediction:
    policy: tuple[float, ...]
    value_probabilities: tuple[float, ...]
    value: float


def prediction_from_logits(policy, value):
    validate_logits(policy, value, 1)
    p = probability_vector(torch.softmax(policy, dim=1)[0].detach().numpy())
    wdl = probability_vector(torch.softmax(value, dim=1)[0].detach().numpy(), 3)
    return Prediction(tuple(p), tuple(wdl), float(wdl[0] - wdl[2]))


class NeuralInference:
    """Shared read-only canonical CPU model. Caller must put it in eval mode.

    In-memory fake networks must explicitly declare representation_version.
    Training a shared model concurrently with inference is unsupported.
    """
    def __init__(self, model):
        if getattr(model, "representation_version", None) != ENCODING:
            raise ValueError("Model must declare the canonical encoding; historical weights are incompatible")
        for tensor in (*model.parameters(), *model.buffers()):
            if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError("Model tensors must be finite CPU float32")
        self.model = model
        self._check_eval()

    def _check_eval(self):
        if any(module.training for module in self.model.modules()):
            raise ValueError("Inference requires model.eval(); concurrent training is unsupported")

    def predict(self, game):
        self._check_eval()
        with torch.inference_mode():
            return prediction_from_logits(*self.model(encode_current_player(game)))

    def predict_legal(self, game):
        """Reusable policy/value interface for MCTS and future NN Only agents."""
        if game.is_game_over() or not game.get_valid_moves():
            raise ValueError("Cannot predict actions for a terminal position or without legal moves")
        prediction = self.predict(game)
        return Prediction(tuple(legal_policy(prediction.policy, game.get_valid_moves())),
                          prediction.value_probabilities, prediction.value)


def _load_state(state, *, representation):
    model = Connect4Net()
    expected = model.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("Checkpoint parameter keys do not match Connect4Net")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape
                or tensor.dtype != torch.float32 or tensor.device.type != "cpu"
                or not torch.isfinite(tensor).all()):
            raise ValueError(f"Invalid checkpoint tensor: {name}")
    model.load_state_dict(state, strict=True)
    model.representation_version = representation
    return model.eval()


def load_checkpoint(path):
    """Load explicitly versioned canonical inference artifacts on CPU.

    No default historical path or unsafe pickle fallback. Future training must
    author provenance/resume artifacts separately from this inference schema.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or set(checkpoint) != {"contract", "model_state_dict"}
            or checkpoint["contract"] != CHECKPOINT_CONTRACT):
        raise ValueError("Expected a versioned canonical checkpoint; historical checkpoints are forensic only")
    return NeuralInference(_load_state(checkpoint["model_state_dict"], representation=ENCODING))


def checkpoint_payload(model):
    """Validated independent CPU tensors; provenance belongs outside this envelope."""
    if type(model) is not Connect4Net or model.representation_version != ENCODING:
        raise ValueError("Checkpoint requires the canonical Connect4Net")
    NeuralInference(model)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    # Reuse the reader's authoritative shape/key/dtype checks without advancing RNG.
    with torch.random.fork_rng(devices=[]):
        _load_state(state, representation=ENCODING)
    return {"contract": deepcopy(CHECKPOINT_CONTRACT), "model_state_dict": state}


def save_checkpoint(path, model):
    """Write a canonical inference artifact, refusing to overwrite any file."""
    payload = checkpoint_payload(model)
    with open(path, "xb") as stream:
        torch.save(payload, stream)


def load_historical_network_for_research(path):
    """Forensic path returning logits on encode_historical_absolute.

    Value perspective/strategic validity is not established. This tagged model
    is rejected by NeuralInference and must never initialize fresh training.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model_state_dict", "iteration"}:
        raise ValueError("Expected a historical model_state_dict/iteration checkpoint")
    return _load_state(checkpoint["model_state_dict"], representation=HISTORICAL_ENCODING)
