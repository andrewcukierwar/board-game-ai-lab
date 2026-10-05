"""V2 scalar-value network, legal-logit inference and minimal inference artifact.

The model declares ``alphazero_v2_contract`` and deliberately no v1
``representation_version``, so v1 inference/training rejects it and v2 rejects
v1 Connect4Net. Fresh initialization only: nothing here reads v1 weights.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..neural_mcts import encode_current_player, validate_input
from .config import ARCHITECTURE, INFERENCE_CONTRACT


class AlphaZeroV2Net(nn.Module):
    """Retained v1 trunk/policy head; scalar tanh value head (325,896 parameters)."""
    alphazero_v2_contract = ARCHITECTURE

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.policy_conv = nn.Conv2d(128, 32, 1)
        self.policy_fc = nn.Linear(32 * 6 * 7, 7)
        self.value_conv = nn.Conv2d(128, 32, 1)
        self.value_fc1 = nn.Linear(32 * 6 * 7, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, states):
        """Return (raw policy logits (N,7), tanh value (N,1)) for the player to move."""
        validate_input(states)
        shared = F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(states))))))
        policy = self.policy_fc(F.relu(self.policy_conv(shared)).flatten(1))
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(F.relu(self.value_conv(shared)).flatten(1)))))
        validate_outputs(policy, value, states.shape[0])
        return policy, value


def validate_outputs(policy, value, batch_size):
    for name, tensor, width in (("policy logits", policy, 7), ("value", value, 1)):
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != (batch_size, width)
                or tensor.dtype != torch.float32 or tensor.device.type != "cpu"
                or not torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite CPU float32 ({batch_size},{width})")
    if (value.abs() > 1).any():
        raise ValueError("value must lie in [-1,1]")


def legal_moves_tuple(legal_moves):
    moves = tuple(legal_moves)
    if (not moves or len(set(moves)) != len(moves)
            or any(type(m) is not int or not 0 <= m < 7 for m in moves)):
        raise ValueError("Expected nonempty distinct legal action indices 0..6")
    return moves


def legal_policy_from_logits(logits, legal_moves):
    """Mask illegal logits, then stable float64 softmax over legal actions only.

    Illegal probabilities are exactly zero; an illegal logit, however large,
    cannot change relative legal probabilities. No uniform repair path.
    """
    raw = np.asarray(logits, dtype=np.float64)
    if raw.shape != (7,) or not np.isfinite(raw).all():
        raise ValueError("Expected seven finite raw policy logits")
    legal = sorted(legal_moves_tuple(legal_moves))
    shifted = raw[legal] - raw[legal].max()
    weights = np.exp(shifted)  # max term is exactly 1, so the sum is in [1, 7]
    policy = np.zeros(7, dtype=np.float64)
    policy[legal] = weights / weights.sum()
    return policy


@dataclass(frozen=True)
class V2Prediction:
    logits: tuple[float, ...]  # raw, all seven physical columns
    policy: tuple[float, ...]  # legal-masked softmax
    value: float               # tanh scalar for the player to move


def require_v2_model(model):
    if getattr(model, "alphazero_v2_contract", None) != ARCHITECTURE:
        raise ValueError("Model must declare the AlphaZero v2 contract; v1 networks are incompatible")
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Model tensors must be finite CPU float32")


class V2Inference:
    """Read-only CPU inference over a v2 model that the caller keeps in eval mode."""

    def __init__(self, model):
        require_v2_model(model)
        self.model = model
        self._check_eval()

    def _check_eval(self):
        if any(module.training for module in self.model.modules()):
            raise ValueError("Inference requires model.eval(); concurrent training is unsupported")

    def predict(self, game):
        """Legal-masked policy and scalar value; terminal/no-legal states are rejected."""
        if game.is_game_over() or not game.get_valid_moves():
            raise ValueError("Cannot predict actions for a terminal position or without legal moves")
        self._check_eval()
        with torch.inference_mode():
            policy, value = self.model(encode_current_player(game))
        validate_outputs(policy, value, 1)
        logits = tuple(float(x) for x in policy[0])
        return V2Prediction(logits, tuple(float(p) for p in legal_policy_from_logits(logits, game.get_valid_moves())),
                            float(value[0, 0]))


def weights_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_copy(model):
    """Independent eval-mode snapshot with gradients disabled."""
    require_v2_model(model)
    snapshot = deepcopy(model).eval()
    for parameter in snapshot.parameters():
        parameter.requires_grad_(False)
    return snapshot


def model_from_state(state):
    with torch.random.fork_rng(devices=[]):  # Loading never advances the global torch RNG.
        model = AlphaZeroV2Net()
    expected = model.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("State keys do not match AlphaZeroV2Net")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape
                or tensor.dtype != torch.float32 or tensor.device.type != "cpu"
                or not torch.isfinite(tensor).all()):
            raise ValueError(f"Invalid v2 tensor: {name}")
    model.load_state_dict(state, strict=True)
    return model.eval()


def model_state(model):
    if type(model) is not AlphaZeroV2Net:
        raise ValueError("Artifacts require AlphaZeroV2Net")
    require_v2_model(model)
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def inference_payload(model):
    """Contract plus model tensors only: no optimizer, replay, RNG or provenance."""
    return {"contract": deepcopy(INFERENCE_CONTRACT), "model_state_dict": model_state(model)}


def save_inference_checkpoint(path, model):
    """Write a minimal v2 inference artifact, refusing to overwrite any file."""
    from .artifacts import atomic_torch_save
    payload = inference_payload(model)

    def validate(temporary):
        state = load_inference_checkpoint(temporary).model.state_dict()
        if any(not torch.equal(state[name], tensor) for name, tensor in payload["model_state_dict"].items()):
            raise ValueError("Reloaded inference tensors differ")
    return atomic_torch_save(path, payload, validate)


def load_inference_checkpoint(path):
    """Weights-only CPU load of a v2 inference artifact; rejects v1/resume artifacts."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or set(checkpoint) != {"contract", "model_state_dict"}
            or checkpoint["contract"] != INFERENCE_CONTRACT):
        raise ValueError("Expected an AlphaZero v2 inference checkpoint")
    return V2Inference(model_from_state(checkpoint["model_state_dict"]))
