"""Compact residual policy/value networks behind the experimental model factory.

Two registered variants (three or five residual blocks, 64 channels) keep the
CNN's encoding, output, loss, inference and replay contracts but each declares
its own architecture identity: every CNN-only path rejects them, and every
ResNet path rejects the CNN and the other variant. The historical network,
trainer and configuration modules are untouched and stay the default.

Batch normalization makes the model state more than its parameters: running
means/variances and update counters are saved, hashed and validated with the
weights, a training step changes them before the optimizer runs, and inference
always uses the running statistics (eval mode). Inert on import.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from ..neural_mcts import validate_input
from .config import INFERENCE_CONTRACT, V2Config
from .generation import INITIALIZATION_DOMAIN
from .network import V2Inference, validate_outputs, weights_sha256
from .training import V2Trainer

RESNET3X64 = "connect4-resnet3x64-bn-policy32-value32x64-scalar-tanh-v1"
RESNET5X64 = "connect4-resnet5x64-bn-policy32-value32x64-scalar-tanh-v1"
NORMALIZATION = "batchnorm2d-eps1e-5-momentum0.1-eval-running-statistics"


@dataclass(frozen=True)
class ResNetSpec:
    """Trunk depth/width plus the CNN's head geometry (32-filter 1x1 heads, 64 hidden value units)."""
    blocks: int
    channels: int = 64
    policy_channels: int = 32
    value_channels: int = 32
    value_hidden: int = 64


RESNET_SPECS = {RESNET3X64: ResNetSpec(blocks=3), RESNET5X64: ResNetSpec(blocks=5)}


def resnet_spec(architecture):
    if not isinstance(architecture, str) or architecture not in RESNET_SPECS:
        raise ValueError(f"Unregistered AlphaZero ResNet architecture: {architecture}")
    return RESNET_SPECS[architecture]


class ResidualBlock(nn.Module):
    """conv3x3-BN-ReLU-conv3x3-BN, identity skip, then ReLU (AlphaGo Zero post-activation)."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))


class AlphaZeroResNet(nn.Module):
    """Residual trunk with the v2 policy/value heads; 322,440 (3x64) or 470,408 (5x64) parameters."""

    def __init__(self, architecture):
        super().__init__()
        spec = resnet_spec(architecture)
        # Never the CNN's ARCHITECTURE, so V2Inference/V2Trainer/frozen_copy reject this model.
        self.alphazero_v2_contract = architecture
        self.stem_conv = nn.Conv2d(1, spec.channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(spec.channels)
        self.tower = nn.Sequential(*(ResidualBlock(spec.channels) for _ in range(spec.blocks)))
        self.policy_conv = nn.Conv2d(spec.channels, spec.policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(spec.policy_channels)
        self.policy_fc = nn.Linear(spec.policy_channels * 6 * 7, 7)
        self.value_conv = nn.Conv2d(spec.channels, spec.value_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(spec.value_channels)
        self.value_fc1 = nn.Linear(spec.value_channels * 6 * 7, spec.value_hidden)
        self.value_fc2 = nn.Linear(spec.value_hidden, 1)

    def forward(self, states):
        """Return (raw policy logits (N,7), tanh value (N,1)) for the player to move."""
        validate_input(states)
        shared = self.tower(F.relu(self.stem_bn(self.stem_conv(states))))
        policy = self.policy_fc(F.relu(self.policy_bn(self.policy_conv(shared))).flatten(1))
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(
            F.relu(self.value_bn(self.value_conv(shared))).flatten(1)))))
        validate_outputs(policy, value, states.shape[0])
        return policy, value


@lru_cache(maxsize=None)
def _layout(architecture):
    """Expected state-dict name -> (shape, dtype), built without advancing the global torch RNG."""
    with torch.random.fork_rng(devices=[]):
        reference = AlphaZeroResNet(architecture)
    return {name: (tuple(tensor.shape), tensor.dtype) for name, tensor in reference.state_dict().items()}


def validate_state(state, architecture):
    """Exact names, shapes and dtypes; finite CPU floats; valid batch-norm statistics/counters."""
    resnet_spec(architecture)
    layout = _layout(architecture)
    if not isinstance(state, dict) or state.keys() != layout.keys():
        raise ValueError(f"State keys do not match {architecture}")
    for name, tensor in state.items():
        shape, dtype = layout[name]
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype
                or tensor.device.type != "cpu" or not torch.isfinite(tensor).all()
                or (name.endswith(("running_var", "num_batches_tracked")) and (tensor < 0).any())):
            raise ValueError(f"Invalid {architecture} tensor: {name}")


def require_resnet_model(model, architecture=None):
    """A registered ResNet whose declared identity matches its actual tensors (and the caller's)."""
    declared = getattr(model, "alphazero_v2_contract", None)
    if (type(model) is not AlphaZeroResNet or not isinstance(declared, str) or declared not in RESNET_SPECS
            or (architecture is not None and declared != architecture)):
        raise ValueError("Model must be a registered AlphaZero ResNet of the expected architecture")
    validate_state(model.state_dict(), declared)


def initial_resnet(architecture, seed):
    """Fresh PyTorch-default initialization, seed- and architecture-determined; global RNG untouched."""
    resnet_spec(architecture)
    init_seed = int(hashlib.sha256(
        f"{INITIALIZATION_DOMAIN}:{architecture}:{seed}".encode()).hexdigest(), 16) % 2 ** 63
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        return AlphaZeroResNet(architecture).eval()


def model_state(model, architecture=None):
    require_resnet_model(model, architecture)
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def model_from_state(state, architecture):
    validate_state(state, architecture)
    with torch.random.fork_rng(devices=[]):  # Loading never advances the global torch RNG.
        model = AlphaZeroResNet(architecture)
    model.load_state_dict(state, strict=True)
    return model.eval()


def frozen_copy(model, architecture=None):
    """Independent eval-mode snapshot (weights and running statistics) with gradients disabled."""
    require_resnet_model(model, architecture)
    snapshot = deepcopy(model).eval()
    for parameter in snapshot.parameters():
        parameter.requires_grad_(False)
    return snapshot


class ResNetInference(V2Inference):
    """V2Inference's predict/predict_batch over a ResNet; PUCT and batching accept it unchanged."""

    def __init__(self, model):
        require_resnet_model(model)
        self.model = model
        self._check_eval()


def parameter_groups(model):
    """(decay names, no-decay names): conv/linear weights decay; biases and batch-norm affines never do."""
    decay, no_decay = [], []
    for prefix, module in model.named_modules():
        for name, _ in module.named_parameters(recurse=False):
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name == "weight":
                decay.append(f"{prefix}.{name}")
            elif isinstance(module, (nn.Linear, nn.BatchNorm2d)) and name in ("weight", "bias"):
                no_decay.append(f"{prefix}.{name}")
            else:
                raise ValueError(f"Unclassified parameter for weight decay: {prefix}.{name}")
    return decay, no_decay


def make_optimizer(model, config):
    decay, no_decay = parameter_groups(model)
    parameters = dict(model.named_parameters())
    return torch.optim.AdamW(
        [dict(params=[parameters[n] for n in decay], weight_decay=config.weight_decay),
         dict(params=[parameters[n] for n in no_decay], weight_decay=0.0)],
        lr=config.learning_rate, betas=config.adam_betas, eps=config.adam_eps)


class ResNetTrainer(V2Trainer):
    """V2Trainer's loss, clipping and AdamW settings with batch-norm-aware grouping and failure.

    The train-mode forward updates running statistics before the optimizer
    step, so ANY failed step leaves the model off its last valid boundary and
    invalidates the trainer, not only nonfinite parameters.
    """

    def __init__(self, model, config=V2Config()):
        require_resnet_model(model)
        model.eval()
        self.model, self.config = model, config
        self.optimizer = make_optimizer(model, config)
        self.steps = 0
        self.failed = False
        self.last_metrics = None

    def step(self, examples):
        try:
            metrics = super().step(examples)
            if any(not torch.isfinite(buffer).all() for buffer in self.model.buffers()):
                raise ValueError("Nonfinite batch-norm statistics after update; discard this trainer")
            return metrics
        except BaseException:
            self.failed = True
            raise


def inference_contract(architecture):
    return dict(deepcopy(INFERENCE_CONTRACT), architecture=architecture,
                architecture_spec=asdict(resnet_spec(architecture)), normalization=NORMALIZATION)


class ResNetFactory:
    """CNNFactory's adapter surface, bound to exactly one registered ResNet identity."""
    hash = staticmethod(weights_sha256)

    def __init__(self, architecture):
        self.spec = resnet_spec(architecture)
        self.architecture = architecture
        self.output_contract = inference_contract(architecture)

    def construct(self, seed):
        return initial_resnet(self.architecture, seed)

    def restore(self, state):
        return model_from_state(state, self.architecture)

    def state(self, model):
        return model_state(model, self.architecture)

    def freeze(self, model):
        return frozen_copy(model, self.architecture)

    def inference(self, model):
        require_resnet_model(model, self.architecture)
        return ResNetInference(model)

    def trainer(self, model, config):
        require_resnet_model(model, self.architecture)
        return ResNetTrainer(model, config)

    def restore_optimizer(self, trainer, state, steps):
        """Also binds batch-norm statistics to the counters: one train-mode forward per trainer step."""
        from .experimental import validate_optimizer
        require_resnet_model(trainer.model, self.architecture)
        if any(int(buffer) != steps for name, buffer in trainer.model.named_buffers()
               if name.endswith("num_batches_tracked")):
            raise ValueError("Batch-norm update counters differ from trainer steps")
        validate_optimizer(trainer, state, steps)


def save_inference_checkpoint(path, model):
    """Write a minimal ResNet inference artifact, refusing to overwrite any file."""
    from .artifacts import atomic_torch_save
    require_resnet_model(model)
    architecture = model.alphazero_v2_contract
    payload = {"contract": inference_contract(architecture), "model_state_dict": model_state(model)}

    def validate(temporary):
        state = load_inference_checkpoint(temporary, architecture).model.state_dict()
        if any(not torch.equal(state[name], tensor) for name, tensor in payload["model_state_dict"].items()):
            raise ValueError("Reloaded inference tensors differ")
    return atomic_torch_save(path, payload, validate)


def load_inference_checkpoint(path, architecture):
    """Weights-only CPU load; the caller must name the one architecture it expects."""
    contract = inference_contract(architecture)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or set(checkpoint) != {"contract", "model_state_dict"}
            or checkpoint["contract"] != contract):
        raise ValueError(f"Expected a {architecture} inference checkpoint")
    return ResNetInference(model_from_state(checkpoint["model_state_dict"], architecture))
