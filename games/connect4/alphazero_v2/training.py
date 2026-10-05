"""V2 loss and persistent AdamW trainer; no tactical anchoring or v1 W/D/L targets."""
import torch
from torch.nn import functional as F

from .config import V2Config
from .data import V2Example
from .network import require_v2_model, validate_outputs


def batch_tensors(examples):
    """States (N,1,6,7), visit targets (N,7) and scalar z (N,1), all CPU float32."""
    if not examples or any(not isinstance(e, V2Example) or e.outcome is None for e in examples):
        raise ValueError("A nonempty batch of completed V2Examples is required")
    states = torch.tensor([e.observation for e in examples], dtype=torch.float32).unsqueeze(1)
    policies = torch.tensor([e.policy_target for e in examples], dtype=torch.float32)
    outcomes = torch.tensor([[float(e.outcome)] for e in examples], dtype=torch.float32)
    return states, policies, outcomes


def training_loss(policy_logits, values, policies, outcomes):
    """Return (combined, policy CE, value MSE), each a minibatch mean, equal weight 1.

    Policy CE uses all seven FINITE raw logits; illegal target entries are zero,
    so no -inf masked log-probabilities ever enter the loss.
    """
    if not isinstance(policies, torch.Tensor) or policies.ndim != 2 or policies.shape[0] < 1:
        raise ValueError("Policy targets require a nonempty (N,7) tensor")
    n = policies.shape[0]
    validate_outputs(policy_logits, values, n)
    if (policies.shape != (n, 7) or policies.dtype != torch.float32 or policies.device.type != "cpu"
            or not torch.isfinite(policies).all() or (policies < 0).any()
            or not torch.allclose(policies.sum(1), torch.ones(n), rtol=0, atol=1e-6)):
        raise ValueError("Policy targets must be finite normalized CPU float32 (N,7)")
    if (not isinstance(outcomes, torch.Tensor) or outcomes.shape != (n, 1) or outcomes.dtype != torch.float32
            or outcomes.device.type != "cpu" or not ((outcomes == -1) | (outcomes == 0) | (outcomes == 1)).all()):
        raise ValueError("Value targets must be CPU float32 (N,1) in {-1,0,+1}")
    policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(1).mean()
    value_mse = F.mse_loss(values, outcomes)
    combined = policy_loss + value_mse
    if not torch.isfinite(combined):
        raise ValueError("Nonfinite training loss")
    return combined, policy_loss, value_mse


def parameter_groups(model):
    """(decay names, no-decay names): conv/linear weights decay; biases never do."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if name.endswith(".bias") and parameter.ndim == 1:
            no_decay.append(name)
        elif name.endswith(".weight") and parameter.ndim > 1:
            decay.append(name)
        else:
            raise ValueError(f"Unclassified parameter for weight decay: {name}")
    return decay, no_decay


def make_optimizer(model, config):
    decay, no_decay = parameter_groups(model)
    parameters = dict(model.named_parameters())
    return torch.optim.AdamW(
        [dict(params=[parameters[n] for n in decay], weight_decay=config.weight_decay),
         dict(params=[parameters[n] for n in no_decay], weight_decay=0.0)],
        lr=config.learning_rate, betas=config.adam_betas, eps=config.adam_eps)


class V2Trainer:
    """Owns one persistent AdamW for the learner; never recreated across generations.

    Mode boundary: eval between updates, train during an update, eval on return
    (including failure). Nonfinite loss/gradients abort before the optimizer
    step; nonfinite parameters after a step invalidate the trainer.
    """

    def __init__(self, model, config=V2Config()):
        require_v2_model(model)
        model.eval()
        self.model, self.config = model, config
        self.optimizer = make_optimizer(model, config)
        self.steps = 0
        self.failed = False
        self.last_metrics = None

    def step(self, examples):
        if self.failed:
            raise RuntimeError("Trainer failed a numerical check; recover from a valid boundary")
        states, policies, outcomes = batch_tensors(examples)
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            combined, policy_loss, value_mse = training_loss(*self.model(states), policies, outcomes)
            combined.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("Missing or nonfinite training gradients")
            unclipped = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm,
                                                       error_if_nonfinite=True)
            clipped = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in parameters]))
            self.optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                self.failed = True
                raise ValueError("Nonfinite parameters after update; discard this trainer")
            self.steps += 1
            self.last_metrics = dict(
                combined_loss=float(combined.detach()), policy_loss=float(policy_loss.detach()),
                value_mse=float(value_mse.detach()), gradient_norm=float(unclipped),
                clipped_gradient_norm=float(clipped), clipped=bool(unclipped > self.config.max_grad_norm))
            return self.last_metrics
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval()
