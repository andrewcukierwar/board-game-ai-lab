"""One changed objective; historical raw-logit CE, MSE and optimizer stay intact."""
import torch
from games.connect4.alphazero_v2.training import training_loss, batch_tensors

COEFFICIENT = {'control': 0.0, 'treatment': 0.2}


def loss(logits, values, policies, outcomes, arm):
    original, ce, mse = training_loss(logits, values, policies, outcomes)
    penalty = COEFFICIENT[arm] * values.square().mean()
    return original + penalty, ce, mse, penalty


def step(runner, examples, arm):
    """Same train/eval modes, clipping, AdamW, step counts and finite checks as V2Trainer."""
    trainer, model, config = runner.trainer, runner.model, runner.config.training
    if trainer.failed:
        raise RuntimeError('Failed trainer cannot continue')
    tensors = batch_tensors(examples)
    parameters = [p for p in model.parameters() if p.requires_grad]
    model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    try:
        total, ce, mse, penalty = loss(*model(tensors[0]), *tensors[1:], arm)
        total.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('Missing/nonfinite gradient')
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
        trainer.optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            trainer.failed = True
            raise ValueError('Nonfinite optimizer result')
        trainer.steps += 1
        return dict(loss=float(total.detach()), ce=float(ce.detach()), mse=float(mse.detach()),
                    penalty=float(penalty.detach()), gradient_norm=float(norm),
                    clipped=bool(norm > config.max_grad_norm))
    finally:
        trainer.optimizer.zero_grad(set_to_none=True)
        model.eval()
