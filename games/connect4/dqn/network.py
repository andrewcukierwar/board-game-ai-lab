"""Historical 42 -> 128 -> 128 -> 7 MLP, with raw (unbounded) Q outputs."""

import torch
from torch import nn

ARCHITECTURE = 'connect4-dqn-42-128-128-7-relu-v1'


def validate_batch(states):
    if (not isinstance(states, torch.Tensor) or states.dtype != torch.float32
            or states.ndim != 2 or states.shape[1] != 42 or states.shape[0] == 0):
        raise ValueError('Expected nonempty float32 states of shape (N, 42)')
    if not ((states == -1) | (states == 0) | (states == 1)).all().item():
        raise ValueError('State values must be -1, 0 or 1')


def checked_q(network, states):
    validate_batch(states)
    values = network(states)
    if (not isinstance(values, torch.Tensor) or values.shape != (len(states), 7)
            or values.dtype != torch.float32 or values.device != states.device
            or not torch.isfinite(values).all().item()):
        raise ValueError('Network must return finite float32 Q-values of shape (N, 7)')
    return values


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(42, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 7)

    def forward(self, states):
        validate_batch(states)
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(states)))))
