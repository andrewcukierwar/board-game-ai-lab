"""Standalone deterministic inference; importing this module requires torch."""

import torch

from ..dqn.encoding import playable_state
from ..dqn.network import checked_q


class DQNAgent:
    """Accept a preloaded CPU float32 Q-network using the canonical contract.

    Ties choose the lowest legal physical column. The caller owns the network;
    do not train it concurrently with inference. No environment or replay is held.
    """

    def __init__(self, network):
        if not isinstance(network, torch.nn.Module):
            raise TypeError('network must be a torch.nn.Module')
        if any(t.device.type != 'cpu' or (t.is_floating_point() and t.dtype != torch.float32)
               for t in (*network.parameters(), *network.buffers())):
            raise ValueError('DQN inference currently requires a CPU float32 network')
        self.network = network
        self.network.eval()

    def choose_move(self, game):
        state, mask = playable_state(game)
        self.network.eval()
        with torch.inference_mode():
            values = checked_q(self.network, torch.tensor([state], dtype=torch.float32))[0]
            return int(values.masked_fill(~torch.tensor(mask), -torch.inf).argmax().item())
