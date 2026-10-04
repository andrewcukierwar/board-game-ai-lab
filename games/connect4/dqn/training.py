"""One-ply adversarial DQN primitives; no training loop or import-time work.

Q(s,a) is for the player to move in s. The successor is encoded for the
opponent, so the bootstrap is SUBTRACTED: r - gamma * max_legal Q_target(s').
A win by either actor earns +1; draws/ongoing moves earn 0. Terminal samples
never invoke the target network. Connect4.step's fixed-X rewards are unused.
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
import random

import torch
from torch.nn import functional as F

from .encoding import encode_game, legal_mask, playable_state, validate_state
from .network import DQN, checked_q
from ..agents.dqn_agent import DQNAgent


def positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def unit_interval(name, value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be finite and in [0, 1]')


def copied_state(state):
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().clone().numpy()
    return validate_state(state)


@dataclass(frozen=True)
class Transition:
    state: tuple
    action: int
    reward: float
    next_state: tuple
    done: bool
    next_legal: tuple
    actor: int

    def __post_init__(self):
        object.__setattr__(self, 'state', copied_state(self.state))
        object.__setattr__(self, 'next_state', copied_state(self.next_state))
        mask = tuple(self.next_legal)
        if len(mask) != 7 or any(type(v) is not bool for v in mask):
            raise ValueError('next_legal must contain seven booleans')
        object.__setattr__(self, 'next_legal', mask)
        if type(self.actor) is not int or self.actor not in (0, 1):
            raise ValueError('actor must be 0 or 1')
        if type(self.action) is not int or not 0 <= self.action < 7:
            raise ValueError('action must be a column in 0..6')
        if not legal_mask(self.state)[self.action]:
            raise ValueError('Selected action is illegal')
        # Verify the representation boundary, including the perspective flip.
        # A legal drop adds the actor's +1 before negating for the next player.
        expected_next = list(self.state)
        row = next(r for r in range(5, -1, -1) if self.state[r * 7 + self.action] == 0)
        expected_next[row * 7 + self.action] = 1
        if self.next_state != tuple(-v for v in expected_next):
            raise ValueError('next_state must be one legal ply in the next player perspective')
        if type(self.done) is not bool:
            raise ValueError('done must be boolean')
        if type(self.reward) not in (int, float) or self.reward not in (0, 1):
            raise ValueError('Actor reward must be 0 or 1')
        if not self.done and self.reward != 0:
            raise ValueError('Nonterminal reward must be 0')
        if not self.done and not any(mask):
            raise ValueError('Nonterminal position has no legal actions')
        expected = (False,) * 7 if self.done else legal_mask(self.next_state)
        if mask != expected:
            raise ValueError('next_legal disagrees with next state / terminal status')


def reflect_transition(transition):
    """Reflect physical columns only, reconstructing the validated contract.

    Row order, piece signs and actor identity are unchanged. Replay snapshots
    are immutable; the reflected transition is a new independent instance.
    """
    if not isinstance(transition, Transition):
        raise TypeError('Reflection requires a validated Transition')

    def reflect(state):
        return tuple(state[row * 7 + col] for row in range(6) for col in range(6, -1, -1))

    return Transition(state=reflect(transition.state), action=6 - transition.action,
                      reward=transition.reward, next_state=reflect(transition.next_state),
                      done=transition.done, next_legal=transition.next_legal[::-1],
                      actor=transition.actor)


def collect_transition(game, action):
    """Advance game by exactly one legal ply and snapshot both perspectives.

    This is the explicit mutating collection boundary. Inference never uses it.
    Capture actor BEFORE make_move, which flips the engine turn even on a win.
    """
    state, mask = playable_state(game)
    if type(action) is not int or not 0 <= action < 7 or not mask[action]:
        raise ValueError('action must be a legal column')
    actor = game.current_player
    if not game.make_move(action):
        raise ValueError('Engine rejected a legal action')
    next_state = encode_game(game)
    winner = game.check_winner()
    if winner not in (-1, actor):
        raise ValueError('A legal move cannot award a win to the opponent')
    done = game.is_game_over()
    return Transition(state, action, float(winner == actor), next_state, done,
                      (False,) * 7 if done else legal_mask(next_state), actor)


class ReplayMemory:
    def __init__(self, capacity, *, seed=None):
        positive_int('capacity', capacity)
        self._items = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self._items)

    def append(self, transition):
        if not isinstance(transition, Transition):
            raise TypeError('Replay accepts validated Transition instances')
        # Reconstruct at the ownership boundary, including detached tuple copies.
        self._items.append(Transition(**vars(transition)))

    def composition(self):
        """Read-only snapshot; does not consume replay RNG or change sampling."""
        return dict(size=len(self), terminal=sum(t.done for t in self._items),
                    nonterminal=sum(not t.done for t in self._items),
                    rewards={str(r): sum(t.reward == r for t in self._items) for r in (0, 1)},
                    actors={str(a): sum(t.actor == a for t in self._items) for a in (0, 1)})

    def sample(self, batch_size):
        positive_int('batch_size', batch_size)
        if batch_size > len(self):
            raise ValueError('Not enough replay samples')
        return self._rng.sample(list(self._items), batch_size)


def bellman_targets(target_network, transitions, gamma):
    unit_interval('gamma', gamma)
    if not transitions:
        raise ValueError('Empty transition batch')
    with torch.no_grad():
        targets = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        active = [i for i, t in enumerate(transitions) if not t.done]
        if active:
            masks = torch.tensor([transitions[i].next_legal for i in active], dtype=torch.bool)
            if not masks.any(dim=1).all().item():
                raise ValueError('Nonterminal position has no legal actions')
            states = torch.tensor([transitions[i].next_state for i in active], dtype=torch.float32)
            target_network.eval()
            q = checked_q(target_network, states)
            targets[active] -= gamma * q.masked_fill(~masks, -torch.inf).max(dim=1).values
        return targets


def selected_action_loss(q_values, actions, targets):
    """Mean Huber loss on Q(s,a) only; target values never carry gradients."""
    if q_values.ndim != 2 or q_values.shape[1] != 7 or len(q_values) == 0:
        raise ValueError('Expected Q-values of shape (N, 7)')
    if actions.dtype != torch.int64 or actions.shape != (len(q_values),):
        raise ValueError('Expected int64 actions of shape (N,)')
    if ((actions < 0) | (actions > 6)).any().item():
        raise ValueError('Action outside 0..6')
    if targets.shape != (len(q_values),) or not torch.isfinite(targets).all().item():
        raise ValueError('Expected finite targets of shape (N,)')
    if not torch.isfinite(q_values).all().item():
        raise ValueError('Q-values must be finite')
    return F.smooth_l1_loss(q_values.gather(1, actions[:, None]).squeeze(1), targets.detach())


@dataclass(frozen=True)
class DQNConfig:
    replay_capacity: int = 10000
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    target_sync_interval: int = 100
    seed: int = 0
    horizontal_symmetry_probability: float = 0.0

    def __post_init__(self):
        for name in ('replay_capacity', 'batch_size', 'target_sync_interval'):
            positive_int(name, getattr(self, name))
        if self.batch_size > self.replay_capacity:
            raise ValueError('batch_size exceeds replay_capacity')
        for name in ('gamma', 'epsilon_start', 'epsilon_min', 'epsilon_decay',
                     'horizontal_symmetry_probability'):
            unit_interval(name, getattr(self, name))
        if self.epsilon_min > self.epsilon_start:
            raise ValueError('epsilon_min exceeds epsilon_start')
        if (type(self.learning_rate) not in (int, float)
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError('learning_rate must be positive and finite')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')


class DQNTrainer:
    """CPU foundation. One optimize call = one sampled batch / optimizer step.

    Epsilon decays (clamped at its floor) and target sync is scheduled only on
    successful optimizer steps. No episode driver, self-play or automatic save.
    """

    def __init__(self, config=None):
        self.config = DQNConfig() if config is None else config
        if not isinstance(self.config, DQNConfig):
            raise TypeError('config must be DQNConfig')
        # Seed initialization without changing the caller's global CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.online = DQN()
        self.target = deepcopy(self.online).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.memory = ReplayMemory(self.config.replay_capacity, seed=self.config.seed)
        self.rng = random.Random(self.config.seed)
        # Domain-separated deterministic stream; never share exploration/replay RNGs.
        self.augmentation_seed = f'connect4-horizontal-symmetry-v1:{self.config.seed}'
        self.augmentation_rng = random.Random(self.augmentation_seed)
        self.transformed_samples = 0
        self.untransformed_samples = 0
        self.epsilon = self.config.epsilon_start
        self.updates = 0

    def choose_move(self, game):
        _, mask = playable_state(game)
        if self.rng.random() < self.epsilon:
            return self.rng.choice([i for i, allowed in enumerate(mask) if allowed])
        return DQNAgent(self.online).choose_move(game)

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict(), strict=True)
        self.target.eval().requires_grad_(False)

    def augment_batch(self, batch):
        """One independent decision per sampled item; never insert into replay.

        Counts describe constructed optimization samples, even if a subsequent
        optimizer invariant fails. Disabled augmentation consumes no RNG draws.
        """
        probability = self.config.horizontal_symmetry_probability
        if probability == 0:
            self.untransformed_samples += len(batch)
            return batch
        augmented = []
        for transition in batch:
            if self.augmentation_rng.random() < probability:
                augmented.append(reflect_transition(transition))
                self.transformed_samples += 1
            else:
                augmented.append(transition)
                self.untransformed_samples += 1
        return augmented

    def augmentation_summary(self):
        total = self.transformed_samples + self.untransformed_samples
        return dict(horizontal_symmetry_probability=self.config.horizontal_symmetry_probability,
                    rng='python.random.Random', seed=self.augmentation_seed,
                    transformed=self.transformed_samples, untransformed=self.untransformed_samples,
                    total=total, observed_fraction=self.transformed_samples / total if total else None)

    def optimize(self):
        if len(self.memory) < self.config.batch_size:
            return None
        batch = self.augment_batch(self.memory.sample(self.config.batch_size))
        states = torch.tensor([t.state for t in batch], dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
        targets = bellman_targets(self.target, batch, self.config.gamma)
        self.online.train()
        loss = selected_action_loss(checked_q(self.online, states), actions, targets)
        if not torch.isfinite(loss).item():
            raise ValueError('Nonfinite loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all().item()
               for p in self.online.parameters()):
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError('Nonfinite gradients')
        self.optimizer.step()
        self.updates += 1
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        if self.updates % self.config.target_sync_interval == 0:
            self.sync_target()
        return float(loss.detach().item())
