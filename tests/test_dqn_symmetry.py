"""Synthetic symmetry verification; no learning-strength-dependent assertions."""

from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path
from random import Random

import pytest

torch = pytest.importorskip('torch')

from games.connect4.connect4 import Connect4
from games.connect4.dqn.diagnostics import FIXTURES, position
from games.connect4.dqn.experiment import Bounds, parser, run_training
from games.connect4.dqn.network import checked_q
from games.connect4.dqn.training import (
    DQNConfig, DQNTrainer, Transition, bellman_targets, collect_transition,
    reflect_transition, selected_action_loss,
)

DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]
WIN = [0, 6, 1, 6, 2, 5, 3]


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('moves,action,actor,reward,done', [
    ([], 0, 0, 0, False), ([3], 1, 1, 0, False),
    ([0] * 6, 2, 0, 0, False), ([0] * 6 + [2], 4, 1, 0, False),
    ([0, 1, 0, 1, 0, 2], 0, 0, 1, True),
    ([0, 1, 0, 1, 2, 1, 2], 1, 1, 1, True),
    (DRAW[:-1], DRAW[-1], 1, 0, True),
])
def test_reflection_contract_and_independent_legal_sequences(moves, action, actor, reward, done):
    original = collect_transition(position(moves), action)
    before = deepcopy(original)
    mirrored = reflect_transition(original)
    independent = collect_transition(position([6 - c for c in moves]), 6 - action)
    assert mirrored == independent
    assert mirrored is not original and original == before
    assert (mirrored.actor, mirrored.reward, mirrored.done) == (actor, reward, done)
    assert mirrored.action == 6 - action
    assert mirrored.next_legal == original.next_legal[::-1]
    for row in range(6):
        for col in range(7):
            assert mirrored.state[row * 7 + col] == original.state[row * 7 + 6 - col]
            assert mirrored.next_state[row * 7 + col] == original.next_state[row * 7 + 6 - col]
    assert reflect_transition(mirrored) == original
    with pytest.raises(FrozenInstanceError):
        mirrored.action = 0
    with pytest.raises(TypeError):
        mirrored.state[0] = 1


@pytest.mark.parametrize('fixture', FIXTURES, ids=lambda f: f['name'])
def test_every_frozen_legal_position_successor_mirrors(fixture):
    for action in position(fixture['moves']).get_valid_moves():
        original = collect_transition(position(fixture['moves']), action)
        mirrored = collect_transition(position([6 - c for c in fixture['moves']]), 6 - action)
        assert reflect_transition(original) == mirrored
        assert reflect_transition(mirrored) == original


def test_reflection_reconstructs_validation(monkeypatch):
    transition = collect_transition(Connect4(), 0)
    original = Transition.__post_init__
    calls = []
    def checked(self):
        calls.append(self)
        original(self)
    monkeypatch.setattr(Transition, '__post_init__', checked)
    assert reflect_transition(transition) == calls[0]
    assert len(calls) == 1
    with pytest.raises(TypeError):
        reflect_transition(object())


@pytest.mark.parametrize('value', [-0.01, 1.01, True, None, float('inf'), float('nan')])
def test_probability_validation(value):
    with pytest.raises(ValueError, match='horizontal_symmetry_probability'):
        DQNConfig(horizontal_symmetry_probability=value)


def test_cli_default_and_experimental_probability():
    assert DQNConfig().horizontal_symmetry_probability == 0.0
    assert parser().parse_args(['--output', '/tmp/test']).horizontal_symmetry_probability == 0.0
    assert parser().parse_args(['--output', '/tmp/test', '--horizontal-symmetry-probability', '0.5']).horizontal_symmetry_probability == 0.5


def test_seeded_independent_per_sample_decisions_and_rng_ownership():
    config = DQNConfig(seed=42, horizontal_symmetry_probability=0.5)
    first, second = DQNTrainer(config), DQNTrainer(config)
    batch = [collect_transition(Connect4(), 0)] * 100
    reference = Random('connect4-horizontal-symmetry-v1:42')
    decisions = [reference.random() < 0.5 for _ in batch]
    assert any(decisions) and not all(decisions)
    exploration, replay = first.rng.getstate(), first.memory._rng.getstate()
    global_state, torch_state = __import__('random').getstate(), torch.get_rng_state().clone()
    actual = first.augment_batch(batch)
    assert actual == second.augment_batch(batch)
    assert [t.action == 6 for t in actual] == decisions
    assert first.augmentation_rng is not first.rng
    assert first.augmentation_rng is not first.memory._rng
    assert first.rng.getstate() == exploration and first.memory._rng.getstate() == replay
    assert __import__('random').getstate() == global_state and torch.equal(torch.get_rng_state(), torch_state)
    assert first.augmentation_summary()['transformed'] == sum(decisions)
    assert first.augmentation_summary()['untransformed'] == 100 - sum(decisions)
    # Exploration and sampling draws cannot perturb future augmentation decisions.
    first.rng.random()
    first.memory._rng.random()
    assert first.augment_batch(batch) == second.augment_batch(batch)
    third = DQNTrainer(replace(config, seed=43))
    assert third.augment_batch(batch) != actual


def legacy_step(trainer):
    """Pre-augmentation numerical optimization path, independent of augment_batch."""
    batch = trainer.memory.sample(trainer.config.batch_size)
    states = torch.tensor([t.state for t in batch], dtype=torch.float32)
    actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
    targets = bellman_targets(trainer.target, batch, trainer.config.gamma)
    trainer.online.train()
    loss = selected_action_loss(checked_q(trainer.online, states), actions, targets)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainer.optimizer.step()
    trainer.updates += 1
    trainer.epsilon = max(trainer.config.epsilon_min, trainer.epsilon * trainer.config.epsilon_decay)
    if trainer.updates % trainer.config.target_sync_interval == 0:
        trainer.sync_target()
    return loss.item()


def test_disabled_exact_legacy_weights_losses_rng_and_sync():
    config = DQNConfig(seed=42, batch_size=4, target_sync_interval=2)
    current, legacy = DQNTrainer(config), DQNTrainer(config)
    augmentation_rng = current.augmentation_rng.getstate()
    for c in range(7):
        for trainer in (current, legacy):
            trainer.memory.append(collect_transition(Connect4(), c))
    for _ in range(4):
        assert current.optimize() == legacy_step(legacy)
        assert current.choose_move(Connect4()) == legacy.choose_move(Connect4())
        assert current.rng.getstate() == legacy.rng.getstate()
        assert current.memory._rng.getstate() == legacy.memory._rng.getstate()
        assert current.epsilon == legacy.epsilon and current.updates == legacy.updates
        for name in ('online', 'target'):
            assert all(torch.equal(a, b) for a, b in zip(getattr(current, name).parameters(), getattr(legacy, name).parameters()))
    assert current.augmentation_rng.getstate() == augmentation_rng
    assert current.augmentation_summary()['transformed'] == 0
    assert current.augmentation_summary()['untransformed'] == 16


@pytest.mark.parametrize('probability', [0.0, 0.5, 1.0])
def test_replay_replacement_sample_composition_and_driver_accounting(probability, monkeypatch):
    trainer = DQNTrainer(DQNConfig(seed=42, batch_size=2, replay_capacity=5,
                                   target_sync_interval=3, horizontal_symmetry_probability=probability))
    actions = iter(WIN * 2)
    trainer.choose_move = lambda game: next(actions)
    original_augment = trainer.augment_batch
    def observed(batch):
        before = tuple(trainer.memory._items)
        result = original_augment(batch)
        assert tuple(trainer.memory._items) == before
        assert len(result) == len(batch) == 2
        assert Counter((t.reward, t.done, t.actor) for t in result) == Counter((t.reward, t.done, t.actor) for t in batch)
        return result
    monkeypatch.setattr(trainer, 'augment_batch', observed)
    report = run_training(trainer, Bounds(max_games=2))
    assert report['status'] == 'bounded_stop' and report['stop_reason'] == 'max_games'
    assert report['plies'] == 14 and report['updates'] == 13
    assert report['completed_games'] == 2 and report['partial_episode'] is None
    assert report['replay_size'] == 5 and report['collected_terminal'] == 2
    assert report['target_sync_updates'] == [3, 6, 9, 12]
    assert report['augmentation']['total'] == 26
    game = Connect4()
    expected = [collect_transition(game, c) for c in WIN]
    assert list(trainer.memory._items) == expected[-5:]
    if probability in (0, 1):
        assert report['augmentation']['transformed'] == 26 * probability


def test_augmented_optimization_is_deterministic_and_warmup_draws_nothing():
    trainers = [DQNTrainer(DQNConfig(seed=42, batch_size=4, horizontal_symmetry_probability=0.5)) for _ in range(2)]
    for trainer in trainers:
        rng = trainer.augmentation_rng.getstate()
        assert trainer.optimize() is None
        assert trainer.augmentation_rng.getstate() == rng
        assert trainer.augmentation_summary()['total'] == 0
        for c in range(7):
            trainer.memory.append(collect_transition(Connect4(), c))
    for _ in range(3):
        assert trainers[0].optimize() == trainers[1].optimize()
    assert trainers[0].augmentation_summary() == trainers[1].augmentation_summary()
    assert all(torch.equal(a, b) for a, b in zip(trainers[0].online.parameters(), trainers[1].online.parameters()))


def test_mirrored_signed_bellman_targets_mask_and_terminal_exclusion():
    class FixedQ(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = torch.tensor(values, dtype=torch.float32)
            self.seen = []
        def forward(self, states):
            assert not torch.is_grad_enabled()
            self.seen.append(states.clone())
            return self.values.expand(len(states), -1)
    batch = [collect_transition(position([0] * 6), 2),
             collect_transition(position([0] * 6 + [2]), 3),
             collect_transition(position([0, 1, 0, 1, 0, 2]), 0),
             collect_transition(position([0, 1, 0, 1, 2, 1, 2]), 1),
             collect_transition(position(DRAW[:-1]), DRAW[-1])]
    mirrored = [reflect_transition(t) for t in batch]
    values = [999, -3, 0.5, -1, -2, -4, -5]
    original_net, mirrored_net = FixedQ(values), FixedQ(values[::-1])
    expected = torch.tensor([-0.495, -0.495, 1, 1, 0], dtype=torch.float32)
    assert torch.equal(bellman_targets(original_net, batch, 0.99), expected)
    assert torch.equal(bellman_targets(mirrored_net, mirrored, 0.99), expected)
    assert mirrored_net.seen[0].tolist() == [list(t.next_state) for t in mirrored[:2]]
    assert len(mirrored_net.seen) == 1


def test_frozen_suite_hash():
    path = Path(__file__).resolve().parents[1] / 'games/connect4/dqn/diagnostic_positions.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'ec99e0ee080370ab8cfc678c00a49ff552466de73ffe0955778ac87ee340543a'
