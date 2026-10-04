"""Controlled correctness checks, never self-play or learned-strength claims."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import math
from random import Random

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from games.connect4.connect4 import Connect4
from games.connect4.agents.dqn_agent import DQNAgent
from games.connect4.dqn.encoding import encode_game, legal_mask, validate_state
from games.connect4.dqn.network import DQN
from games.connect4.dqn.training import (
    DQNConfig, DQNTrainer, ReplayMemory, Transition, bellman_targets,
    collect_transition, selected_action_loss,
)
from games.connect4.dqn.checkpoint import CONTRACT, load_checkpoint, save_checkpoint

DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]


def position(moves):
    game = Connect4()
    for move in moves:
        assert not game.is_game_over()
        assert game.is_valid_move(move)
        assert game.make_move(move)
    return game


class FixedQ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.seen = []

    def forward(self, states):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.seen.append(states.clone())
        return self.values.expand(len(states), -1)


class NeverCalled(torch.nn.Module):
    def forward(self, states):
        pytest.fail('Terminal targets must not evaluate the target network')


@pytest.mark.parametrize('moves', [[3], [3, 2]])
def test_encoding_both_players_row_major(moves):
    game = position(moves)
    state = encode_game(game)
    assert len(state) == 42
    assert state[5 * 7 + 3] == (-1 if game.current_player == 1 else 1)
    assert state[5 * 7 + 2] == (0 if len(moves) == 1 else -1)
    assert state[:35] == (0,) * 35
    assert state == encode_game(deepcopy(game))


@pytest.mark.parametrize('bad', [np.zeros((6, 7)), np.zeros(41), np.zeros((1, 42)),
                                [2] * 42, [float('nan')] * 42, ['0'] * 42, [True] * 42])
def test_strict_state_validation(bad):
    with pytest.raises(ValueError):
        validate_state(bad)


@pytest.mark.parametrize('corruption', ['shape', 'symbol', 'player', 'piece'])
def test_engine_validation(corruption):
    game = Connect4()
    if corruption == 'shape':
        game.board.pop()
    elif corruption == 'symbol':
        game.board[0][0] = '?'
    elif corruption == 'player':
        game.current_player = True
    else:
        game.piece = 'O'
    with pytest.raises(ValueError):
        encode_game(game)


def test_architecture_raw_q_and_batch_validation():
    net = DQN()
    assert sum(p.numel() for p in net.parameters()) == 22919
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.fc3.bias.copy_(torch.tensor([-3, 2, 4, 8, -2, 0, 1]))
    assert net(torch.zeros(2, 42)).tolist() == [[-3, 2, 4, 8, -2, 0, 1]] * 2
    for states in (torch.zeros(42), torch.zeros(1, 6, 7), torch.zeros(0, 42),
                   torch.zeros(1, 42, dtype=torch.float64), torch.full((1, 42), 0.5)):
        with pytest.raises(ValueError):
            net(states)


@pytest.mark.parametrize('moves', [[0] * 6, [0] * 6 + [2]])
def test_greedy_mask_perspective_and_no_mutation(moves):
    game = position(moves)
    before = deepcopy(game.__dict__)
    net = FixedQ([1000, -3, -2, -1, -4, -5, -6])
    agent = DQNAgent(net)
    assert [agent.choose_move(game) for _ in range(3)] == [3] * 3
    assert net.seen[0].tolist()[0] == list(encode_game(game))
    assert game.__dict__ == before
    assert not net.training and net.values.grad is None


def test_fixed_weight_greedy_tie_uses_physical_column_order():
    net = DQN()
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    assert DQNAgent(net).choose_move(Connect4()) == 0


@pytest.mark.parametrize('moves', [[0, 1, 0, 1, 0, 1, 0], DRAW])
def test_inference_and_collection_reject_terminal(moves):
    game = position(moves)
    before = deepcopy(game.__dict__)
    with pytest.raises(ValueError, match='terminal'):
        DQNAgent(NeverCalled()).choose_move(game)
    with pytest.raises(ValueError, match='terminal'):
        collect_transition(game, 3)
    assert game.__dict__ == before


@pytest.mark.parametrize('moves,action,actor', [([0, 1, 0, 1, 0, 2], 0, 0),
                                             ([0, 1, 0, 1, 2, 1, 2], 1, 1)])
def test_win_rewards_and_terminal_targets_for_both_players(moves, action, actor):
    game = position(moves)
    before = encode_game(game)
    t = collect_transition(game, action)
    assert t.actor == actor and game.current_player == 1 - actor
    assert t.state == before and t.next_state == encode_game(game)
    assert t.done and t.reward == 1 and t.next_legal == (False,) * 7
    assert bellman_targets(NeverCalled(), [t], 0.99).tolist() == [1]


def test_terminal_draw_target():
    t = collect_transition(position(DRAW[:-1]), DRAW[-1])
    assert t.done and t.reward == 0
    assert bellman_targets(NeverCalled(), [t], 0.9).tolist() == [0]


@pytest.mark.parametrize('moves', [[0] * 6, [0] * 6 + [2]])
@pytest.mark.parametrize('values,expected', [([999, 0.2, 0.5, -0.1, 0, 0.1, 0.3], -0.45),
                                          ([999, -2, -3, -4, -5, -6, -7], 1.8)])
def test_signed_legal_bellman_targets_both_actors(moves, values, expected):
    game = position(moves)
    actor = game.current_player
    t = collect_transition(game, 3)
    net = FixedQ(values)
    targets = bellman_targets(net, [t], 0.9)
    assert t.actor == actor and t.reward == 0 and not t.done
    assert targets.item() == pytest.approx(expected)
    assert net.seen[0].tolist()[0] == list(encode_game(game))
    assert not targets.requires_grad and net.values.grad is None


def test_mixed_batch_only_evaluates_nonterminal_rows():
    win = collect_transition(position([0, 1, 0, 1, 0, 2]), 0)
    draw = collect_transition(position(DRAW[:-1]), DRAW[-1])
    ongoing = collect_transition(Connect4(), 3)
    net = FixedQ([1, 2, 3, 4, 5, 6, 7])
    assert bellman_targets(net, [win, ongoing, draw], 0.5).tolist() == [1, -3.5, 0]
    assert net.seen[0].shape == (1, 42)


def test_invariant_no_legal_nonterminal_and_bad_mask():
    t = collect_transition(Connect4(), 3)
    with pytest.raises(ValueError, match='no legal'):
        replace(t, next_legal=(False,) * 7)
    with pytest.raises(ValueError, match='disagrees'):
        replace(t, next_legal=(False,) + (True,) * 6)


def test_replay_copies_tensor_numpy_game_data_and_is_bounded():
    game = Connect4()
    first = collect_transition(game, 3)
    state = torch.tensor(first.state, requires_grad=True)
    next_state = np.array(first.next_state)
    mask = list(first.next_legal)
    t = replace(first, state=state, next_state=next_state, next_legal=mask)
    memory = ReplayMemory(2, seed=7)
    memory.append(t)
    with torch.no_grad():
        state.fill_(1)
    next_state.fill(1)
    mask[0] = False
    game.make_move(2)
    saved = memory.sample(1)[0]
    assert saved == first and saved is not t
    with pytest.raises(FrozenInstanceError):
        saved.reward = 1
    second = collect_transition(game, 4)
    memory.append(second)
    memory.append(second)
    assert len(memory) == 2 and all(s == second for s in memory.sample(2))


def test_selected_action_huber_loss_and_gradients():
    q = torch.tensor([[0., 20, 30, 40, 50, 60, 70], [80., 0, 90, 100, 110, 120, 130]], requires_grad=True)
    targets = torch.tensor([2., -0.5], requires_grad=True)
    before = q.detach().clone()
    loss = selected_action_loss(q, torch.tensor([0, 1]), targets)
    assert loss.item() == pytest.approx((1.5 + 0.125) / 2)
    loss.backward()
    expected = torch.zeros_like(q)
    expected[0, 0], expected[1, 1] = -0.5, 0.25
    assert torch.equal(q.grad, expected)
    assert torch.equal(q.detach(), before) and targets.grad is None


@pytest.mark.parametrize('change', [dict(batch_size=0), dict(batch_size=True), dict(replay_capacity=-1),
                                  dict(batch_size=10001), dict(gamma=1.1), dict(gamma=float('nan')),
                                  dict(learning_rate=0), dict(learning_rate=float('inf')),
                                  dict(epsilon_start=-1), dict(epsilon_min=1, epsilon_start=0),
                                  dict(epsilon_decay=2), dict(target_sync_interval=0), dict(seed=True), dict(seed=-1)])
def test_config_validation(change):
    with pytest.raises(ValueError):
        DQNConfig(**change)


def test_seeded_exploration_legal_and_repeatable():
    config = DQNConfig(seed=73)
    a, b = DQNTrainer(config), DQNTrainer(config)
    game = position([0] * 6)
    expected_rng = Random(config.seed)
    expected = []
    for _ in range(20):
        expected_rng.random()
        expected.append(expected_rng.choice(list(range(1, 7))))
    assert [a.choose_move(game) for _ in range(20)] == expected
    assert [b.choose_move(game) for _ in range(20)] == expected
    a.epsilon = 0
    assert a.choose_move(game) == DQNAgent(a.online).choose_move(game)


def test_tiny_optimization_target_independence_schedule_and_seed():
    config = DQNConfig(batch_size=2, target_sync_interval=2, epsilon_start=0.2,
                       epsilon_min=0.15, epsilon_decay=0.5, seed=19)
    global_rng = torch.random.get_rng_state().clone()
    a, b = DQNTrainer(config), DQNTrainer(config)
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    assert a.optimize() is None and a.updates == 0 and a.epsilon == 0.2
    for online, target in zip(a.online.parameters(), a.target.parameters()):
        assert online.data_ptr() != target.data_ptr() and torch.equal(online, target)
        assert not target.requires_grad
    before = deepcopy(a.target.state_dict())
    samples = [collect_transition(position([0, 1, 0, 1, 0, 2]), 0),
               collect_transition(position([0, 1, 0, 1, 2, 1, 2]), 1)]
    for trainer in (a, b):
        for t in samples:
            trainer.memory.append(t)
    first_loss = a.optimize()
    assert math.isfinite(first_loss) and first_loss > 0
    assert first_loss == b.optimize()
    assert a.epsilon == 0.15 and a.updates == 1
    assert all(torch.equal(before[k], v) for k, v in a.target.state_dict().items())
    assert any(not torch.equal(before[k], v) for k, v in a.online.state_dict().items())
    assert all(p.grad is None for p in a.target.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.online.parameters())
    assert any(p.grad.abs().sum() > 0 for p in a.online.parameters())
    assert a.optimize() < first_loss
    assert all(torch.equal(v, a.target.state_dict()[k]) for k, v in a.online.state_dict().items())
    assert a.updates == 2 and a.epsilon == 0.15


def test_checkpoint_round_trip_and_safe_load(tmp_path, monkeypatch):
    net = DQN()
    path = tmp_path / 'untrained-test.pt'
    metadata = {'seed': 7, 'updates': 0, 'config': {'gamma': 0.99}, 'trained': False}
    save_checkpoint(path, net, training_metadata=metadata)
    original_load = torch.load
    calls = []
    def observed(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', observed)
    restored, provenance = load_checkpoint(path)
    assert calls == [{'weights_only': True, 'map_location': 'cpu'}]
    assert provenance == metadata and not restored.training
    states = torch.tensor([encode_game(Connect4()), encode_game(position([3]))])
    assert torch.equal(net(states), restored(states))
    assert DQNAgent(net).choose_move(position([0] * 6)) == DQNAgent(restored).choose_move(position([0] * 6))


@pytest.mark.parametrize('key', list(CONTRACT))
def test_checkpoint_rejects_incompatible_metadata(tmp_path, key):
    path = tmp_path / 'bad.pt'
    save_checkpoint(path, DQN())
    payload = torch.load(path, weights_only=True)
    payload[key] = 'incompatible'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='Incompatible'):
        load_checkpoint(path)


@pytest.mark.parametrize('bad', ['missing', 'extra', 'shape', 'dtype', 'nan', 'historical', 'bool_version', 'bool_actions'])
def test_checkpoint_rejects_invalid_weights_and_schema(tmp_path, bad):
    path = tmp_path / 'bad.pt'
    save_checkpoint(path, DQN())
    payload = torch.load(path, weights_only=True)
    state = payload['model_state_dict']
    if bad == 'missing':
        del state['fc1.bias']
    elif bad == 'extra':
        state['other'] = torch.zeros(1)
    elif bad == 'shape':
        state['fc1.bias'] = torch.zeros(1)
    elif bad == 'dtype':
        state['fc1.bias'] = state['fc1.bias'].double()
    elif bad == 'nan':
        state['fc1.bias'][0] = float('nan')
    elif bad == 'historical':
        payload = {'model_state_dict': state, 'iteration': 99}
    elif bad == 'bool_version':
        payload['format_version'] = True
    else:
        payload['action_order'][0] = False
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_checkpoint_rejects_nonportable_metadata(tmp_path):
    for metadata in ({'loss': float('nan')}, {'tensor': torch.zeros(1)}, {1: 'not a string'}, []):
        with pytest.raises(ValueError):
            save_checkpoint(tmp_path / 'bad.pt', DQN(), training_metadata=metadata)


def test_rejects_unflipped_successor_encoding():
    t = collect_transition(position([3]), 2)
    with pytest.raises(ValueError, match='next player perspective'):
        replace(t, next_state=tuple(-v for v in t.next_state))


@pytest.mark.parametrize('action', [-1, 7, True, 2.0, 0])
def test_invalid_collection_action_preserves_game(action):
    game = position([0] * 6)
    before = deepcopy(game.__dict__)
    with pytest.raises(ValueError, match='legal column'):
        collect_transition(game, action)
    assert game.__dict__ == before


@pytest.mark.parametrize('values', [[0] * 6, [float('nan')] * 7, [float('inf')] * 7])
def test_inference_rejects_invalid_network_outputs(values):
    with pytest.raises(ValueError, match='finite float32'):
        DQNAgent(FixedQ(values)).choose_move(Connect4())


def test_replay_sampling_seed_and_nonterminal_optimization():
    a, b = ReplayMemory(7, seed=42), ReplayMemory(7, seed=42)
    trainer = DQNTrainer(DQNConfig(batch_size=3))
    for col in range(7):
        t = collect_transition(Connect4(), col)
        a.append(t)
        b.append(t)
        trainer.memory.append(t)
    assert a.sample(4) == b.sample(4)
    assert math.isfinite(trainer.optimize())
    assert all(p.grad is None for p in trainer.target.parameters())


def test_checkpoint_save_rejects_altered_architecture(tmp_path):
    net = DQN()
    net.fc1 = torch.nn.Linear(42, 64)
    with pytest.raises(ValueError, match='Invalid state_dict tensor'):
        save_checkpoint(tmp_path / 'bad.pt', net)
