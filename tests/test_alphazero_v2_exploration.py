"""Optional action-temperature schedules in the experimental batched collector."""
import random

import pytest
import torch

from games.connect4.connect4 import Connect4
from games.connect4.agents.mcts_nn_agent import visit_policy
from games.connect4.alphazero_v2 import batching as b
from games.connect4.alphazero_v2.batching import TemperatureSchedule
from games.connect4.alphazero_v2.config import V2Config, action_temperature
from games.connect4.alphazero_v2.data import capture_example, encode_board, finalize_game, visit_target
from games.connect4.alphazero_v2.network import (AlphaZeroV2Net, V2Inference, V2Prediction,
    legal_policy_from_logits, weights_sha256)
from games.connect4.alphazero_v2.search import PUCTSearch, SelfPlayDecision, select_action
from research.alphazero_exploration import schedules as s

CONFIG = V2Config(self_play_simulations=8)


class FakeInference(V2Inference):
    """Deterministic state-dependent predictions, as in the Phase 4D.4C batching tests."""
    def __init__(self):
        pass

    def predict(self, state):
        n = state.occupied.bit_count()
        logits = tuple((a - 3) * .125 + (n % (a + 1)) * .0625 for a in range(7))
        return V2Prediction(logits, tuple(legal_policy_from_logits(logits, state.get_valid_moves())),
                            ((n % 5) - 2) * .125)

    def predict_batch(self, states):
        return tuple(self.predict(state) for state in states)


def serial_oracle(inference, config, generation, index, schedule):
    """Independent serial game: unchanged Phase 4D.4B search, explicit temperature."""
    streams = b.game_streams(config, generation, index)
    engine = PUCTSearch(inference, config.self_play_simulations, backend='bitboard')
    game, moves, pending = Connect4(), [], []
    while not game.is_game_over():
        result = engine.run(game, rng=streams.search, root_noise=streams.noise)
        action = select_action(result.visits, schedule.temperature(len(moves)), streams.action)
        pending.append(capture_example(game, SelfPlayDecision(len(moves), result, action)))
        assert game.make_move(action.move)
        moves.append(action.move)
    return finalize_game(generation, index, moves, pending), streams.state()


def test_declared_schedules_by_ply():
    expected = dict(original=[1.]*8 + [0.]*34, extended=[1.]*16 + [0.]*26,
                    smoothed=[1.]*8 + [.3]*22 + [0.]*12, extended_smoothed=[1.]*16 + [.3]*14 + [0.]*12)
    assert [x.name for x in s.SCHEDULES] == list(expected)
    for schedule in s.SCHEDULES:
        assert [schedule.temperature(p) for p in range(42)] == expected[schedule.name]
        assert TemperatureSchedule.from_dict(schedule.to_dict()) == schedule
    assert all(s.ORIGINAL.temperature(p) == action_temperature(p, V2Config().exploratory_plies) for p in range(42))
    assert V2Config().exploratory_plies == 8


@pytest.mark.parametrize('name,segments', [
    ('', ((0, 1.),)), (3, ((0, 1.),)), ('é', ((0, 1.),)), ('x', ()), ('x', None), ('x', ((1, 1.),)),
    ('x', ((0, 1.), (0, 0.))), ('x', ((0, 1.), (8, 1.))), ('x', ((0, 1.), (9, .5), (8, 0.))),
    ('x', ((0, 1.), (42, 0.))), ('x', ((0, -1.),)), ('x', ((0, float('nan')),)), ('x', ((0, float('inf')),)),
    ('x', ((0, True),)), ('x', ((0., 1.),)), ('x', ((True, 1.),)), ('x', ((0, 1., 2),)), ('x', ((0, '1'),)),
    ('x', (0, 1.))])
def test_invalid_schedules(name, segments):
    with pytest.raises(ValueError):
        TemperatureSchedule(name, segments)


@pytest.mark.parametrize('ply', [-1, 42, 1.0, True, None])
def test_invalid_ply_and_schedule_argument(ply):
    with pytest.raises(ValueError):
        s.SMOOTHED.temperature(ply)
    for bad in ('original', ((0, 1.),), s.ORIGINAL.to_dict(), 8):
        with pytest.raises(ValueError):
            b.collect_batched(FakeInference(), CONFIG, 21, range(2), batch_size=2, temperature_schedule=bad)
    for bad in ({}, dict(name='x'), dict(name='x', segments=[[0, 1.]], extra=1), None):
        with pytest.raises(ValueError):
            TemperatureSchedule.from_dict(bad)


@pytest.mark.parametrize('size', [1, 3, 8])
def test_explicit_original_equals_default_and_serial_reference(size):
    default_stats, explicit_stats, serial_stats = b.CollectionStats(), b.CollectionStats(), b.CollectionStats()
    default = b.collect_batched(FakeInference(), CONFIG, 21, range(9), batch_size=size, stats=default_stats)
    explicit = b.collect_batched(FakeInference(), CONFIG, 21, range(9), batch_size=size, stats=explicit_stats,
                                 temperature_schedule=s.ORIGINAL)
    serial = b.collect_serial_reference(FakeInference(), CONFIG, 21, range(9), stats=serial_stats)
    assert default == explicit == serial
    assert default_stats.rng_states == explicit_stats.rng_states == serial_stats.rng_states
    assert default_stats.batch_sizes == explicit_stats.batch_sizes


def test_schedule_overrides_config_exploratory_plies_only_when_given():
    config = V2Config(self_play_simulations=8, exploratory_plies=3)
    default = b.collect_batched(FakeInference(), config, 21, range(4), batch_size=4)
    assert all(e.action_temperature == (1. if e.ply < 3 else 0.) for g in default for e in g.examples)
    explicit = b.collect_batched(FakeInference(), config, 21, range(4), batch_size=4, temperature_schedule=s.EXTENDED)
    assert all(e.action_temperature == (1. if e.ply < 16 else 0.) for g in explicit for e in g.examples)


@pytest.mark.parametrize('schedule', s.SCHEDULES, ids=lambda x: x.name)
@pytest.mark.parametrize('size', [1, 4, 8])
def test_schedules_match_independent_serial_oracle_and_contracts(schedule, size):
    inference, stats, decisions = FakeInference(), b.CollectionStats(), []
    games = b.collect_batched(inference, CONFIG, 21, range(10), batch_size=size, stats=stats,
                              observer=lambda i, d, t: decisions.append((i, d)), temperature_schedule=schedule)
    assert stats.simulations == stats.plies * CONFIG.self_play_simulations == 8 * sum(len(g.moves) for g in games)
    for game in games:
        expected, rng = serial_oracle(FakeInference(), CONFIG, 21, game.index, schedule)
        assert game == expected and stats.rng_states[game.index] == rng
        replay = Connect4()
        for example in game.examples:
            assert example.observation == encode_board(replay)
            assert example.ply == sum(c != ' ' for row in replay.board for c in row)
            assert example.actor == replay.current_player
            assert example.action_temperature == schedule.temperature(example.ply)
            assert sum(example.visits) == 8 and example.policy_target == visit_target(example.visits)
            if example.action_temperature == 0:
                assert example.visits[example.action] == max(example.visits)
            assert example.outcome == (0 if game.winner == -1 else 1 if example.actor == game.winner else -1)
            assert replay.make_move(example.action)
        assert replay.is_game_over() and replay.check_winner() == game.winner
    # Recorded execution is the existing visit mathematics; targets ignore temperature.
    assert len(decisions) == stats.plies
    for index, decision in decisions:
        action, search = decision.action, decision.search
        assert search.simulations == sum(search.visits) == search.root.visits == 8
        assert action.temperature == schedule.temperature(decision.ply)
        assert action.distribution == tuple(visit_policy(search.visits, action.temperature))
        assert action.distribution[action.move] > 0
        assert search.visit_target == tuple(v / 8 for v in search.visits)
    assert {e.action_temperature for g in games for e in g.examples} <= {t for _, t in schedule.segments}


def test_one_action_draw_per_move_and_shared_prefix_under_matched_identities():
    """No schedule-dependent draws: streams and searched examples agree until temperatures differ."""
    collected = {}
    for schedule in s.SCHEDULES:
        stats = b.CollectionStats()
        games = b.collect_batched(FakeInference(), CONFIG, 21, range(12), batch_size=8, stats=stats,
                                  temperature_schedule=schedule)
        collected[schedule.name] = games
        for game in games:
            fresh = random.Random(b.game_seed(CONFIG.seed, 21, game.index, 'action'))
            for _ in game.moves:
                fresh.random()
            assert stats.rng_states[game.index]['action'] == fresh.getstate()
            assert stats.rng_states[game.index]['noise']['draws'] == len(game.moves)
    content = lambda g, n: [(e.observation, e.visits, e.action, e.action_temperature) for e in g.examples[:n]]
    for name, shared in (('extended', 8), ('smoothed', 8), ('extended_smoothed', 8)):
        assert all(content(x, shared) == content(y, shared)
                   for x, y in zip(collected['original'], collected[name]))
    assert all(content(x, 16) == content(y, 16)
               for x, y in zip(collected['extended'], collected['extended_smoothed']))
    # The schedules do change later play in this sample; otherwise the test is vacuous.
    assert any(x.moves != y.moves for x, y in zip(collected['original'], collected['smoothed']))
    assert any(x.moves != y.moves for x, y in zip(collected['original'], collected['extended']))


def test_pool_and_order_do_not_change_scheduled_games():
    reference = b.collect_batched(FakeInference(), CONFIG, 21, range(9), batch_size=1,
                                  temperature_schedule=s.EXTENDED_SMOOTHED)
    for size, ids in ((2, range(9)), (8, reversed(range(9))), (16, [4, 0, 8, 2, 6, 1, 7, 3, 5])):
        assert reference == b.collect_batched(FakeInference(), CONFIG, 21, list(ids), batch_size=size,
                                              temperature_schedule=s.EXTENDED_SMOOTHED)


def test_scheduled_cancellation_and_partial_failure_are_safe():
    completed = []
    def check():
        if len(completed) >= 2:
            raise TimeoutError('deadline')
    with pytest.raises(TimeoutError):
        b.collect_batched(FakeInference(), CONFIG, 21, range(8), batch_size=4, check=check,
                          on_game=completed.append, temperature_schedule=s.SMOOTHED)
    assert 2 <= len(completed) < 8
    full = b.collect_batched(FakeInference(), CONFIG, 21, range(8), batch_size=4, temperature_schedule=s.SMOOTHED)
    by_index = {g.index: g for g in full}
    assert all(g == by_index[g.index] and all(e.outcome is not None for e in g.examples) for g in completed)


def test_frozen_network_reproduces_and_stays_frozen():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(1144)
        inference = V2Inference(AlphaZeroV2Net().eval())
    config = V2Config(self_play_simulations=16)
    before = weights_sha256(inference.model)
    a, c = b.CollectionStats(), b.CollectionStats()
    first = b.collect_batched(inference, config, 21, range(6), batch_size=4, stats=a, temperature_schedule=s.SMOOTHED)
    again = b.collect_batched(inference, config, 21, range(6), batch_size=4, stats=c, temperature_schedule=s.SMOOTHED)
    assert first == again and a.rng_states == c.rng_states
    assert weights_sha256(inference.model) == before
