"""Independent serial search oracle, full trees/RNGs and collection contracts."""
import random

import numpy as np
import pytest
import torch

from games.connect4.connect4 import Connect4
from games.connect4.alphazero_v2 import batching as b
from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.network import (AlphaZeroV2Net, V2Inference, V2Prediction,
    encode_bitboard_current_player, legal_policy_from_logits, weights_sha256)
from games.connect4.alphazero_v2.search import PUCTSearch
from research.alphazero_bitboard.validation import assert_result


class FakeInference(V2Inference):
    def __init__(self):
        self.calls = []

    def predict(self, state):
        # State-dependent exact predictions; ties and positive/negative backups.
        n = state.occupied.bit_count()
        logits = tuple((a - 3) * .125 + (n % (a + 1)) * .0625 for a in range(7))
        return V2Prediction(logits, tuple(legal_policy_from_logits(logits, state.get_valid_moves())),
                            ((n % 5) - 2) * .125)

    def predict_batch(self, states):
        self.calls.append(len(states))
        return tuple(self.predict(state) for state in states)


def position(moves):
    game = Connect4()
    for move in moves:
        assert game.make_move(move)
    return BitboardState.from_game(game)


@pytest.fixture
def inference():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(1144)
        return V2Inference(AlphaZeroV2Net().eval())


@pytest.mark.parametrize('size', [1, 2, 4, 8, 16])
def test_batch_encoding_outputs_order_and_weights(inference, size):
    fixtures = [[], [3], [3]*6, [0,1,0,1,0,2], [0,1,0,1,2,1],
                [6,5,6,5,6,4], [3,2,4,2,5,1], [3,4,2,4,1,5]]
    states = [position(fixtures[i % len(fixtures)]) for i in range(size)]
    before = weights_sha256(inference.model)
    captured = []
    def hook(module, args, outputs):
        assert torch.is_inference_mode_enabled()
        assert not any(m.training for m in module.modules())
        captured.append(args[0].numpy().tobytes())
    handle = inference.model.register_forward_hook(hook)
    try:
        actual = inference.predict_batch(states)
    finally:
        handle.remove()
    assert len(captured) == 1
    assert captured[0] == b''.join(encode_bitboard_current_player(s).numpy().tobytes() for s in states)
    expected = [inference.predict(s) for s in states]
    if size == 1:
        assert tuple(expected) == actual
    for state, a, e in zip(states, actual, expected):
        assert max(abs(x-y) for x,y in zip(a.logits,e.logits)) <= 2e-5
        assert abs(a.value-e.value) <= 2e-6
        assert max(abs(x-y) for x,y in zip(a.policy,e.policy)) <= 2e-6
        assert abs(sum(a.policy)-1) < 1e-14
        assert all(a.policy[m] == 0 for m in range(7) if m not in state.get_valid_moves())
    assert actual == inference.predict_batch(states)
    assert before == weights_sha256(inference.model)


@pytest.mark.parametrize('invalid', [None, 1, 'x', {}, iter([]), [None], [Connect4()],
    [BitboardState((1,1),1,0)], [BitboardState((0,0),1,0)],
    [BitboardState((0,0),0,True)], [BitboardState((2,0),2,1)],
    [BitboardState((0,0),0,1)], [BitboardState((0,0),0,0,0)],
    [BitboardState((1<<6,0),1<<6,1)], [BitboardState((0,0),0,0,-1.0)]])
def test_invalid_batches(inference, invalid):
    with pytest.raises(ValueError):
        inference.predict_batch(invalid)


def test_terminal_empty_and_eval_rejection(inference):
    assert inference.predict_batch([]) == ()
    with pytest.raises(ValueError):
        inference.predict_batch([position([0,1,0,1,0,1,0])])
    inference.model.conv1.train()
    for states in ([], [position([])], [position([])]*2):
        with pytest.raises(ValueError):
            inference.predict_batch(states)


@pytest.mark.parametrize('kind', ['nan', 'inf', 'shape', 'dtype', 'range'])
def test_bad_outputs(inference, kind):
    def forward(x):
        n = len(x)
        logits, values = torch.zeros(n,7), torch.zeros(n,1)
        if kind == 'nan': logits[0,0] = float('nan')
        if kind == 'inf': values[0,0] = float('inf')
        if kind == 'shape': logits = logits[:,:6]
        if kind == 'dtype': values = values.double()
        if kind == 'range': values[0,0] = 1.001
        return logits, values
    inference.model.forward = forward
    with pytest.raises(ValueError):
        inference.predict_batch([position([]), position([3])])


def drive_search(inference, state, config, streams):
    gen = b.search_requests(state, config.self_play_simulations, streams)
    leaf = next(gen)
    while True:
        try:
            leaf = gen.send(inference.predict_batch([leaf])[0])
        except StopIteration as done:
            return done.value


@pytest.mark.parametrize('moves', [[], [3], [3]*6, [0,1,0,1,0,2],
                                   [0,1,0,1,2,1], [6,5,6,5,6,4]])
@pytest.mark.parametrize('epsilon', [0., .25])
def test_resumable_search_exact_every_node_and_rng(moves, epsilon):
    config = V2Config(self_play_simulations=32, root_noise_epsilon=epsilon)
    a, c = b.game_streams(config, 21, 3), b.game_streams(config, 21, 3)
    inference = FakeInference()
    state = position(moves)
    expected = PUCTSearch(inference,32,backend='bitboard').run(state,rng=a.search,root_noise=a.noise)
    actual = drive_search(inference,state,config,c)
    assert_result(expected,actual)
    assert a.state() == c.state()


@pytest.mark.parametrize('size', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('reverse', [False, True])
def test_fake_full_games_exact_trees_rng_order_refill_labels(size, reverse):
    inference = FakeInference()
    config = V2Config(self_play_simulations=8)
    ids = list(range(9))
    serial_stats, batch_stats = b.CollectionStats(), b.CollectionStats()
    trees = {}
    expected = b.collect_serial_reference(inference,config,21,ids,stats=serial_stats,
        observer=lambda i,d,t: trees.__setitem__((i,d.ply),d))
    if reverse: ids.reverse()
    count = 0
    def compare(i,d,t):
        nonlocal count
        e = trees[i,d.ply]
        assert d.action == e.action
        assert_result(e.search,d.search)
        count += 1
    actual = b.collect_batched(inference,config,21,ids,batch_size=size,stats=batch_stats,observer=compare)
    assert expected == actual
    assert serial_stats.rng_states == batch_stats.rng_states
    assert count == sum(len(g.moves) for g in expected)
    assert batch_stats.simulations == config.self_play_simulations*count
    assert batch_stats.max_active == min(size,len(ids))
    assert batch_stats.forward_calls == sum(batch_stats.batch_sizes.values())
    assert batch_stats.neural_positions == sum(n*c for n,c in batch_stats.batch_sizes.items())
    assert len({len(g.moves) for g in actual}) > 1
    for g in actual:
        game = Connect4()
        for e in g.examples:
            assert e.ply == sum(c!=' ' for row in game.board for c in row)
            assert e.actor == game.current_player
            assert abs(sum(e.policy_target)-1) < 1e-14
            assert e.action_temperature == (1. if e.ply<8 else 0.)
            assert e.outcome == (0 if g.winner==-1 else 1 if e.actor==g.winner else -1)
            assert game.make_move(e.action)
        assert game.is_game_over() and game.check_winner() == g.winner


def test_actual_network_complete_games_batch1_exact_and_batch4_repeat(inference):
    config = V2Config(self_play_simulations=32)
    a, c = b.CollectionStats(), b.CollectionStats()
    expected = b.collect_serial_reference(inference,config,21,range(4),stats=a)
    actual = b.collect_batched(inference,config,21,range(4),batch_size=1,stats=c)
    assert expected == actual and a.rng_states == c.rng_states
    before = weights_sha256(inference.model)
    x = b.collect_batched(inference,config,21,range(4),batch_size=4)
    assert x == b.collect_batched(inference,config,21,range(4),batch_size=4)
    assert before == weights_sha256(inference.model)


@pytest.mark.parametrize('size', [0, -1, True, 1.0, None])
def test_invalid_pool(size):
    with pytest.raises(ValueError):
        b.collect_batched(FakeInference(),V2Config(),21,[],batch_size=size)


@pytest.mark.parametrize('ids', [[1,1], [-1], [True], [1.0]])
def test_invalid_ids(ids):
    for collector in (b.collect_serial_reference, b.collect_batched):
        with pytest.raises(ValueError):
            collector(FakeInference(),V2Config(),21,ids,**({'batch_size':2} if collector==b.collect_batched else {}))


def test_zero_and_single_active():
    inference = FakeInference()
    assert b.collect_batched(inference,V2Config(),21,[],batch_size=8) == ()
    assert not inference.calls
    config = V2Config(self_play_simulations=4)
    assert b.collect_batched(inference,config,21,[99],batch_size=8) == b.collect_serial_reference(inference,config,21,[99])


def test_rng_domain_and_game_isolation():
    config = V2Config()
    seeds = {b.game_seed(config.seed,g,i,d) for g in (1,2) for i in (0,1) for d in ('search','action','noise')}
    assert len(seeds) == 12
    a, c = b.game_streams(config,1,0), b.game_streams(config,1,0)
    a.search.random()
    assert a.action.getstate() == c.action.getstate() and a.noise.get_state() == c.noise.get_state()
    a.noise.mix([1/7]*7, range(7))
    assert a.action.getstate() == c.action.getstate()


@pytest.mark.parametrize('failure', ['inference','search','observer','callback','length'])
def test_exceptions_discard_collection_and_allow_next_call(monkeypatch, failure):
    inference = FakeInference()
    config = V2Config(self_play_simulations=4)
    def fail(*args,**kwargs): raise RuntimeError('injected')
    options = {}
    if failure == 'inference': monkeypatch.setattr(inference,'predict_batch',fail)
    if failure == 'length': monkeypatch.setattr(inference,'predict_batch',lambda s: ())
    if failure == 'search': monkeypatch.setattr(PUCTSearch,'_select_child',fail)
    if failure == 'observer': options['observer'] = fail
    if failure == 'callback': options['on_game'] = fail
    with pytest.raises(RuntimeError):
        b.collect_batched(inference,config,21,range(4),batch_size=2,**options)
    monkeypatch.undo()
    assert len(b.collect_batched(FakeInference(),config,21,range(2),batch_size=2)) == 2


def test_bounded_cancellation_terminal_simulations_and_partial_failure():
    inference = FakeInference()
    checks = 0
    def check():
        nonlocal checks
        checks += 1
        if checks == 10: raise TimeoutError('cancel')
    with pytest.raises(TimeoutError):
        b.collect_batched(inference,V2Config(),21,range(8),batch_size=4,check=check)
    assert checks == 10 and len(inference.calls) < 10
    completed = []
    def on_game(game):
        completed.append(game)
    def partial_check():
        if completed: raise TimeoutError('after complete game')
    with pytest.raises(TimeoutError):
        b.collect_batched(FakeInference(),V2Config(self_play_simulations=4),21,range(5),
                          batch_size=2,check=partial_check,on_game=on_game)
    assert 0 < len(completed) < 5
    assert all(e.outcome is not None for g in completed for e in g.examples)
    # Terminal-only tail of a search must still call check each simulation.
    checks = 0
    streams = b.game_streams(V2Config(),21,0)
    gen = b.search_requests(position([0,1,0,1,0,2]),1000,streams,check=check)
    leaf = next(gen)
    with pytest.raises(TimeoutError):
        while True: leaf = gen.send(inference.predict(leaf))


def test_existing_collector_defaults_unchanged(inference):
    from games.connect4.alphazero_v2.search import SelfPlayer, V2RootNoise
    from games.connect4.alphazero_v2.selfplay import collect_generation
    config = V2Config(self_play_simulations=4)
    def player():
        return SelfPlayer(inference,config,search_rng=random.Random(1),action_rng=random.Random(2),root_noise=V2RootNoise(seed=3))
    original = player()
    assert original.search.backend == 'engine'
    expected = collect_generation(original,1,1)
    b.collect_batched(inference,config,21,range(2),batch_size=2)
    assert collect_generation(player(),1,1) == expected


def test_terminal_draw_rejected_and_cancellation_does_not_mutate_model(inference):
    draw = [2,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,3,6,
            4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6]
    with pytest.raises(ValueError): inference.predict_batch([position(draw)])
    before = weights_sha256(inference.model)
    def check(): raise TimeoutError('stop before forward')
    with pytest.raises(TimeoutError):
        b.collect_batched(inference,V2Config(),21,range(8),batch_size=8,check=check)
    assert weights_sha256(inference.model) == before


def test_per_game_protocol_is_pinned_and_global_rng_is_untouched():
    import hashlib
    from copy import deepcopy
    before_random = random.getstate()
    before_numpy = deepcopy(np.random.get_state())
    before_torch = torch.get_rng_state().clone()
    expected = int.from_bytes(hashlib.sha256(
        b'connect4-az-independent-games-v1:42:21:3:search').digest(),'big')
    assert b.game_seed(42,21,3,'search') == expected
    config = V2Config(self_play_simulations=4)
    b.collect_batched(FakeInference(),config,21,range(8),batch_size=8)
    assert random.getstate() == before_random
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    assert np.array_equal(before_numpy[1],after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert torch.equal(before_torch,torch.get_rng_state())
