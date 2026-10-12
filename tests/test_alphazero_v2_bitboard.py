"""Independent engine oracle and exact neural-PUCT differential regressions."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import random

import pytest
import torch

from games.connect4.connect4 import Connect4
from games.connect4.alphazero_v2.bitboard import BitboardState, BOARD_MASK, has_four
from games.connect4.alphazero_v2.network import AlphaZeroV2Net, V2Inference, encode_bitboard_current_player
from games.connect4.alphazero_v2.search import PUCTSearch, SelfPlayer, EvaluationAgent, V2RootNoise, select_action
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.selfplay import play_game
from games.connect4.neural_mcts import encode_current_player

DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6]


def position(moves):
    game = Connect4()
    for move in moves:
        assert game.make_move(move)
    return game


from research.alphazero_bitboard.validation import assert_state, assert_tree, assert_result


def test_at_least_10000_legal_transitions_and_mirrors():
    rng = random.Random(2026101144)
    checked = 0
    actors, winners = set(), set()
    full_column = False
    while checked < 10000:
        game = Connect4()
        while not game.is_game_over():
            for source in (game, position([6 - a for a in history(game)])):
                state = BitboardState.from_game(source)
                assert_state(state, source)
                actors.add(source.current_player)
                full_column |= len(source.get_valid_moves()) < 7
                for move in source.get_valid_moves():
                    after = deepcopy(source)
                    assert after.make_move(move)
                    successor = state.drop(move)
                    assert_state(successor, after)
                    winners.add(after.check_winner())
                    checked += 1
            move = rng.choice(game.get_valid_moves())
            game.make_move(move)
            game._test_history = (*getattr(game, '_test_history', ()), move)
        assert_state(BitboardState.from_game(game), game)
    assert actors == {0, 1} and winners == {-1, 0, 1} and full_column


def history(game):
    return getattr(game, '_test_history', ())


@pytest.mark.parametrize('moves', [[], [3], [3] * 6, DRAW[:40], DRAW[:41], DRAW,
                                  [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 2, 1, 2, 1],
                                  [4, 0, 3, 5, 2, 4], [3, 4, 1, 1, 0, 2, 3, 3, 2, 1, 4, 2]])
def test_edges_and_detachment(moves):
    game = position(moves)
    state = BitboardState.from_game(game)
    before = deepcopy(game.__dict__)
    assert_state(state, game)
    for invalid in (True, False, -1, 7, 1.0, '3', None):
        with pytest.raises(ValueError):
            state.drop(invalid)
    if game.is_game_over():
        with pytest.raises(ValueError):
            state.drop(3)
    else:
        for move in game.get_valid_moves():
            after = deepcopy(game)
            after.make_move(move)
            assert_state(state.drop(move), after)
    assert game.__dict__ == before
    with pytest.raises(FrozenInstanceError):
        state.current_player = 0


def test_all_69_windows_and_sentinel_wraparound():
    windows = [tuple((r + i * dr, c + i * dc) for i in range(4))
               for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
               for r in range(6) for c in range(7)
               if 0 <= r + 3 * dr < 6 and 0 <= c + 3 * dc < 7]
    assert len(windows) == 69
    for player in (0, 1):
        for window in windows:
            for missing in (None, 0, 1, 2, 3):
                board = [[' '] * 7 for _ in range(6)]
                bits = 0
                for i, (r, c) in enumerate(window):
                    if i != missing:
                        board[r][c] = 'XO'[player]
                        bits |= 1 << (7 * c + 5 - r)
                assert has_four(bits) == (Connect4(board).check_winner() == player) == (missing is None)
    for col in range(6):
        bits = sum(1 << (7 * col + row) for row in (4, 5))
        bits |= sum(1 << (7 * (col + 1) + row) for row in (0, 1))
        assert not has_four(bits)


@pytest.fixture
def inference():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(1144)
        return V2Inference(AlphaZeroV2Net().eval())


@pytest.mark.parametrize('moves', [[], [3], [3] * 6, DRAW[:40], [4, 0, 3, 5, 2, 4],
                                  [3, 4, 1, 1, 0, 2, 3, 3, 2, 1, 4, 2]])
@pytest.mark.parametrize('noisy', [False, True])
@pytest.mark.parametrize('budget', [1, 32])
def test_search_full_tree_rng_predictions_and_temperatures(inference, moves, noisy, budget):
    game = position(moves)
    before = deepcopy(game.__dict__)
    traces, results, states, noises = [], [], [], []
    for backend in ('engine', 'bitboard'):
        trace = []
        def capture(module, args, output):
            trace.append((args[0].numpy().tobytes(), *(t.numpy().tobytes() for t in output)))
        hook = inference.model.register_forward_hook(capture)
        rng = random.Random(881)
        noise = V2RootNoise(seed=992) if noisy else None
        try:
            result = PUCTSearch(inference, budget, backend=backend).run(game, rng=rng, root_noise=noise)
        finally:
            hook.remove()
        traces.append(trace); results.append(result); states.append(rng.getstate())
        noises.append(noise.get_state() if noise else None)
    assert traces[0] == traces[1]
    assert_result(*results)
    assert states[0] == states[1] and noises[0] == noises[1]
    assert sum(results[0].visits) == results[0].root.visits == budget
    for temperature in (0.0, 1.0):
        a, b = random.Random(77), random.Random(77)
        assert select_action(results[0].visits, temperature, a) == select_action(results[1].visits, temperature, b)
        assert a.getstate() == b.getstate()
    assert game.__dict__ == before


def test_complete_selfplay_targets_and_all_streams(inference):
    players = [SelfPlayer(inference, V2Config(self_play_simulations=8),
                          search_rng=random.Random(41), action_rng=random.Random(44),
                          root_noise=V2RootNoise(seed=47), backend=backend)
               for backend in ('engine', 'bitboard')]
    games = [play_game(p, 1, 0) for p in players]
    assert games[0] == games[1]
    assert {e.action_temperature for e in games[0].examples} == {0.0, 1.0}
    assert players[0].search_rng.getstate() == players[1].search_rng.getstate()
    assert players[0].action_rng.getstate() == players[1].action_rng.getstate()
    assert players[0].root_noise.get_state() == players[1].root_noise.get_state()


def test_bitboard_search_never_calls_engine_or_materializes_board(inference, monkeypatch):
    game = position([3, 2])
    def forbidden(*a, **kw):
        pytest.fail('Engine operation or diagnostic board in bitboard search')
    for name in ('make_move', 'is_game_over', 'check_winner', '_has_terminated'):
        monkeypatch.setattr(Connect4, name, forbidden)
    monkeypatch.setattr(BitboardState, 'board', property(forbidden))
    result = PUCTSearch(inference, 32, backend='bitboard').run(game, rng=random.Random(19))
    assert sum(result.visits) == 32


def test_backend_validation_default_and_terminal_rejection(inference):
    assert PUCTSearch(inference, 1).backend == 'engine'
    assert EvaluationAgent(inference, 1, backend='bitboard').search_engine.backend == 'bitboard'
    with pytest.raises(ValueError):
        PUCTSearch(inference, 1, backend='unknown')
    for backend in ('engine', 'bitboard'):
        with pytest.raises(ValueError):
            PUCTSearch(inference, 1, backend=backend).run(position(DRAW), rng=random.Random(0))


def test_conversion_rejects_bad_piece_symbols_and_floating_pieces():
    game = Connect4(); game.piece = 'O'
    with pytest.raises(ValueError):
        BitboardState.from_game(game)
    game = Connect4(); game.board[5][0] = '?'
    with pytest.raises(ValueError):
        BitboardState.from_game(game)
    game = Connect4(); game.board[4][0] = 'X'
    with pytest.raises(ValueError):
        BitboardState.from_game(game)


def test_bitboard_capture_rejects_other_position_at_same_ply(inference):
    from games.connect4.alphazero_v2.data import capture_example
    player = SelfPlayer(inference, V2Config(self_play_simulations=1),
                        search_rng=random.Random(1), action_rng=random.Random(2),
                        root_noise=V2RootNoise(seed=3), backend='bitboard')
    decision = player.decide(position([3]))
    capture_example(position([3]), decision)
    with pytest.raises(ValueError, match='does not belong'):
        capture_example(position([2]), decision)


def test_ambiguous_two_winner_conversion_rejected():
    board = [[' '] * 7 for _ in range(6)]
    for col in range(4):
        board[5][col] = 'X'
        board[4][col] = 'O'
    with pytest.raises(ValueError, match='two winners'):
        BitboardState.from_game(Connect4(board))


@pytest.mark.parametrize('backend', ['engine', 'bitboard'])
def test_profile_probe_compares_same_representation_and_restores(inference, backend):
    from research.alphazero_bitboard.benchmark import profile_probe
    before = PUCTSearch._select_child
    result = profile_probe(inference, position([3]), 4, 'regression-profile', backend)
    assert result['semantics_verified']
    assert result['calls']['neural_forward'] == 5
    assert PUCTSearch._select_child is before
