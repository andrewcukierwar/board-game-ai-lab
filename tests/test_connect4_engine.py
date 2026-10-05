"""Connect4 engine boundary: invalid columns and post-terminal moves are rejected."""
from copy import deepcopy
from random import Random

import numpy as np
import pytest

from games.connect4.connect4 import Connect4

X_WIN = [0, 1, 0, 1, 0, 1, 0]
O_WIN = [0, 1, 0, 1, 2, 1, 2, 1]
DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]


def play(moves):
    game = Connect4()
    for move in moves:
        assert not game.is_game_over()
        assert game.make_move(move)
    return game


def test_terminal_fixtures():
    assert play(X_WIN).check_winner() == 0
    assert play(O_WIN).check_winner() == 1
    draw = play(DRAW)
    assert draw.check_winner() == -1 and draw.is_board_full()


@pytest.mark.parametrize('column', [-1, -7, 7, 8, 100])
def test_out_of_range_columns_rejected_without_mutation(column):
    game = play([3, 2])
    before = deepcopy(game.__dict__)
    assert not game.is_valid_move(column)
    assert game.make_move(column) is False
    assert game.__dict__ == before
    with pytest.raises(ValueError):
        game.step(column)
    assert game.__dict__ == before


@pytest.mark.parametrize('column', [True, False, 1.0, 3.5, '3', None, [3], np.bool_(True)])
def test_non_integer_and_bool_columns_rejected(column):
    game = Connect4()
    before = deepcopy(game.__dict__)
    assert not game.is_valid_move(column)
    assert game.make_move(column) is False
    assert game.__dict__ == before


def test_numpy_integer_columns_remain_accepted():
    game = Connect4()
    assert game.is_valid_move(np.int64(3))
    assert game.make_move(np.int64(3))
    assert game.board[5][3] == 'X' and game.current_player == 1


@pytest.mark.parametrize('history', [X_WIN, O_WIN, DRAW], ids=['x_win', 'o_win', 'draw'])
def test_moves_after_termination_rejected(history):
    game = play(history)
    before = deepcopy(game.__dict__)
    for column in range(7):
        assert game.make_move(column) is False
        assert game.__dict__ == before
        with pytest.raises(ValueError):
            game.step(column)
        assert game.__dict__ == before


def test_valid_moves_on_finished_boards_keep_historical_physical_meaning():
    # Callers check is_game_over(); make_move now enforces it independently.
    assert play(X_WIN).get_valid_moves() == [3, 2, 4, 1, 5, 0, 6]
    assert play(DRAW).get_valid_moves() == []


def test_full_column_rejected():
    game = play([0] * 6)
    assert not game.is_valid_move(0) and game.make_move(0) is False


def test_gravity_and_turn_alternation_unchanged():
    game = Connect4()
    assert game.current_player == 0 and game.piece == 'X'
    for i, column in enumerate([3, 3, 3, 4]):
        assert game.make_move(column)
        assert game.current_player == (i + 1) % 2
        assert game.piece == ('X' if game.current_player == 0 else 'O')
    assert [game.board[r][3] for r in (5, 4, 3, 2)] == ['X', 'O', 'X', ' ']
    assert game.board[5][4] == 'O'


def test_winning_move_still_flips_turn_and_step_rewards_unchanged():
    game = play(X_WIN[:-1])
    _, reward, done, _ = game.step(0)
    assert (reward, done, game.current_player, game.check_winner()) == (1, True, 1, 0)


def test_fast_terminal_guard_matches_is_game_over():
    rng = Random(7)
    for _ in range(300):
        game = Connect4()
        while True:
            assert game._has_terminated() == game.is_game_over()
            if game.is_game_over():
                break
            assert game.make_move(rng.choice(game.get_valid_moves()))
    for _ in range(300):  # Arbitrary, possibly unreachable boards.
        board = [[rng.choice('XO   ') for _ in range(7)] for _ in range(6)]
        game = Connect4(board=board)
        assert game._has_terminated() == game.is_game_over()
