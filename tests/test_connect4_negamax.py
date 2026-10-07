"""Independent engine/array minimax oracle; no pruning, table, or bitboards."""
from copy import deepcopy
from math import inf
from random import Random

import pytest

from games.connect4.connect4 import Connect4
from games.connect4.agents.negamax_agent import (
    EXACT, LOWER, UPPER, WIN_SCORE, NegamaxAgent, SearchState, SearchTable, negamax)

DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]


def position(moves):
    game = Connect4()
    for col in moves:
        assert game.make_move(col)
    return game


def oracle(game, depth):
    winner = game.check_winner()
    if winner != -1:
        return (1 if winner == game.current_player else -1) * (WIN_SCORE + depth)
    if game.is_board_full():
        return 0
    if depth == 0:
        score = 0
        own = 'X' if game.current_player == 0 else 'O'
        other = 'O' if own == 'X' else 'X'
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for row in range(6):
                for col in range(7):
                    if not (0 <= row + 3 * dr < 6 and 0 <= col + 3 * dc < 7):
                        continue
                    window = [game.board[row + i * dr][col + i * dc] for i in range(4)]
                    if other not in window:
                        score += (0, 1, 3, 9, 81)[window.count(own)]
                    if own not in window:
                        score -= (0, 1, 3, 9, 81)[window.count(other)]
        return score
    values = []
    for col in game.get_valid_moves():
        child = Connect4(game.board, game.current_player)
        assert child.make_move(col)
        values.append(-oracle(child, depth - 1))
    return max(values)


def root_oracle(game, depth):
    scores = {}
    for col in game.get_valid_moves():
        child = Connect4(game.board, game.current_player)
        assert child.make_move(col)
        scores[col] = -oracle(child, depth - 1)
    return scores


@pytest.mark.parametrize('moves,depth', [
    ([], 3), ([0, 1, 0, 1, 0, 2], 4),  # immediate / slower wins
    ([0, 1, 0, 1, 0], 3),  # required block for O
    ([5, 4, 3, 6, 2, 4], 3),  # Phase 4 unsafe-cache reproduction
    ([6, 4, 4, 2, 2, 2, 6, 3], 2),  # tied best moves
    ([5, 6, 6, 5, 4, 3, 0, 1, 0, 2, 4, 6, 6, 2, 3, 0], 3),
    (DRAW[:36], 4), (DRAW[:41], 4),
])
def test_every_root_move_matches_independent_oracle(moves, depth):
    game = position(moves)
    before = deepcopy(game.board)
    agent = NegamaxAgent(depth)
    scores = agent.score_moves(game)
    assert scores == root_oracle(game, depth)
    assert agent.choose_move(game) == max(scores, key=scores.get)
    assert game.board == before
    if moves == [6, 4, 4, 2, 2, 2, 6, 3]:
        assert sum(value == max(scores.values()) for value in scores.values()) > 1


def test_seeded_legal_positions_all_root_values_and_bitboard_terminals():
    rng = Random(815)
    for sample in range(12):
        game = Connect4()
        for _ in range(sample * 3):
            if game.is_game_over():
                break
            assert game.make_move(rng.choice(game.get_valid_moves()))
            state = SearchState(game)
            assert state.terminal_value(2) == (oracle(game, 2) if game.is_game_over() else None)
            if not game.is_game_over():
                assert state.heuristic() == oracle(game, 0)
        if not game.is_game_over():
            depth = 1 + sample % 3
            assert NegamaxAgent(depth).score_moves(game) == root_oracle(game, depth)


@pytest.mark.parametrize('depth', [1, 2, 3, 4])
def test_cutoffs_then_full_window_cache_reuse(depth):
    game = position([5, 4, 3, 6, 2, 4])
    expected = oracle(game, depth)
    state, table = SearchState(game), SearchTable()
    key = (*state.pieces, state.mover, depth)
    # Fail-high / beta cutoff must produce LOWER; fail-low UPPER.
    assert negamax(state, depth, expected - 2, expected - 1, table) >= expected - 1
    assert table.entries[key][0] == LOWER
    assert negamax(state, depth, -inf, inf, table) == expected
    assert table.entries[key] == (EXACT, expected)
    assert negamax(state, depth, -inf, inf, table) == expected
    assert table.hits > 0 and table.cutoffs > 0
    table = SearchTable()
    assert negamax(state, depth, expected + 1, expected + 2, table) <= expected + 1
    assert table.entries[key][0] == UPPER
    assert negamax(state, depth, -inf, inf, table) == expected
    # Further arbitrary windows, reusing the same table, then exact root values.
    table = SearchTable()
    for alpha, beta in [(-10, -9), (0, 1), (20, 21), (-5, 60)]:
        value = negamax(state, depth, alpha, beta, table)
        assert value <= alpha if expected <= alpha else value >= beta if expected >= beta else value == expected
    assert negamax(state, depth, -inf, inf, table) == expected


def test_transposition_key_includes_mover_and_depth_and_fresh_decisions():
    first, second = position([0, 1, 2, 3]), position([2, 3, 0, 1])
    assert first.board == second.board
    table = SearchTable()
    assert negamax(SearchState(first), 3, table=table) == oracle(first, 3)
    hits = table.hits
    assert negamax(SearchState(second), 3, table=table) == oracle(second, 3)
    assert table.hits > hits
    assert negamax(SearchState(first), 2, table=table) == oracle(first, 2)
    first.current_player = 1
    assert negamax(SearchState(first), 3, table=table) == oracle(first, 3)
    assert any(key[2] == 0 for key in table.entries) and any(key[2] == 1 for key in table.entries)
    # A terminal winner from either mover perspective must be exact, before depth 0.
    win = position([0, 1, 0, 1, 0, 1, 0])
    for mover in [0, 1]:
        win.current_player = mover
        assert negamax(SearchState(win), 0, table=table) == (WIN_SCORE if mover == 0 else -WIN_SCORE)
        assert negamax(SearchState(win), 4, table=table) == (WIN_SCORE + 4) * (1 if mover == 0 else -1)
    draw = position(DRAW)
    assert negamax(SearchState(draw), 0, table=table) == 0
    assert negamax(SearchState(draw), 4, table=table) == 0
    with pytest.raises(ValueError, match='terminal'):
        NegamaxAgent(2).choose_move(draw)
    agent = NegamaxAgent(3)
    agent.choose_move(first)
    initial = agent.last_stats.copy()
    agent.choose_move(first)
    assert agent.last_stats == initial


def test_terminal_dominance_fast_win_and_required_block():
    game = position([0, 1, 0, 1, 0, 2, 4, 2])
    agent = NegamaxAgent(4)
    assert agent.choose_move(game) == 0
    assert agent.last_scores[0] == WIN_SCORE + 3
    assert NegamaxAgent(2).choose_move(position([0, 1, 0, 1, 0])) == 0
    assert 69 * 81 < WIN_SCORE
