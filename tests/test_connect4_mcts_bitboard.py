"""Independent engine oracle for MCTS geometry, random play, and tree state."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from random import Random

import pytest

from games.connect4.agents.mcts_agent import MCTSAgent, Node
from games.connect4.agents.mcts_bitboard import (
    BitboardState, BOARD_MASK, has_four, random_rollout,
)
from games.connect4.connect4 import Connect4
from scripts.benchmark_public_agents import DRAW, POSITIONS, position


def assert_equivalent(state, game):
    assert state.board == tuple(tuple(row) for row in game.board)
    assert state.current_player == game.current_player
    assert state.get_valid_moves() == game.get_valid_moves()
    assert state.check_winner() == game.check_winner()
    assert state.is_game_over() == game.is_game_over()
    assert not (state.pieces[0] & state.pieces[1])
    assert state.occupied == state.pieces[0] | state.pieces[1]
    assert not state.occupied & ~BOARD_MASK
    for col in range(7):
        assert state.is_valid_move(col) == game.is_valid_move(col)


WINDOWS = [tuple((row + i * dr, col + i * dc) for i in range(4))
           for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
           for row in range(6) for col in range(7)
           if 0 <= row + 3 * dr < 6 and 0 <= col + 3 * dc < 7]


@pytest.mark.parametrize('player', [0, 1])
@pytest.mark.parametrize('window', WINDOWS)
def test_every_four_window_and_each_missing_cell_against_array_engine(player, window):
    # Enumerate all 69 array windows independently of the bitboard constants.
    for missing in (None, 0, 1, 2, 3):
        board = [[' '] * 7 for _ in range(6)]
        bits = 0
        for index, (row, col) in enumerate(window):
            if index != missing:
                board[row][col] = 'XO'[player]
                bits |= 1 << (7 * col + 5 - row)
        game = Connect4(board, 1 - player)
        assert has_four(bits) == (game.check_winner() == player) == (missing is None)
        assert BitboardState.from_game(game).check_winner() == game.check_winner()


def test_sentinel_prevents_wraparound_false_four():
    # Adjacent linear bits at a column boundary are not adjacent board cells.
    for col in range(6):
        bits = sum(1 << (7 * col + row) for row in (4, 5))
        bits |= sum(1 << (7 * (col + 1) + row) for row in (0, 1))
        assert not has_four(bits)


@pytest.mark.parametrize('history', [*POSITIONS.values(), DRAW[:38], DRAW[:40], DRAW[:41],
                                   DRAW, [3] * 6, [0, 1, 0, 1, 0, 1, 0],
                                   [0, 1, 0, 1, 2, 1, 2, 1]])
def test_fixtures_conversion_legal_drops_terminals_and_isolation(history):
    game = Connect4()
    state = BitboardState.from_game(game)
    for move in history:
        previous = state
        before = deepcopy(game.__dict__)
        state = state.drop(move)
        assert game.__dict__ == before
        assert previous.current_player == game.current_player
        assert game.make_move(move)
        assert_equivalent(state, game)
    assert_equivalent(BitboardState.from_game(game), game)
    before = deepcopy(game.__dict__)
    for col in range(7):
        if game.is_game_over() or not game.is_valid_move(col):
            with pytest.raises(ValueError, match='Invalid or terminal'):
                state.drop(col)
        else:
            after = deepcopy(game)
            assert after.make_move(col)
            assert_equivalent(state.drop(col), after)
    assert game.__dict__ == before
    with pytest.raises(FrozenInstanceError):
        state.current_player = 1 - state.current_player


@pytest.mark.parametrize('move', [True, False, -1, 7, 1.0, '3', None])
def test_invalid_drops_leave_state_unchanged(move):
    state = BitboardState.from_game(Connect4())
    assert not state.is_valid_move(move)
    with pytest.raises(ValueError):
        state.drop(move)
    assert state.occupied == 0


@pytest.mark.parametrize('seed', range(20))
def test_random_legal_games_and_all_alternative_drops_match_engine(seed):
    rng = Random(seed + 38219)
    for _ in range(8):
        game = Connect4()
        state = BitboardState.from_game(game)
        while not game.is_game_over():
            assert_equivalent(state, game)
            for move in game.get_valid_moves():
                after = deepcopy(game)
                after.make_move(move)
                assert_equivalent(state.drop(move), after)
            move = rng.choice(game.get_valid_moves())
            game.make_move(move)
            state = state.drop(move)
        assert_equivalent(state, game)
        with pytest.raises(ValueError):
            state.drop(0)


@pytest.mark.parametrize('history', [*POSITIONS.values(), [3] * 6, DRAW[:40], DRAW[:41], DRAW,
                                   [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 2, 1, 2, 1]])
@pytest.mark.parametrize('seed', range(8))
def test_rollouts_match_engine_winner_and_rng_consumption(history, seed):
    game = Connect4()
    for move in history:
        game.make_move(move)
    before = deepcopy(game.__dict__)
    engine = deepcopy(game)
    engine_rng, bits_rng = Random(seed), Random(seed)
    while not engine.is_game_over():
        engine.make_move(engine_rng.choice(engine.get_valid_moves()))
    assert random_rollout(BitboardState.from_game(game), bits_rng) == engine.check_winner()
    assert bits_rng.getstate() == engine_rng.getstate()
    assert game.__dict__ == before


@pytest.mark.parametrize('budget', [100, 400, 800, 1000])
def test_search_uses_compact_states_exact_accounting_and_no_engine_operations(budget, monkeypatch):
    game = position(POSITIONS['near-opening'])
    before = deepcopy(game.__dict__)
    def forbidden(*args):
        pytest.fail('Search must not invoke engine move or terminal operations')
    for name in ('make_move', 'is_game_over', 'check_winner', '_has_terminated'):
        monkeypatch.setattr(Connect4, name, forbidden)
    agent = MCTSAgent(budget, rng=Random(19))
    roots = []
    backpropagate = agent._backpropagate
    def record(node, winner):
        assert isinstance(node.game_state, BitboardState)
        backpropagate(node, winner)
        while node.parent is not None:
            node = node.parent
        roots.append(node)
    monkeypatch.setattr(agent, '_backpropagate', record)
    assert agent.choose_move(game) in game.get_valid_moves()
    root = roots[-1]
    assert root.visits == len(roots) == budget
    assert sum(child.visits for child in root.children.values()) == budget
    count, pending = 0, [root]
    while pending:
        node = pending.pop()
        count += 1
        assert isinstance(node.game_state, BitboardState)
        assert node.player_just_moved == 1 - node.game_state.current_player
        assert node.visits > 0
        assert 0 <= node.wins <= node.visits
        pending.extend(node.children.values())
    assert count <= budget + 1
    assert game.__dict__ == before


@pytest.mark.parametrize('winner', [0, 1, -1])
@pytest.mark.parametrize('history', [[], [3]])
def test_compact_tree_previous_player_rewards(history, winner):
    root = Node(BitboardState.from_game(position(history)))
    first = Node(root.game_state.drop(0), parent=root, move=0)
    second = Node(first.game_state.drop(1), parent=first, move=1)
    MCTSAgent(1)._backpropagate(second, winner)
    for node in (root, first, second):
        assert node.visits == 1
        assert node.wins == (0.5 if winner == -1 else float(winner == node.player_just_moved))


def test_selection_matches_independent_ucb_scores_and_seeded_tie_breaks():
    root = Node(BitboardState.from_game(Connect4()))
    for col in root.untried_moves:
        root.children[col] = Node(root.game_state.drop(col), parent=root, move=col)
    rng = Random(973)
    for _ in range(100):
        root.visits = 2000
        for child in root.children.values():
            child.visits = rng.randrange(0, 100)
            child.wins = rng.randrange(child.visits + 1) * 0.5
        scores = {move: child.ucb1() for move, child in root.children.items()}
        expected_rng = Random(81)
        expected = expected_rng.choice([move for move, score in scores.items() if score == max(scores.values())])
        agent = MCTSAgent(1, rng=Random(81))
        assert agent._select_child(root) is root.children[expected]
        assert agent.rng.getstate() == expected_rng.getstate()
