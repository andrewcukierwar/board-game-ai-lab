"""Array-only evaluation and rollback checks, independent of cached scoring."""
from copy import deepcopy
from itertools import product
from random import Random

import pytest

from games.connect4.connect4 import Connect4
from games.connect4.agents import negamax_agent as engine
from tests.test_connect4_negamax import DRAW, position, root_oracle
from scripts.benchmark_negamax_incremental import load_baseline
from scripts.negamax_evaluation_variants import VARIANTS


def array_windows():
    return tuple(tuple((r + i * dr, c + i * dc) for i in range(4))
                 for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
                 for r in range(6) for c in range(7)
                 if 0 <= r + 3 * dr < 6 and 0 <= c + 3 * dc < 7)


def array_evaluation(board, mover):
    score = 0
    for cells in array_windows():
        values = [board[r][c] for r, c in cells]
        x, o = values.count('X'), values.count('O')
        # Deliberately reproduce the original two independent branches,
        # including both branches on empty windows.
        if not o:
            score += (0, 1, 3, 9, 81)[x]
        if not x:
            score -= (0, 1, 3, 9, 81)[o]
    return score if mover == 0 else -score


def snapshot(state):
    return deepcopy(tuple(getattr(state, name) for name in state.__slots__))


def assert_state(state, board, mover, stack):
    pieces = [sum(1 << (7 * c + 5 - r) for r in range(6) for c in range(7)
                  if board[r][c] == player) for player in 'XO']
    assert state.pieces == pieces
    assert state.heights == [sum(board[r][c] != ' ' for r in range(6)) for c in range(7)]
    assert state.count == sum(v != ' ' for row in board for v in row)
    assert state.mover == mover
    expected = {}
    for cells in array_windows():
        values = [board[r][c] for r, c in cells]
        mask = sum(1 << (7 * c + 5 - r) for r, c in cells)
        expected[mask] = values.count('X') + 5 * values.count('O')
    assert len(expected) == 69
    assert set(expected) == set(engine.WINDOWS)
    assert state.window_counts == [expected[mask] for mask in engine.WINDOWS]
    assert state.score == array_evaluation(board, 0)
    assert state.heuristic() == array_evaluation(board, mover)
    assert state.score_history == stack


def test_all_geometries_all_piece_patterns_both_perspectives():
    # Floating stones here intentionally test all 3**4 configurations in
    # every geometry; legal gravity transitions are tested separately below.
    assert len(array_windows()) == 69
    for cells in array_windows():
        for values in product(' XO', repeat=4):
            board = [[' '] * 7 for _ in range(6)]
            for (r, c), value in zip(cells, values):
                board[r][c] = value
            for mover in (0, 1):
                state = engine.SearchState(Connect4(board, mover))
                assert_state(state, board, mover, [])
    assert engine.WINDOW_SCORES[0] == 0


@pytest.mark.parametrize('moves', [[], [0], [0, 1, 0, 1, 0, 2],
                                  [0, 1, 0, 1, 0], [1, 0, 1, 0, 2, 0, 2, 6, 4, 6, 4, 6],
                                  DRAW[:30], DRAW[:38], DRAW[:41], DRAW])
def test_tactics_blocked_dense_draw_and_every_column(moves):
    game = position(moves)
    for mover in (0, 1):
        assert_state(engine.SearchState(Connect4(game.board, mover)), game.board, mover, [])
    for col in range(7):
        state = engine.SearchState(position([]))
        before = snapshot(state)
        game = position([col])
        state.play(col)
        assert_state(state, game.board, game.current_player, [0])
        state.undo(col)
        assert snapshot(state) == before


def test_thousands_of_seeded_legal_branching_transitions_and_complete_undo():
    rng = Random(31012026)
    transitions = 0

    def explore(state, game, remaining, stack):
        nonlocal transitions
        if not remaining or game.is_game_over():
            return
        # Different sibling orders and branch depths exercise overwritten
        # incremental state after undo, rather than only one linear path.
        cols = game.get_valid_moves()
        rng.shuffle(cols)
        for col in cols[:2]:
            before = snapshot(state)
            child = Connect4(game.board, game.current_player)
            assert child.make_move(col)
            child_stack = stack + [array_evaluation(game.board, 0)]
            state.play(col)
            transitions += 1
            assert_state(state, child.board, child.current_player, child_stack)
            explore(state, child, remaining - 1, child_stack)
            state.undo(col)
            transitions += 1
            assert_state(state, game.board, game.current_player, stack)
            assert snapshot(state) == before

    for sample in range(160):
        game = Connect4()
        state = engine.SearchState(game)
        initial = snapshot(state)
        games, moves, stack = [game], [], []
        assert_state(state, game.board, game.current_player, stack)
        while not game.is_game_over():
            if len(moves) % 5 == sample % 5:
                explore(state, game, 1 + sample % 4, stack)
            col = rng.choice(game.get_valid_moves())
            child = Connect4(game.board, game.current_player)
            assert child.make_move(col)
            stack.append(array_evaluation(game.board, 0))
            state.play(col)
            transitions += 1
            moves.append(col)
            games.append(child)
            game = child
            assert_state(state, game.board, game.current_player, stack)
        while moves:
            state.undo(moves.pop())
            transitions += 1
            stack.pop()
            games.pop()
            game = games[-1]
            assert_state(state, game.board, game.current_player, stack)
        assert snapshot(state) == initial
    assert transitions >= 10_000
    print(f'Validated {transitions} seeded legal play/undo transitions')


@pytest.mark.parametrize('method', ['heuristic', 'terminal_value', 'ordered_moves'])
def test_nested_and_root_exception_restore_all_incremental_state(method, monkeypatch):
    game = position([3, 2, 4, 3])
    state = engine.SearchState(game)
    before, caller = snapshot(state), deepcopy((game.board, game.current_player, game.piece))
    original = getattr(engine.SearchState, method)

    def fail(self, *args, **kwargs):
        if self.count >= 7:
            raise RuntimeError('injected nested failure')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(engine.SearchState, method, fail)
    with pytest.raises(RuntimeError, match='injected'):
        engine.negamax(state, 4, table=engine.SearchTable())
    assert snapshot(state) == before
    held = []
    factory = engine.SearchState

    def capture(game):
        result = factory(game)
        held.append(result)
        return result

    monkeypatch.setattr(engine, 'SearchState', capture)
    with pytest.raises(RuntimeError, match='injected'):
        engine.NegamaxAgent(4).choose_move(game)
    assert snapshot(held[0]) == before
    assert (game.board, game.current_player, game.piece) == caller


@pytest.mark.parametrize('depth', [1, 2, 3, 4, 6, 8, 10])
def test_unpruned_array_minimax_retained_for_deep_tractable_trees(depth):
    game = position(DRAW[:36])
    assert engine.NegamaxAgent(depth).score_moves(game) == root_oracle(game, depth)


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_ablation_legal_counts_scores_and_rollback(variant):
    rng = Random(731)
    cls = VARIANTS[variant]
    for sample in range(80):
        game, moves, saved = Connect4(), [], []
        state = cls(game)
        initial = deepcopy((snapshot(state), vars(state) if hasattr(state, '__dict__') else None))
        while not game.is_game_over():
            col = rng.choice(game.get_valid_moves())
            saved.append((game, deepcopy((snapshot(state), vars(state) if hasattr(state, '__dict__') else None))))
            child = Connect4(game.board, game.current_player)
            assert child.make_move(col)
            state.play(col)
            game = child
            moves.append(col)
            rebuilt = engine.SearchState(game)
            codes = ([x + 5 * o for x, o in zip(*state.counts)] if variant == 'arrays-stack'
                     else state.window_counts)
            assert codes == rebuilt.window_counts
            assert state.score == array_evaluation(game.board, 0)
            assert state.heuristic() == array_evaluation(game.board, game.current_player)
            assert state.pieces == rebuilt.pieces and state.heights == rebuilt.heights
            assert state.mover == game.current_player and state.count == len(moves)
        while moves:
            state.undo(moves.pop())
            game, before = saved.pop()
            assert (snapshot(state), vars(state) if hasattr(state, '__dict__') else None) == before
            assert state.heuristic() == array_evaluation(game.board, game.current_player)
        assert (snapshot(state), vars(state) if hasattr(state, '__dict__') else None) == initial


@pytest.mark.parametrize('depth', [1, 2, 4, 6, 8, 10])
@pytest.mark.parametrize('moves', [[], [3, 2, 4, 3], [0, 1, 0, 1, 0, 2],
                                  [1, 0, 1, 0, 2, 0], [5, 4, 3, 6, 2, 4],
                                  DRAW[:30], DRAW[:38]])
def test_pinned_baseline_entire_root_vector_choice_and_counters(moves, depth):
    baseline = load_baseline()
    game = position(moves)
    before, after = baseline.NegamaxAgent(depth), engine.NegamaxAgent(depth)
    assert before.choose_move(game) == after.choose_move(game)
    assert list(before.last_scores.items()) == list(after.last_scores.items())
    assert before.last_stats == after.last_stats
