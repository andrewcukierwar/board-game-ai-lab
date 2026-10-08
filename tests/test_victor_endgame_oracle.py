"""Independent mechanics, exact values, full trees, isolation and strict cutoffs."""
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

import pytest

from victor_validation.exact_oracle import EndgamePosition, has_four, replay, solve, verify_replay
from victor_validation import reference_game as reference
from victor_validation.report import FIXTURES

CASES = json.loads(FIXTURES.read_text())['inputs']['cases']


@pytest.mark.parametrize('moves,winner', [
    ([0, 1, 0, 1, 0, 1, 0], 0),  # Vertical White.
    ([0, 1, 0, 1, 2, 1, 2, 1], 1),  # Vertical Black.
    ([0, 0, 1, 1, 2, 2, 3], 0),  # Horizontal White.
])
def test_independent_terminal_replays(moves, winner):
    p = replay(moves)
    assert p.winner == reference.winner(p.board) == winner
    assert p.terminal and p.legal_columns == ()
    result = solve(p, max_remaining=0, position_budget=1)
    assert result.status == 'exact' and result.move_values == ()
    assert result.value_for(winner) == 1 and result.value_for(1 - winner) == -1
    assert result.visited_positions == 1
    with pytest.raises(ValueError, match='post-terminal'):
        replay(moves + [6])


def test_all_four_cell_lines_both_colors_and_reflections_independently():
    # Endpoint interpolation, not the oracle's shifts or production GROUPS.
    cells = [(r, c) for r in range(6) for c in range(7)]
    lines = []
    for (r, c), (rr, cc) in combinations(cells, 2):
        if (rr - r, cc - c) not in ((0, 3), (3, 0), (3, 3), (3, -3)):
            continue
        line = [(r + i * (rr - r) // 3, c + i * (cc - c) // 3) for i in range(4)]
        lines.append(line)
        for piece in ('X', 'O'):
            board = [[' '] * 7 for _ in range(6)]
            for a, b in line:
                board[a][b] = piece
            bits = sum(1 << (7 * b + 5 - a) for a, b in line)
            assert has_four(bits) and reference.winner(board) == 'XO'.index(piece)
            for a, b in line:
                assert not has_four(bits ^ (1 << (7 * b + 5 - a)))
            mirrored = sum(1 << (7 * (6 - b) + 5 - a) for a, b in line)
            assert has_four(mirrored)
    assert len(lines) == 69
    # Across a column boundary: the sentinel must prevent a false vertical four.
    assert not has_four(sum(1 << i for i in (4, 5, 7, 8)))


def test_exhaustive_two_column_occupancies_against_matrix_run_detection():
    # All 4096 occupancies in two complete columns, including nongravity patterns.
    for mask in range(1 << 12):
        board = [[' '] * 7 for _ in range(6)]
        bits = 0
        for i in range(12):
            if mask & (1 << i):
                h, c = i % 6, i // 6
                board[5 - h][c] = 'X'
                bits |= 1 << (7 * c + h)
        assert has_four(bits) == (reference.winner(board) == 0)


def test_direct_constructed_draws_both_turns_and_full_legal_enumeration():
    # Basic-valid matrices, not claimed replay fixtures. Values hand-checkable.
    full = [list(r) for r in ['XXOOXXO', 'OOXXOOX'] * 3]
    p = EndgamePosition.from_board(full, 0)
    assert solve(p).value_for(0) == solve(p).value_for(1) == 0
    assert p.winner is None and p.terminal and not p.legal_columns
    full[0][2] = ' '
    p = EndgamePosition.from_board(full, 1)
    assert solve(p).move_values == ((2, 0),)
    full[0][0] = ' '
    p = EndgamePosition.from_board(full, 0)
    assert solve(p).move_values == ((0, 0), (2, 0))
    assert p.board == tuple(tuple(row) for row in full)


def test_direct_constructed_tactical_positions_with_hand_checkable_wins():
    # Top a6 empty, b6/c6/d6 White: a6 wins immediately; f6 only draws.
    # Basic-valid construction, not a claim of historical reachability.
    rows = [' XXXO O', 'OOXXOOX', 'XXOOXXO', 'OOXXOOX', 'XXOOXXO', 'OOXXOOX']
    p = EndgamePosition.from_board([list(r) for r in rows], 0)
    assert not p.terminal and p.remaining == 2
    assert solve(p).move_values == ((0, 1), (5, 0))
    assert p.drop(0).winner == reference.winner(p.drop(0).board) == 0
    assert solve(p.drop(5)).move_values == ((0, 0),)  # Black must fill a6.


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_small_exhaustive_reference_all_descendants_at_four_remaining(case):
    # Visit every legal continuation until <=4 empties, then compare uncached
    # matrix minimax, bitboard minimax and ALL root move values, on both turns.
    root = replay(case['moves'])
    seen = set()
    compared = set()

    def inspect(p):
        if p in seen:
            return
        seen.add(p)
        assert p.winner == reference.winner(p.board)
        assert p.legal_columns == reference.legal_moves(p.board)
        if p.remaining <= 4:
            result = solve(p, max_remaining=4, position_budget=1000)
            assert result.status == 'exact'
            assert result.mover_value == reference.exhaustive_value(p.board, p.turn)
            assert result.move_values == tuple((c, -reference.exhaustive_value(
                reference.drop(p.board, p.turn, c), 1 - p.turn)) for c in p.legal_columns)
            assert result.value_for(0) == -result.value_for(1)
            compared.add(p)
            for c in p.legal_columns:
                inspect(p.drop(c))
        elif not p.terminal:
            for c in p.legal_columns:
                assert p.drop(c).board == reference.drop(p.board, p.turn, c)
                inspect(p.drop(c))

    inspect(root)
    assert len(seen) <= 1000 and compared


def test_direct_tactical_move_values_and_perspective_after_forced_reply():
    case = next(c for c in CASES if c['id'] == 'immediate-white-win')
    p = replay(case['moves'])
    assert solve(p).move_values == ((2, -1), (5, -1), (6, 1))
    assert p.drop(6).winner == 0
    q = p.drop(2)
    result = solve(q)
    assert q.turn == 1 and result.mover_value == result.value_for(1) == 1
    assert result.value_for(0) == -1
    draw = replay(next(c['moves'] for c in CASES if c['id'] == 'no-cover-forced-draw'))
    assert solve(draw).move_values == ((0, -1), (5, 0), (6, -1))


def test_exact_budget_boundary_memoization_determinism_and_no_partial_value():
    p = replay(CASES[0]['moves'])
    complete = solve(p)
    assert complete.status == 'exact' and complete.cache_hits > 0
    for budget in (0, 1, complete.visited_positions - 1):
        result = solve(p, position_budget=budget)
        assert result.status == 'unknown_position_budget'
        assert result.visited_positions == budget
        assert result.mover_value is result.value_for(0) is result.value_for(1) is None
        assert result.move_values == ()
    assert solve(p, position_budget=complete.visited_positions).mover_value == complete.mover_value
    assert solve(p) == complete
    assert solve(p, max_remaining=6).status == 'unknown_remaining_cap'
    assert solve(p, max_remaining=6).visited_positions == 0
    assert solve(p, max_remaining=10).mover_value == complete.mover_value


@pytest.mark.parametrize('kwargs', [
    {'max_remaining': True}, {'max_remaining': -1}, {'max_remaining': 11},
    {'max_remaining': 1.5}, {'position_budget': True}, {'position_budget': -1},
    {'position_budget': 1_000_001}, {'position_budget': None},
])
def test_invalid_resource_configuration(kwargs):
    with pytest.raises(ValueError):
        solve(replay(()), **kwargs)


@pytest.mark.parametrize('moves', [[True], [-1], [7], [0.0], [None], [0] * 7, [0] * 43, '012'])
def test_replay_rejects_invalid_columns_full_columns_and_length(moves):
    with pytest.raises(ValueError):
        replay(moves)


def test_replay_exact_binding_and_basic_validation_is_not_reachability():
    p = replay([0, 1])
    assert verify_replay(p.board, p.turn, [0, 1]) == p
    with pytest.raises(ValueError, match='differs'):
        verify_replay(p.board, p.turn, [1, 0])
    # No bottom White stone: White's very first move cannot exist in this board.
    unreachable = tuple(tuple(r) for r in ['       '] * 4 + ['XX     ', 'OO     '])
    assert EndgamePosition.from_board(unreachable, 0).turn == 0
    with pytest.raises(ValueError, match='differs'):
        verify_replay(unreachable, 0, [0, 0, 1, 1])
    with pytest.raises(ValueError):
        verify_replay(p.board, 1, [0, 1])


def test_independent_snapshot_validation_and_forgery_rejection():
    empty = replay(()).board
    for board, turn in [(empty, 1), (empty, True), (empty[:-1], 0),
                        (['       '] * 6, 0), ([['?'] * 7] * 6, 0)]:
        with pytest.raises(ValueError):
            EndgamePosition.from_board(board, turn)
    floating = [list(row) for row in empty]
    floating[4][0] = 'X'
    with pytest.raises(ValueError, match='gravity'):
        EndgamePosition.from_board(floating, 1)
    for p in [replace(replay(()), white=True), replace(replay(()), white=1 << 48),
              replace(replay(()), heights=(True,) + (0,) * 6),
              replace(replay(()), heights=(0,)), replace(replay(()), black=-1)]:
        with pytest.raises(ValueError):
            solve(p)
    terminal = replay([0, 1, 0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match='counts/turn'):
        EndgamePosition.from_board(terminal.board, 0)


def test_no_engine_victor_alphazero_or_neural_imports_in_fresh_process():
    tests_dir = str(Path(__file__).parent.resolve())
    code = f'''
import sys, importlib.abc
sys.path.insert(0, {tests_dir!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('games', 'torch', 'numpy', 'api'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
from victor_validation.exact_oracle import replay, solve
p = replay([0,1,0,1,0,1,0])
assert solve(p).value_for(0) == 1
'''
    run = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
