"""Actual move values, interrupted searches, tactical decisions and full games."""
from dataclasses import replace
import json
from pathlib import Path
from random import Random

import pytest

from games.connect4.victor import Position, SearchBudget, SolverBudget, VictorSolver, select_move, solve_exact
from games.connect4.victor.cli import play_game
from victor_validation.exact_oracle import replay, solve
from victor_validation.reference_game import exhaustive_value, drop

CASES = json.loads((Path(__file__).parent / 'victor_validation/positions.json').read_text())['inputs']['cases']
EXACT = SearchBudget(nodes=100_000, seconds=None, max_remaining=10)
FAST = SolverBudget(exact=EXACT, cover_nodes=0)


def endgames(seed=1988, count=80, remaining=(8, 9, 10)):
    rng = Random(seed)
    for _ in range(count):
        target = rng.choice(remaining)
        while True:
            p, history = replay(()), []
            while p.remaining > target:
                choices = [c for c in p.legal_columns if not p.drop(c).terminal]
                if not choices:
                    break
                c = rng.choice(choices)
                p, history = p.drop(c), history + [c]
            if p.remaining == target:
                yield tuple(history), p
                break


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_exact_every_move_against_independent_oracle(case):
    p = replay(case['moves'])
    result = solve_exact(Position.from_board(p.board, p.turn), EXACT)
    expected = solve(p, max_remaining=10)
    assert result.status == expected.status == 'exact'
    assert result.value == expected.mover_value
    assert dict(result.move_values) == dict(expected.move_values)
    selected = select_move(p.board, p.turn, FAST)
    assert selected.exact_value == expected.mover_value
    if not p.terminal:
        assert dict(expected.move_values)[selected.move] == expected.mover_value


def test_seeded_both_turns_all_root_values_and_matrix_oracle():
    turns = set()
    for _, p in endgames(count=100):
        turns.add(p.turn)
        result = solve_exact(Position.from_board(p.board, p.turn), EXACT)
        expected = solve(p, max_remaining=10)
        assert result.status == expected.status == 'exact'
        assert result.value == expected.mover_value
        assert dict(result.move_values) == dict(expected.move_values)
    assert turns == {0, 1}
    for _, p in endgames(count=20, remaining=(3, 4)):
        result = solve_exact(Position.from_board(p.board, p.turn), EXACT)
        assert result.value == exhaustive_value(p.board, p.turn)
        assert dict(result.move_values) == {c: -exhaustive_value(drop(p.board, p.turn, c), 1-p.turn)
                                           for c in p.legal_columns}


def test_cutoffs_discard_partial_values_and_respect_exact_boundaries():
    p = replay(CASES[0]['moves'])
    position = Position.from_board(p.board, p.turn)
    complete = solve_exact(position, EXACT)
    for nodes in (0, 1, complete.nodes - 1):
        r = solve_exact(position, replace(EXACT, nodes=nodes))
        assert r.status == 'unknown_node_budget'
        assert r.value is r.best_move is None and r.move_values == ()
        assert r.nodes == nodes
    r = solve_exact(position, replace(EXACT, nodes=complete.nodes))
    assert r.status == 'exact' and r.move_values == complete.move_values
    assert solve_exact(position, replace(EXACT, seconds=0)).status == 'unknown_time_budget'
    assert solve_exact(position, replace(EXACT, table_entries=0)).status == 'unknown_table_budget'
    assert solve_exact(position, replace(EXACT, max_remaining=0)).status == 'unknown_remaining_cap'
    again = solve_exact(position, EXACT)
    assert (again.value, again.move_values, again.nodes, again.cache_hits) == (
        complete.value, complete.move_values, complete.nodes, complete.cache_hits)


@pytest.mark.parametrize('kwargs', [dict(nodes=True), dict(nodes=-1), dict(seconds=float('nan')),
                                   dict(seconds=-1), dict(seconds=True), dict(max_remaining=25),
                                   dict(table_entries=1.5)])
def test_invalid_exact_budgets(kwargs):
    with pytest.raises(ValueError):
        SearchBudget(**kwargs)


@pytest.mark.parametrize('history,column,kind', [
    ([0, 1, 0, 1, 0, 2], 0, 'terminal_win'),
    ([0, 1, 0, 1, 2, 1, 2], 1, 'terminal_win'),
    ([0, 1, 0, 1, 2, 1], 1, 'forced_defense'),
    ([0, 1, 0, 1, 0], 0, 'forced_defense'),
])
def test_tactics_both_sides_without_full_search(history, column, kind):
    p = replay(history)
    budget = replace(FAST, exact=replace(EXACT, nodes=0))
    result = select_move(p.board, p.turn, budget)
    assert result.move == column and result.move_kind == kind
    assert result.justified_move
    assert result.bound is None
    assert result.exact_value == (1 if kind == 'terminal_win' else None)


def test_no_cover_and_exact_no_cover_draw_are_different():
    p = replay(next(c['moves'] for c in CASES if c['id'] == 'no-cover-forced-draw'))
    r = select_move(p.board, p.turn, FAST)
    assert r.move == 5 and r.exact_value == 0
    empty = replay(())
    # Without the opening book the empty board is beyond exact search: no value.
    r = select_move(empty.board, 0, SolverBudget(strategic_children=0, opening_book=False))
    assert r.move in empty.legal_columns and r.exact_value is None and r.bound is None
    assert r.black_cover.witness is None
    # The exact opening book proves it a White win; only the centre wins.
    r = select_move(empty.board, 0, SolverBudget(strategic_children=0))
    assert r.move_kind == 'opening_book' and r.exact_value == 1 and r.move == 3


def test_exact_avoids_a_quiet_tactical_trap_that_greedy_fallback_loses():
    # Found by the reproducible quiet sampler (seed 6105, ten remaining).
    # No immediate win for either player; g-column loses while four moves draw.
    history = (4,3,2,1,4,3,3,6,1,2,1,0,0,6,3,5,2,4,4,4,4,2,2,0,0,2,1,3,0,1,6,5)
    p = replay(history)
    expected = solve(p,max_remaining=10)
    assert dict(expected.move_values) == {0:0,1:0,3:0,5:0,6:-1}
    greedy = select_move(p.board,0,replace(FAST,exact=replace(EXACT,nodes=0),fallback_depth=1))
    assert greedy.move == 6 and greedy.move_kind == 'heuristic' and greedy.exact_value is None
    exact = select_move(p.board,0,FAST)
    assert exact.move in (0,1,3,5) and exact.exact_value == 0 and exact.justified_move


def test_terminal_and_invalid_inputs_never_select():
    p = replay([0,1,0,1,0,1,0])
    assert select_move(p.board, p.turn, FAST).move is None
    with pytest.raises(ValueError):
        select_move(p.board, 0, FAST)
    with pytest.raises(ValueError):
        select_move([[' '] * 7] * 6, True, FAST)


@pytest.mark.parametrize('white,black', [('victor','random'), ('random','victor'),
    ('victor','negamax:2'), ('negamax:2','victor'), ('victor','mcts:8'),
    ('mcts:8','victor'), ('victor','victor')])
def test_complete_game_legal_terminal_and_reproducible(white, black):
    budget = SolverBudget(exact=SearchBudget(nodes=20_000, seconds=None, max_remaining=8),
                          cover_nodes=1000, strategic_children=1, white_contexts=2,
                          fallback_depth=2)
    r = play_game(white, black, seed=1988, budget=budget)
    p = replay(r['moves'])
    assert p.terminal and r['winner'] == (-1 if p.winner is None else p.winner)
    assert len(r['moves']) <= 42
    for d in r['decisions']:
        before = replay(r['moves'][:d['ply']])
        assert d['move'] in before.legal_columns and d['player'] == before.turn
        if d['kind'] in ('heuristic', 'exploratory_nine_rule', 'exploratory_white_context'):
            assert d['exact_value'] is None
    again = play_game(white, black, seed=1988, budget=budget)
    assert again['moves'] == r['moves']


def test_stateful_solver_mismatch_discards_previous_game_policy():
    p = replay([0])
    agent = VictorSolver(SolverBudget(exact=replace(EXACT, nodes=0), strategic_children=1))
    first = agent.select_move(p.board, p.turn)
    assert first.move in p.legal_columns
    other = replay([6])
    result = agent.select_move(other.board, 1)
    assert result.position.board == other.board and result.move in other.legal_columns


def test_established_three_rule_bound_and_retained_sigma_r_moves():
    from victor_validation.nine_rule_reference import DIAGRAMS
    p = replay(DIAGRAMS['6.1'] + [0])
    agent = VictorSolver(SolverBudget(exact=replace(EXACT,nodes=0)))
    r = agent.select_move(p.board,1)
    assert r.move_kind == 'strategic_nonloss' and r.bound is not None
    assert r.exact_value is None and r.certificate is not None
    assert agent.cert is not None
    q = p.drop(r.move).drop(0)
    r = agent.select_move(q.board,1)
    assert r.move in q.legal_columns and r.move_kind == 'strategic_nonloss'
    assert r.reason == 'response replays original rule instances from exact anchor'
