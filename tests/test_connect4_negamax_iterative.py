"""Independent array-oracle, horizon/bound and lifecycle tests for research variants."""
from copy import deepcopy
from math import inf
from random import Random
import ast
import json

import pytest

from games.connect4.agents.negamax_tt import pack_key, unpack_key, unpack_entry
from games.connect4.connect4 import Connect4
from scripts import benchmark_negamax_iterative as bench
from scripts.negamax_iterative_variants import (
    DIAGNOSTICS, IDENTITY_MASK, VARIANTS, git_source, load_variant, schedule, variant_source)
from tests.test_connect4_negamax import DRAW, oracle, position, root_oracle
from tests.test_connect4_incremental_evaluation import assert_state, snapshot

FORCED_LOSS = [1, 0, 3, 0, 5, 0, 1, 6, 3, 6, 5, 6]
CASES = [([], 3), ([0, 1, 0, 1, 0, 2], 4), ([0, 1, 0, 1, 0], 3),
         ([5, 4, 3, 6, 2, 4], 3), ([6, 4, 4, 2, 2, 2, 6, 3], 3),
         (FORCED_LOSS, 4), (DRAW[:36], 4), (DRAW[:41], 10)]


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('history,depth', CASES)
def test_complete_ordered_roots_oracle_ties_caller_and_determinism(variant, history, depth):
    module, game = load_variant(variant), position(history)
    caller = deepcopy((game.board, game.current_player, game.piece))
    agent = module.NegamaxAgent(depth)
    expected = root_oracle(game, depth)
    assert list(agent.score_moves(game).items()) == list(expected.items())
    initial = deepcopy(agent.last_stats)
    assert agent.choose_move(game) == max(expected, key=expected.get)
    assert agent.last_stats == initial
    assert (game.board, game.current_player, game.piece) == caller
    assert len(agent.last_scores) == len(game.get_valid_moves())
    if history == FORCED_LOSS:
        assert max(expected.values()) < -module.WIN_SCORE
    if history == [6, 4, 4, 2, 2, 2, 6, 3]:
        assert sum(v == max(expected.values()) for v in expected.values()) > 1


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('depth', [1, 2, 3, 4, 6])
def test_horizons_are_independent_and_iterations_complete(variant, depth):
    module = load_variant(variant)
    game = position([5, 4, 3, 6, 2, 4])
    agent = module.NegamaxAgent(depth)
    with bench.capture(module) as (held, _):
        agent.choose_move(game)
    iterations = getattr(agent, 'last_iterations', [dict(depth=depth, **agent.last_stats)])
    assert [s['depth'] for s in iterations] == list(schedule(depth, variant))
    for key in held[0].entries:
        # Piece count fixes ply, so the remaining depth proves final horizon only.
        x, o, _, remaining = unpack_key(key)
        assert remaining + x.bit_count() + o.bit_count() == depth + 6
    assert list(agent.last_scores.items()) == list(load_variant('direct').NegamaxAgent(depth).score_moves(game).items())
    assert all(agent.last_stats[k] == sum(s[k] for s in iterations)
               for k in ('nodes', 'entries', 'hits', 'cutoffs'))
    hints = getattr(held[0], 'hints', None)
    assert (hints is not None) == (variant in ('hints', 'combined') and depth > 2)
    if hints:
        assert all(type(k) is int and 0 <= k <= IDENTITY_MASK and type(v) is int and 0 <= v < 7
                   for k, v in hints.items())


@pytest.mark.parametrize('variant', ['hints', 'combined'])
@pytest.mark.parametrize('bad', [None, -1, 7, 99, True, False, 3.0, '3', (3,), {'move': 3}, object()])
def test_invalid_hints_do_not_supply_values_or_skip_moves(variant, bad):
    module = load_variant(variant)
    game = position([5, 4, 3, 6, 2, 4])
    state, table = module.SearchState(game), module.SearchTable()
    identity = pack_key(*state.pieces, state.mover, 0)
    table.hints = {identity: bad}
    assert module.negamax(state, 3, table=table) == oracle(game, 3)
    assert snapshot(state) == snapshot(module.SearchState(game))
    # Full columns and foreign position/mover identities cannot reorder this node.
    game = position(DRAW[:30])
    state, table = module.SearchState(game), module.SearchTable()
    identity = pack_key(*state.pieces, state.mover, 0)
    assert state.heights[3] == 6
    table.hints = {identity: 3, identity ^ (1 << 98): 5, 0: 6}
    assert module.negamax(state, 3, table=table) == oracle(game, 3)


@pytest.mark.parametrize('variant', ['hints', 'combined'])
@pytest.mark.parametrize('depth', [1, 2, 3, 4])
def test_transpositions_horizon_mover_bound_interpretation_and_all_retained_bounds(variant, depth):
    module = load_variant(variant)
    game = position([5, 4, 3, 6, 2, 4])
    state, table = module.SearchState(game), module.SearchTable()
    shallow = module.SearchTable()
    module.negamax(state, 1, table=shallow)
    table.hints = module.export_hints(shallow)
    expected = oracle(game, depth)
    for alpha, beta, flag in ((expected - 2, expected - 1, module.LOWER),
                              (expected + 1, expected + 2, module.UPPER)):
        table.entries.clear()
        value = module.negamax(state, depth, alpha, beta, table)
        assert value >= beta if flag == module.LOWER else value <= alpha
        key = pack_key(*state.pieces, state.mover, depth)
        assert unpack_entry(table.entries[key])[0] == flag
        assert module.negamax(state, depth, table=table) == expected
        assert unpack_entry(table.entries[key])[:2] == (module.EXACT, expected)
    for a, b in ((-10, -9), (0, 1), (20, 21), (-inf, inf)):
        value = module.negamax(state, depth, a, b, table)
        assert value <= a if expected <= a else value >= b if expected >= b else value == expected
    for key, entry in table.entries.items():
        x, o, mover, remaining = unpack_key(key)
        flag, value, move = unpack_entry(entry)
        board = [['X' if x & (1 << (7 * c + 5 - r)) else
                  'O' if o & (1 << (7 * c + 5 - r)) else ' ' for c in range(7)] for r in range(6)]
        child = Connect4(board, mover)
        exact = oracle(child, remaining)
        assert value == exact if flag == module.EXACT else value <= exact if flag == module.LOWER else value >= exact
        assert move in child.get_valid_moves()
    for mover in (0, 1):
        for remaining in (1, 2, 3):
            child = Connect4(game.board, mover)
            assert module.negamax(module.SearchState(child), remaining, table=table) == oracle(child, remaining)
    first, second = position([0, 1, 2, 3]), position([2, 3, 0, 1])
    table = module.SearchTable()
    table.hints = {pack_key(*module.SearchState(first).pieces, 0, 0): 6}
    assert module.negamax(module.SearchState(first), 3, table=table) == oracle(first, 3)
    hits = table.hits
    assert module.negamax(module.SearchState(second), 3, table=table) == oracle(second, 3)
    assert table.hits == hits + 1


@pytest.mark.parametrize('variant', VARIANTS)
def test_terminals_before_leaves_every_perspective_and_large_depth(variant):
    module = load_variant(variant)
    for history in (DRAW, [0, 1, 0, 1, 0, 1, 0]):
        for mover in (0, 1):
            game = position(history)
            game.current_player = mover
            for depth in (0, 4, 10**80):
                assert module.negamax(module.SearchState(game), depth, table=module.SearchTable()) == oracle(game, depth)
            with pytest.raises(ValueError, match='terminal'):
                module.NegamaxAgent(4).choose_move(game)


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('method', ['heuristic', 'ordered_moves', 'terminal_value'])
def test_recursive_and_root_exception_restore_exact_incremental_state(variant, method, monkeypatch):
    module = load_variant(variant)
    game = position([3, 2, 4, 3])
    caller = deepcopy((game.board, game.current_player, game.piece))
    state = module.SearchState(game)
    before = snapshot(state)
    original = getattr(module.SearchState, method)

    def fail(self, *args, **kwargs):
        if self.count >= 6:
            raise RuntimeError('injected nested failure')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(module.SearchState, method, fail)
    table = module.SearchTable()
    table.hints = {pack_key(*state.pieces, state.mover, 0): 2}
    with pytest.raises(RuntimeError, match='injected'):
        module.negamax(state, 4, table=table)
    assert snapshot(state) == before
    with pytest.raises(RuntimeError, match='injected'):
        module.NegamaxAgent(6).choose_move(game)
    assert (game.board, game.current_player, game.piece) == caller


@pytest.mark.parametrize('variant', ['hints', 'combined'])
def test_preparation_export_and_final_iteration_exception_restoration(variant, monkeypatch):
    module = load_variant(variant)
    game = position([3, 2, 4, 3])
    states, tables = [], []
    original_state, original_table = module.SearchState, module.SearchTable

    def state_factory(game):
        state = original_state(game)
        states.append(state)
        return state

    def table_factory():
        table = original_table()
        tables.append(table)
        return table

    monkeypatch.setattr(module, 'SearchState', state_factory)
    monkeypatch.setattr(module, 'SearchTable', table_factory)
    original = module.negamax

    def fail(state, depth, *args):
        if len(tables) == len(schedule(6, variant)) and state.count >= 6:
            raise RuntimeError('final iteration failure')
        return original(state, depth, *args)

    monkeypatch.setattr(module, 'negamax', fail)
    with pytest.raises(RuntimeError, match='final'):
        module.NegamaxAgent(6).choose_move(game)
    assert snapshot(states[0]) == snapshot(original_state(game))

    def export_fail(table):
        raise RuntimeError('export failure')

    monkeypatch.setattr(module, 'export_hints', export_fail)
    with pytest.raises(RuntimeError, match='export'):
        module.NegamaxAgent(6).choose_move(game)
    assert snapshot(states[-1]) == snapshot(original_state(game))


@pytest.mark.parametrize('variant', VARIANTS)
def test_seeded_incremental_state_and_search_parity(variant):
    module, rng = load_variant(variant), Random(338310)
    for sample in range(12):
        game = position([])
        for _ in range(sample * 2):
            if game.is_game_over():
                break
            game.make_move(rng.choice(game.get_valid_moves()))
        if game.is_game_over():
            continue
        state = module.SearchState(game)
        assert_state(state, game.board, game.current_player, [])
        initial = snapshot(state)
        for col in state.legal():
            child = Connect4(game.board, game.current_player)
            child.make_move(col)
            old_score = state.score
            state.play(col)
            assert_state(state, child.board, child.current_player, [old_score])
            state.undo(col)
            assert snapshot(state) == initial
        assert list(module.NegamaxAgent(3).score_moves(game).items()) == list(root_oracle(game, 3).items())


@pytest.mark.parametrize('variant', VARIANTS)
def test_diagnostic_instrumentation_matches_unwrapped_decision(variant):
    game = position([3, 2, 4, 3])
    module = load_variant(variant)
    normal = bench.decision(module, game, 6)
    counted = bench.counted_run(variant, game, 6)
    bench.assert_same(normal, counted, True)
    assert counted['diagnostic_totals']['invalid_hints'] == 0
    assert counted['diagnostic_totals']['leaves'] > 0
    if variant in ('direct', 'iterative'):
        assert counted['diagnostic_totals']['hint_lookups'] == 0
    else:
        assert counted['diagnostic_totals']['legal_hints'] > 0


def test_source_scope_schedule_and_exclusive_evidence_writes(tmp_path):
    base = ast.parse(git_source())
    for variant in VARIANTS:
        tree = ast.parse(variant_source(variant))
        for name in ('SearchState', 'has_four', 'winning_squares', 'NegamaxAgent'):
            old = next(n for n in base.body if getattr(n, 'name', None) == name)
            new = next(n for n in tree.body if getattr(n, 'name', None) == name)
            assert ast.dump(old) == ast.dump(new)
    assert variant_source('direct').encode() == git_source()
    assert schedule(10, 'hints') == (8, 10)
    assert schedule(10, 'iterative') == (2, 4, 6, 8, 10)
    assert schedule(9, 'combined') == (1, 3, 5, 7, 9)
    for invalid in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            schedule(invalid, 'direct')
    path = tmp_path / 'evidence.json'
    bench.write_new(path, {'ok': True})
    with pytest.raises(FileExistsError):
        bench.write_new(path, {'ok': False})
    assert json.loads(path.read_text()) == {'ok': True}
