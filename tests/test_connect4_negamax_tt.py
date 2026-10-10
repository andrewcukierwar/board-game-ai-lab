"""Independent TT identity, representation, bound and eviction correctness."""
from copy import deepcopy
from math import inf
from random import Random
import sys

import pytest

from games.connect4.agents.negamax_tt import (
    DirectMappedEntries, pack_key, unpack_key, pack_entry, unpack_entry)
from scripts.negamax_tt_variants import VARIANTS, load_variant, variant_source
from scripts import benchmark_negamax_tt as bench
from tests.test_connect4_negamax import DRAW, position, oracle, root_oracle


def key_for(module, state, depth):
    identity = (*state.pieces, state.mover, depth)
    return identity if 'baseline' in module.__name__ else pack_key(*identity)


def entry_for(module, flag, score, hint):
    return pack_entry(flag, score, hint) if 'packed-entry' in module.__name__ else (flag, score, hint)


def snapshot(state):
    return deepcopy(tuple(getattr(state, name) for name in state.__slots__))


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('capacity', [None, 1, 17, 16384])
@pytest.mark.parametrize('history,depth', [
    ([], 3), ([0, 1, 0, 1, 0, 2], 4), ([0, 1, 0, 1, 0], 3),
    ([5, 4, 3, 6, 2, 4], 3), ([6, 4, 4, 2, 2, 2, 6, 3], 3),
    ([5, 6, 6, 5, 4, 3, 0, 1, 0, 2, 4, 6, 6, 2, 3, 0], 3),
    (DRAW[:36], 4), (DRAW[:41], 4),
])
def test_all_exact_roots_and_historical_ties(variant, capacity, history, depth):
    module = load_variant(variant, capacity)
    game = position(history)
    before = deepcopy((game.board, game.current_player, game.piece))
    expected = root_oracle(game, depth)
    agent = module.NegamaxAgent(depth)
    assert list(agent.score_moves(game).items()) == list(expected.items())
    assert agent.choose_move(game) == max(expected, key=expected.get)
    assert (game.board, game.current_player, game.piece) == before
    assert len(agent.last_scores) == len(game.get_valid_moves())
    assert capacity is None or agent.last_stats['entries'] <= capacity


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('capacity', [None, 1, 17])
@pytest.mark.parametrize('depth', [1, 2, 3, 4])
def test_bounds_windows_transpositions_depth_mover_and_terminals(variant, capacity, depth):
    module = load_variant(variant, capacity)
    game = position([5, 4, 3, 6, 2, 4])
    expected, state = oracle(game, depth), module.SearchState(game)
    before = snapshot(state)
    for window, flag in [((expected - 2, expected - 1), module.LOWER),
                         ((expected + 1, expected + 2), module.UPPER)]:
        table = module.SearchTable()
        a, b = window
        value = module.negamax(state, depth, a, b, table)
        entry = dict(table.entries.items())[key_for(module, state, depth)]
        decoded = unpack_entry(entry) if variant == 'packed-entry' else entry
        assert decoded[0] == flag
        assert value >= b if flag == module.LOWER else value <= a
        assert module.negamax(state, depth, table=table) == expected
        entry = dict(table.entries.items())[key_for(module, state, depth)]
        decoded = unpack_entry(entry) if variant == 'packed-entry' else entry
        assert decoded[:2] == (module.EXACT, expected)
        hits = table.hits
        assert module.negamax(state, depth, table=table) == expected
        assert table.hits == hits + 1
    table = module.SearchTable()
    for a, b in [(-10, -9), (0, 1), (20, 21), (-5, 60), (-inf, inf)]:
        value = module.negamax(state, depth, a, b, table)
        assert value <= a if expected <= a else value >= b if expected >= b else value == expected
    assert snapshot(state) == before
    for remaining in (2, 3):
        for mover in (0, 1):
            game.current_player = mover
            assert module.negamax(module.SearchState(game), remaining, table=table) == oracle(game, remaining)
    first, second = position([0, 1, 2, 3]), position([2, 3, 0, 1])
    table = module.SearchTable()
    assert module.negamax(module.SearchState(first), 3, table=table) == oracle(first, 3)
    hits = table.hits
    assert module.negamax(module.SearchState(second), 3, table=table) == oracle(second, 3)
    assert table.hits == hits + 1
    for history in (DRAW, [0, 1, 0, 1, 0, 1, 0]):
        game = position(history)
        for mover in (0, 1):
            game.current_player = mover
            for remaining in (0, 4, 10**80):
                assert module.negamax(module.SearchState(game), remaining, table=table) == oracle(game, remaining)
        with pytest.raises(ValueError, match='terminal'):
            module.NegamaxAgent(4).choose_move(game)


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('capacity', [None, 1])
@pytest.mark.parametrize('flag', ['exact', 'lower', 'upper'])
@pytest.mark.parametrize('hint', [None, 0, 3, 8])
def test_hints_preserve_bounds_and_legal_validation(variant, capacity, flag, hint):
    module = load_variant(variant, capacity)
    for history in ([5, 4, 3, 6, 2, 4], DRAW[:30]):
        game, table = position(history), module.SearchTable()
        expected = oracle(game, 3)
        state = module.SearchState(game)
        table.entries[key_for(module, state, 3)] = entry_for(module, flag, expected, hint)
        assert module.negamax(state, 3, table=table) == expected
        assert state.ordered_moves(99, 'none') == state.legal()
        if history == DRAW[:30]:
            assert 3 not in state.legal()


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('capacity', [None, 1])
def test_every_retained_bound_against_array_oracle_and_exception_rollback(variant, capacity, monkeypatch):
    module = load_variant(variant, capacity)
    game = position([5, 4, 3, 6, 2, 4])
    state, table = module.SearchState(game), module.SearchTable()
    for a, b in [(-10, -9), (20, 21), (-inf, inf)]:
        module.negamax(state, 3, a, b, table)
        for key, entry in table.entries.items():
            x, o, mover, depth = key if variant == 'baseline' else unpack_key(key)
            flag, value, hint = unpack_entry(entry) if variant == 'packed-entry' else entry
            board = [['X' if x & (1 << (7 * col + 5 - row)) else
                      'O' if o & (1 << (7 * col + 5 - row)) else ' '
                      for col in range(7)] for row in range(6)]
            child = type(game)(board, mover)
            exact = oracle(child, depth)
            assert value == exact if flag == module.EXACT else value <= exact if flag == module.LOWER else value >= exact
            assert hint in child.get_valid_moves()
    before, caller = snapshot(state), deepcopy((game.board, game.current_player, game.piece))

    def fail(self):
        raise RuntimeError('injected leaf failure')

    monkeypatch.setattr(module.SearchState, 'heuristic', fail)
    with pytest.raises(RuntimeError, match='injected'):
        module.negamax(state, 3, table=module.SearchTable())
    assert snapshot(state) == before
    with pytest.raises(RuntimeError, match='injected'):
        module.NegamaxAgent(3).choose_move(game)
    assert (game.board, game.current_player, game.piece) == caller


def test_injective_legal_identities_exhaustive_openings_and_seeded_play_undo():
    module = load_variant('baseline')
    seen = {}

    def check(state):
        for mover in (0, 1):
            for depth in (0, 1, 4, 10, 42, 10**100):
                identity = (*state.pieces, mover, depth)
                key = pack_key(*identity)
                assert unpack_key(key) == identity
                assert seen.setdefault(key, identity) == identity

    state = module.SearchState(position([]))

    def visit(remaining):
        check(state)
        if remaining:
            for col in state.legal():
                before = snapshot(state)
                state.play(col)
                visit(remaining - 1)
                state.undo(col)
                assert snapshot(state) == before

    visit(5)
    rng = Random(3102)
    for _ in range(200):
        state = module.SearchState(position([]))
        history = []
        while state.terminal_value(0) is None:
            check(state)
            col = rng.choice(state.legal())
            history.append(col)
            state.play(col)
        check(state)
        for col in reversed(history):
            state.undo(col)
            check(state)
    assert len(seen) > 100000
    # All 49 individual bits, including sentinel-field boundaries, are disjoint.
    for bit in range(49):
        assert pack_key(1 << bit, 0, 0, 0) != pack_key(0, 1 << bit, 0, 0)
        assert unpack_key(pack_key((1 << 49) - 1, (1 << 49) - 1, 1, 10**100)) == (
            (1 << 49) - 1, (1 << 49) - 1, 1, 10**100)


@pytest.mark.parametrize('flag', ['exact', 'lower', 'upper'])
@pytest.mark.parametrize('score', [0, -1, 1, -1000000, 1000000, -1000042, 1000042,
                                  -(1000000 + 10**100), 1000000 + 10**100])
@pytest.mark.parametrize('hint', [None, *range(15)])
def test_entry_roundtrips_all_fields_and_signed_boundaries(flag, score, hint):
    assert unpack_entry(pack_entry(flag, score, hint)) == (flag, score, hint)


def test_encoding_rejects_ambiguous_or_overflowing_fields():
    for identity in [(-1, 0, 0, 0), (1 << 49, 0, 0, 0), (0, 1 << 49, 0, 0),
                     (0, 0, 2, 0), (0, 0, 0, -1), (False, 0, 0, 0)]:
        with pytest.raises(ValueError):
            pack_key(*identity)
    for entry in [('bad', 0, 0), ('exact', .5, 0), ('exact', 1, -1),
                  ('exact', 1, 15), ('exact', 1, 99)]:
        with pytest.raises(ValueError):
            pack_entry(*entry)
    for entry in [-1, 3, 7, True]:
        with pytest.raises(ValueError):
            unpack_entry(entry)


def test_full_equality_on_forced_collisions_and_eviction_accounting():
    entries = DirectMappedEntries(1)
    a, b = pack_key(1, 2, 0, 3), pack_key(1, 2, 1, 3)
    entries[a] = pack_entry('exact', 0, None)
    assert entries.get(a) == pack_entry('exact', 0, None)
    assert entries.get(b) is None
    entries[b] = pack_entry('lower', -1000003, 2)
    assert entries.get(a) is None
    assert entries.get(b) == pack_entry('lower', -1000003, 2)
    entries[b] = pack_entry('upper', 1000003, 3)
    assert len(entries) == 1 and entries.evictions == 1 and entries.replacements == 1
    assert list(entries.items()) == [(b, pack_entry('upper', 1000003, 3))]
    for bad in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            DirectMappedEntries(bad)


@pytest.mark.parametrize('variant', VARIANTS)
def test_baseline_all_counters_and_leaves_match(variant):
    baseline, module = load_variant('baseline'), load_variant(variant)
    for history in ([], [1, 4, 6, 0, 6], DRAW[:36]):
        for depth in (2, 4, 6):
            a = bench.counted_run(baseline, position(history), depth)
            b = bench.counted_run(module, position(history), depth)
            bench.assert_same(a, b)
            assert a['leaf_evaluations'] == b['leaf_evaluations']


def test_only_tt_expressions_changed_in_ablation_sources():
    # Reverse declared substitutions: evaluation/search/pruning bodies are exact.
    base = variant_source('baseline')
    for variant in ('packed-key', 'packed-entry'):
        source = variant_source(variant)
        source = source.replace('\nfrom games.connect4.agents.negamax_tt import pack_entry, unpack_entry', '')
        source = source.replace('key = (state.pieces[0] | (state.pieces[1] << 49) |\n'
                                '           (state.mover << 98) | (depth << 99))',
                                'key = (*state.pieces, state.mover, depth)')
        source = source.replace('unpack_entry(table.entries[key])', 'table.entries[key]')
        source = source.replace('pack_entry(flag, best, best_move)', '(flag, best, best_move)')
        assert source == base


def test_reachable_memory_matches_independent_unique_graph():
    module = load_variant('baseline')
    bench.warm(module)
    with bench.capture_table(module) as held:
        bench.decision(module, position([]), 4)
    seen = set()

    def size(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        total = sys.getsizeof(obj)
        if isinstance(obj, dict):
            total += sum(size(k) + size(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            total += sum(size(v) for v in obj)
        elif hasattr(obj, '__dict__'):
            total += size(vars(obj))
        return total

    breakdown = bench.memory_breakdown(held[0])
    assert breakdown['total_bytes'] == size(held[0])
    assert breakdown['categories']['key_tuples'] == 72 * len(held[0].entries)


def test_manifest_freezes_fixtures_sources_and_prevents_overwrite(tmp_path):
    (tmp_path / 'DESIGN.md').write_bytes((bench.ROOT / 'DESIGN.md').read_bytes())
    config = bench.declare(tmp_path)
    bench.check_source(config, tmp_path)
    assert len(config['positions']) == 24
    assert sum(p['group'] == 'phase3a' for p in config['positions']) == 11
    assert [len(p['history']) for p in config['positions'] if p['group'] == 'additional'] == [10, 14, 18, 22]
    assert [p['history'] for p in config['positions'] if p['group'] == 'post-hoc-diagnostic'] == [[1, 4, 6, 0, 6]]
    with pytest.raises(FileExistsError):
        bench.declare(tmp_path)
    (tmp_path / 'DESIGN.md').write_text('changed')
    with pytest.raises(AssertionError):
        bench.check_source(config, tmp_path)


@pytest.mark.parametrize('field', ['move', 'scores', 'stats'])
def test_fail_closed_result_counter_parity(field):
    a = dict(move=3, scores=[(3, 0), (2, 0)], stats={'nodes': 2})
    b = deepcopy(a)
    b[field] = None
    with pytest.raises(AssertionError):
        bench.assert_same(a, b)
    if field == 'stats':
        bench.assert_same(a, b, counters=False)


def test_fresh_process_json_preserves_exact_vector_shape():
    import json
    result = bench.decision(load_variant('baseline'), position([]), 2)
    bench.assert_same(result, json.loads(json.dumps(result)))


def analysis_row(group='phase3a', depth=10, ratio=1.0, memory_ratio=.5, node_ratio=1.0):
    base = dict(median_wall_seconds=.1, median_cpu_seconds=.1,
                counted=dict(stats=dict(entries=20000, nodes=50000)),
                memory=[dict(retained=dict(total_bytes=100000), traced_peak_bytes=100000)])
    new = deepcopy(base)
    new.update(median_wall_seconds=.1 * ratio, median_cpu_seconds=.1 * ratio)
    new['memory'][0]['retained']['total_bytes'] = int(100000 * memory_ratio)
    new['memory'][0]['traced_peak_bytes'] = int(100000 * memory_ratio)
    new['counted']['stats']['nodes'] = int(50000 * node_ratio)
    return dict(group=group, depth=depth, variants=dict(baseline=base,
        **{'packed-key': deepcopy(new), 'packed-entry': deepcopy(new)}))


def test_A_acceptance_excludes_posthoc_tail_but_enforces_latency_and_memory():
    rows = [analysis_row(depth=d) for d in (4, 6, 8, 10)]
    rows.append(analysis_row(group='post-hoc-diagnostic', ratio=100))
    assert bench.analyze(rows, 'A')['selected'] is not None
    rows[0] = analysis_row(depth=4, ratio=1.3)
    assert bench.analyze(rows, 'A')['selected'] is None
    rows = [analysis_row(depth=d, memory_ratio=.8) for d in (4, 6, 8, 10)]
    assert bench.analyze(rows, 'A')['selected'] is None


@pytest.mark.parametrize('tail_ratio,tail_nodes', [(1.3, 1), (1, 1.6)])
def test_B_rejects_expensive_tail_latency_or_recomputation(tail_ratio, tail_nodes):
    rows = [analysis_row(depth=d) for d in (4, 6, 8, 10)]
    rows.append(analysis_row(group='post-hoc-diagnostic', ratio=tail_ratio, node_ratio=tail_nodes))
    for row in rows:
        old = row['variants']
        row['variants'] = dict(unbounded=old['baseline'], **{'16384': old['packed-key']})
    assert bench.analyze(rows, 'B')['selected'] is None
    assert not all(bench.analyze(rows, 'B')['comparisons']['16384']['checks'].values())


def test_dictionary_structure_and_spare_capacity_formula_matches_runtime():
    from scripts.audit_negamax_tt import dictionary_layout
    for count in (1, 5, 6, 50, 128, 1000, 10000, 119958):
        table = dict.fromkeys(range(count), 0)
        layout = dictionary_layout(sys.getsizeof(table), count)
        assert layout['allocated_bytes'] == (layout['headers_bytes'] + layout['hash_index_bytes'] +
                layout['occupied_dense_bytes'] + layout['spare_dense_bytes'])
        assert layout['spare_dense_bytes'] >= 0


def test_bounded_zero_entry_is_a_hit_not_absence():
    entries = DirectMappedEntries(1)
    entries[100] = pack_entry('exact', 0, 0)
    assert entries.get(100) == 0
    assert unpack_entry(entries.get(100)) == ('exact', 0, 0)
    assert entries.get(101) is None


def test_python_hash_collision_does_not_alias_full_packed_identities():
    modulus = sys.hash_info.modulus
    first = pack_key(0, 0, 0, 1)
    second = pack_key(0, 0, 0, 1 + modulus)
    assert first != second and hash(first) == hash(second)
    entries = {first: pack_entry('exact', 7, 3), second: pack_entry('lower', -9, None)}
    assert len(entries) == 2 and unpack_entry(entries[first]) == ('exact', 7, 3)
    bounded = DirectMappedEntries(16384)
    bounded[first] = entries[first]
    assert bounded.get(second) is None
    bounded[second] = entries[second]
    assert bounded.get(first) is None and bounded.get(second) == entries[second]


@pytest.mark.parametrize('capacity', [1, 16, 16384, 32768, 65536])
def test_mixed_index_full_key_collision_safety_and_zero_entries(capacity):
    from scripts.benchmark_negamax_tt_mixed import MixedEntries
    entries = MixedEntries(capacity)
    first = pack_key(0, 0, 0, 1)
    second = pack_key(0, 0, 0, 1 + sys.hash_info.modulus)
    assert hash(first) == hash(second)
    entries[first] = 0
    assert entries.get(first) == 0 and entries.get(second) is None
    entries[second] = pack_entry('upper', -(1000000 + 10**100), None)
    assert entries.get(first) is None
    assert unpack_entry(entries.get(second)) == ('upper', -(1000000 + 10**100), None)
    assert len(entries) == 1 and entries.evictions == 1
    entries[second] = pack_entry('exact', 1, 6)
    assert entries.replacements == 1


@pytest.mark.parametrize('capacity', [1, 16, 16384, 32768, 65536])
@pytest.mark.parametrize('history,depth', [([], 3), ([0, 1, 0, 1, 0], 3),
                                         ([0, 1, 0, 1, 0, 2], 4), (DRAW[:36], 4)])
def test_mixed_index_exact_complete_roots_against_array_oracle(capacity, history, depth):
    from scripts.benchmark_negamax_tt_mixed import load_mixed_variant
    module = load_mixed_variant('packed-entry', capacity)
    game = position(history)
    before = deepcopy((game.board, game.current_player, game.piece))
    agent = module.NegamaxAgent(depth)
    expected = root_oracle(game, depth)
    assert agent.choose_move(game) == max(expected, key=expected.get)
    assert list(agent.last_scores.items()) == list(expected.items())
    assert agent.last_stats['entries'] <= capacity
    assert (game.board, game.current_player, game.piece) == before


def test_tt_probe_accounting_separates_terminals_and_heuristic_leaves():
    from scripts.diagnose_negamax_tt import diagnose
    for history in ([], [0, 1, 0, 1, 0, 2], DRAW[:41]):
        result, table = diagnose(load_variant('packed-entry'), position(history), 4)
        assert result['stats']['nodes'] == (result['tt_probes'] + result['terminal_nodes'] +
                                           result['leaf_evaluations'])
        assert 0 <= result['hit_per_probe'] <= 1
        assert result['stats']['entries'] == len(table.entries)
        if history == DRAW[:41]:
            assert result['terminal_nodes'] == result['stats']['nodes']
            assert result['tt_probes'] == result['leaf_evaluations'] == 0


def test_mixed_strategy_changes_only_slot_expression_in_probe_and_store():
    import ast
    import inspect
    import textwrap
    from scripts.benchmark_negamax_tt_mixed import MixedEntries
    for name in ('get', '__setitem__'):
        nodes = []
        for cls in (DirectMappedEntries, MixedEntries):
            node = ast.parse(textwrap.dedent(inspect.getsource(getattr(cls, name)))).body[0]
            assert isinstance(node.body[0], ast.Assign)
            assert ast.dump(node.body[0].targets[0]) == "Name(id='slot', ctx=Store())"
            node.body[0].value = ast.Constant(value='SLOT')
            nodes.append(ast.dump(node, include_attributes=False))
        assert nodes[0] == nodes[1]
