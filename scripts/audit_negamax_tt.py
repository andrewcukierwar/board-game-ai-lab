"""Independently validate TT evidence coverage, parity and export comparisons."""
import argparse
import ast
import csv
import json
from pathlib import Path

from games.connect4.agents import negamax_agent as production
from scripts import benchmark_negamax_tt as bench
from scripts.negamax_tt_variants import AGENT_PATH, VARIANTS, git_source, variant_source


def ast_definition(source, name):
    node = next(n for n in ast.parse(source).body if getattr(n, 'name', None) == name)
    return ast.dump(node, include_attributes=False)


def dictionary_layout(size, entries):
    """CPython 3.11 general-key combined table, no deletions in experiment A.

    Infer capacity from measured __sizeof__, checking the exact byte formula.
    Local pycore_dict.h defines 32-byte keys header, 24-byte general entries
    and index widths. Headers (including GC) are 64+32 bytes on this runtime.
    Dense spare capacity is distinct from allocated sparse hash-index overhead.
    """
    import sys
    assert sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 11)
    assert sys.maxsize == (1 << 63) - 1
    for log_size in range(3, 32):
        slots = 1 << log_size
        width = 1 if slots <= 255 else 2 if slots <= 65535 else 4
        dense_capacity = slots * 2 // 3
        if 96 + slots * width + dense_capacity * 24 == size:
            assert entries <= dense_capacity
            return dict(allocated_bytes=size, headers_bytes=96,
                hash_index_bytes=slots * width, hash_index_slots=slots,
                dense_capacity=dense_capacity, occupied_dense_bytes=entries * 24,
                spare_dense_bytes=(dense_capacity - entries) * 24,
                note='Capacity inferred from measured size; no dictionary deletions')
    raise AssertionError(('Unrecognized dictionary layout', size, entries))


def audit(directory, integration=True):
    config = json.loads((directory / 'manifest.json').read_text())
    if (directory / 'mixed-manifest.json').exists():
        supplemental = json.loads((directory / 'mixed-manifest.json').read_text())
        assert supplemental['runner_sha256'] == bench.digest(Path(
            'scripts/benchmark_negamax_tt_mixed.py').read_bytes())
        assert supplemental['design_sha256'] == bench.digest((directory / 'DESIGN.md').read_bytes())
        for name in ('A_results', 'A_analysis'):
            path = directory / ('A-results.jsonl' if name == 'A_results' else 'A-analysis.json')
            assert supplemental[name + '_sha256'] == bench.digest(path.read_bytes())
    bench.check_source(config, directory)
    conditions = {(p['id'], d) for p in config['positions'] for d in config['depths']}
    prior = [json.loads(line) for line in git_source(
        'docs/search-negamax-v2/incremental-evaluation/results.jsonl').decode().splitlines()]
    old = {(r['position'], r['depth']): r['variants']['compact-stack']['counted'] for r in prior}
    all_rows, checked, prior_checked = {}, 0, 0
    for experiment in ('A', 'B'):
        rows = [json.loads(line) for line in (directory / f'{experiment}-results.jsonl').read_text().splitlines()]
        assert len(rows) == len(conditions)
        assert {(r['position'], r['depth']) for r in rows} == conditions
        analysis = json.loads((directory / f'{experiment}-analysis.json').read_text())
        assert analysis == bench.analyze(rows, experiment)
        all_rows[experiment] = rows
        for row in rows:
            reference = row['variants']['baseline' if experiment == 'A' else 'unbounded']['counted']
            if experiment == 'A' and (row['position'], row['depth']) in old:
                bench.assert_same(old[row['position'], row['depth']], reference)
                assert old[row['position'], row['depth']]['leaf_evaluations'] == reference['leaf_evaluations']
                prior_checked += 1
            if experiment == 'B':
                a = next(r for r in all_rows['A'] if (r['position'], r['depth']) ==
                         (row['position'], row['depth']))
                selected = json.loads((directory / 'A-analysis.json').read_text())['selected'] or 'baseline'
                bench.assert_same(a['variants'][selected]['counted'], reference)
            assert len(row['variants']) == (3 if experiment == 'A' else 4)
            for name, data in row['variants'].items():
                own = data['counted']
                bench.assert_same(reference, own, experiment == 'A')
                if experiment == 'A':
                    assert own['leaf_evaluations'] == reference['leaf_evaluations']
                runs = [*data['samples'], own, *data['memory'], data['rss']]
                assert len(data['samples']) == config['samples']
                assert len(data['memory']) == config['memory_repetitions']
                for run in runs:
                    bench.assert_same(own, run)
                    checked += 1
                for memory in data['memory']:
                    retained = memory['retained']
                    assert sum(retained['categories'].values()) == retained['total_bytes']
                    assert retained['entries'] == own['stats']['entries']
                assert data['memory'][0]['retained'] == data['memory'][1]['retained']
                if experiment == 'B' and name != 'unbounded':
                    assert own['storage']['capacity'] == int(name)
                    assert own['stats']['entries'] <= int(name)
                    for run in [*data['memory'], data['rss']]:
                        assert run['storage'] == own['storage']
    layouts = []
    for row in all_rows['A']:
        for name, data in row['variants'].items():
            retention = data['memory'][0]['retained']
            layouts.append(dict(position=row['position'], depth=row['depth'], variant=name,
                **dictionary_layout(retention['categories']['dictionary_structure_and_capacity'],
                                    retention['entries'])))
    bench.write_new(directory / 'dictionary-layout.json', layouts)
    for experiment in ('A', 'B'):
        path = directory / f'{experiment}-probe-rates.jsonl'
        if path.exists():
            probes = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(probes) == len(conditions)
            lookup = {(r['position'], r['depth']): r for r in all_rows[experiment]}
            for row in probes:
                original = lookup[row['position'], row['depth']]['variants']
                for name, run in row['variants'].items():
                    bench.assert_same(original[name]['counted'], run)
                    assert run['leaf_evaluations'] == original[name]['counted']['leaf_evaluations']
                    assert run['tt_probes'] + run['terminal_nodes'] + run['leaf_evaluations'] == run['stats']['nodes']
    selected = json.loads((directory / 'A-analysis.json').read_text())['selected'] or 'baseline'
    expected_source = variant_source(selected)
    actual_source = Path(AGENT_PATH).read_text()
    if integration:
        for name in ('has_four', 'winning_squares', 'SearchState', 'SearchTable', 'negamax', 'NegamaxAgent'):
            assert ast_definition(expected_source, name) == ast_definition(actual_source, name), name
        for row in all_rows['A']:
            fixture = next(p for p in config['positions'] if p['id'] == row['position'])
            actual = bench.counted_run(production, bench.position(fixture['history']), row['depth'])
            expected = row['variants'][selected]['counted']
            bench.assert_same(expected, actual)
            assert expected['leaf_evaluations'] == actual['leaf_evaluations']
    b_manifest = json.loads((directory / 'B-manifest.json').read_text())
    assert b_manifest['representation'] == selected
    assert b_manifest['A_results_sha256'] == bench.digest((directory / 'A-results.jsonl').read_bytes())
    for source in (actual_source, expected_source):
        for name in ('SearchState', 'has_four', 'winning_squares', 'NegamaxAgent'):
            assert ast_definition(source, name) == ast_definition(git_source().decode(), name)
    with (directory / 'performance.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['experiment', 'position', 'group', 'depth', 'variant', 'wall_ms', 'cpu_ms',
            'wall_ratio', 'added_ms', 'nodes', 'node_ratio', 'extra_nodes', 'entries', 'hits', 'hit_per_node',
            'cutoffs', 'leaves', 'capacity', 'evictions', 'replacements', 'reachable_tt_bytes',
            'traced_current_bytes', 'traced_peak_bytes', 'rss_before_bytes', 'rss_retained_bytes',
            'rss_high_water_bytes'])
        for experiment, rows in all_rows.items():
            for row in rows:
                ref = row['variants']['baseline' if experiment == 'A' else 'unbounded']
                for name, data in row['variants'].items():
                    counters, storage = data['counted']['stats'], data['counted']['storage']
                    writer.writerow([experiment, row['position'], row['group'], row['depth'], name,
                        data['median_wall_seconds'] * 1000, data['median_cpu_seconds'] * 1000,
                        data['median_wall_seconds'] / ref['median_wall_seconds'],
                        (data['median_wall_seconds'] - ref['median_wall_seconds']) * 1000,
                        counters['nodes'], counters['nodes'] / ref['counted']['stats']['nodes'],
                        counters['nodes'] - ref['counted']['stats']['nodes'],
                        counters['entries'], counters['hits'], counters['hits'] / counters['nodes'],
                        counters['cutoffs'], data['counted']['leaf_evaluations'], storage['capacity'],
                        storage['evictions'], storage['replacements'], data['memory'][0]['retained']['total_bytes'],
                        max(m['traced_current_bytes'] for m in data['memory']),
                        max(m['traced_peak_bytes'] for m in data['memory']),
                        data['rss']['rss_before_bytes'], data['rss']['rss_retained_bytes'],
                        data['rss']['high_water_bytes']])
    result = dict(conditions_per_experiment=len(conditions), saved_decisions_checked=checked,
        previous_phase_vectors_checked=prior_checked, selected_representation=selected,
        production_all_conditions_checked=integration, baseline_sha256=bench.digest(git_source()),
        production_sha256=bench.digest(Path(AGENT_PATH).read_bytes()),
        evidence_hashes={p.name: bench.digest(p.read_bytes()) for p in directory.iterdir()
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.py', '.csv')},
        all_checks_passed=True)
    bench.write_new(directory / 'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=bench.ROOT)
    parser.add_argument('--no-integration', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, not args.no_integration), indent=2))


if __name__ == '__main__':
    main()
