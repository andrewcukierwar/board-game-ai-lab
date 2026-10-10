"""Check all saved decisions, coverage, frozen source and independent root oracles."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

from scripts import benchmark_negamax_iterative as bench
from scripts.negamax_iterative_variants import VARIANTS, git_source, load_variant, schedule
from tests.test_connect4_negamax import DRAW, position, root_oracle


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(directory):
    start = time.perf_counter()
    config = json.loads((directory / 'manifest.json').read_text())
    bench.check_source(config, directory)
    rows = read_rows(directory / 'results.jsonl')
    identities = [(r['position'], r['depth']) for r in rows]
    expected = {(f['id'], d) for f in config['positions'] for d in config['depths']}
    assert len(identities) == len(set(identities)) and set(identities) == expected
    assert bench.analyze(rows) == json.loads((directory / 'analysis.json').read_text())
    assert Path(bench.AGENT_PATH).read_bytes() == git_source(), 'Production source differs from baseline'
    # Reuse prior frozen decisions, independently of this phase's generated code.
    old = {(r['position'], r['depth']): r['variants']['packed-entry']
           for r in read_rows_from_bytes(git_source('docs/search-negamax-v2/transposition-table/A-results.jsonl'))}
    prior_matches, decisions, csv_rows, roots = 0, 0, [], []
    all_rows = list(rows)
    depth12_preflight = json.loads((directory / 'depth12-preflight.json').read_text())
    if (directory / 'depth12-results.jsonl').exists():
        assert depth12_preflight['passed']
        extra = read_rows(directory / 'depth12-results.jsonl')
        assert {(r['position'], r['depth']) for r in extra} == {(p, 12) for p in bench.PREFLIGHT_IDS}
        empty = next(r for r in extra if r['position'] == 'empty')
        bench.assert_same(empty['variants']['direct']['samples'][0], depth12_preflight['probe'], True)
        decisions += 1
        all_rows += extra
    else:
        assert not depth12_preflight['passed'], 'Declared feasible D12 must be completed'
    for r in all_rows:
        assert tuple(r['variants']) == VARIANTS
        base = r['variants']['direct']['samples'][0]
        if (r['position'], r['depth']) in old:
            prior = old[r['position'], r['depth']]
            for key in ('scores', 'move', 'stats'):
                assert base[key] == prior['samples'][0][key]
            assert r['variants']['direct']['counted']['diagnostic_totals']['leaves'] == prior['counted']['leaf_evaluations']
            prior_matches += 1
        root = dict(position=r['position'], depth=r['depth'], move=base['move'], scores=base['scores'])
        root['variant_final_stats'] = {}
        for v, data in r['variants'].items():
            assert len(data['samples']) == 7
            assert len(data['memory']) == (1 if r['depth'] == 12 else 2)
            reference = data['samples'][0]
            assert [i['depth'] for i in reference['iterations']] == list(schedule(r['depth'], v))
            assert data['median_wall_seconds'] == statistics.median(s['wall_seconds'] for s in data['samples'])
            assert data['median_cpu_seconds'] == statistics.median(s['cpu_seconds'] for s in data['samples'])
            for s in data['samples'] + data['memory'] + [data['counted']]:
                bench.assert_same(base, s)
                bench.assert_same(reference, s, True)
                assert s['wall_seconds'] > 0 and s['cpu_seconds'] > 0
                assert all(s['stats'][k] == sum(i[k] for i in s['iterations'])
                           for k in ('nodes', 'entries', 'hits', 'cutoffs'))
                decisions += 1
            final = reference['iterations'][-1]
            root['variant_final_stats'][v] = final
            if v == 'iterative':
                for k in ('nodes', 'entries', 'hits', 'cutoffs'):
                    assert final[k] == base['stats'][k]
            counted = data['counted']
            assert len(counted['diagnostics']) == len(reference['iterations'])
            assert counted['diagnostic_totals']['invalid_hints'] == 0
            assert all(counted['diagnostic_totals'][k] == sum(d[k] for d in counted['diagnostics'])
                       for k in counted['diagnostic_totals'])
            if v in ('direct', 'iterative'):
                assert counted['diagnostic_totals']['hint_lookups'] == 0
            assert data['memory'][0]['retained'] == data['memory'][-1]['retained']
            memory = data['memory'][0]
            csv_rows.append(dict(position=r['position'], group=r['group'], depth=r['depth'], variant=v,
                wall_ms=data['median_wall_seconds'] * 1000, cpu_ms=data['median_cpu_seconds'] * 1000,
                wall_speedup=r['variants']['direct']['median_wall_seconds'] / data['median_wall_seconds'],
                total_nodes=reference['stats']['nodes'], final_nodes=final['nodes'],
                total_entries_at_iteration_end=reference['stats']['entries'], final_entries=final['entries'],
                total_hits=reference['stats']['hits'], final_hits=final['hits'],
                total_cutoffs=reference['stats']['cutoffs'], final_cutoffs=final['cutoffs'],
                final_leaves=counted['diagnostics'][-1]['leaves'], **counted['diagnostic_totals'],
                retained_bytes=memory['retained']['total_bytes'],
                score_tt_bytes=memory['retained']['categories']['score_tt'],
                hint_bytes=memory['retained']['categories'].get('hint_cache', 0),
                hint_entries=memory['retained']['hint_entries'],
                peak_bytes=max(m['traced_peak_bytes'] for m in data['memory']),
                move=base['move'], scores=json.dumps(base['scores']),
                minimum_wall_ms=min(s['wall_seconds'] for s in data['samples']) * 1000,
                maximum_wall_ms=max(s['wall_seconds'] for s in data['samples']) * 1000))
        roots.append(root)
    assert prior_matches == 96
    preflight = json.loads((directory / 'preflight.json').read_text())
    assert preflight['passed'] and len(preflight['rows']) == 8
    for r in preflight['rows']:
        normal = r['variants']['direct']['normal']
        for data in r['variants'].values():
            bench.assert_same(normal, data['normal'])
            bench.assert_same(data['normal'], data['memory'], True)
            decisions += 2
    rss = json.loads((directory / 'rss.json').read_text())
    assert len(rss) == 16
    references = {(r['position'], r['depth']): r for r in rows}
    for r in rss:
        expected = references[r['position'], r['depth']]['variants'][r['variant']]['samples'][0]
        bench.assert_same(expected, r['result'], True)
        decisions += 1
    cross = json.loads((directory / 'cross-hint-effectiveness.json').read_text())
    assert cross['script_sha256'] == bench.digest(Path('scripts/diagnose_negamax_iterative.py').read_bytes())
    for v, expected_hash in cross['generated_source_hashes'].items():
        assert bench.digest((directory / f'{v}-cross-diagnostic.py').read_bytes()) == expected_hash
    expected_cross = {(r['position'], r['depth'], v) for r in all_rows for v in ('hints', 'combined')}
    cross_ids = [(r['position'], r['depth'], r['variant']) for r in cross['records']]
    assert len(cross_ids) == len(set(cross_ids)) and set(cross_ids) == expected_cross
    all_references = {(r['position'], r['depth']): r for r in all_rows}
    for r in cross['records']:
        primary = all_references[r['position'], r['depth']]['variants'][r['variant']]['counted']
        assert all(r['totals'][k] == value for k, value in primary['diagnostic_totals'].items())
        assert r['totals']['cross_changed_first'] <= r['totals']['cross_first'] <= r['totals']['legal_hints']
        assert r['totals']['cross_first_cutoffs'] <= r['totals']['cross_first']
    oracle_vectors = []
    for f in config['positions']:
        game = position(f['history'])
        for depth in (1, 2, 3):
            expected = [list(item) for item in root_oracle(game, depth).items()]
            for v in VARIANTS:
                result = bench.decision(load_variant(v), game, depth)
                assert result['scores'] == expected
                assert result['move'] == max(dict(expected), key=dict(expected).get)
            oracle_vectors.append(dict(position=f['id'], depth=depth, scores=expected))
    for prefix in (36, 38, 41):
        for depth in (4, 6, 8, 10, 12):
            game = position(DRAW[:prefix])
            expected = [list(item) for item in root_oracle(game, depth).items()]
            for v in VARIANTS:
                assert [list(item) for item in load_variant(v).NegamaxAgent(depth).score_moves(game).items()] == expected
            oracle_vectors.append(dict(position=f'draw-prefix-{prefix}', depth=depth, scores=expected))
    with (directory / 'performance.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(csv_rows)
    bench.write_new(directory / 'root-parity.json', roots)
    bench.write_new(directory / 'oracle-vectors.json', oracle_vectors)
    paths = ['DESIGN.md', 'manifest.json', 'preflight.json', 'results.jsonl', 'analysis.json',
             'rss.json', 'runtime-end.json', 'performance.csv', 'root-parity.json', 'oracle-vectors.json',
             'depth12-preflight.json', 'cross-hint-effectiveness.json',
             'hints-cross-diagnostic.py', 'combined-cross-diagnostic.py']
    if (directory / 'depth12-results.jsonl').exists():
        paths += ['depth12-results.jsonl', 'depth12-runtime.json']
    result = dict(conditions=len(rows), extra_depth12_conditions=len(all_rows)-len(rows),
                  verified_saved_decisions=decisions, prior_score_move_counter_leaf_matches=prior_matches,
                  independent_oracle_vectors=len(oracle_vectors),
                  independent_oracle_variant_decisions=len(oracle_vectors) * 4,
                  supplementary_cross_hint_decisions=len(cross['records']),
                  production_source_byte_identical_to_baseline=True,
                  no_invalid_hints=True, all_ordered_final_scores_and_moves_identical=True,
                  within_variant_counters_deterministic=True,
                  iterative_without_hints_final_counters_equal_direct=True,
                  elapsed_seconds=time.perf_counter()-start,
                  auditor_sha256=bench.digest(Path(__file__).read_bytes()),
                  evidence_hashes={p: bench.digest((directory / p).read_bytes()) for p in paths})
    bench.write_new(directory / 'audit.json', result)
    return result


def read_rows_from_bytes(data):
    return [json.loads(line) for line in data.splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=bench.ROOT)
    print(json.dumps(audit(parser.parse_args().directory), indent=2))


if __name__ == '__main__':
    main()
