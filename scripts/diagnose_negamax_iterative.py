"""Supplementary cross-hint-only usefulness counters; excluded from all timing.

Distinguishes cross-depth suggestions from the unchanged same-depth TT hints
in the primary diagnostic's aggregate ordering/cutoff counters.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from scripts import benchmark_negamax_iterative as bench
from scripts.negamax_iterative_variants import (
    DIAGNOSTICS, diagnostic_source, load_variant, replace_once)


def run(directory):
    config = json.loads((directory / 'manifest.json').read_text())
    bench.check_source(config, directory)
    rows = [json.loads(l) for l in (directory / 'results.jsonl').read_text().splitlines()]
    if (directory / 'depth12-results.jsonl').exists():
        rows += [json.loads(l) for l in (directory / 'depth12-results.jsonl').read_text().splitlines()]
    fixtures = {f['id']: f for f in config['positions']}
    records = []
    sources = {}
    for variant in ('hints', 'combined'):
        source = diagnostic_source(variant)
        source = replace_once(source, '    hint = None', '    hint = None\n    cross_hint = None')
        source = replace_once(source, '            hint = suggestion',
                              '            hint = suggestion\n            cross_hint = suggestion')
        source = replace_once(source, '    hinted_first = hint is not None and moves[0] == hint',
            '    hinted_first = hint is not None and moves[0] == hint\n'
            '    cross_first = cross_hint is not None and moves[0] == cross_hint\n'
            '    if cross_first:\n'
            '        table.diagnostic["cross_first"] += 1\n'
            '        if moves[0] != state.ordered_moves(None, tactical if depth > 1 else "none")[0]:\n'
            '            table.diagnostic["cross_changed_first"] += 1')
        source = replace_once(source, '                table.cutoffs += 1',
            '                table.cutoffs += 1\n'
            '                table.diagnostic["cross_first_cutoffs"] += int(index == 0 and cross_first)')
        sources[variant] = bench.digest(source.encode())
        with (directory / f'{variant}-cross-diagnostic.py').open('x') as stream:
            stream.write(source)
        module = load_variant(variant)
        exec(compile(source, f'{variant}:cross-only-diagnostic', 'exec'), module.__dict__)
        bench.warm(module)
        factory = module.SearchTable
        for row in rows:
            metrics = []

            def create():
                table = factory()
                table.diagnostic = {name: 0 for name in (*DIAGNOSTICS, 'cross_first',
                                    'cross_changed_first', 'cross_first_cutoffs')}
                metrics.append(table.diagnostic)
                return table

            with patch.object(module, 'SearchTable', create):
                result = bench.decision(module, bench.position(fixtures[row['position']]['history']), row['depth'])
            expected = row['variants'][variant]['samples'][0]
            bench.assert_same(expected, result, True)
            totals = {name: sum(d[name] for d in metrics) for name in metrics[0]}
            primary = row['variants'][variant]['counted']['diagnostic_totals']
            assert all(totals[k] == primary[k] for k in DIAGNOSTICS)
            assert totals['cross_changed_first'] <= totals['cross_first'] <= totals['legal_hints']
            assert totals['cross_first_cutoffs'] <= totals['cross_first']
            records.append(dict(position=row['position'], depth=row['depth'], variant=variant,
                                iterations=metrics, totals=totals))
    bench.write_new(directory / 'cross-hint-effectiveness.json',
        dict(records=records, generated_source_hashes=sources,
             script_sha256=bench.digest(Path(__file__).read_bytes()),
             method='Post-primary instrumentation only; full self-counter and vector parity; no timed inference'))
    return dict(verified_decisions=len(records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=bench.ROOT)
    print(json.dumps(run(parser.parse_args().directory), indent=2))
