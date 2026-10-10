"""Separate, untimed TT probe rates and index distribution; never acceptance timings."""
from collections import Counter
import json
from unittest.mock import patch

from scripts import benchmark_negamax_tt as bench
from scripts.benchmark_negamax_tt_mixed import load_mixed_variant, MULTIPLIER, WORD_MASK
from scripts.negamax_tt_variants import load_variant


def diagnose(module, game, depth):
    leaves = terminals = 0
    heuristic, terminal = module.SearchState.heuristic, module.SearchState.terminal_value

    def count_leaf(state):
        nonlocal leaves
        leaves += 1
        return heuristic(state)

    def count_terminal(state, remaining):
        nonlocal terminals
        value = terminal(state, remaining)
        terminals += value is not None
        return value

    with bench.capture_table(module) as held, patch.object(module.SearchState, 'heuristic', count_leaf), \
            patch.object(module.SearchState, 'terminal_value', count_terminal):
        result = bench.decision(module, game, depth)
    probes = result['stats']['nodes'] - leaves - terminals
    assert 0 <= result['stats']['hits'] <= probes
    result.update(leaf_evaluations=leaves, terminal_nodes=terminals, tt_probes=probes,
                  hit_per_probe=result['stats']['hits'] / probes if probes else 0)
    return result, held[0]


def run(root=bench.ROOT):
    config = json.loads((root / 'manifest.json').read_text())
    fixtures = {p['id']: p for p in config['positions']}
    selected = json.loads((root / 'A-analysis.json').read_text())['selected'] or 'baseline'
    histograms = []
    for directory, experiment, mixed in [(root, 'A', False), (root, 'B', False),
                                          (root / 'mixed-index', 'B', True)]:
        bench.check_source(json.loads((directory / 'manifest.json').read_text()), directory)
        rows = [json.loads(line) for line in (directory / f'{experiment}-results.jsonl').read_text().splitlines()]
        names = list(rows[0]['variants'])
        specs = {n: (n, None) for n in names} if experiment == 'A' else {
            n: (selected, None if n == 'unbounded' else int(n)) for n in names}
        modules = {n: (load_mixed_variant if mixed else load_variant)(*spec) for n, spec in specs.items()}
        for module in modules.values():
            bench.warm(module)
        with (directory / f'{experiment}-probe-rates.jsonl').open('x') as stream:
            for row in rows:
                results = {}
                for name, module in modules.items():
                    actual, table = diagnose(module, bench.position(fixtures[row['position']]['history']), row['depth'])
                    expected = row['variants'][name]['counted']
                    bench.assert_same(expected, actual)
                    assert expected['leaf_evaluations'] == actual['leaf_evaluations']
                    results[name] = actual
                    if experiment == 'B' and not mixed and name == 'unbounded' and row['depth'] == 10:
                        for capacity in config['capacities']:
                            keys = [key for key, value in table.entries.items()]
                            for label in ('modulo', 'mixed'):
                                indices = Counter((hash(key) % capacity if label == 'modulo' else
                                    ((hash(key) * MULTIPLIER) & WORD_MASK) >> (65 - capacity.bit_length()))
                                    for key in keys)
                                histograms.append(dict(position=row['position'], depth=row['depth'],
                                    capacity=capacity, indexing=label, population_entries=len(keys),
                                    reachable_slots=len(indices), maximum_identities_per_slot=max(indices.values())))
                stream.write(json.dumps(dict(position=row['position'], depth=row['depth'], variants=results)) + '\n')
                stream.flush()
    bench.write_new(root / 'index-distribution.json', histograms)


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=bench.ROOT)
    run(parser.parse_args().directory)
