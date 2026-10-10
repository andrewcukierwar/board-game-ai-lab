"""Predeclared, fail-closed TT-only experiments. Run A fully before B.

All outputs use exclusive creation. Timings measure whole unwrapped decisions;
leaf counts, retention, tracemalloc, RSS and microbenchmarks are separate.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from random import Random
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from unittest.mock import patch

from games.connect4.agents.negamax_tt import DirectMappedEntries, pack_key, unpack_key
from scripts.benchmark_public_agents import position
from scripts.negamax_tt_variants import BASELINE, AGENT_PATH, VARIANTS, git_source, load_variant, variant_source

ROOT = Path('docs/search-negamax-v2/transposition-table')
HASH_PATHS = ('scripts/benchmark_negamax_tt.py', 'scripts/negamax_tt_variants.py',
              'games/connect4/agents/negamax_tt.py', 'games/connect4/connect4.py',
              'games/connect4/board.py', 'scripts/benchmark_public_agents.py')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    with path.open('x') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')


def declare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    fixture_source = git_source('docs/search-negamax-v2/incremental-evaluation/manifest.json')
    fixtures = json.loads(fixture_source)['positions']
    rng = Random(20261010)
    for sample, plies in enumerate((10, 14, 18, 22)):
        while True:
            game, history = position([]), []
            for _ in range(plies):
                col = rng.choice(game.get_valid_moves())
                assert game.make_move(col)
                history.append(col)
                if game.is_game_over():
                    break
            if not game.is_game_over():
                break
        fixtures.append(dict(id=f'additional-{sample:02d}', history=history, group='additional'))
    config = dict(baseline_commit=BASELINE, baseline_sha256=digest(git_source()),
        fixture_source_sha256=digest(fixture_source), positions=fixtures,
        source_hashes={p: digest(Path(p).read_bytes()) for p in HASH_PATHS},
        design_sha256=digest((directory / 'DESIGN.md').read_bytes()),
        variants=list(VARIANTS), capacities=[16384, 32768, 65536], depths=[4, 6, 8, 10],
        samples=7, warmups=1, memory_repetitions=2, table_constructor_warmups=64,
        micro_calls=20000, python=sys.version, executable=sys.executable,
        platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count(),
        load_average=os.getloadavg(), seed=20261010)
    write_new(directory / 'manifest.json', config)
    for variant in VARIANTS:
        with (directory / f'{variant}-source.py').open('x') as stream:
            stream.write(variant_source(variant))
    return config


def check_source(config, directory):
    assert config['baseline_commit'] == BASELINE
    assert config['baseline_sha256'] == digest(git_source())
    assert config['fixture_source_sha256'] == digest(git_source(
        'docs/search-negamax-v2/incremental-evaluation/manifest.json'))
    assert config['python'] == sys.version
    assert config['design_sha256'] == digest((directory / 'DESIGN.md').read_bytes())
    for path, expected in config['source_hashes'].items():
        assert digest(Path(path).read_bytes()) == expected, f'Source drift: {path}'
    for variant in VARIANTS:
        assert (directory / f'{variant}-source.py').read_text() == variant_source(variant)


def warm(module, count=64):
    for _ in range(count):
        module.SearchTable()


def decision(module, game, depth):
    agent = module.NegamaxAgent(depth)
    wall, cpu = time.perf_counter(), time.process_time()
    move = agent.choose_move(game)
    return dict(wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu,
                move=move, scores=[list(item) for item in agent.last_scores.items()], stats=agent.last_stats)


def assert_same(a, b, counters=True):
    for key in ('move', 'scores', 'stats') if counters else ('move', 'scores'):
        assert a[key] == b[key], (key, a[key], b[key])


@contextmanager
def capture_table(module):
    held, constructor = [], module.SearchTable

    def factory():
        table = constructor()
        held.append(table)
        return table

    with patch.object(module, 'SearchTable', factory):
        yield held


def storage_stats(table):
    entries = table.entries
    if isinstance(entries, DirectMappedEntries):
        return dict(capacity=entries.capacity, occupancy=len(entries),
                    evictions=entries.evictions, replacements=entries.replacements)
    return dict(capacity=None, occupancy=len(entries), evictions=0, replacements=None)


def counted_run(module, game, depth):
    leaves, original = 0, module.SearchState.heuristic

    def count(state):
        nonlocal leaves
        leaves += 1
        return original(state)

    with capture_table(module) as held, patch.object(module.SearchState, 'heuristic', count):
        result = decision(module, game, depth)
    result.update(leaf_evaluations=leaves, storage=storage_stats(held[0]))
    return result


def memory_breakdown(table):
    """Logical instance graph, unique identities; no module/class referents.

    Attribution order: container, keys, entries, table attributes. Small cached
    integers/shared strings count once, in the first applicable category.
    """
    seen, totals = set(), {}

    def add(obj, category):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        totals[category] = totals.get(category, 0) + sys.getsizeof(obj)

    entries = table.entries
    if isinstance(entries, dict):
        add(entries, 'dictionary_structure_and_capacity')
    else:
        add(entries, 'bounded_storage_object')
        add(entries.keys, 'bounded_slot_lists')
        add(entries.values, 'bounded_slot_lists')
        for name in ('capacity', 'occupancy', 'evictions', 'replacements'):
            add(getattr(entries, name), 'bounded_counters')
        add(None, 'hints_flags_shared')
    for key, value in entries.items():
        if isinstance(key, tuple):
            add(key, 'key_tuples')
            for board in key[:2]:
                add(board, 'key_bitboard_integers')
            for field in key[2:]:
                add(field, 'key_mover_depth_integers')
        else:
            add(key, 'packed_key_integers')
        if isinstance(value, tuple):
            add(value, 'value_tuples')
            add(value[1], 'stored_scores')
            add(value[0], 'hints_flags_shared')
            add(value[2], 'hints_flags_shared')
        else:
            add(value, 'packed_entry_integers')
    add(table, 'search_table_object')
    add(vars(table), 'search_table_attributes')
    for key, value in vars(table).items():
        add(key, 'attribute_names')
        if key != 'entries':
            add(value, 'surrounding_settings_counters')
    return dict(total_bytes=sum(totals.values()), categories=totals,
                attribute_dictionary_bytes=sys.getsizeof(vars(table)), entries=len(entries))


def memory_run(module, game, depth):
    gc.collect()
    tracemalloc.start()
    try:
        with capture_table(module) as held:
            result = decision(module, game, depth)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    result.update(traced_current_bytes=current, traced_peak_bytes=peak,
                  retained=memory_breakdown(held[0]), storage=storage_stats(held[0]))
    return result


def rss_bytes():
    return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())])) * 1024


def rss_worker(variant, capacity, history, depth):
    module = load_variant(variant, capacity)
    warm(module)
    game = position(history)
    before = rss_bytes()
    with capture_table(module) as held:
        result = decision(module, game, depth)
    after = rss_bytes()
    high = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != 'Darwin':
        high *= 1024
    result.update(rss_before_bytes=before, rss_retained_bytes=after,
                  high_water_bytes=high, storage=storage_stats(held[0]))
    return result


def fresh_rss(variant, capacity, history, depth):
    output = subprocess.check_output([sys.executable, '-m', 'scripts.benchmark_negamax_tt',
        'rss-worker', '--variant', variant, '--capacity', str(capacity or 0),
        '--history', json.dumps(history), '--depth', str(depth)])
    return json.loads(output)


def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def analyze(rows, experiment):
    reference = 'baseline' if experiment == 'A' else 'unbounded'
    primary = [r for r in rows if r['group'] != 'post-hoc-diagnostic']
    comparisons = {}
    for name in rows[0]['variants']:
        if name == reference:
            continue
        pairs = [(r, r['variants'][reference], r['variants'][name]) for r in primary]
        ratios = [b['median_wall_seconds'] / a['median_wall_seconds'] for _, a, b in pairs]
        summed_ratio = sum(b['median_wall_seconds'] for _, a, b in pairs) / sum(
            a['median_wall_seconds'] for _, a, b in pairs)
        large = [(a, b) for _, a, b in pairs if a['counted']['stats']['entries'] >=
                 (2000 if experiment == 'A' else 16385)]
        memory_ratio = sum(b['memory'][0]['retained']['total_bytes'] for a, b in large) / sum(
            a['memory'][0]['retained']['total_bytes'] for a, b in large)
        limit = 1.08 if experiment == 'A' else 1.05
        checks = dict(memory=memory_ratio <= .7, geometric_latency=geometric(ratios) <= limit,
            summed_latency=summed_ratio <= limit,
            per_condition=all(b['median_wall_seconds'] <= a['median_wall_seconds'] * 1.2 or
                b['median_wall_seconds'] - a['median_wall_seconds'] <= .00025 for _, a, b in pairs),
            peak=all(max(m['traced_peak_bytes'] for m in b['memory']) <=
                max(m['traced_peak_bytes'] for m in a['memory']) * 1.05 or
                max(m['traced_peak_bytes'] for m in b['memory']) -
                max(m['traced_peak_bytes'] for m in a['memory']) <= 32768 for _, a, b in pairs))
        if experiment == 'B':
            costly = [(r['variants'][reference], r['variants'][name]) for r in rows
                      if r['variants'][reference]['median_wall_seconds'] >= .05]
            checks['expensive_latency'] = all(b['median_wall_seconds'] <=
                a['median_wall_seconds'] * 1.2 for a, b in costly)
            checks['expensive_nodes'] = all(b['counted']['stats']['nodes'] <=
                a['counted']['stats']['nodes'] * 1.5 for a, b in costly)
        comparisons[name] = dict(eligible=all(checks.values()), checks=checks,
            geometric_wall_ratio=geometric(ratios), summed_wall_ratio=summed_ratio,
            geometric_cpu_ratio=geometric(b['median_cpu_seconds'] / a['median_cpu_seconds']
                                         for _, a, b in pairs),
            retained_large_ratio=memory_ratio, large_conditions=len(large),
            depth_wall_ratios={str(d): geometric(b['median_wall_seconds'] /
                a['median_wall_seconds'] for r, a, b in pairs if r['depth'] == d)
                for d in (4, 6, 8, 10)},
            worst_wall_ratio=max(ratios))
    eligible = [name for name, c in comparisons.items() if c['eligible']]
    selected = (min(eligible, key=lambda n: comparisons[n]['retained_large_ratio'])
                if experiment == 'A' else max(eligible, key=int)) if eligible else None
    return dict(experiment=experiment, selected=selected, comparisons=comparisons)


def micro_run(module, variant, game, depth, calls):
    with capture_table(module) as held:
        decision(module, game, depth)
    pairs = list(held[0].entries.items())
    identities = [key if isinstance(key, tuple) else unpack_key(key) for key, _ in pairs]
    if not pairs:
        return None
    start, cpu = time.perf_counter(), time.process_time()
    checksum = 0
    for i in range(calls):
        x, o, mover, remaining = identities[i % len(identities)]
        key = (x, o, mover, remaining) if variant == 'baseline' else (
            x | (o << 49) | (mover << 98) | (remaining << 99))
        checksum ^= hash(key)
    creation = dict(wall_seconds=time.perf_counter() - start,
                    cpu_seconds=time.process_time() - cpu, checksum=checksum)
    start, cpu = time.perf_counter(), time.process_time()
    for i in range(calls):
        key = pairs[i % len(pairs)][0]
        assert key in held[0].entries
        value = held[0].entries[key]
    lookup = dict(wall_seconds=time.perf_counter() - start,
                  cpu_seconds=time.process_time() - cpu)
    return dict(creation=creation, lookup=lookup, population_entries=len(pairs))


def run(directory, experiment):
    config = json.loads((directory / 'manifest.json').read_text())
    check_source(config, directory)
    if experiment == 'A':
        names = config['variants']
        specs = {name: (name, None) for name in names}
    else:
        # Existence and complete coverage of A are checked before any B work.
        a_rows = [json.loads(line) for line in (directory / 'A-results.jsonl').read_text().splitlines()]
        assert len(a_rows) == len(config['positions']) * len(config['depths'])
        a_analysis = json.loads((directory / 'A-analysis.json').read_text())
        assert a_analysis == analyze(a_rows, 'A')
        selected = a_analysis['selected'] or 'baseline'
        names = ['unbounded', *map(str, config['capacities'])]
        specs = {'unbounded': (selected, None), **{str(c): (selected, c) for c in config['capacities']}}
        write_new(directory / 'B-manifest.json', dict(representation=selected,
            A_results_sha256=digest((directory / 'A-results.jsonl').read_bytes()), specs=specs))
    modules = {name: load_variant(*specs[name]) for name in names}
    for module in modules.values():
        warm(module, config['table_constructor_warmups'])
    rows = []
    with (directory / f'{experiment}-results.jsonl').open('x') as stream:
        for fixture in config['positions']:
            game = position(fixture['history'])
            assert not game.is_game_over()
            for depth in config['depths']:
                variants = {name: dict(samples=[]) for name in names}
                expected = decision(modules[names[0]], game, depth)
                for repetition in range(config['samples'] + config['warmups']):
                    offset = repetition % len(names)
                    for name in names[offset:] + names[:offset]:
                        result = decision(modules[name], game, depth)
                        assert_same(expected, result, experiment == 'A')
                        if repetition >= config['warmups']:
                            variants[name]['samples'].append(result)
                for name, data in variants.items():
                    data['counted'] = counted_run(modules[name], game, depth)
                    assert_same(expected, data['counted'], experiment == 'A')
                    data['memory'] = [memory_run(modules[name], game, depth)
                                      for _ in range(config['memory_repetitions'])]
                    for result in data['memory']:
                        assert_same(expected, result, experiment == 'A')
                    data['rss'] = fresh_rss(*specs[name], fixture['history'], depth)
                    assert_same(expected, data['rss'], experiment == 'A')
                    data.update(median_wall_seconds=statistics.median(
                        s['wall_seconds'] for s in data['samples']),
                        median_cpu_seconds=statistics.median(s['cpu_seconds'] for s in data['samples']))
                if experiment == 'A':
                    assert len({v['counted']['leaf_evaluations'] for v in variants.values()}) == 1
                row = dict(position=fixture['id'], group=fixture['group'], depth=depth, variants=variants)
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(f'{experiment} {fixture["id"]} d{depth}: ' + ', '.join(
                    f'{n}={v["median_wall_seconds"] * 1000:.3f}ms' for n, v in variants.items()), flush=True)
    write_new(directory / f'{experiment}-analysis.json', analyze(rows, experiment))
    if experiment == 'A':
        micros = []
        for fixture in config['positions']:
            game = position(fixture['history'])
            samples = {name: [] for name in names}
            for repetition in range(config['samples'] + config['warmups']):
                offset = repetition % len(names)
                for name in names[offset:] + names[:offset]:
                    result = micro_run(modules[name], name, game, 6, config['micro_calls'])
                    if repetition >= config['warmups']:
                        samples[name].append(result)
            micros.append(dict(position=fixture['id'], variants=samples))
        write_new(directory / 'A-microbenchmark.json', micros)
    print(json.dumps(analyze(rows, experiment), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['declare', 'A', 'B', 'rss-worker'])
    parser.add_argument('--directory', type=Path, default=ROOT)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--capacity', type=int, default=0)
    parser.add_argument('--history')
    parser.add_argument('--depth', type=int)
    args = parser.parse_args()
    if args.command == 'declare':
        declare(args.directory)
    elif args.command == 'rss-worker':
        print(json.dumps(rss_worker(args.variant, args.capacity or None,
                                   json.loads(args.history), args.depth)))
    else:
        run(args.directory, args.command)


if __name__ == '__main__':
    main()
