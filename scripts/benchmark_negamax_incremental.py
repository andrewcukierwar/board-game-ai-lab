"""Pinned-baseline, evaluation-only ablations with fail-closed parity checks.

Declare into a new directory, then run. Existing evidence is never overwritten.
No search implementation is copied: the baseline is executed from Git's object
database and candidates use the ordinary production search body.
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
import statistics
import subprocess
import sys
import time
import tracemalloc
import types
from unittest.mock import patch

from games.connect4.agents import negamax_agent as candidate
from scripts.benchmark_negamax_ordering import table_bytes
from scripts.benchmark_public_agents import position
from scripts.negamax_evaluation_variants import VARIANTS
from scripts.pinned_git_source import pinned_source

BASELINE = '93ee943d8744651064dcf9a3a79108d6ca73af53'
AGENT_PATH = 'games/connect4/agents/negamax_agent.py'
EVIDENCE = 'docs/search-negamax-v2/deeper-search/'
HASH_PATHS = (AGENT_PATH, 'games/connect4/connect4.py', 'games/connect4/board.py',
              'scripts/benchmark_negamax_incremental.py',
              'scripts/negamax_evaluation_variants.py',
              'scripts/benchmark_negamax_ordering.py', 'scripts/benchmark_public_agents.py',
              'tests/test_connect4_incremental_evaluation.py',
              'tests/test_negamax_incremental_benchmark.py')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git_source(path):
    return pinned_source(BASELINE, path)


def load_baseline():
    source = git_source(AGENT_PATH)
    module = types.ModuleType('negamax_incremental_baseline')
    exec(compile(source, f'{BASELINE}:{AGENT_PATH}', 'exec'), module.__dict__)
    return module


def write_new(path, data):
    with path.open('x') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')


def declare(directory, samples=7):
    directory.mkdir(parents=True, exist_ok=True)
    fixtures = []
    sources = {}
    for name in ('candidate.json', 'profiling-supplement/candidate.json'):
        source = git_source(EVIDENCE + name)
        sources[name] = digest(source)
        fixtures += [dict(id=f['id'], history=f['history'], group='phase3a')
                     for f in json.loads(source)['profile_positions']]
    rng = Random(20261009)
    for sample in range(8):
        plies = (8, 12, 16, 20)[sample % 4]
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
        fixtures.append(dict(id=f'seeded-{sample:02d}', history=history, group='seeded'))
    fixtures.append(dict(id='post-hoc-tail', history=[1, 4, 6, 0, 6], group='post-hoc-diagnostic'))
    hardware = []
    if platform.system() == 'Darwin':
        info = subprocess.check_output(['system_profiler', 'SPHardwareDataType'], text=True)
        hardware = [line.strip() for line in info.splitlines() if any(label in line for label in (
            'Model Name:', 'Model Identifier:', 'Chip:', 'Total Number of Cores:', 'Memory:'))]
    design = directory / 'DESIGN.md'
    result = dict(baseline_commit=BASELINE, baseline_sha256=digest(git_source(AGENT_PATH)),
                  source_hashes={path: digest(Path(path).read_bytes()) for path in HASH_PATHS},
                  fixture_source_hashes=sources, design_sha256=digest(design.read_bytes()),
                  python=sys.version, executable=sys.executable, platform=platform.platform(),
                  machine=platform.machine(), hardware=hardware, cpu_count=os.cpu_count(),
                  load_average=os.getloadavg(), seed=20261009, positions=fixtures,
                  variants=['baseline', *VARIANTS], depths=[4, 6, 8, 10],
                  samples=samples, warmups=1, memory_repetitions=2,
                  table_constructor_warmups=64,
                  micro_calls=20_000, micro_cycles=2_000,
                  method='Rotating variant order; timings of unwrapped ordinary choose_move; '
                         'fresh TT; separate leaf counting and twice-repeated traced TT retention; '
                         'equal table-constructor warm-up stabilizes CPython split dictionaries; '
                         'all results and counters checked against immutable baseline; '
                         'post-hoc tail excluded from acceptance.')
    write_new(directory / 'manifest.json', result)
    return result


def check_source(config, directory):
    assert config['baseline_commit'] == BASELINE
    assert digest(git_source(AGENT_PATH)) == config['baseline_sha256']
    assert sys.version == config['python']
    for path, expected in config['source_hashes'].items():
        assert digest(Path(path).read_bytes()) == expected, f'Source drift: {path}'
    for path, expected in config['fixture_source_hashes'].items():
        assert digest(git_source(EVIDENCE + path)) == expected
    assert digest((directory / 'DESIGN.md').read_bytes()) == config['design_sha256']


@contextmanager
def variant_module(baseline, variant):
    if variant == 'baseline':
        yield baseline
    else:
        with patch.object(candidate, 'SearchState', VARIANTS[variant]):
            yield candidate


def decision(module, game, depth):
    agent = module.NegamaxAgent(depth)
    wall, cpu = time.perf_counter(), time.process_time()
    move = agent.choose_move(game)
    result = dict(wall_seconds=time.perf_counter() - wall,
                  cpu_seconds=time.process_time() - cpu, move=move,
                  scores=list(agent.last_scores.items()), stats=agent.last_stats)
    return result


def assert_same(a, b):
    for key in ('move', 'scores', 'stats'):
        assert a[key] == b[key], (key, a[key], b[key])


def counted_run(module, game, depth):
    leaves = 0
    original = module.SearchState.heuristic

    def count(state):
        nonlocal leaves
        leaves += 1
        return original(state)

    with patch.object(module.SearchState, 'heuristic', count):
        result = decision(module, game, depth)
    result['leaf_evaluations'] = leaves
    return result


def memory_run(module, game, depth):
    held, factory = [], module.SearchTable

    def capture():
        table = factory()
        held.append(table)
        return table

    gc.collect()
    tracemalloc.start()
    try:
        with patch.object(module, 'SearchTable', capture):
            result = decision(module, game, depth)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    result.update(retained_bytes=current, peak_bytes=peak, tt_bytes=table_bytes(held[0]),
                  table_attributes_bytes=sys.getsizeof(vars(held[0])))
    return result


def micro(module, game, config):
    state = module.SearchState(game)
    expected = state.heuristic()
    start, cpu = time.perf_counter(), time.process_time()
    checksum = 0
    for _ in range(config['micro_calls']):
        checksum += state.heuristic()
    calls = dict(wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu,
                 checksum=checksum)
    assert checksum == expected * config['micro_calls']
    col = state.legal()[0]
    start, cpu = time.perf_counter(), time.process_time()
    checksum = 0
    for _ in range(config['micro_cycles']):
        state.play(col)
        checksum += state.heuristic()
        state.undo(col)
    cycles = dict(wall_seconds=time.perf_counter() - start,
                  cpu_seconds=time.process_time() - cpu, checksum=checksum)
    assert state.heuristic() == expected
    return dict(calls=calls, cycles=cycles)


def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def analyze(rows):
    non_tail = [r for r in rows if r['group'] != 'post-hoc-diagnostic']
    comparisons = {}
    for variant in VARIANTS:
        speedups, broad, peak, slowdowns = [], [], [], []
        for row in non_tail:
            base, new = row['variants']['baseline'], row['variants'][variant]
            speedup = base['median_wall_seconds'] / new['median_wall_seconds']
            speedups.append(speedup)
            if (row['depth'] in (6, 8, 10) and base['counted']['stats']['nodes'] >= 2000
                    and base['counted']['leaf_evaluations'] / base['counted']['stats']['nodes'] >= .35):
                broad.append(speedup)
            old_peak = max(m['peak_bytes'] for m in base['memory'])
            new_peak = max(m['peak_bytes'] for m in new['memory'])
            peak.append(new_peak <= old_peak * 1.05 or new_peak - old_peak <= 32768)
            slowdowns.append(new['median_wall_seconds'] <= base['median_wall_seconds'] * 1.5
                             or new['median_wall_seconds'] - base['median_wall_seconds'] <= .00025)
        sum_ratio = sum(r['variants'][variant]['median_wall_seconds'] for r in non_tail) / sum(
            r['variants']['baseline']['median_wall_seconds'] for r in non_tail)
        tt_equal = all(r['variants'][variant]['memory'][0]['tt_bytes'] ==
                       r['variants']['baseline']['memory'][0]['tt_bytes'] for r in non_tail)
        checks = dict(broad_speedup=geometric(broad) >= 1.15,
                      broad_improved_fraction=sum(s > 1 for s in broad) / len(broad) >= .8,
                      distribution_speedup=geometric(speedups) >= 1.05,
                      workload_reduction=sum_ratio <= .9, per_condition_overhead=all(slowdowns),
                      peak_memory=all(peak), tt_equal=tt_equal)
        comparisons[variant] = dict(eligible=all(checks.values()), checks=checks,
            broad_conditions=len(broad), broad_geometric_speedup=geometric(broad),
            broad_improved_fraction=sum(s > 1 for s in broad) / len(broad),
            geometric_speedup=geometric(speedups), summed_median_ratio=sum_ratio,
            depth_geometric_speedups={str(d): geometric(
                r['variants']['baseline']['median_wall_seconds'] /
                r['variants'][variant]['median_wall_seconds'] for r in non_tail if r['depth'] == d)
                for d in (4, 6, 8, 10)})
    eligible = [v for v, c in comparisons.items() if c['eligible']]
    return dict(comparisons=comparisons,
                selected=max(eligible, key=lambda v: comparisons[v]['geometric_speedup']) if eligible else None,
                correctness='Every run asserted exact ordered scores, moves and counters; leaf counts matched')


def run(directory):
    config = json.loads((directory / 'manifest.json').read_text())
    check_source(config, directory)
    output = directory / 'results.jsonl'
    baseline, rows = load_baseline(), []
    for module in (baseline, candidate):
        for _ in range(config['table_constructor_warmups']):
            module.SearchTable()
    # Opening exclusively prevents accidental clobbering or concurrent writers.
    with output.open('x') as stream:
        for fixture in config['positions']:
            game = position(fixture['history'])
            assert not game.is_game_over()
            for depth in config['depths']:
                variants = {v: dict(samples=[]) for v in config['variants']}
                expected = None
                for repetition in range(config['samples'] + config['warmups']):
                    offset = repetition % len(variants)
                    order = config['variants'][offset:] + config['variants'][:offset]
                    for variant in order:
                        with variant_module(baseline, variant) as module:
                            result = decision(module, game, depth)
                        if expected is None:
                            expected = result
                        assert_same(expected, result)
                        if repetition >= config['warmups']:
                            variants[variant]['samples'].append(result)
                for variant in variants:
                    with variant_module(baseline, variant) as module:
                        counted = counted_run(module, game, depth)
                        assert_same(expected, counted)
                        variants[variant]['counted'] = counted
                        memory = []
                        for _ in range(config['memory_repetitions']):
                            result = memory_run(module, game, depth)
                            assert_same(expected, result)
                            memory.append(result)
                        variants[variant]['memory'] = memory
                    samples = variants[variant]['samples']
                    variants[variant].update(
                        median_wall_seconds=statistics.median(s['wall_seconds'] for s in samples),
                        median_cpu_seconds=statistics.median(s['cpu_seconds'] for s in samples))
                leaves = {v['counted']['leaf_evaluations'] for v in variants.values()}
                assert len(leaves) == 1, 'Unexpected leaf count change'
                row = dict(position=fixture['id'], group=fixture['group'], depth=depth, variants=variants)
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(f'{fixture["id"]} d{depth}: ' + ', '.join(
                    f'{v}={data["median_wall_seconds"] * 1000:.3f}ms' for v, data in variants.items()), flush=True)
    micro_rows = []
    for fixture in config['positions']:
        game = position(fixture['history'])
        variants = {v: [] for v in config['variants']}
        for repetition in range(config['samples'] + config['warmups']):
            offset = repetition % len(variants)
            for variant in config['variants'][offset:] + config['variants'][:offset]:
                with variant_module(baseline, variant) as module:
                    result = micro(module, game, config)
                if repetition >= config['warmups']:
                    variants[variant].append(result)
        for kind in ('calls', 'cycles'):
            assert len({s[kind]['checksum'] for data in variants.values() for s in data}) == 1
        micro_rows.append(dict(position=fixture['id'], variants=variants))
    write_new(directory / 'microbenchmark.json', dict(rows=micro_rows))
    write_new(directory / 'analysis.json', analyze(rows))
    # Reachable lookup size, outside traced decision allocations. Reuse the
    # unique-object accounting used for TT storage.
    holder = types.SimpleNamespace(lookups=(candidate.CELL_WINDOWS,
                                           candidate.WINDOW_SCORES, candidate.PLAY_DELTAS))
    write_new(directory / 'runtime-end.json', dict(load_average=os.getloadavg(),
        evaluator_lookup_bytes=table_bytes(holder), memory_note='TT retained diagnostically; '
        'production releases it. Peak is traced Python allocation, not RSS. Lookup '
        'storage allocated at module import is reported separately.'))
    print(json.dumps(analyze(rows), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['declare', 'run'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=7)
    args = parser.parse_args()
    if args.samples < 5:
        parser.error('Use at least five timed samples')
    if args.command == 'declare':
        declare(args.directory, args.samples)
    else:
        run(args.directory)


if __name__ == '__main__':
    main()
