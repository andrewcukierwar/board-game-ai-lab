"""Four-way, source-frozen complete-decision experiment; exclusive evidence writes."""
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

from scripts.benchmark_public_agents import position
from scripts.negamax_iterative_variants import (
    AGENT_PATH, BASELINE, DIAGNOSTICS, VARIANTS, diagnostic_source,
    git_source, load_variant, schedule, variant_source)

ROOT = Path('docs/search-negamax-v2/iterative-deepening')
FIXTURE_PATH = 'docs/search-negamax-v2/transposition-table/manifest.json'
HASH_PATHS = (AGENT_PATH, 'games/connect4/agents/negamax_tt.py',
              'games/connect4/connect4.py', 'games/connect4/board.py',
              'scripts/negamax_iterative_variants.py', 'scripts/benchmark_negamax_iterative.py',
              'scripts/benchmark_public_agents.py', 'tests/test_connect4_negamax_iterative.py')
PREFLIGHT_IDS = ('empty', 'near-opening', 'seeded-04', 'post-hoc-tail')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def runtime():
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                machine=platform.machine(), cpus=os.cpu_count(), load=os.getloadavg(),
                unix_seconds=time.time())


def declare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    prior = git_source(FIXTURE_PATH)
    fixtures = json.loads(prior)['positions']
    identities = set()
    for f in fixtures:
        state = load_variant('direct').SearchState(position(f['history']))
        identities.add((*state.pieces, state.mover))
    rng = Random(20261011)
    for sample, plies in enumerate((6, 10, 18, 26)):
        while True:
            game, history = position([]), []
            for _ in range(plies):
                col = rng.choice(game.get_valid_moves())
                assert game.make_move(col)
                history.append(col)
                if game.is_game_over():
                    break
            state = load_variant('direct').SearchState(game)
            identity = (*state.pieces, state.mover)
            if not game.is_game_over() and identity not in identities:
                identities.add(identity)
                break
        fixtures.append(dict(id=f'generalization-{sample:02d}', history=history, group='generalization'))
    config = dict(baseline_commit=BASELINE, baseline_sha256=digest(git_source()),
                  fixture_source_sha256=digest(prior), positions=fixtures,
                  source_hashes={p: digest(Path(p).read_bytes()) for p in HASH_PATHS},
                  design_sha256=digest((directory / 'DESIGN.md').read_bytes()),
                  variants=list(VARIANTS), depths=[4, 6, 8, 10], samples=7,
                  warmups=1, memory_repetitions=2, constructor_warmups=64,
                  generalization_seed=20261011, runtime=runtime())
    if platform.system() == 'Darwin':
        hardware = subprocess.check_output(['system_profiler', 'SPHardwareDataType'], text=True)
        config['hardware'] = [l.strip() for l in hardware.splitlines() if any(s in l for s in
            ('Model Name:', 'Model Identifier:', 'Chip:', 'Total Number of Cores:', 'Memory:'))]
    write_new(directory / 'manifest.json', config)
    for variant in VARIANTS:
        for label, source in (('source', variant_source(variant)), ('diagnostic', diagnostic_source(variant))):
            with (directory / f'{variant}-{label}.py').open('x') as stream:
                stream.write(source)
    return config


def check_source(config, directory):
    assert config['baseline_commit'] == BASELINE
    assert digest(git_source()) == config['baseline_sha256']
    assert digest(git_source(FIXTURE_PATH)) == config['fixture_source_sha256']
    assert sys.version == config['runtime']['python']
    assert digest((directory / 'DESIGN.md').read_bytes()) == config['design_sha256']
    for p, expected in config['source_hashes'].items():
        assert digest(Path(p).read_bytes()) == expected, p
    for variant in VARIANTS:
        assert (directory / f'{variant}-source.py').read_text() == variant_source(variant)
        assert (directory / f'{variant}-diagnostic.py').read_text() == diagnostic_source(variant)


def warm(module):
    for _ in range(64):
        module.SearchTable()


def decision(module, game, depth):
    agent = module.NegamaxAgent(depth)
    wall, cpu = time.perf_counter(), time.process_time()
    move = agent.choose_move(game)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    iterations = getattr(agent, 'last_iterations', [dict(depth=depth, **agent.last_stats)])
    return dict(wall_seconds=wall, cpu_seconds=cpu, move=move,
                scores=[list(item) for item in agent.last_scores.items()],
                stats=agent.last_stats, iterations=iterations)


def assert_same(a, b, counters=False):
    for key in ('move', 'scores', 'stats', 'iterations') if counters else ('move', 'scores'):
        assert a[key] == b[key], (key, a[key], b[key])


@contextmanager
def capture(module, diagnostic=False):
    held, metrics, factory = [], [], module.SearchTable

    def create():
        # Keep only the final table; previous tables die on the normal schedule.
        held.clear()
        table = factory()
        if diagnostic:
            table.diagnostic = {name: 0 for name in DIAGNOSTICS}
            metrics.append(table.diagnostic)
        held.append(table)
        return table

    with patch.object(module, 'SearchTable', create):
        yield held, metrics


def retained(table):
    seen, categories = set(), {}

    def add(obj, category):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        categories[category] = categories.get(category, 0) + sys.getsizeof(obj)

    for attr, category in (('entries', 'score_tt'), ('hints', 'hint_cache')):
        value = getattr(table, attr, None)
        if value is None:
            continue
        add(value, category)
        for key, entry in value.items():
            add(key, category)
            add(entry, category)
    add(table, 'table_overhead')
    add(vars(table), 'table_overhead')
    for key, value in vars(table).items():
        add(key, 'table_overhead')
        if key not in ('entries', 'hints'):
            add(value, 'table_overhead')
    return dict(total_bytes=sum(categories.values()), categories=categories,
                final_entries=len(table.entries), hint_entries=len(getattr(table, 'hints', None) or {}),
                attribute_bytes=sys.getsizeof(vars(table)))


def memory_run(module, game, depth):
    gc.collect()
    tracemalloc.start()
    try:
        with capture(module) as (held, _):
            result = decision(module, game, depth)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    result.update(traced_current_bytes=current, traced_peak_bytes=peak, retained=retained(held[0]))
    return result


def counted_run(variant, game, depth):
    module = load_variant(variant, diagnostic=True)
    warm(module)
    with capture(module, True) as (_, metrics):
        result = decision(module, game, depth)
    assert len(metrics) == len(result['iterations'])
    result['diagnostics'] = metrics
    result['diagnostic_totals'] = {name: sum(s[name] for s in metrics) for name in DIAGNOSTICS}
    return result


def rss_worker(variant, history, depth):
    module = load_variant(variant)
    warm(module)
    game = position(history)

    def rss():
        try:
            return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())],
                                               stderr=subprocess.DEVNULL)) * 1024
        except (OSError, subprocess.CalledProcessError):
            return None

    before = rss()
    with capture(module) as (held, _):
        result = decision(module, game, depth)
    high = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.update(rss_before_bytes=before, rss_after_bytes=rss(),
                  high_water_bytes=high if platform.system() == 'Darwin' else high * 1024,
                  retained=retained(held[0]))
    return result


def fresh(variant, history, depth, timeout=30):
    cmd = [sys.executable, '-m', 'scripts.benchmark_negamax_iterative', 'worker',
           '--variant', variant, '--history', json.dumps(history), '--depth', str(depth)]
    return json.loads(subprocess.check_output(cmd, text=True, timeout=timeout))


def preflight(directory):
    config = json.loads((directory / 'manifest.json').read_text())
    check_source(config, directory)
    rows = []
    start = time.perf_counter()
    modules = {v: load_variant(v) for v in VARIANTS}
    for m in modules.values():
        warm(m)
    for f in config['positions']:
        if f['id'] not in PREFLIGHT_IDS:
            continue
        for depth in (8, 10):
            variants = {}
            expected = None
            for v, m in modules.items():
                normal = decision(m, position(f['history']), depth)
                traced = memory_run(m, position(f['history']), depth)
                expected = expected or normal
                assert_same(expected, normal)
                assert_same(normal, traced, True)
                variants[v] = dict(normal=normal, memory=traced)
            rows.append(dict(position=f['id'], depth=depth, variants=variants))
            print(f'preflight {f["id"]} D{depth}', flush=True)
    # Conservative mean on the expensive four-board set applied to ALL conditions.
    per_condition = statistics.mean(sum(8 * v['normal']['wall_seconds'] +
        2 * v['memory']['wall_seconds'] + 3 * v['normal']['wall_seconds']
        for v in row['variants'].values()) for row in rows)
    projection = per_condition * len(config['positions']) * len(config['depths']) * 2
    practical = all(v['normal']['wall_seconds'] <= 10 and
        v['memory']['traced_peak_bytes'] <= 256 * 1024**2
        for r in rows for v in r['variants'].values())
    result = dict(rows=rows, main_projected_seconds=projection,
                  passed=practical and projection <= 7200, elapsed_seconds=time.perf_counter() - start)
    write_new(directory / 'preflight.json', result)
    return {k: v for k, v in result.items() if k != 'rows'}


def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def analyze(rows):
    non_tail = [r for r in rows if r['group'] != 'post-hoc-diagnostic']

    def latency(r, v, cpu=False):
        return r['variants'][v]['median_cpu_seconds' if cpu else 'median_wall_seconds']

    broad = [r for r in non_tail if r['depth'] in (8, 10) and
             r['variants']['direct']['counted']['stats']['nodes'] >= 2000 and
             r['variants']['direct']['counted']['diagnostic_totals']['leaves'] /
             r['variants']['direct']['counted']['stats']['nodes'] >= .35]
    result = {}
    for v in VARIANTS[1:]:
        speeds = [latency(r, 'direct') / latency(r, v) for r in non_tail]
        bs = [latency(r, 'direct') / latency(r, v) for r in broad]
        sum_ratio = sum(latency(r, v) for r in non_tail) / sum(latency(r, 'direct') for r in non_tail)
        cpu_ratio = sum(latency(r, v, True) for r in non_tail) / sum(latency(r, 'direct', True) for r in non_tail)

        def mem(r, name, field):
            return max((m['retained']['total_bytes'] if field == 'retained' else m['traced_peak_bytes'])
                       for m in r['variants'][name]['memory'])

        retained_ratio = sum(mem(r, v, 'retained') for r in broad) / sum(mem(r, 'direct', 'retained') for r in broad)
        checks = dict(broad_speedup=geometric(bs) >= 1.15,
                      broad_fraction=sum(s > 1 for s in bs) / len(bs) >= .75,
                      distribution_speedup=geometric(speeds) >= 1.05,
                      summed_latency=sum_ratio <= .9, summed_cpu=cpu_ratio <= .95,
                      per_condition=all(latency(r, v) <= 1.2 * latency(r, 'direct') or
                          latency(r, v) - latency(r, 'direct') <= .00025 for r in non_tail),
                      expensive=all(latency(r, v) <= 1.2 * latency(r, 'direct') and
                          r['variants'][v]['counted']['stats']['nodes'] <= 1.25 *
                          r['variants']['direct']['counted']['stats']['nodes']
                          for r in rows if latency(r, 'direct') >= .05),
                      memory=all(mem(r, v, field) <= 1.25 * mem(r, 'direct', field) or
                          mem(r, v, field) - mem(r, 'direct', field) <= 65536
                          for r in non_tail for field in ('retained', 'peak')),
                      broad_retained=retained_ratio <= 1.15)
        result[v] = dict(eligible=all(checks.values()), checks=checks,
                        broad_conditions=len(broad), broad_speedup=geometric(bs),
                        broad_faster_fraction=sum(s > 1 for s in bs) / len(bs),
                        geometric_speedup=geometric(speeds), summed_latency_ratio=sum_ratio,
                        summed_cpu_ratio=cpu_ratio, broad_retained_ratio=retained_ratio,
                        depth_speedups={str(d): geometric(latency(r, 'direct') / latency(r, v)
                            for r in non_tail if r['depth'] == d) for d in (4, 6, 8, 10)},
                        generalization_speedup=geometric(latency(r, 'direct') / latency(r, v)
                            for r in non_tail if r['group'] == 'generalization'))
    eligible = [v for v in result if result[v]['eligible']]
    return dict(comparisons=result, selected=min(eligible, key=lambda v: result[v]['summed_latency_ratio'])
                if eligible else 'direct')


def run(directory, depth12=False):
    config = json.loads((directory / 'manifest.json').read_text())
    check_source(config, directory)
    assert json.loads((directory / 'preflight.json').read_text())['passed']
    if depth12:
        assert json.loads((directory / 'depth12-preflight.json').read_text())['passed']
    modules = {v: load_variant(v) for v in VARIANTS}
    for m in modules.values():
        warm(m)
    rows, start = [], time.perf_counter()
    with (directory / ('depth12-results.jsonl' if depth12 else 'results.jsonl')).open('x') as stream:
        conditions = [(f, d) for f in config['positions']
                      if not depth12 or f['id'] in PREFLIGHT_IDS
                      for d in ([12] if depth12 else config['depths'])]
        for index, (f, depth) in enumerate(conditions):
            game = position(f['history'])
            assert not game.is_game_over()
            variants = {v: dict(samples=[]) for v in VARIANTS}
            reference = None
            for rep in range(8):
                offset = (rep + index) % 4
                for v in VARIANTS[offset:] + VARIANTS[:offset]:
                    result = decision(modules[v], game, depth)
                    reference = reference or result
                    assert_same(reference, result)
                    if rep:
                        variants[v]['samples'].append(result)
            for v in VARIANTS:
                expected = variants[v]['samples'][0]
                counted = counted_run(v, game, depth)
                assert_same(expected, counted, True)
                memory = [memory_run(modules[v], game, depth) for _ in range(1 if depth12 else 2)]
                for s in variants[v]['samples'] + memory:
                    assert_same(expected, s, True)
                variants[v].update(counted=counted, memory=memory,
                    median_wall_seconds=statistics.median(s['wall_seconds'] for s in variants[v]['samples']),
                    median_cpu_seconds=statistics.median(s['cpu_seconds'] for s in variants[v]['samples']))
            row = dict(position=f['id'], group=f['group'], depth=depth, variants=variants)
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            print(f'{f["id"]} D{depth}: ' + ', '.join(f'{v}={variants[v]["median_wall_seconds"]*1000:.3f}ms'
                  for v in VARIANTS), flush=True)
    write_new(directory / ('depth12-runtime.json' if depth12 else 'runtime-end.json'),
              dict(runtime=runtime(), elapsed_seconds=time.perf_counter() - start))
    if not depth12:
        write_new(directory / 'analysis.json', analyze(rows))
    return dict(conditions=len(rows), elapsed_seconds=time.perf_counter() - start)


def rss_run(directory):
    config = json.loads((directory / 'manifest.json').read_text())
    check_source(config, directory)
    results = []
    for f in config['positions']:
        if f['id'] in PREFLIGHT_IDS:
            for v in VARIANTS:
                results.append(dict(position=f['id'], variant=v, depth=10,
                                    result=fresh(v, f['history'], 10)))
    write_new(directory / 'rss.json', results)
    return dict(fresh_processes=len(results))


def depth12_preflight(directory):
    config = json.loads((directory / 'manifest.json').read_text())
    check_source(config, directory)
    try:
        probe = fresh('direct', [], 12)
        # Conservatively reserve timings, instrumented counting and tracing.
        projected = probe['wall_seconds'] * 4 * 4 * (8 + 3 + 20)
        passed = (probe['wall_seconds'] <= 10 and probe['high_water_bytes'] <= 256 * 1024**2
                  and projected <= 600)
        result = dict(probe=probe, projected_seconds=projected, passed=passed)
    except subprocess.TimeoutExpired:
        result = dict(passed=False, reason='empty D12 probe exceeded external 30s deadline; incomplete')
    write_new(directory / 'depth12-preflight.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('declare', 'preflight', 'run', 'rss', 'depth12-preflight', 'depth12', 'worker'))
    parser.add_argument('--directory', type=Path, default=ROOT)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--history')
    parser.add_argument('--depth', type=int)
    args = parser.parse_args()
    if args.action == 'worker':
        # External subprocess only; never changes a completed search's semantics.
        if args.depth == 12 and platform.system() != 'Darwin':
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
        result = rss_worker(args.variant, json.loads(args.history), args.depth)
    elif args.action == 'depth12':
        result = run(args.directory, True)
    else:
        result = {'declare': declare, 'preflight': preflight, 'run': run, 'rss': rss_run,
                  'depth12-preflight': depth12_preflight}[args.action](args.directory)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
