"""Source-frozen paired decisions, exclusive evidence, audit and fixed gates."""
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
from unittest.mock import patch

from scripts.benchmark_public_agents import position
from scripts.negamax_v3_variants import (BASELINE, AGENT_PATH, variant_source,
                                        diagnostic_source, load_source)

ROOT = Path('docs/search-negamax-v3')
HASH_PATHS = ('scripts/negamax_v3_variants.py', 'scripts/benchmark_negamax_v3.py',
              'games/connect4/agents/negamax_tt.py', 'games/connect4/connect4.py',
              'games/connect4/board.py', 'scripts/with_benchmark_lock.sh')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    with path.open('x') as f:
        f.write(json.dumps(data, indent=2) + '\n')


def runtime():
    return dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                cpus=os.cpu_count(), load=os.getloadavg(), unix_seconds=time.time())


def fixtures():
    source = subprocess.check_output(['git', 'show', BASELINE + ':docs/search-negamax-v2/iterative-deepening/manifest.json'])
    positions = json.loads(source)['positions']
    rng = Random(20261012)
    seen = {tuple(load_source(variant_source('direct')).SearchState(position(p['history'])).pieces) for p in positions}
    for i, plies in enumerate((7, 13, 21, 29)):
        while True:
            game, history = position([]), []
            for _ in range(plies):
                col = rng.choice(game.get_valid_moves())
                assert game.make_move(col)
                history.append(col)
                if game.is_game_over():
                    break
            bits = tuple(load_source(variant_source('direct')).SearchState(game).pieces)
            if not game.is_game_over() and bits not in seen:
                seen.add(bits)
                break
        positions.append(dict(id=f'heldout-{i:02d}', history=history, group='heldout'))
    return positions, digest(source)


def declare(directory, variants):
    directory.mkdir(parents=True, exist_ok=True)
    positions, fixture_hash = fixtures()
    config = dict(baseline_commit=BASELINE, variants=variants, positions=positions,
                  fixture_source_hash=fixture_hash, depths=[4, 6, 8, 10], samples=7,
                  memory_repetitions=2, warmups=1, maximum_batch_seconds=450,
                  design_sha256=digest((directory / 'DESIGN.md').read_bytes()),
                  source_hashes={p: digest(Path(p).read_bytes()) for p in HASH_PATHS},
                  runtime=runtime(), sources={})
    for variant in variants:
        candidate = directory / f'{variant}-input.py'
        source = candidate.read_text() if candidate.exists() else variant_source(variant)
        for suffix, data in [('source', source), ('diagnostic', diagnostic_source(source))]:
            name = f'{variant}-{suffix}.py'
            with (directory / name).open('x') as f:
                f.write(data)
            config['sources'][name] = digest(data.encode())
    write_new(directory / 'manifest.json', config)
    return config


def check(directory):
    config = json.loads((directory / 'manifest.json').read_text())
    assert config['baseline_commit'] == BASELINE
    assert config['runtime']['python'] == sys.version
    assert digest((directory / 'DESIGN.md').read_bytes()) == config['design_sha256']
    for p, expected in config['source_hashes'].items():
        assert digest(Path(p).read_bytes()) == expected, p
    for p, expected in config['sources'].items():
        assert digest((directory / p).read_bytes()) == expected, p
    assert (directory / 'direct-source.py').read_text() == variant_source('direct')
    return config


def decision(module, game, depth):
    agent = module.NegamaxAgent(depth)
    wall, cpu = time.perf_counter(), time.process_time()
    move = agent.choose_move(game)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    return dict(wall_seconds=wall, cpu_seconds=cpu, move=move,
                scores=[list(item) for item in agent.last_scores.items()], stats=agent.last_stats)


def same(a, b, counters=False):
    for k in ('scores', 'move', 'stats') if counters else ('scores', 'move'):
        assert a[k] == b[k], (k, a[k], b[k])


@contextmanager
def capture(module):
    held, factory = [], module.SearchTable
    def create():
        table = factory()
        held.append(table)
        return table
    with patch.object(module, 'SearchTable', create):
        yield held


def retained(table):
    seen = set()
    def size(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        n = sys.getsizeof(obj)
        if isinstance(obj, dict):
            n += sum(size(k) + size(v) for k, v in obj.items())
        return n
    return size(table) + size(vars(table))


def memory_run(module, game, depth):
    gc.collect()
    tracemalloc.start()
    try:
        with capture(module) as held:
            result = decision(module, game, depth)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    result.update(current_bytes=current, peak_bytes=peak, retained_bytes=retained(held[0]))
    return result


def require_lock():
    owner = Path.home() / '.cache/bgai-laptop-benchmark.lock/owner'
    assert owner.exists(), 'Use scripts/with_benchmark_lock.sh'
    fields = dict(line.split('=', 1) for line in owner.read_text().splitlines())
    assert fields['branch'] == 'research/negamax-v3' and fields['worktree'] == str(Path.cwd())
    assert int(fields['pid']) == os.getppid(), 'Lock owner must be the invoking wrapper'
    return fields


def run(directory, start, stop):
    owner = require_lock()
    config = check(directory)
    modules = {v: load_source((directory / f'{v}-source.py').read_text(), v) for v in config['variants']}
    for module in modules.values():
        for _ in range(64):
            module.SearchTable()
    diagnostic = {v: load_source((directory / f'{v}-diagnostic.py').read_text(), v + '-diagnostic') for v in modules}
    started = time.monotonic()
    for index, fixture in enumerate(config['positions'][start:stop], start):
        for depth in config['depths']:
            assert time.monotonic() - started < config['maximum_batch_seconds'], 'Batch budget exhausted'
            path = directory / f'condition-{index:02d}-{depth:02d}.json'
            assert not path.exists(), 'Never overwrite evidence'
            game = position(fixture['history'])
            samples = {v: [] for v in modules}
            references = {v: decision(m, game, depth) for v, m in modules.items()}
            for r in references.values():
                same(references['direct'], r)
            names = list(modules)
            for repetition in range(config['samples']):
                offset = (index + depth + repetition) % len(names)
                for v in names[offset:] + names[:offset]:
                    sample = decision(modules[v], game, depth)
                    same(references[v], sample, True)
                    samples[v].append(sample)
            variants = {}
            for v in names:
                memories = [memory_run(modules[v], game, depth) for _ in range(config['memory_repetitions'])]
                with capture(diagnostic[v]) as held:
                    counted = decision(diagnostic[v], game, depth)
                counted['diagnostics'] = {k: getattr(held[0], k) for k in
                    ('leaves', 'terminals', 'probes', 'canonicalizations', 'reflected_hits')}
                for r in memories + [counted]:
                    same(references[v], r, True)
                variants[v] = dict(warmup=references[v], samples=samples[v], memory=memories,
                                   counted=counted, median_wall_seconds=statistics.median(s['wall_seconds'] for s in samples[v]),
                                   median_cpu_seconds=statistics.median(s['cpu_seconds'] for s in samples[v]))
            write_new(path, dict(position=fixture['id'], group=fixture['group'], depth=depth,
                                variants=variants, runtime=runtime(), lock_owner=owner))
            print(f"saved {fixture['id']} D{depth}", flush=True)
    print(f'batch complete {time.monotonic()-started:.1f}s', flush=True)


def rows(directory):
    return [json.loads(p.read_text()) for p in sorted(directory.glob('condition-*.json'))]


def analyze(data):
    primary = [r for r in data if r['position'] != 'post-hoc-tail']
    broad = [r for r in primary if r['depth'] in (8, 10) and r['variants']['direct']['samples'][0]['stats']['nodes'] >= 2000]
    def geometric(values):
        values = list(values)
        return math.exp(statistics.mean(math.log(x) for x in values)) if values else None
    result = {}
    for v in data[0]['variants']:
        if v == 'direct':
            continue
        def ratio(r, metric='median_wall_seconds'):
            return r['variants'][v][metric] / r['variants']['direct'][metric]
        geo = geometric(1 / ratio(r) for r in primary)
        broad_geo = geometric(1 / ratio(r) for r in broad)
        wall_sum = sum(r['variants'][v]['median_wall_seconds'] for r in primary) / sum(r['variants']['direct']['median_wall_seconds'] for r in primary)
        cpu_sum = sum(r['variants'][v]['median_cpu_seconds'] for r in primary) / sum(r['variants']['direct']['median_cpu_seconds'] for r in primary)
        heldout_geo = geometric(1 / ratio(r) for r in primary if r['group'] == 'heldout')
        failures = []
        for name, passed in [('geometric', geo >= 1.05), ('wall_sum', wall_sum <= .95),
                             ('cpu_sum', cpu_sum <= .97), ('broad_geometric', broad_geo >= 1.10),
                             ('broad_fraction', sum(ratio(r) < 1 for r in broad) / len(broad) >= .70),
                             ('heldout', heldout_geo >= 1)]:
            if not passed:
                failures.append(name)
        regressions = []
        memory_failures = []
        for r in data:
            a, b = r['variants']['direct'], r['variants'][v]
            label = f"{r['position']}/D{r['depth']}"
            if ratio(r) > 1.2 and b['median_wall_seconds'] - a['median_wall_seconds'] > .00025:
                regressions.append(label)
            if a['median_wall_seconds'] >= .05 and (ratio(r) > 1.2 or b['samples'][0]['stats']['nodes'] > 1.25 * a['samples'][0]['stats']['nodes']):
                failures.append('expensive:' + label)
            for k in ('retained_bytes', 'peak_bytes'):
                am = max(m[k] for m in a['memory'])
                bm = max(m[k] for m in b['memory'])
                if bm > 1.25 * am and bm - am > 65536:
                    memory_failures.append(label + '/' + k)
        retained_ratio = sum(r['variants'][v]['memory'][0]['retained_bytes'] for r in broad) / sum(r['variants']['direct']['memory'][0]['retained_bytes'] for r in broad)
        if regressions:
            failures.append('per_condition')
        if memory_failures or retained_ratio > 1.15:
            failures.append('memory')
        result[v] = dict(accepted=not failures, geometric_speedup=geo, broad_speedup=broad_geo,
                         wall_sum_ratio=wall_sum, cpu_sum_ratio=cpu_sum, heldout_speedup=heldout_geo,
                         broad_retained_ratio=retained_ratio, regressions=regressions,
                         memory_failures=memory_failures, failed_gates=failures,
                         baseline_sum_median_seconds=sum(r['variants']['direct']['median_wall_seconds'] for r in primary),
                         candidate_sum_median_seconds=sum(r['variants'][v]['median_wall_seconds'] for r in primary))
    return result


def audit(directory):
    config = check(directory)
    data = rows(directory)
    assert {(r['position'], r['depth']) for r in data} == {(p['id'], d) for p in config['positions'] for d in config['depths']}
    assert len(data) == len(config['positions']) * len(config['depths'])
    decisions = 0
    for r in data:
        a = r['variants']['direct']['samples'][0]
        for v, b in r['variants'].items():
            assert len(b['samples']) == config['samples'] and len(b['memory']) == config['memory_repetitions']
            ref = b['samples'][0]
            for sample in [b['warmup'], b['counted']] + b['samples'] + b['memory']:
                same(a, sample)
                same(ref, sample, True)
                decisions += 1
            assert b['median_wall_seconds'] == statistics.median(s['wall_seconds'] for s in b['samples'])
            assert b['median_cpu_seconds'] == statistics.median(s['cpu_seconds'] for s in b['samples'])
            assert b['memory'][0]['retained_bytes'] == b['memory'][1]['retained_bytes']
    # Independent unpruned array oracle, not a bitboard implementation.
    from tests.test_connect4_negamax import root_oracle, DRAW
    modules = {v: load_source((directory / f'{v}-source.py').read_text(), v) for v in config['variants']}
    vectors = []
    cases = [(p['id'], p['history'], d) for p in config['positions'] for d in (1, 2, 3)]
    cases += [('draw-prefix', DRAW[:n], d) for n in (36, 40, 41) for d in (4, 6, 8, 10, 12)]
    for name, history, depth in cases:
        game = position(history)
        expected = root_oracle(game, depth)
        for m in modules.values():
            assert m.NegamaxAgent(depth).score_moves(game) == expected
        vectors.append(dict(position=name, history=history, depth=depth, scores=list(expected.items())))
    # Compare inherited canonical score/counter vectors without rewriting them.
    prior = { (r['position'], r['depth']): r['variants']['direct']['samples'][0]
             for r in map(json.loads, Path('docs/search-negamax-v2/iterative-deepening/results.jsonl').read_text().splitlines()) }
    matched = 0
    for r in data:
        if (r['position'], r['depth']) in prior:
            same(prior[(r['position'], r['depth'])], r['variants']['direct']['samples'][0], True)
            matched += 1
    write_new(directory / 'oracle-vectors.json', vectors)
    write_new(directory / 'analysis.json', analyze(data))
    write_new(directory / 'audit.json', dict(passed=True, conditions=len(data), decisions=decisions,
                                            oracle_vectors=len(vectors), inherited_matches=matched,
                                            evidence_hashes={p.name: digest(p.read_bytes()) for p in directory.glob('condition-*.json')}))
    print(f'audit passed: {decisions} decisions, {len(vectors)} oracle vectors, {matched} inherited vectors')


def summarize(directory):
    data = rows(directory)
    analysis = json.loads((directory / 'analysis.json').read_text())
    assert analyze(data) == analysis
    lines = ['# Complete-decision results', '', 'Source-frozen baseline: `' + BASELINE + '`.',
             'Research-only variants; acceptance uses every fixed gate in DESIGN.md.', '',
             '| Variant | Decision | Geometric | Broad | Wall sum/A | CPU sum/A | Retained broad/A |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for v, a in analysis.items():
        lines.append(f"| {v} | {'eligible' if a['accepted'] else 'reject'} | {a['geometric_speedup']:.3f}× | {a['broad_speedup']:.3f}× | {a['wall_sum_ratio']:.3f} | {a['cpu_sum_ratio']:.3f} | {a['broad_retained_ratio']:.3f} |")
    lines += ['', 'Seven rotated paired warmed samples; two separate traced memory decisions.',
              'Geometric ratios use non-tail conditions. Broad: D8/D10 and ≥2,000 baseline nodes.',
              'Single laptop; finite workload and host activity limit generalization. No strength inference.', '',
              '| Position | Depth | Variant | ms | Speedup | Nodes | Leaves | Hits | Entries | Retained MiB | Peak MiB |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in data:
        for v, b in r['variants'].items():
            stats = b['samples'][0]['stats']
            lines.append(f"| {r['position']} | {r['depth']} | {v} | {b['median_wall_seconds']*1000:.3f} | {r['variants']['direct']['median_wall_seconds']/b['median_wall_seconds']:.3f}× | {stats['nodes']} | {b['counted']['diagnostics']['leaves']} | {stats['hits']} | {stats['entries']} | {b['memory'][0]['retained_bytes']/2**20:.3f} | {max(m['peak_bytes'] for m in b['memory'])/2**20:.3f} |")
    lines += ['', '## Fixed-gate failures and individual regressions', '']
    for v, a in analysis.items():
        lines += [f"- {v}: gates `{a['failed_gates']}`; latency regressions `{a['regressions']}`; memory `{a['memory_failures']}`."]
    with (directory / 'RESULTS.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['declare', 'run', 'audit', 'summarize'])
    parser.add_argument('--phase', required=True)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=32)
    args = parser.parse_args()
    directory = args.directory or ROOT / args.phase
    if args.command == 'declare':
        variants = args.variants or (['direct', 'mirror', 'mirror-selective'] if args.phase == 'mirror' else ['direct', 'pvs'])
        declare(directory, variants)
    elif args.command == 'run':
        run(directory, args.start, args.stop)
    elif args.command == 'audit':
        audit(directory)
    else:
        summarize(directory)


if __name__ == '__main__':
    main()
