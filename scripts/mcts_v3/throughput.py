"""Fixture latency, throughput and traced memory for MCTS v3 configurations.

    python -m scripts.mcts_v3.throughput --output docs/search-mcts-v3/preflight/throughput.json

Descriptive fixture measurements only. Equal-time budgets in the design come
from real game timings, not from this file. Run under the benchmark lock.
"""
import argparse
import gc
import json
import os
import platform
import random
import statistics
import time
import tracemalloc
from pathlib import Path

from scripts.benchmark_public_agents import POSITIONS
from scripts.evaluate_mcts_strength import atomic_json
from scripts.mcts_v3.harness import ROOT, hardware, make_agent, mcts, negamax, position, source_hashes
from scripts.mcts_v3.studies import SINGLES, research

BUDGETS = (400, 2000)
SAMPLES = 5


def fixtures():
    rows = dict(POSITIONS)
    openings = json.loads((ROOT / 'openings.json').read_text())['sets']['preflight']
    rows.update({o['id']: o['history'] for o in openings})
    return rows


def measure(spec, game, seed):
    agent = make_agent(spec, seed)
    wall, cpu = time.perf_counter(), time.process_time()
    move = agent.choose_move(game)
    cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
    stats = getattr(agent, 'last_stats', None)
    simulations = stats['simulations'] if isinstance(stats, dict) and 'simulations' in stats else None
    if simulations is None and spec['type'] == 'mcts':
        simulations = 0 if agent._winning_moves(game) else spec['simulations']
    return wall, cpu, move, simulations


def traced_peak(spec, game, seed):
    agent = make_agent(spec, seed)
    gc.collect()
    tracemalloc.start()
    agent.choose_move(game)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def run(configs, budgets, samples, extra=()):
    specs = {f'{name}-{b}': (research(b, config) if config or name == 'aa' else mcts(b))
             for b in budgets for name, config in configs.items()}
    specs.update({f'base-{b}': mcts(b) for b in budgets})
    specs.update(extra)
    rows = []
    for position_name, history in fixtures().items():
        game = position(history)
        names = list(specs)
        times = {name: [] for name in names}
        detail = {}
        for repetition in range(samples + 1):
            # Rotate order so no configuration always runs first or last.
            shift = repetition % len(names)
            for name in names[shift:] + names[:shift]:
                wall, cpu, move, simulations = measure(specs[name], game, 9000 + repetition)
                if repetition:
                    times[name].append((wall, cpu))
                    detail.setdefault(name, []).append((move, simulations))
        for name in names:
            walls = [w for w, _ in times[name]]
            simulations = [s for _, s in detail[name]]
            median = statistics.median(walls)
            executed = statistics.mean(simulations) if None not in simulations else None
            rows.append(dict(position=position_name, plies=len(history), config=name, spec=specs[name],
                             wall_ms=[1000 * w for w in walls],
                             median_wall_ms=1000 * median,
                             median_cpu_ms=1000 * statistics.median(c for _, c in times[name]),
                             moves=[m for m, _ in detail[name]], simulations=simulations,
                             simulations_per_second=executed / median if executed else None,
                             traced_peak_bytes=traced_peak(specs[name], game, 9001)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=SAMPLES)
    parser.add_argument('--negamax', type=int, nargs='*', default=[])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    load = os.getloadavg()
    rows = run(SINGLES, BUDGETS, args.samples,
               {f'negamax-{d}': negamax(d) for d in args.negamax})
    atomic_json(args.output, dict(
        python=platform.python_version(), platform=platform.platform(), hardware=hardware(),
        source_hashes=source_hashes(), samples=args.samples, warmups=1,
        load_average_before=load, load_average_after=os.getloadavg(), rows=rows))
    by_config = {}
    for row in rows:
        by_config.setdefault(row['config'], []).append(row)
    base = {(r['position'], r['spec'].get('simulations')): r for r in rows if r['config'].startswith('base-')}
    for name, group in by_config.items():
        ratios = [r['median_wall_ms'] / base[r['position'], r['spec'].get('simulations')]['median_wall_ms']
                  for r in group if (r['position'], r['spec'].get('simulations')) in base]
        print(f'{name:14s} median {statistics.median(r["median_wall_ms"] for r in group):8.3f} ms'
              + (f'  time vs base x{statistics.median(ratios):.2f}' if ratios else '')
              + f'  peak {max(r["traced_peak_bytes"] for r in group) / 2**20:.2f} MiB')


if __name__ == '__main__':
    main()
