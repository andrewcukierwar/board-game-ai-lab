"""Laptop process RSS and synthetic MCTS/Negamax thread-contention diagnostics.

This does not run an HTTP server or change production concurrency reservations.
Each RSS search runs in a fresh process; timings contain no memory tracing.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

from games.connect4.agents.mcts_agent import MCTSAgent
from games.connect4.agents.negamax_agent import NegamaxAgent
from scripts.benchmark_public_agents import POSITIONS, position
from scripts.evaluate_mcts_strength import atomic_json, digest, seed_for, source_hashes


def maxrss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == 'Darwin' else value * 1024


def search(kind, budget, history, seed):
    game = position(history)
    agent = MCTSAgent(budget, rng=random.Random(seed)) if kind == 'mcts' else NegamaxAgent(6)
    start = time.perf_counter()
    move = agent.choose_move(game)
    return dict(kind=kind, move=move, wall_seconds=time.perf_counter() - start)


def worker(budget, history, seed):
    before = maxrss_bytes()
    result = search('mcts', budget, history, seed)
    return dict(**result, maxrss_before_bytes=before, maxrss_after_bytes=maxrss_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--budget', type=int)
    parser.add_argument('--history')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.budget, json.loads(args.history), args.seed)))
        return
    if args.directory is None:
        parser.error('directory is required')
    config = json.loads((args.directory / 'experiment.json').read_text())
    if config['source_hashes'] != source_hashes():
        raise ValueError('Source drift')
    output = args.directory / 'capacity.json'
    if output.exists():
        raise ValueError('Refusing to overwrite capacity evidence')
    result = dict(experiment_sha256=digest(config), python=platform.python_version(),
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rss=[], contention=[])
    for name in ('empty', 'near-opening'):
        history = POSITIONS[name]
        for budget in (400, 2000, 5000, 10000):
            seed = seed_for(config['master_seed'], 'capacity', name, budget)
            child = subprocess.check_output([sys.executable, '-m', 'scripts.measure_mcts_capacity',
                '--worker', '--budget', str(budget), '--history', json.dumps(history), '--seed', str(seed)], text=True)
            result['rss'].append(dict(position=name, budget=budget, seed=seed, **json.loads(child)))
        for budget in (400, 2000, 5000, 10000):
            samples = []
            for repetition in range(6):
                seed = seed_for(config['master_seed'], 'contention', name, repetition)
                # Alternate isolated versus contended order; repetition zero
                # warms both paths. Each search owns its game and RNG.
                sample = {}
                order = ('isolated', 'contended') if repetition % 2 else ('contended', 'isolated')
                for mode in order:
                    if mode == 'isolated':
                        sample[mode] = [search(kind, budget, history, seed) for kind in ('mcts', 'negamax')]
                    else:
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            futures = [pool.submit(search, kind, budget, history, seed) for kind in ('mcts', 'negamax')]
                            sample[mode] = [f.result() for f in futures]
                if [r['move'] for r in sample['isolated']] != [r['move'] for r in sample['contended']]:
                    raise RuntimeError('Thread contention altered deterministic moves')
                if repetition:
                    samples.append(dict(seed=seed, **sample))
            result['contention'].append(dict(position=name, budget=budget, samples=samples,
                median_mcts_isolated_seconds=statistics.median(s['isolated'][0]['wall_seconds'] for s in samples),
                median_mcts_contended_seconds=statistics.median(s['contended'][0]['wall_seconds'] for s in samples)))
    atomic_json(output, result)


if __name__ == '__main__':
    main()
