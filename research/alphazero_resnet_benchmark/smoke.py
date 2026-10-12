"""Pre-freeze pipeline smoke test: tiny budgets through every measured code path and the analysis.

Not evidence. It exists so that the single authorized run does not fail on a
programming error. It trains each architecture for at most 60 updates on seed
42's frozen split; its metrics are written only to the ignored output tree and
are not used to set any budget, endpoint or threshold. Its wall time is
recorded in results/smoke.json and counted against the phase's time limit.
"""
import argparse
import shutil
import time

from . import analyze, bench, learn
from .common import NAMES, OUTPUT, RESULTS, Deadline, preserve, runtime, write

PLAN = dict(updates=dict(cnn=60, resnet3x64=40, resnet5x64=20), common_updates=20, milestones=[0, 10, 20, 40, 60],
            order={'42': list(NAMES)}, equal_update_points=[10, 20], equal_time_points_seconds=[0.5])


def stages(directory):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    deadline = Deadline(time.time() + 600)
    timing = bench.inference_timing(2, scale=0.05, check=deadline)
    memory = [bench.child(['inference-memory', name, 8], 120) for name in NAMES]
    write(directory / 'inference.json', dict(timing=timing, memory=memory))
    rows = []
    for name in NAMES:
        rows.append(bench.child(['training', name, 4], 120))
        if name != 'cnn':
            rows.append(bench.child(['training', name, 4, '--strip'], 120))
    write(directory / 'training.json', dict(rows=rows))
    write(directory / 'collection.json', dict(rows=[bench.child(['collection', name, first, 2], 300)
                                                    for first in (0, 2) for name in NAMES]))
    learn.run(42, PLAN, directory / 'learning-seed-42', deadline, verify_freeze=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-only', action='store_true', help='re-run only the analysis on existing smoke output')
    parser.add_argument('--prior-seconds', type=float, default=0.0, help='wall time of earlier smoke invocations')
    args = parser.parse_args()
    identity = runtime()
    before = preserve()
    started = time.perf_counter()
    directory = OUTPUT / 'smoke'
    if not args.analysis_only:
        stages(directory)
    design = dict(learning=PLAN, collection=dict(reference_plies=30),
                  analysis=dict(bootstrap_replicates=500, bootstrap_seed=1))
    benchmarks, learned = analyze.analyze(directory, directory / 'analysis', design, (42,))
    if preserve() != before:
        raise ValueError('Official inventory changed')
    wall = time.perf_counter() - started + args.prior_seconds
    write(RESULTS / 'smoke.json', dict(
        purpose='pipeline verification only; metrics are not evidence and set no budget or endpoint',
        runtime=identity, wall_seconds=wall, prior_invocation_seconds=args.prior_seconds, seed=42, updates=PLAN['updates'],
        frozen_replay_read=True, collection_games_per_architecture=4, comparisons=sorted(learned['comparisons']),
        stages=['inference', 'training', 'collection', 'learning', 'analysis']))
    print(dict(wall_seconds=wall, comparisons=sorted(learned['comparisons'])))


if __name__ == '__main__':
    main()
