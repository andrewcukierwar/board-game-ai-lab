"""Measured benchmark stages run by the supervisor; each writes one JSON under the run directory."""
import argparse
import time

from . import bench
from .common import NAMES, OUTPUT, RESULTS, Deadline, cpu_seconds, enforce, preserve, read, runtime, write

RUN = OUTPUT / 'run'


def remaining(deadline):
    left = deadline.remaining()
    if left <= 1:
        raise TimeoutError('Phase 4D.4H-B hard deadline reached')
    return left


def inference(plan, deadline):
    timing = bench.inference_timing(plan['repetitions'], check=deadline)
    memory = [bench.child(['inference-memory', name, batch], remaining(deadline))
              for batch in bench.INFERENCE_BATCHES for name in NAMES]
    return dict(timing=timing, memory=memory)


def training(plan, deadline):
    rows = []
    for order in plan['passes']:
        for name in order:
            rows.append(bench.child(['training', name, plan['updates']], remaining(deadline)))
            if name != 'cnn':
                rows.append(bench.child(['training', name, plan['updates'], '--strip'], remaining(deadline)))
    return dict(rows=rows)


def collection(plan, deadline):
    rows = []
    for block in plan['blocks']:
        for name in block['order']:
            rows.append(bench.child(['collection', name, block['first'], block['count']], remaining(deadline)))
    return dict(rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('inference', 'training', 'collection'))
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    identity = runtime()
    enforce()
    before = preserve()
    deadline = Deadline(args.deadline)
    started, cpu = time.perf_counter(), cpu_seconds()
    result = globals()[args.stage](read(RESULTS / 'design.json')[args.stage], deadline)
    if preserve() != before:
        raise ValueError('Official inventory changed')
    result.update(stage=args.stage, runtime=identity, wall_seconds=time.perf_counter() - started,
                  parent_cpu_seconds=cpu_seconds() - cpu, official_files=len(before))
    write(RUN / f'{args.stage}.json', result)


if __name__ == '__main__':
    main()
