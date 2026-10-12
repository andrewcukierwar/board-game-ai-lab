"""Cost-only feasibility pilot: no frozen replay is read, no model is trained for results.

Measures just enough to set the prospective budgets in results/design.json:
batch-128 update time, eval-mode scoring speed and pool-eight search speed for
each architecture, on disposable seed-initialized models and synthetic
random-play positions. Everything it runs counts against the phase's time limit.
"""
import json
import time

from .bench import child, evaluation_throughput
from .common import NAMES, RESULTS, cpu_seconds, preserve, runtime, write

PILOT_UPDATES = 30
PILOT_FORWARDS = 400   # pool-eight forward calls per architecture, then the search is abandoned


def main():
    identity = runtime()
    before = preserve()
    started, cpu = time.perf_counter(), cpu_seconds()
    result = dict(runtime=identity, training={}, collection={}, evaluation={})
    child_cpu = 0.0
    for name in NAMES:
        result['training'][name] = child(['training', name, PILOT_UPDATES], 300)
        result['collection'][name] = child(['collection', name, 0, 8, '--max-forwards', PILOT_FORWARDS], 300)
        result['evaluation'][name] = evaluation_throughput(name)
        child_cpu += result['collection'][name]['cpu_seconds']
    if preserve() != before:
        raise ValueError('Official inventory changed')
    result['wall_seconds'] = time.perf_counter() - started
    result['parent_cpu_seconds'] = cpu_seconds() - cpu
    result['primary_training_updates'] = 0
    result['frozen_replay_positions_read'] = 0
    write(RESULTS / 'pilot.json', result)
    print(json.dumps({name: dict(update_ms=result['training'][name]['trainer_step']['mean_ms'],
                                 staged_ms=result['training'][name]['staged_total_ms'],
                                 search_ms_per_position=1e3 * result['collection'][name]['wall_seconds'] / result['collection'][name]['neural_positions'],
                                 mean_batch=result['collection'][name]['mean_batch'],
                                 eval_positions_per_second=result['evaluation'][name]['positions_per_second'],
                                 train_rss=result['training'][name]['peak_rss_mib'])
                      for name in NAMES} | dict(wall_seconds=result['wall_seconds']), indent=1))


if __name__ == '__main__':
    main()
