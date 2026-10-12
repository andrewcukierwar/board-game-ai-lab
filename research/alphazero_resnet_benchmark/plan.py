"""Prospective design constants, feasibility projection from the pilot, and the source freeze.

Run once, after the dataset freeze, the pilot and the pipeline smoke test and
before any measured stage. It refuses to freeze an infeasible design; it never
shrinks the workload to make it fit.
"""
import json

from .common import NAMES, RESULTS, read, runtime, sha, source_map, write

HARD_MAXIMUM_SECONDS = 75 * 60
TARGET_SECONDS = 60 * 60
RESERVE_SECONDS = 60          # final preservation checks and logging
CONTINGENCY = 1.25
REVERSED = list(reversed(NAMES))

LEARNING = dict(
    # Equal nominal optimizer time (about 305 s) from the pilot's batch-128 update cost; fixed, never adapted.
    updates=dict(cnn=6800, resnet3x64=4700, resnet5x64=3000),
    common_updates=3000,
    milestones=[0, 50, 100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000, 3500, 4000, 4700, 5400, 6100, 6800],
    order={'42': list(NAMES), '314159': REVERSED},
    equal_update_points=[400, 800, 1600, 3000],
    equal_time_points_seconds=[60, 120, 240],
    nominal_train_seconds=305,
    batch_size=128, sampling='uniform over training positions, without replacement within a batch',
    sampling_domain='phase4d4h-b-fixed-data-sampling', reflection_probability=0.5)
INFERENCE = dict(repetitions=7, batches=[1, 8, 16, 32], calls={'1': 400, '8': 250, '16': 160, '32': 100}, warmup=20,
                 allowance_seconds=180)
TRAINING = dict(updates=60, passes=[list(NAMES), REVERSED])
COLLECTION = dict(simulations=256, pool=8, seed=20261012, generation=1, reference_plies=30,
                  blocks=[dict(first=0, count=16, order=list(NAMES)), dict(first=16, count=16, order=REVERSED)])
ANALYSIS = dict(
    bootstrap_replicates=10_000, bootstrap_seed=4404801, interval='two-sided 95% percentile',
    primary_learning_endpoint='novel held-out combined loss = policy cross-entropy + value MSE (the training objective)',
    verdict=('BETTER: both seeds favor the ResNet and the pooled interval excludes zero in its favor; '
             'WORSE: both seeds favor the CNN and the pooled interval excludes zero against the ResNet; '
             'otherwise INCONCLUSIVE'),
    memory_acceptable='peak resident memory at most 1.5x the CNN and at most 2048 MiB in every benchmark',
    throughput_classes='cost per neural position against the CNN: <=1.25x comparable; <=2.0x acceptable with a '
                       'proportionally smaller equal-wall budget; >2.0x costly, needs a compensating equal-update advantage')


def projection(pilot, held_positions, probe_positions, prior_seconds):
    update = {n: pilot['training'][n]['trainer_step']['mean_ms'] / 1e3 for n in NAMES}
    search = {n: pilot['collection'][n]['wall_seconds'] / pilot['collection'][n]['neural_positions'] for n in NAMES}
    scoring = {n: pilot['evaluation'][n]['positions_per_second'] for n in NAMES}
    training_children = sum(len(order) + sum(n != 'cnn' for n in order) for order in TRAINING['passes'])
    training = sum((2.9 * (TRAINING['updates'] + 10) * update[n] + 3) * (1 if n == 'cnn' else 2)
                   for order in TRAINING['passes'] for n in order)
    games = sum(b['count'] for b in COLLECTION['blocks'])
    collection = sum(games * 42 * COLLECTION['simulations'] * search[n] for n in NAMES) + 3 * 2 * len(NAMES)
    learning = {}
    for n in NAMES:
        milestones = sum(m <= LEARNING['updates'][n] for m in LEARNING['milestones'])
        evaluation = milestones * ((held_positions + 3 * probe_positions) / scoring[n] + 1.0)
        learning[n] = dict(optimizer_seconds=LEARNING['updates'][n] * update[n] * 1.03, evaluation_seconds=evaluation,
                           milestones=milestones)
    per_seed = 35 + sum(v['optimizer_seconds'] + v['evaluation_seconds'] for v in learning.values())
    total = INFERENCE['allowance_seconds'] + training + collection + 2 * per_seed
    ceiling = HARD_MAXIMUM_SECONDS - prior_seconds - RESERVE_SECONDS
    return dict(update_seconds=update, search_seconds_per_neural_position=search, scoring_positions_per_second=scoring,
                inference_seconds=INFERENCE['allowance_seconds'], training_benchmark_seconds=training,
                training_benchmark_children=training_children, collection_seconds_upper_bound=collection,
                learning_per_architecture=learning, learning_seconds_per_seed=per_seed, projected_seconds=total,
                contingency=CONTINGENCY, projected_with_contingency=total * CONTINGENCY,
                prior_measured_seconds=prior_seconds, run_ceiling_seconds=ceiling,
                projected_total_with_prior=total + prior_seconds, target_seconds=TARGET_SECONDS,
                hard_maximum_seconds=HARD_MAXIMUM_SECONDS,
                feasible=total * CONTINGENCY <= ceiling and total + prior_seconds <= TARGET_SECONDS)


def main():
    identity = runtime()
    pilot, smoke, freeze = (read(RESULTS / name) for name in ('pilot.json', 'smoke.json', 'dataset-freeze.json'))
    held = max(s['identity']['held_out_positions'] for s in freeze['seeds'].values())
    prior = pilot['wall_seconds'] + smoke['wall_seconds']
    budget = projection(pilot, held, 4096, prior)
    design = dict(learning=LEARNING, inference=INFERENCE, training=TRAINING, collection=COLLECTION, analysis=ANALYSIS,
                  budget=budget, pilot_sha256=sha(RESULTS / 'pilot.json'), smoke_sha256=sha(RESULTS / 'smoke.json'),
                  dataset_freeze_sha256=sha(RESULTS / 'dataset-freeze.json'), validation_sha256=sha(RESULTS / 'validation.json'))
    print(json.dumps(budget, indent=1))
    if not budget['feasible']:
        raise SystemExit('Infeasible within the time limit: do not launch and do not reduce the design')
    write(RESULTS / 'design.json', design)
    from games.connect4.alphazero_v2.provenance import execution_source_identity
    write(RESULTS / 'freeze.json', dict(files=source_map(), runtime=identity,
                                        execution_source_sha256=execution_source_identity()['sha256']))


if __name__ == '__main__':
    main()
