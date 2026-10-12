"""Prospective design, feasibility projection, generation-zero boundaries and the source freeze.

Run once, after calibration, the pipeline smoke test and validation and before
any training. It refuses to freeze an infeasible design; it never shrinks the
workload to make it fit. The budgets below were set by hand from
results/calibration.json BEFORE this freeze and are never adapted afterwards.
"""
import json
from pathlib import Path

from .common import (ARMS, FIXTURES_SHA256, IDENTITY, NAMES, OUTPUT, PARAMETERS, RESULTS, ROOT, SEEDS, arm_label,
                     experimental_config, preserve, protected_inventory, read, runtime, sha, source_map, text_sha,
                     worktree_observations, write)

HARD_MAXIMUM_SECONDS = 120 * 60
TARGET_SECONDS = 90 * 60
RESERVE_SECONDS = 90           # final integrity checks and logging
CONTINGENCY = 1.25
TASK_OVERHEAD_SECONDS = 15     # interpreter start, strict load, validation reload, persistence, per training task
DYNAMICS_SECONDS = 60
POOL_INEFFICIENCY = 1.3

GENERATIONS = 6
# Frozen integer budgets per generation. Ratio 3:2 from the measured cost ratio (see calibration.json);
# the proposed 192/128 was scaled by 0.75 because the proposed size did not fit the time limit.
BUDGETS = dict(cnn=dict(games_per_generation=144, optimizer_updates=144),
               resnet3x64=dict(games_per_generation=96, optimizer_updates=96))
# ABBA over arms within a generation, reversed on even generations, so drift cancels across architectures.
ODD = [[42, 'cnn'], [42, 'resnet3x64'], [314159, 'resnet3x64'], [314159, 'cnn']]
ORDER = [ODD if g % 2 else list(reversed(ODD)) for g in range(1, GENERATIONS + 1)]

EVALUATION = dict(simulations=256, generations=[0, 2, 4, 6], search_seed=4404901, match_seed=4404902, workers=3,
                  opponent='reference negamax (connect4-reference-negamax-v1), seeded uniform ties',
                  games_per_checkpoint=80, suites=dict(solved=300, tactical=200, deep=128))

ANALYSIS = dict(
    bootstrap_replicates=10_000, bootstrap_seed=4404903, interval='two-sided 95% percentile',
    resampling=('paired cluster bootstrap within seed: opening families for matches (both colors stay together), '
                'Phase 4D.4G mirror/prefix clusters for position suites; pooled = equal-weight mean of the two seeds, '
                'resampled independently'),
    multiplicity='none declared; one primary endpoint, everything else is supporting',
    decision_checkpoint=GENERATIONS, consistency_checkpoint=4,
    equal_compute=dict(
        quantity='ResNet / CNN ratio of realized self-play collection seconds plus optimizer-update seconds, all generations',
        approximately_equal=[0.90, 1.10], qualified=[0.80, 1.25],
        rule='inside approximately_equal: equal-compute claim allowed; inside qualified only: comparison described as '
             'roughly comparable computation; outside: no equal-compute claim of any kind'),
    strength=dict(
        endpoint='ResNet minus CNN match score against the fixed opponent at generation 6 of each arm',
        PASS='both seeds > 0, pooled difference >= +0.05, pooled 95% lower bound > 0, and pooled difference at '
             'generation 4 >= 0',
        FAIL='both seeds < 0 and pooled 95% upper bound < 0',
        INCONCLUSIVE='anything else, and always when the endpoint is saturated',
        practical_margin=0.05, saturation=[0.10, 0.90],
        saturated='mean generation-6 score of the four arms outside [0.10, 0.90]'),
    tactical=dict(
        endpoint='ResNet minus CNN raw-PUCT accuracy on tactical200 at generation 6, immediate-win and safe-response '
                 'categories separately; the same rule on the raw policy is reported and is not binding',
        PASS='both categories: pooled difference >= -0.03 and pooled 95% lower bound >= -0.10',
        FAIL='any category: pooled difference <= -0.05 and pooled 95% upper bound < 0',
        INCONCLUSIVE='anything else', pass_point=-0.03, pass_lower=-0.10, fail_point=-0.05),
    value=dict(
        endpoint='ResNet minus CNN raw exact-value MSE on solved300 at generation 6 (supporting only)',
        BETTER='both seeds < 0 and pooled 95% upper bound < 0', WORSE='both seeds > 0 and pooled 95% lower bound > 0',
        INCONCLUSIVE='anything else'),
    supporting_rule='BETTER/WORSE for any other contrast: both seeds agree in sign and the pooled 95% interval '
                    'excludes zero; otherwise INCONCLUSIVE (signs oriented so that lower loss/error and higher '
                    'accuracy/score favor the ResNet)',
    secondary_pairings=dict(
        equal_compute='same generation index (2, 4, 6): the arms have had about the same computation',
        equal_games='CNN generation 4 against ResNet generation 6 (576 games and 576 updates each); raw-network '
                    'measures also CNN generation 2 against ResNet generation 3 (288 each). Descriptive: these '
                    'checkpoints differ in generation count, replay refreshes and wall time and are not a matched '
                    'experiment'),
    recommendation=dict(
        replace='strength PASS, equal-compute claim not invalid, tactical gate not FAIL',
        keep_cnn='strength FAIL or tactical gate FAIL',
        otherwise='no demonstrated advantage: the CNN stays the default because it is cheaper per position; the '
                  'ResNet is not rejected',
        next_investment_indicators=dict(
            value_learning='generation-6 raw exact-value MSE on solved300 at or above 0.9 x the constant-zero MSE in '
                           'both architectures, or prequential held-out value MSE above the constant-zero MSE',
            tactical_search='generation-6 raw-PUCT immediate-win or safe-response accuracy below 0.95 in the better arm',
            replay_optimization='generation-6 replay mirror-duplicate fraction above 0.40, or generation-6 '
                                'post-ply-8 self-play duplicate fraction above 0.30, in the better arm'),
        note='the indicators inform a judgment; they are not an automatic selection'))


def design_record(calibration, prior_seconds):
    depth = calibration['opponent']['chosen_depth']
    if depth is None:
        raise SystemExit('No unsaturated opponent depth: do not freeze')
    from . import evaluate
    openings = [r['id'] for r in evaluate.openings()]
    evaluation = dict(EVALUATION, opponent_depth=depth, openings=openings)
    if 2 * len(openings) != evaluation['games_per_checkpoint']:
        raise ValueError('Opening count differs from the declared games per checkpoint')
    design = dict(smoke=False, seeds=list(SEEDS), generations=GENERATIONS, arms=BUDGETS, order=ORDER,
                  training_overrides={}, schedule='original: temperature 1 at plies 0-7, then 0',
                  evaluation=evaluation, analysis=ANALYSIS)
    design['configurations'] = {arm_label(seed, name): experimental_config(design, seed, name).to_dict()
                                for seed, name in ARMS}
    design['budget'] = projection(calibration, design, prior_seconds)
    return design


def projection(calibration, design, prior_seconds):
    arms, generations, plan = design['arms'], design['generations'], design['evaluation']
    per_architecture = {}
    for name in NAMES:
        games = generations * arms[name]['games_per_generation']
        updates = generations * arms[name]['optimizer_updates']
        self_play = games * calibration['collection'][name]['seconds_per_game']
        update = updates * calibration['update_ms'][name] / 1e3
        per_architecture[name] = dict(games=games, updates=updates, self_play_seconds=self_play, update_seconds=update,
                                      total_seconds=self_play + update)
    training = len(SEEDS) * sum(v['total_seconds'] for v in per_architecture.values())
    overhead = TASK_OVERHEAD_SECONDS * generations * len(ARMS)
    evaluation_cpu = len(ARMS) * DYNAMICS_SECONDS
    for name in NAMES:
        per_checkpoint = (plan['games_per_checkpoint'] * calibration['match_cost'][name]['seconds_per_game']
                          + sum(plan['suites'].values()) * calibration['search'][name]['seconds_per_search'])
        evaluation_cpu += len(SEEDS) * len(plan['generations']) * per_checkpoint
    evaluation_wall = evaluation_cpu / plan['workers'] * POOL_INEFFICIENCY
    total = training + overhead + evaluation_wall + RESERVE_SECONDS
    ceiling = HARD_MAXIMUM_SECONDS - prior_seconds
    return dict(expected_per_architecture_per_seed=per_architecture,
                expected_ratio_resnet_over_cnn=per_architecture['resnet3x64']['total_seconds'] / per_architecture['cnn']['total_seconds'],
                training_seconds=training, training_task_overhead_seconds=overhead,
                evaluation_cpu_seconds=evaluation_cpu, evaluation_wall_seconds=evaluation_wall,
                projected_seconds=total, contingency=CONTINGENCY, projected_with_contingency=total * CONTINGENCY,
                prior_measured_seconds=prior_seconds, run_ceiling_seconds=ceiling,
                projected_total_with_prior=total + prior_seconds, target_seconds=TARGET_SECONDS,
                hard_maximum_seconds=HARD_MAXIMUM_SECONDS,
                feasible=total * CONTINGENCY <= ceiling and total + prior_seconds <= TARGET_SECONDS)


def smoke_design():
    """Tiny disposable design that exercises every task of the real pipeline in about a minute."""
    from . import evaluate
    openings = [r['id'] for r in evaluate.openings()][:3]
    design = dict(smoke=True, seeds=list(SEEDS), generations=2,
                  arms=dict(cnn=dict(games_per_generation=12, optimizer_updates=6),
                            resnet3x64=dict(games_per_generation=8, optimizer_updates=4)),
                  order=[ODD, list(reversed(ODD))], training_overrides=dict(self_play_simulations=16, batch_size=32),
                  schedule='original: temperature 1 at plies 0-7, then 0',
                  evaluation=dict(EVALUATION, simulations=16, generations=[0, 2], opponent_depth=2, openings=openings,
                                  games_per_checkpoint=6),
                  analysis=dict(ANALYSIS, decision_checkpoint=2, consistency_checkpoint=2,
                                secondary_pairings=dict(ANALYSIS['secondary_pairings'])),
                  budget=dict(run_ceiling_seconds=600))
    design['configurations'] = {arm_label(seed, name): experimental_config(design, seed, name).to_dict()
                                for seed, name in ARMS}
    return design


def initial_boundaries(design, directory):
    """Generation-zero boundaries: fresh seed-determined weights, empty replay, zero optimizer steps."""
    from games.connect4.alphazero_v2.experimental import ORIGINAL, ExperimentalRunner, load_experimental
    records = {}
    for seed, name in ARMS:
        config = experimental_config(design, seed, name)
        runner = ExperimentalRunner(config, schedule=ORIGINAL, lineage=f'phase4d4h-c-seed-{seed}', arm=name)
        path = Path(directory) / arm_label(seed, name) / 'generation-0000.pt'
        path.parent.mkdir(parents=True, exist_ok=False)
        digest = runner.save(path)
        loaded = load_experimental(path, expected_sha256=digest, expected_config=config, restore_global_rng=False)
        parameters = sum(p.numel() for p in loaded.model.parameters())
        fresh = loaded.factory.hash(loaded.factory.construct(seed))
        if (loaded.counters() != dict(completed_generations=0, total_games=0, total_positions=0, trainer_steps=0)
                or len(loaded.replay) or loaded.replay.games or loaded.history or loaded.parent is not None
                or loaded.trainer.optimizer.state_dict()['state'] or loaded.factory.hash(loaded.model) != fresh
                or parameters != PARAMETERS[name] or config.architecture != IDENTITY[name]):
            raise ValueError(f'{arm_label(seed, name)} is not a from-scratch generation-zero arm')
        records[arm_label(seed, name)] = dict(
            path=str(path), sha256=digest, weights_sha256=fresh, architecture=config.architecture, parameters=parameters,
            initial_state_sha256=loaded.initial_state_sha256(), replay_positions=0, trainer_steps=0,
            optimizer_state_entries=0, config_sha256=text_sha(config.to_dict()))
    return records


def main():
    identity = runtime()
    official = preserve()
    calibration, smoke, validation = (read(RESULTS / f'{name}.json') for name in ('calibration', 'smoke', 'validation'))
    if not validation['passed'] or smoke['status'] != 'COMPLETE_PENDING_ANALYSIS':
        raise SystemExit('Validation or the pipeline smoke test did not pass: do not freeze')
    prior = calibration['wall_seconds'] + smoke['elapsed_seconds'] + validation['wall_seconds']
    design = design_record(calibration, prior)
    design.update(calibration_sha256=sha(RESULTS / 'calibration.json'), smoke_sha256=sha(RESULTS / 'smoke.json'),
                  validation_sha256=sha(RESULTS / 'validation.json'), fixtures_sha256=FIXTURES_SHA256)
    print(json.dumps(design['budget'], indent=1))
    if not design['budget']['feasible']:
        raise SystemExit('Infeasible within the time limit: do not launch and do not reduce the design here')
    if (RESULTS / 'freeze.json').exists():
        raise SystemExit('A freeze already exists')
    write(RESULTS / 'design.json', design)
    initial = initial_boundaries(design, OUTPUT / 'arms')
    from games.connect4.alphazero_v2.provenance import execution_source_identity
    protected = protected_inventory()
    write(RESULTS / 'freeze.json', dict(
        files=source_map(), runtime=identity, execution_source_sha256=execution_source_identity()['sha256'],
        initial=initial, official_files=len(official),
        protected_at_freeze={label: dict(files=len(files), inventory_sha256=text_sha(files)) for label, files in protected.items()},
        worktrees_at_freeze=worktree_observations(), fixtures_sha256=sha(ROOT / 'fixtures.json')))
    print(json.dumps({label: dict(sha256=r['sha256'], weights_sha256=r['weights_sha256']) for label, r in initial.items()}, indent=1))


if __name__ == '__main__':
    main()
