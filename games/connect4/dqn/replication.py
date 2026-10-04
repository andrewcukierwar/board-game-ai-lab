"""Phase 4C.2e frozen protocol and evaluation; reuse existing experiment/validation.

No training on import. The four explicit experiment CLI commands are frozen
before execution; this module does not introduce a trainer or select a model.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from . import validation_holdout as h

OUTPUT = Path('experiment-output/phase4c2e-replication')
FINAL_PATH = Path(__file__).with_name('replication_positions.json')
SUITES = {'existing': h.OLD_PATH, 'validation': h.PATH, 'final': FINAL_PATH}
SEEDS = (73, 314)


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def training_args(seed, probability, output):
    return ['--output', str(output), '--seed', str(seed),
            '--horizontal-symmetry-probability', str(probability),
            '--max-updates', '90000', '--max-plies', '100000', '--max-games', '7000',
            '--max-seconds', '300', '--replay-capacity', '10000', '--batch-size', '64',
            '--gamma', '0.99', '--learning-rate', '0.001', '--target-sync-interval', '100',
            '--epsilon-start', '1.0', '--epsilon-min', '0.10', '--epsilon-decay', '0.99997',
            '--threads', '1', '--interop-threads', '1', '--evaluation-games', '0',
            '--evaluation-seconds', '120']


def freeze():
    # Construction seed is a data-sampling seed, never an additional training seed.
    data = h.construct(seed=42020501, exclude_paths=(h.OLD_PATH, h.PATH))
    write(FINAL_PATH, data)
    h.validate(data['positions'], exclude_paths=(h.OLD_PATH, h.PATH))
    from .validation import configuration
    old = configuration()
    protocol = {k: old[k] for k in ('opening_seed', 'random_seed', 'openings', 'sides',
                'negamax_depths', 'games_per_model_per_depth', 'random_games_per_model',
                'aggregate_seconds', 'random_seconds', 'negamax_cache', 'tie_break')}
    protocol.update(head_to_head_games=0, schedule='depth, opening, side, model; then Random',
                    aggregate_scope='per seed pair', threads=1, interop_threads=1,
                    deterministic_algorithms=True, device='cpu', dtype='float32',
                    epsilon=0, learning=False, tactical_guards=False, inference_augmentation=False)
    runs = []
    for seed in SEEDS:
        for method, probability in (('baseline', 0.0), ('augmented', 0.5)):
            output = Path(f'experiment-output/phase4c2e-seed{seed}-{method}')
            runs.append(dict(seed=seed, method=method, output=str(output),
                             args=training_args(seed, probability, output)))
    config = dict(runs=runs, evaluation=protocol,
                  suites={k: dict(path=str(p), sha256=h.digest(p),
                                 previously_inspected=k != 'final') for k,p in SUITES.items()},
                  selection='No winner selection; four candidate inference checkpoints retained',
                  stop='First training limit; no retries, extensions, extra seeds or tuning')
    write(OUTPUT/'configuration.json', config)
    write(OUTPUT/'freeze.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
          configuration_sha256=h.digest(OUTPUT/'configuration.json'),
          final_holdout_sha256=h.digest(FINAL_PATH), predictions_inspected=False))
    print(json.dumps(config, indent=2))


def frozen_configuration():
    config = json.loads((OUTPUT/'configuration.json').read_text())
    frozen = json.loads((OUTPUT/'freeze.json').read_text())
    if h.digest(OUTPUT/'configuration.json') != frozen['configuration_sha256']:
        raise ValueError('Frozen configuration changed')
    for suite in config['suites'].values():
        if h.digest(suite['path']) != suite['sha256']:
            raise ValueError('Frozen suite changed')
    return config


def evaluate():
    import torch
    from . import validation as v
    from .checkpoint import load_checkpoint
    from .diagnostics import action_distribution
    config = frozen_configuration()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    # Exclusive marker prevents accidental repeat inference.
    write(OUTPUT/'evaluation-started.json', dict(utc=datetime.now(timezone.utc).isoformat()))
    report = dict(configuration=config, pairs={})
    for seed in SEEDS:
        start = time.monotonic()
        deadline = start + config['evaluation']['aggregate_seconds']
        runs = [r for r in config['runs'] if r['seed'] == seed]
        nets, hashes, weights = {}, {}, {}
        for run in runs:
            path = Path(run['output'])/'candidate.pt'
            training = json.loads((path.parent/'report.json').read_text())
            if (training.get('artifact_error') or training['training']['status'] != 'bounded_stop'
                    or not training['checkpoint']['reload_exact']
                    or training['checkpoint']['sha256'] != h.digest(path)):
                raise ValueError('Invalid training artifact')
            net, _ = load_checkpoint(path)
            nets[run['method']] = net
            hashes[str(path)] = h.digest(path)
            weights[run['method']] = {k:t.clone() for k,t in net.state_dict().items()}
        pair = dict(diagnostics={}, comparisons={}, games=[], matches={})
        for suite, entry in config['suites'].items():
            fixtures = json.loads(Path(entry['path']).read_text())['positions']
            results = {n:v.predictions(net, fixtures) for n,net in nets.items()}
            pair['diagnostics'][suite] = results
            pair['comparisons'][suite] = v.compare(results['baseline'], results['augmented'])
        def record(name, opponent_name, opponent, i, side, limit):
            game = v.play(nets[name], opponent, config['evaluation']['openings'][i], side,
                          config['evaluation']['random_seed']+2*i+side, limit)
            game.update(model=name, opponent=opponent_name, opening_index=i, training_seed=seed)
            pair['games'].append(game)
            log.write(json.dumps(game, allow_nan=False)+'\n')
            log.flush()
        with (OUTPUT/f'games-seed{seed}.jsonl').open('x') as log:
            for depth in config['evaluation']['negamax_depths']:
                for i in range(20):
                    for side in (0,1):
                        for name in nets:
                            if time.monotonic() < deadline:
                                record(name, f'negamax_{depth}', v.NegamaxAgent(depth), i, side, deadline)
            limit = min(deadline, time.monotonic()+config['evaluation']['random_seconds'])
            for i in range(10):
                for side in (0,1):
                    for name in nets:
                        if time.monotonic() < limit:
                            record(name, 'random', v.RandomAgent(), i, side, limit)
        for name in nets:
            for opponent in ('negamax_1','negamax_2','random'):
                games = [g for g in pair['games'] if (g['model'],g['opponent']) == (name,opponent)]
                result = v.summarize_games(games)
                result['requested'] = 20 if opponent == 'random' else 40
                result['greedy_actions'] = action_distribution(d['action'] for g in games
                                                              for d in g['decisions'] if d['agent']=='model')
                pair['matches'][name+'/'+opponent] = result
        pair['weights_unchanged'] = all(torch.equal(t,weights[n][k]) for n,net in nets.items()
                                        for k,t in net.state_dict().items())
        pair['checkpoint_hashes'] = hashes
        pair['files_unchanged'] = all(h.digest(p)==sha for p,sha in hashes.items())
        frozen_configuration()
        if not pair['weights_unchanged'] or not pair['files_unchanged']:
            raise ValueError('Evaluation changed weights/files')
        pair['elapsed_seconds'] = time.monotonic()-start
        report['pairs'][str(seed)] = pair
        write(OUTPUT/f'evaluation-seed{seed}.json', pair)
    write(OUTPUT/'evaluation.json', report)
    print(json.dumps({s:p['matches'] for s,p in report['pairs'].items()}, indent=2))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('operation', choices=('freeze','evaluate'))
    args = cli.parse_args()
    (freeze if args.operation == 'freeze' else evaluate)()
