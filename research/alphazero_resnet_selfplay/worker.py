"""One bounded child task. Every task loads its boundary strictly in a fresh interpreter.

train      one generation of one arm: collect, validate, replay, declared updates, atomic boundary
positions  raw policy and raw PUCT on the frozen development fixtures
matches    both colors on the frozen opening families against the fixed opponent
dynamics   read-only training-dynamics measures over all of one arm's boundaries
"""
import argparse
from collections import Counter
from pathlib import Path
import signal
import time

from .common import (IDENTITY, RESULTS, Deadline, arm_label, cpu_seconds, enforce, experimental_config, peak_rss_mib,
                     read, runtime, sha, write)
from . import evaluate, measures


def state(runner):
    from games.connect4.alphazero_v2.experimental import digest
    content = runner._content()
    return dict(model=runner.factory.hash(runner.model), optimizer=digest(content['optimizer']),
                replay=runner.replay.digest(), sampling=digest(content['sampling']),
                augmentation=digest(content['augmentation']), global_rng=digest(content['global_rng']),
                counters=runner.counters())


def load(path, digest, design, seed, name, *, restore_global_rng):
    from games.connect4.alphazero_v2.experimental import ORIGINAL, load_experimental
    runner = load_experimental(path, expected_sha256=digest, expected_config=experimental_config(design, seed, name),
                               restore_global_rng=restore_global_rng)
    if runner.schedule != ORIGINAL or runner.arm != name or runner.parent is not None:
        raise ValueError('Boundary is not a from-scratch original-schedule arm of this experiment')
    return runner


def train(args, design, check, output):
    runner = load(args.boundary, args.sha256, design, args.seed, args.name, restore_global_rng=True)
    generation = runner.completed_generations + 1
    arm = design['arms'][args.name]
    if generation != args.generation or generation > design['generations']:
        raise ValueError('Generation differs from the declared schedule')
    before, events, marks = state(runner), Counter(), {}
    clock = dict(update_seconds=0.0, started=None)
    origin, cpu_origin = time.perf_counter(), cpu_seconds()

    def progress(event):
        now = time.perf_counter()
        events[event] += 1
        if event == 'before_update':
            clock['started'] = now
        elif event == 'after_update':
            clock['update_seconds'] += now - clock['started']
        if events[event] == 1:
            marks[event] = dict(wall_seconds=now - origin, cpu_seconds=cpu_seconds() - cpu_origin)
        # Durable progress at a coarse grain only, so fsync never dominates a measured interval.
        if (event == 'game_finalized' and events[event] % 16 == 0) or (event == 'after_update' and events[event] % 32 == 0):
            write(output / 'progress.json', dict(events=events, elapsed_seconds=now - origin))
    path = output / f'generation-{generation:04d}.pt'
    result = runner.run_generation(path, check=check, progress=progress)
    if (result['games'], result['updates']) != (arm['games_per_generation'], arm['optimizer_updates']):
        raise ValueError('Generation work differs from the frozen budget')
    after = state(runner)
    reloaded = load(path, result['boundary_sha256'], design, args.seed, args.name, restore_global_rng=False)
    if state(reloaded) != after or sha(path) != result['boundary_sha256']:
        raise ValueError('Strict reload differs from the live runner')
    collection_cpu = marks['validation']['cpu_seconds'] - marks['collection']['cpu_seconds']
    write(output / 'complete.json', dict(
        complete=True, seed=args.seed, architecture=args.name, generation=generation, boundary=str(path),
        sha256=result['boundary_sha256'], parent_boundary_sha256=args.sha256, before=before, after=after,
        result=result, events=events, stage_marks=marks,
        timing=dict(collection_seconds=result['resources']['collection_seconds'],
                    collection_cpu_seconds=collection_cpu, update_seconds=clock['update_seconds'],
                    training_and_persistence_seconds=result['resources']['training_and_persistence_seconds'],
                    generation_wall_seconds=result['resources']['wall_seconds'])))


def frozen(args, design):
    """Independent frozen model of a boundary plus its factory; global RNG untouched."""
    runner = load(args.boundary, args.sha256, design, args.seed, args.name, restore_global_rng=False)
    if runner.completed_generations != args.generation:
        raise ValueError('Boundary generation differs from the declared one')
    model = runner.factory.freeze(runner.model)
    return runner, model, runner.factory.inference(model), runner.factory.hash(model)


def positions(args, design, check, output):
    runner, model, inference, weights = frozen(args, design)
    plan = design['evaluation']
    evaluate.positions(inference, plan['simulations'], plan['search_seed'], check, output)
    if runner.factory.hash(model) != weights:
        raise ValueError('Evaluation mutated checkpoint weights')
    write(output / 'identity.json', dict(seed=args.seed, architecture=args.name, generation=args.generation,
                                         boundary_sha256=args.sha256, weights_sha256=weights, counters=runner.counters()))


def matches(args, design, check, output):
    runner, model, inference, weights = frozen(args, design)
    plan = design['evaluation']
    rows = [r for r in evaluate.openings() if r['id'] in set(plan['openings'])]
    if [r['id'] for r in rows] != plan['openings']:
        raise ValueError('Opening families differ from the frozen list')
    agent = evaluate.RawAgent(inference, plan['simulations'], name=f'{arm_label(args.seed, args.name)}-g{args.generation}',
                              weights_sha256=weights)
    evaluate.matches(agent, plan['opponent_depth'], plan['match_seed'], check, output, rows=rows)
    if runner.factory.hash(model) != weights:
        raise ValueError('Evaluation mutated checkpoint weights')
    write(output / 'identity.json', dict(seed=args.seed, architecture=args.name, generation=args.generation,
                                         boundary_sha256=args.sha256, weights_sha256=weights, counters=runner.counters()))


def dynamics(args, design, check, output):
    rows, previous, runner = [], None, None
    for entry in read(args.boundaries):
        check()
        runner = load(entry['path'], entry['sha256'], design, args.seed, args.name, restore_global_rng=False)
        generation = runner.completed_generations
        if generation != entry['generation']:
            raise ValueError('Boundary generation differs from the ledger')
        model = runner.factory.freeze(runner.model)
        weights = runner.factory.hash(model)
        row = dict(generation=generation, counters=runner.counters(), boundary_sha256=entry['sha256'],
                   weights_sha256=weights, raw=evaluate.raw_suites(runner.factory.inference(model), check),
                   replay=measures.replay_composition(runner))
        if generation:
            games = [g for g in runner.replay.iter_games() if g.generation == generation]
            examples = [e for g in games for e in g.examples]
            row['self_play'] = measures.diversity(games)
            scored, data, logits, values = measures.score_examples(model, measures.probe_sample(runner, generation))
            row['train_probe'] = measures.summary(scored)
            row['batch_norm'] = measures.batch_norm_probe(model, data, logits, values)
            row['newest_generation_after_training'] = measures.summary(measures.score_examples(model, examples)[0])
            # Prequential held-out: the previous network on the games it generated and was never trained on.
            earlier_model, earlier_keys, earlier = previous
            scored, data, logits, values = measures.score_examples(earlier_model, examples)
            novel = measures.torch.tensor([measures.family(e) not in earlier_keys for e in examples])
            earlier['heldout_next_generation'] = dict(all=measures.summary(scored), novel=measures.summary(scored, novel),
                                                      novel_share=float(novel.double().mean()))
            if 'train_probe' in earlier:
                earlier['generalization_gap'] = {
                    k: earlier['heldout_next_generation']['all'][k] - earlier['train_probe'][k]
                    for k in ('policy_ce', 'value_mse')}
            else:  # generation 0: the untrained running statistics against real self-play positions
                earlier['batch_norm'] = measures.batch_norm_probe(earlier_model, data, logits, values)
        if runner.factory.hash(model) != weights:
            raise ValueError('Diagnostics mutated checkpoint weights')
        rows.append(row)
        previous = (model, {measures.family(e) for g in runner.replay.iter_games() for e in g.examples}, row)
    write(output / 'dynamics.json', dict(complete=True, seed=args.seed, architecture=args.name, rows=rows,
                                         history=runner.history, resources=runner.resources))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('train', 'positions', 'matches', 'dynamics'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--name', choices=tuple(IDENTITY), required=True)
    parser.add_argument('--generation', type=int)
    parser.add_argument('--boundary')
    parser.add_argument('--sha256')
    parser.add_argument('--boundaries')
    args = parser.parse_args()
    identity = runtime()
    root = Path(args.root)
    design = read(root / 'design.json')
    if design['smoke']:
        if root.name != 'smoke':
            raise ValueError('A smoke design may only run in the smoke directory')
    else:
        enforce()
        if sha(root / 'design.json') != sha(RESULTS / 'design.json'):
            raise ValueError('Run design differs from the frozen design')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started, cpu = time.perf_counter(), cpu_seconds()

    def stop(*_):
        raise InterruptedError('Cooperative cancellation (signal)')
    signal.signal(signal.SIGTERM, stop)
    check = Deadline(args.deadline, root / 'CANCEL')
    check()
    globals()[args.task](args, design, check, output)
    check()
    write(output / 'task-complete.json', dict(complete=True, task=args.task, seed=args.seed, architecture=args.name,
                                              generation=args.generation, wall_seconds=time.perf_counter() - started,
                                              cpu_seconds=cpu_seconds() - cpu, peak_rss_mib=peak_rss_mib(),
                                              runtime_sha256=__import__('hashlib').sha256(repr(sorted(identity.items())).encode()).hexdigest()))


if __name__ == '__main__':
    main()
