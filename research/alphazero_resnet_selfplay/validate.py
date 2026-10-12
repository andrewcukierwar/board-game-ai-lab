"""Pre-freeze validation: model-factory and boundary compatibility, and fresh-process resume.

Runs after the pipeline smoke test and before the freeze. The smoke run
produced every generation of every arm in its own fresh interpreter through
the real worker. Here each arm is replayed from its generation-zero boundary
in ONE uninterrupted interpreter, and the two chains must agree bit for bit at
every generation: weights, batch-norm buffers, optimizer, replay, counters and
every random stream. That is the fresh-process resume guarantee for exactly
the code path the measured run uses, for both architectures.
"""
import tempfile
import time
from pathlib import Path

from .common import (ARMS, IDENTITY, NAMES, OUTPUT, PARAMETERS, RESULTS, Deadline, arm_label, experimental_config,
                     preserve, read, runtime, text_sha, write)

SMOKE = OUTPUT / 'smoke'


def comparable(runner):
    from games.connect4.alphazero_v2 import experimental as E
    content = runner._content()
    content.pop('resources')   # measured seconds differ between runs by construction
    return dict(content=E.digest(content), model=runner.factory.hash(runner.model),
                optimizer=E.digest(runner.trainer.optimizer.state_dict()), replay=runner.replay.digest(),
                counters=runner.counters(), global_rng=E.digest(runner.global_rng),
                sampling=text_sha(repr(runner.sampling_rng.getstate())),
                augmentation=text_sha(repr(runner.augmenter.rng.getstate())),
                buffers=text_sha({n: b.tolist() for n, b in runner.model.named_buffers()}))


def main():
    from games.connect4.alphazero_v2 import experimental as E
    from games.connect4.alphazero_v2.experimental_models import SUPPORTED_ARCHITECTURES, model_factory
    runtime()
    official = preserve()
    started = time.time()
    record = []

    def check(name, condition, detail=None):
        record.append(dict(check=name, passed=bool(condition), detail=detail))
        if not condition:
            raise AssertionError(f'{name}: {detail}')

    # 1. Model factory.
    for name in NAMES:
        factory = model_factory(IDENTITY[name])
        model = factory.construct(42)
        check(f'{name}: registered architecture identity', IDENTITY[name] in SUPPORTED_ARCHITECTURES
              and factory.architecture == IDENTITY[name])
        check(f'{name}: parameter count', sum(p.numel() for p in model.parameters()) == PARAMETERS[name])
        check(f'{name}: seed-determined initialization', factory.hash(model) == factory.hash(factory.construct(42))
              and factory.hash(model) != factory.hash(factory.construct(314159)))
        other = model_factory(IDENTITY[[n for n in NAMES if n != name][0]])
        refused = 0
        for action in (lambda: other.state(model), lambda: other.freeze(model), lambda: other.inference(model),
                       lambda: other.restore(factory.state(model))):
            try:
                action()
            except Exception:
                refused += 1
        check(f'{name}: the other architecture adapter refuses this model', refused == 4, refused)

    # 2. The smoke run itself.
    status, design = read(SMOKE / 'status.json'), read(SMOKE / 'design.json')
    analysis = read(SMOKE / 'analysis.json')
    check('smoke: supervisor completed', status['status'] == 'COMPLETE_PENDING_ANALYSIS', status['status'])
    check('smoke: every task complete, none retried', all(t['state'] == 'COMPLETE' for t in status['tasks'])
          and status['retries'] == 0, len(status['tasks']))
    check('smoke: integrity checks passed', all(v == 'PASS' or v.startswith('NOT APPLICABLE')
                                               for v in status['integrity'].values()), status['integrity'])
    check('smoke: frozen analysis ran end to end', {'strength', 'tactical_puct', 'value', 'equal_compute_claim',
                                                    'recommendation_rule'} <= set(analysis['verdicts']),
          sorted(analysis['verdicts']))
    expected_tasks = len(ARMS) * (design['generations'] + 1 + 2 * len(design['evaluation']['generations']))
    check('smoke: task count equals the declared schedule', len(status['tasks']) == expected_tasks, len(status['tasks']))

    # 3. Boundary compatibility and fresh-process resume against one uninterrupted interpreter.
    chains = {}
    for seed, name in ARMS:
        label = arm_label(seed, name)
        config = experimental_config(design, seed, name)
        rows = status['boundaries'][label]
        zero = E.load_experimental(rows['0']['path'], expected_sha256=rows['0']['sha256'], expected_config=config,
                                   restore_global_rng=False)
        check(f'{label}: generation zero has empty replay and zero optimizer steps',
              zero.counters() == dict(completed_generations=0, total_games=0, total_positions=0, trainer_steps=0)
              and len(zero.replay) == 0 and not zero.trainer.optimizer.state_dict()['state'] and zero.parent is None)
        check(f'{label}: generation zero is the fresh seed-determined initialization',
              zero.factory.hash(zero.model) == zero.factory.hash(zero.factory.construct(seed)))
        wrong = experimental_config(design, seed, [n for n in NAMES if n != name][0])
        try:
            E.load_experimental(rows['0']['path'], expected_sha256=rows['0']['sha256'], expected_config=wrong,
                                restore_global_rng=False)
            refused = False
        except ValueError:
            refused = True
        check(f'{label}: boundary refuses the other architecture\'s configuration', refused)
        try:
            E.load_experimental(rows['0']['path'], expected_sha256='0' * 64, restore_global_rng=False)
            refused = False
        except ValueError:
            refused = True
        check(f'{label}: boundary refuses a wrong file hash', refused)
        with tempfile.TemporaryDirectory(prefix=f'phase4d4h-c-resume-{label}-') as temporary:
            runner = E.load_experimental(rows['0']['path'], expected_sha256=rows['0']['sha256'], expected_config=config)
            chain = []
            for generation in range(1, design['generations'] + 1):
                runner.run_generation(Path(temporary) / f'generation-{generation}.pt')
                fresh = E.load_experimental(rows[str(generation)]['path'], expected_sha256=rows[str(generation)]['sha256'],
                                            expected_config=config, restore_global_rng=False)
                same = comparable(runner) == comparable(fresh)
                check(f'{label}: generation {generation} from fresh interpreters is bit-identical to one uninterrupted '
                      f'interpreter', same, dict(uninterrupted=comparable(runner), fresh=comparable(fresh)))
                if name != 'cnn':
                    steps = fresh.counters()['trainer_steps']
                    check(f'{label}: generation {generation} batch-norm counters equal trainer steps',
                          all(int(b) == steps for n, b in fresh.model.named_buffers() if n.endswith('num_batches_tracked')))
                chain.append(comparable(fresh))
            try:
                runner.run_generation(Path(temporary) / 'beyond.pt')
                refused = False
            except RuntimeError:
                refused = True
            check(f'{label}: a generation beyond the declaration is refused', refused)
            chains[label] = chain

    # 4. Cooperative cancellation and the deadline.
    with tempfile.TemporaryDirectory(prefix='phase4d4h-c-cancel-') as temporary:
        flag = Path(temporary) / 'CANCEL'
        Deadline(time.time() + 60, flag)()
        flag.touch()
        try:
            Deadline(time.time() + 60, flag)()
            stopped = False
        except InterruptedError:
            stopped = True
        check('cancel file stops a worker check', stopped)
    try:
        Deadline(time.time() - 1)()
        stopped = False
    except TimeoutError:
        stopped = True
    check('passed deadline stops a worker check', stopped)
    check('official campaign unchanged', preserve() == official)

    trains = [t for t in status['tasks'] if t['kind'] == 'train']
    write(RESULTS / 'smoke.json', dict(
        status=status['status'], elapsed_seconds=status['elapsed_seconds'], tasks=len(status['tasks']),
        integrity=status['integrity'], design=design, work=status['work'],
        train_task_overhead_seconds=[t['wall_seconds'] - t['timing']['collection_seconds'] - t['timing']['update_seconds']
                                     for t in trains],
        note='Disposable tiny-budget pipeline test in a separate directory. Its networks, games and numbers are '
             'not results and are not used for any design choice other than confirming that the pipeline runs.'))
    write(RESULTS / 'validation.json', dict(passed=True, checks=record, count=len(record), chains=chains,
                                            wall_seconds=time.time() - started))
    print(f'{len(record)} checks passed in {time.time() - started:.1f}s')


if __name__ == '__main__':
    main()
