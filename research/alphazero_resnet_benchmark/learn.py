"""Fixed-data learning: train each architecture from scratch on one seed's frozen replay split.

One process per seed loads and verifies the frozen dataset once, then trains
the three architectures in the declared order. Minibatch identities and
reflection flags come from streams that depend only on the seed, so every
architecture receives the same batch at the same update number. Losses,
optimizer settings, gradient clipping and mode handling are the unmodified
factory trainer (V2Trainer.step / ResNetTrainer.step).
"""
import argparse
import hashlib
import json
import statistics
import time

import torch

from . import dataset, metrics
from .common import (IDENTITY, OUTPUT, RESULTS, Deadline, cpu_seconds, enforce, peak_rss_mib, preserve, read, runtime,
                     sha, write)

SAMPLING_DOMAIN = 'phase4d4h-b-fixed-data-sampling'


def window(series, start, stop):
    """Gradient and loss stability over the updates (start, stop]."""
    rows = series[start:stop]
    if not rows:
        return None
    norms = sorted(r[3] for r in rows)
    return dict(updates=len(rows), combined_loss_mean=statistics.fmean(r[0] for r in rows),
                policy_loss_mean=statistics.fmean(r[1] for r in rows), value_mse_mean=statistics.fmean(r[2] for r in rows),
                combined_loss_stdev=statistics.pstdev(r[0] for r in rows),
                gradient_norm_mean=statistics.fmean(norms), gradient_norm_median=statistics.median(norms),
                gradient_norm_p95=norms[min(len(norms) - 1, int(0.95 * len(norms)))], gradient_norm_max=norms[-1],
                gradient_norm_min=norms[0], clipped_share=statistics.fmean(r[4] for r in rows))


def train_architecture(name, seed, data, held, suites, plan, output, deadline):
    from games.connect4.alphazero_v2.config import V2Config
    from games.connect4.alphazero_v2.data import ReflectionAugmenter, seeded_rng
    from games.connect4.alphazero_v2.experimental_models import model_factory
    budget = plan['updates'][name]
    milestones = [m for m in plan['milestones'] if m <= budget]
    detailed = {0, plan['common_updates'], budget}
    if milestones[-1] != budget or plan['common_updates'] not in milestones:
        raise ValueError('Milestones must contain the common and final update counts')
    factory = model_factory(IDENTITY[name])
    config = V2Config(seed=seed)
    model = factory.construct(seed)
    trainer = factory.trainer(model, config)
    sampling = seeded_rng(SAMPLING_DOMAIN, seed)
    augmenter = ReflectionAugmenter(config.reflection_probability, seed=seed)
    examples = [e for e, _, _ in data['train']]
    count = len(examples)
    identity = hashlib.sha256()
    series, rows = [], []
    train_seconds = step_seconds = evaluation_seconds = 0.0
    started, cpu = time.perf_counter(), cpu_seconds()
    initial_hash = factory.hash(model)

    def milestone(updates, previous):
        nonlocal evaluation_seconds
        tick = time.perf_counter()
        before = factory.hash(model)
        scored = held.evaluate(model, detail=updates in detailed)
        positions = scored.pop('positions', None)
        suite = suites.evaluate(factory.inference(model))
        if updates not in detailed:
            for value in suite.values():
                value.pop('rows')
        if factory.hash(model) != before or model.training:
            raise RuntimeError('Evaluation changed the model')
        if positions is not None:
            torch.save(positions, output / f'{name}-positions-{updates:05d}.pt')
        seconds = time.perf_counter() - tick
        evaluation_seconds += seconds
        row = dict(updates=updates, train_seconds=train_seconds, step_seconds=step_seconds,
                   evaluation_seconds=seconds, sampling_sha256=identity.hexdigest(), weights_sha256=before,
                   held_out=scored, suites=suite, optimization=window(series, previous, updates),
                   reflected=augmenter.reflected, unreflected=augmenter.unreflected)
        rows.append(row)
        print(json.dumps(dict(seed=seed, architecture=name, updates=updates, train_seconds=round(train_seconds, 2),
                              novel_ce=round(scored['novel']['policy_ce'], 4), novel_mse=round(scored['novel']['value_mse'], 4),
                              probe_ce=round(scored['train_probe']['policy_ce'], 4),
                              solved=round(suite['solved']['accuracy'], 4), tactical=round(suite['tactical']['accuracy'], 4))),
              flush=True)

    milestone(0, 0)
    previous = 0
    for update in range(1, budget + 1):
        deadline()
        tick = time.perf_counter()
        positions = sampling.sample(range(count), config.batch_size)
        batch, flags = augmenter.batch([examples[p] for p in positions])
        inner = time.perf_counter()
        m = trainer.step(batch)
        done = time.perf_counter()
        train_seconds += done - tick
        step_seconds += done - inner
        identity.update(json.dumps([positions, flags]).encode())
        series.append((m['combined_loss'], m['policy_loss'], m['value_mse'], m['gradient_norm'], int(m['clipped'])))
        if update in milestones:
            milestone(update, previous)
            previous = update
            if update in (plan['common_updates'], budget):
                torch.save(factory.state(model), output / f'{name}-model-{update:05d}.pt')
    norms = [m.num_batches_tracked for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if trainer.steps != budget or trainer.failed or any(int(n) != budget for n in norms):
        raise RuntimeError('Training counters differ from the declared budget')
    result = dict(seed=seed, architecture=name, identity=IDENTITY[name], updates=budget, batch_size=config.batch_size,
                  parameters=sum(p.numel() for p in model.parameters()), initial_weights_sha256=initial_hash,
                  final_weights_sha256=factory.hash(model), train_seconds=train_seconds, step_seconds=step_seconds,
                  evaluation_seconds=evaluation_seconds, wall_seconds=time.perf_counter() - started,
                  cpu_seconds=cpu_seconds() - cpu, peak_rss_mib=peak_rss_mib(), milestones=rows,
                  optimizer=dict(type=type(trainer.optimizer).__name__, learning_rate=config.learning_rate,
                                 betas=list(config.adam_betas), eps=config.adam_eps, weight_decay=config.weight_decay,
                                 max_grad_norm=config.max_grad_norm, reflection_probability=config.reflection_probability),
                  models={str(u): sha(output / f'{name}-model-{u:05d}.pt') for u in sorted({plan['common_updates'], budget})},
                  series_columns=['combined_loss', 'policy_loss', 'value_mse', 'gradient_norm', 'clipped'])
    write(output / f'{name}.json', result)
    write(output / f'{name}-series.json', dict(columns=result['series_columns'], rows=series))
    return result


def run(seed, plan, output, deadline, *, verify_freeze=True):
    identity = runtime()
    if verify_freeze:
        enforce()
    before = preserve()
    output.mkdir(parents=True, exist_ok=False)
    started, cpu = time.perf_counter(), cpu_seconds()
    data = dataset.frozen(seed)
    loaded = time.perf_counter()
    held = metrics.HeldOut(data)
    suites = metrics.Suites(data['train_keys'])
    prepared = time.perf_counter()
    summary = dict(seed=seed, runtime=identity, dataset=dataset.identity(data), order=plan['order'][str(seed)],
                   load_seconds=loaded - started, preparation_seconds=prepared - loaded, architectures={})
    for name in plan['order'][str(seed)]:
        result = train_architecture(name, seed, data, held, suites, plan, output, deadline)
        summary['architectures'][name] = {k: result[k] for k in (
            'updates', 'train_seconds', 'step_seconds', 'evaluation_seconds', 'wall_seconds', 'cpu_seconds',
            'peak_rss_mib', 'final_weights_sha256')}
    if preserve() != before:
        raise ValueError('Official inventory changed during training')
    summary.update(wall_seconds=time.perf_counter() - started, cpu_seconds=cpu_seconds() - cpu,
                   peak_rss_mib=peak_rss_mib(), official_files=len(before))
    write(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int)
    parser.add_argument('--deadline', type=float, required=True, help='absolute epoch seconds')
    args = parser.parse_args()
    plan = read(RESULTS / 'design.json')['learning']
    run(args.seed, plan, OUTPUT / f'run/learning-seed-{args.seed}', Deadline(args.deadline))


if __name__ == '__main__':
    main()
