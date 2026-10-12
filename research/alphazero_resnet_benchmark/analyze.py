"""Evidence-only analysis of a completed run: cost summaries, learning contrasts and uncertainty.

Reads the run directory, never trains or evaluates a model, and writes compact
tracked evidence. Every contrast is ResNet minus CNN on identical positions.
Uncertainty resamples whole opening families (held-out replay) or whole
development clusters (solved300/tactical200); it is conditional on these six
trained networks and says nothing about other initializations.
"""
import argparse
import statistics

import numpy as np

from .common import NAMES, OUTPUT, RESULTS, SEEDS, read, write

RESNETS = ('resnet3x64', 'resnet5x64')
LOWER_IS_BETTER = {'combined': True, 'policy_ce': True, 'value_mse': True, 'top1': False,
                   'accuracy': False, 'optimal_set_ce': True, 'unique_win_accuracy': False,
                   'unique_safe_accuracy': False, 'mistake_rate': True}


# Cost benchmarks ------------------------------------------------------------------------------

def inference_summary(stage):
    table = {}
    rows = stage['timing']['rows']
    for path in ('forward', 'predict_batch'):
        for batch in sorted({r['batch'] for r in rows}):
            cell = {}
            reference = {r['repetition']: r['mean_ms'] for r in rows
                         if r['path'] == path and r['batch'] == batch and r['architecture'] == 'cnn'}
            for name in NAMES:
                chosen = [r for r in rows if r['path'] == path and r['batch'] == batch and r['architecture'] == name]
                means = [r['mean_ms'] for r in chosen]
                ratios = [r['mean_ms'] / reference[r['repetition']] for r in chosen]
                cell[name] = dict(repetitions=len(chosen), latency_ms=statistics.median(means), latency_ms_min=min(means),
                                  latency_ms_max=max(means), p95_ms=statistics.median(r['p95_ms'] for r in chosen),
                                  positions_per_second=batch / statistics.median(means) * 1e3,
                                  cost_ratio_to_cnn=statistics.median(ratios), cost_ratio_min=min(ratios),
                                  cost_ratio_max=max(ratios))
            table[f'{path}:{batch}'] = cell
    memory = {f"{r['architecture']}:{r['batch']}": dict(peak_rss_mib=r['peak_rss_mib'], model_rss_mib=r['model_peak_rss_mib'],
                                                        import_rss_mib=r['imported_peak_rss_mib'],
                                                        state_megabytes=r['state_megabytes']) for r in stage['memory']}
    return dict(latency=table, memory=memory)


def training_summary(stage):
    result = {}
    for name in NAMES:
        full = [r for r in stage['rows'] if r['architecture'] == name and not r['stripped_batch_norm']]
        stripped = [r for r in stage['rows'] if r['architecture'] == name and r['stripped_batch_norm']]

        def average(rows, getter):
            return statistics.fmean(getter(r) for r in rows)
        stages = {k: average(full, lambda r, k=k: r['staged'][k]) for k in full[0]['staged']}
        total = average(full, lambda r: r['staged_total_ms'])
        row = dict(passes=len(full), update_ms=average(full, lambda r: r['trainer_step']['mean_ms']),
                   update_ms_by_pass=[r['trainer_step']['mean_ms'] for r in full],
                   update_median_ms=average(full, lambda r: r['trainer_step']['median_ms']),
                   update_p95_ms=average(full, lambda r: r['trainer_step']['p95_ms']),
                   examples_per_second=128e3 / average(full, lambda r: r['trainer_step']['mean_ms']),
                   staged_ms=stages, staged_total_ms=total, shares={k: v / total for k, v in stages.items()},
                   eval_forward_ms=average(full, lambda r: r['eval_forward_ms']),
                   train_forward_ms=average(full, lambda r: r['train_forward_ms']),
                   peak_rss_mib=max(r['peak_rss_mib'] for r in full),
                   import_rss_mib=statistics.fmean(r['imported_peak_rss_mib'] for r in full))
        if stripped:
            bare = {k: average(stripped, lambda r, k=k: r['staged'][k]) for k in stages}
            bare_total = average(stripped, lambda r: r['staged_total_ms'])
            row['batch_norm'] = dict(
                update_ms_without=bare_total, update_overhead_ms=total - bare_total, update_overhead_share=(total - bare_total) / total,
                forward_overhead_ms=stages['forward'] - bare['forward'], backward_overhead_ms=stages['backward'] - bare['backward'],
                optimizer_overhead_ms=stages['optimizer_step'] - bare['optimizer_step'],
                eval_forward_overhead_ms=row['eval_forward_ms'] - average(stripped, lambda r: r['eval_forward_ms']),
                train_minus_eval_forward_ms=row['train_forward_ms'] - row['eval_forward_ms'])
        result[name] = row
    for name in NAMES:
        result[name]['cost_ratio_to_cnn'] = result[name]['update_ms'] / result['cnn']['update_ms']
    return result


def collection_summary(stage, reference_plies):
    result = {}
    for name in NAMES:
        rows = [r for r in stage['rows'] if r['architecture'] == name]
        if not all(r['complete'] for r in rows):
            raise ValueError('Incomplete collection block')
        total = {k: sum(r[k] for r in rows) for k in ('wall_seconds', 'cpu_seconds', 'forward_calls', 'neural_positions',
                                                      'inference_seconds', 'game_work_seconds', 'simulations',
                                                      'terminal_leaves', 'plies', 'games')}
        lengths = [n for r in rows for n in r['game_lengths']]
        sizes = {}
        for r in rows:
            for size, count in r['batch_sizes'].items():
                sizes[size] = sizes.get(size, 0) + count
        per_position = total['wall_seconds'] / total['neural_positions']
        per_ply = total['wall_seconds'] / total['plies']
        result[name] = dict(
            **total, blocks=len(rows), games_per_second=total['games'] / total['wall_seconds'],
            neural_positions_per_second=total['neural_positions'] / total['wall_seconds'],
            simulations_per_second=total['simulations'] / total['wall_seconds'],
            seconds_per_neural_position=per_position, seconds_per_ply=per_ply,
            neural_positions_per_ply=total['neural_positions'] / total['plies'],
            mean_batch_occupancy=total['neural_positions'] / total['forward_calls'],
            full_batch_share_of_calls=sizes.get('8', 0) / total['forward_calls'],
            full_batch_share_of_positions=8 * sizes.get('8', 0) / total['neural_positions'],
            batch_sizes=dict(sorted(sizes.items(), key=lambda kv: int(kv[0]))),
            inference_share_of_wall=total['inference_seconds'] / total['wall_seconds'],
            mean_game_length=statistics.fmean(lengths), min_game_length=min(lengths), max_game_length=max(lengths),
            length_normalized_games_per_second=1 / (per_ply * reference_plies),
            by_block=[dict(game_ids=r['game_ids'], wall_seconds=r['wall_seconds'], games_per_second=r['games'] / r['wall_seconds'],
                           seconds_per_neural_position=r['wall_seconds'] / r['neural_positions'],
                           mean_batch=r['mean_batch'], games_sha256=r['games_sha256']) for r in rows],
            peak_rss_mib=max(r['peak_rss_mib'] for r in rows), expected_simulations=256 * total['plies'],
            outcomes={str(w): sum(x == w for r in rows for x in r['winners']) for w in (0, 1, -1)})
        if total['simulations'] != 256 * total['plies']:
            raise ValueError('Simulation count differs from 256 per move')
    for name in NAMES:
        for key in ('seconds_per_neural_position', 'seconds_per_ply'):
            result[name][f'{key}_ratio_to_cnn'] = result[name][key] / result['cnn'][key]
        result[name]['games_per_second_ratio_to_cnn'] = result[name]['games_per_second'] / result['cnn']['games_per_second']
    return result


# Learning ---------------------------------------------------------------------------------------

def row_at(run, updates):
    for row in run['milestones']:
        if row['updates'] == updates:
            return row
    raise ValueError(f"No milestone at {updates} updates for {run['architecture']}")


def interpolate(a, b, weight):
    """Linear in optimizer seconds between two adjacent milestones, for numbers and nested numeric maps."""
    if isinstance(a, dict):
        return {k: interpolate(a[k], b[k], weight) for k in a if k in b and isinstance(a[k], (dict, int, float, list))
                and not isinstance(a[k], bool)}
    if isinstance(a, list):
        return [interpolate(x, y, weight) for x, y in zip(a, b)]
    return a + (b - a) * weight


def view(row):
    """The quantities compared between architectures at one checkpoint."""
    held, suites = row['held_out'], row['suites']
    return dict(families=held['novel_families'],
                novel=dict(policy_ce=held['novel']['policy_ce'], value_mse=held['novel']['value_mse'],
                           top1=held['novel']['top1'], combined=held['novel']['policy_ce'] + held['novel']['value_mse']),
                solved=dict(accuracy=suites['solved']['accuracy'], optimal_set_ce=suites['solved']['optimal_set_ce'],
                            value_mse=suites['solved']['value_mse'], mistake_rate=suites['solved']['mistake_rate']),
                tactical=dict(accuracy=suites['tactical']['accuracy'],
                              unique_win_accuracy=suites['tactical']['unique_win_accuracy'],
                              unique_safe_accuracy=suites['tactical']['unique_safe_accuracy']))


def at_seconds(run, seconds):
    """Checkpoint view at an optimizer-time budget; None when the run never reached it."""
    rows = run['milestones']
    if seconds > rows[-1]['train_seconds']:
        return None
    for a, b in zip(rows, rows[1:]):
        if a['train_seconds'] <= seconds <= b['train_seconds']:
            weight = (seconds - a['train_seconds']) / (b['train_seconds'] - a['train_seconds'])
            result = interpolate(view(a), view(b), weight)
            result['updates'] = a['updates'] + (b['updates'] - a['updates']) * weight
            return result
    raise ValueError('Milestone times are not increasing')


def family_arrays(view_a, view_b):
    keys = sorted(view_a['families'], key=int)
    if keys != sorted(view_b['families'], key=int):
        raise ValueError('Architectures were scored on different families')
    return (np.array([view_a['families'][k] for k in keys], dtype=np.float64),
            np.array([view_b['families'][k] for k in keys], dtype=np.float64))


def family_statistics(weights, a, b):
    """ResNet minus CNN per replicate: columns policy CE, value MSE, top-1 agreement, combined loss."""
    count = weights @ a[:, 0]
    means_a, means_b = (weights @ a[:, 1:]) / count[:, None], (weights @ b[:, 1:]) / count[:, None]
    delta = means_a - means_b
    return np.column_stack([delta, delta[:, 0] + delta[:, 1]])


FAMILY_METRICS = ('policy_ce', 'value_mse', 'top1', 'combined')


def held_out_contrast(views, replicates, seed):
    """views: {seed: (resnet view, cnn view)}. Per-seed and equal-seed-weight pooled contrasts."""
    result, pooled = {}, None
    for index, (training_seed, (resnet, cnn)) in enumerate(sorted(views.items())):
        a, b = family_arrays(resnet, cnn)
        rng = np.random.Generator(np.random.PCG64([seed, training_seed]))
        weights = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=replicates).astype(np.float64)
        samples = family_statistics(weights, a, b)
        point = family_statistics(np.ones((1, len(a))), a, b)[0]
        result[str(training_seed)] = {m: dict(difference=float(point[k]), low=float(np.percentile(samples[:, k], 2.5)),
                                              high=float(np.percentile(samples[:, k], 97.5)), families=len(a),
                                              positions=int(a[:, 0].sum())) for k, m in enumerate(FAMILY_METRICS)}
        pooled = samples if pooled is None else pooled + samples
    pooled = pooled / len(views)
    result['pooled'] = {m: dict(difference=statistics.fmean(result[str(s)][m]['difference'] for s in views),
                                low=float(np.percentile(pooled[:, k], 2.5)), high=float(np.percentile(pooled[:, k], 97.5)))
                        for k, m in enumerate(FAMILY_METRICS)}
    return result


SUITE_METRICS = {'solved': (('accuracy', 'correct', None), ('optimal_set_ce', 'ce', None), ('value_mse', 'se', None),
                            ('mistake_rate', 'mistake', None)),
                 'tactical': (('accuracy', 'correct', None), ('unique_win_accuracy', 'correct', 'unique_win'),
                              ('unique_safe_accuracy', 'correct', 'unique_safe'), ('avoidable_rate', 'avoidable', None))}


def suite_contrast(rows, suite, replicates, seed):
    """rows: {seed: (resnet per-row records, cnn per-row records)}; clusters are shared by both seeds."""
    result = {}
    for metric, field, category in SUITE_METRICS[suite]:
        contrasts, clusters = {}, None
        for training_seed, (resnet, cnn) in sorted(rows.items()):
            if [r['id'] for r in resnet] != [r['id'] for r in cnn]:
                raise ValueError('Suite rows differ between architectures')
            chosen = [(a, b) for a, b in zip(resnet, cnn) if category is None or a['category'] == category]
            contrasts[training_seed] = np.array([a[field] - b[field] for a, b in chosen], dtype=np.float64)
            clusters = np.array([a['cluster'] for a, _ in chosen])
        ids = np.unique(clusters)
        membership = (clusters[None, :] == ids[:, None]).astype(np.float64)   # clusters x rows
        rng = np.random.Generator(np.random.PCG64([seed, len(ids), len(metric)]))
        weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=replicates).astype(np.float64)
        row_weights = weights @ membership
        entry = {}
        stacked = np.mean([contrasts[s] for s in sorted(contrasts)], axis=0)
        for label, values in [*((str(s), contrasts[s]) for s in sorted(contrasts)), ('pooled', stacked)]:
            samples = (row_weights @ values) / row_weights.sum(1)
            entry[label] = dict(difference=float(values.mean()), low=float(np.percentile(samples, 2.5)),
                                high=float(np.percentile(samples, 97.5)), rows=len(values), clusters=len(ids))
        result[metric] = entry
    return result


def verdict(entry, metric, seeds):
    """Declared rule: both seeds agree in sign and the pooled interval excludes zero."""
    sign = -1 if LOWER_IS_BETTER.get(metric, True) else 1
    points = [sign * entry[str(s)]['difference'] for s in seeds]
    low, high = sorted((sign * entry['pooled']['low'], sign * entry['pooled']['high']))
    if all(p > 0 for p in points) and low > 0:
        return 'BETTER'
    if all(p < 0 for p in points) and high < 0:
        return 'WORSE'
    return 'INCONCLUSIVE'


def compact(row):
    """A milestone without per-family sums, per-row suite records or per-layer statistics."""
    held, suites = row['held_out'], row['suites']
    norm = held['batch_norm']
    return dict(
        updates=row['updates'], train_seconds=row['train_seconds'], step_seconds=row['step_seconds'],
        evaluation_seconds=row['evaluation_seconds'], weights_sha256=row['weights_sha256'],
        sampling_sha256=row['sampling_sha256'],
        novel={k: held['novel'][k] for k in ('positions', 'policy_ce', 'policy_kl', 'value_mse', 'top1')},
        all={k: held['all'][k] for k in ('positions', 'policy_ce', 'value_mse', 'top1')},
        seen={k: held['seen'][k] for k in ('positions', 'policy_ce', 'value_mse', 'top1')},
        novel_by_ply=held['novel_by_ply'], train_probe=held['train_probe'], generalization_gap=held['generalization_gap'],
        calibration=dict(policy_ece=held['calibration']['policy']['ece'], value_error=held['calibration']['value']['error'],
                         value_saturated_share=held['calibration']['value']['saturated_share'],
                         mean_abs_value=held['calibration']['value']['mean_abs_value']),
        batch_norm=None if norm is None else {k: norm[k] for k in norm if k != 'layers'},
        solved={k: v for k, v in suites['solved'].items() if k not in ('rows', 'calibration')}
        | dict(policy_ece=suites['solved']['calibration']['policy']['ece'],
               value_error=suites['solved']['calibration']['value']['error']),
        tactical={k: v for k, v in suites['tactical'].items() if k != 'rows'},
        optimization=row['optimization'])


def learning(run_directory, design, seeds):
    plan, analysis = design['learning'], design['analysis']
    replicates, seed = analysis['bootstrap_replicates'], analysis['bootstrap_seed']
    runs = {s: {n: read(run_directory / f'learning-seed-{s}' / f'{n}.json') for n in NAMES} for s in seeds}
    summaries = {s: read(run_directory / f'learning-seed-{s}' / 'summary.json') for s in seeds}
    common = plan['common_updates']
    # Identical minibatch and reflection identities at every shared update count.
    identity = {}
    for s in seeds:
        shared = {}
        for milestone in plan['milestones']:
            hashes = {n: row_at(runs[s][n], milestone)['sampling_sha256'] for n in NAMES if milestone <= runs[s][n]['updates']}
            if len(hashes) > 1:
                shared[str(milestone)] = len(set(hashes.values())) == 1
        identity[str(s)] = dict(all_shared_milestones_identical=all(shared.values()), milestones=shared)
        if not all(shared.values()):
            raise ValueError('Architectures did not receive identical minibatches')
    curves = {str(s): {n: dict(updates=runs[s][n]['updates'], train_seconds=runs[s][n]['train_seconds'],
                                step_seconds=runs[s][n]['step_seconds'], evaluation_seconds=runs[s][n]['evaluation_seconds'],
                                wall_seconds=runs[s][n]['wall_seconds'], cpu_seconds=runs[s][n]['cpu_seconds'],
                                seconds_per_update=runs[s][n]['train_seconds'] / runs[s][n]['updates'],
                                parameters=runs[s][n]['parameters'], initial_weights_sha256=runs[s][n]['initial_weights_sha256'],
                                final_weights_sha256=runs[s][n]['final_weights_sha256'], models=runs[s][n]['models'],
                                optimizer=runs[s][n]['optimizer'],
                                milestones=[compact(r) for r in runs[s][n]['milestones']]) for n in NAMES} for s in seeds}
    comparisons = {}

    def compare(label, kind, views, rows=None):
        entry = dict(kind=kind, checkpoints={str(s): {n: dict(updates=views[s][n].get('updates'), **views[s][n]['novel'],
                                                             solved_accuracy=views[s][n]['solved']['accuracy'],
                                                             tactical_accuracy=views[s][n]['tactical']['accuracy'])
                                                      for n in NAMES} for s in seeds}, resnets={})
        for name in RESNETS:
            held = held_out_contrast({s: (views[s][name], views[s]['cnn']) for s in seeds}, replicates, seed)
            block = dict(held_out=held, verdicts={m: verdict({k: held[k][m] for k in held}, m, seeds) for m in FAMILY_METRICS})
            if rows is not None:
                for suite in ('solved', 'tactical'):
                    block[suite] = suite_contrast({s: (rows[s][name][suite], rows[s]['cnn'][suite]) for s in seeds},
                                                  suite, replicates, seed)
                    block['verdicts'].update({f'{suite}_{m}': verdict(block[suite][m], m, seeds) for m in block[suite]})
            else:
                # Interpolated checkpoints have no per-row suite records: point differences only.
                block['suite_point_differences'] = {
                    suite: {m: {str(s): views[s][name][suite][m] - views[s]['cnn'][suite][m] for s in seeds}
                            for m in views[seeds[0]][name][suite]} for suite in ('solved', 'tactical')}
            entry['resnets'][name] = block
        comparisons[label] = entry

    def milestone_views(updates_of):
        views, rows = {}, {}
        for s in seeds:
            views[s], rows[s] = {}, {}
            for n in NAMES:
                row = row_at(runs[s][n], updates_of(n))
                views[s][n] = dict(view(row), updates=row['updates'])
                rows[s][n] = {suite: row['suites'][suite].get('rows') for suite in ('solved', 'tactical')}
        return views, rows

    for point in plan['equal_update_points']:
        views, rows = milestone_views(lambda n: point)
        compare(f'equal_updates_{point}', 'equal optimizer updates', views, rows if point == common else None)
    views, rows = milestone_views(lambda n: plan['updates'][n])
    compare('equal_wall_final', 'final models at the declared equal nominal optimizer time', views, rows)
    limit = {s: min(runs[s][n]['train_seconds'] for n in NAMES) for s in seeds}
    for point in [*plan['equal_time_points_seconds'], 'matched']:
        views = {s: {n: at_seconds(runs[s][n], limit[s] if point == 'matched' else point) for n in NAMES} for s in seeds}
        if all(views[s][n] is not None for s in seeds for n in NAMES):
            compare(f'equal_seconds_{point}', 'linear interpolation in measured optimizer seconds', views)
            comparisons[f'equal_seconds_{point}']['seconds'] = {str(s): limit[s] if point == 'matched' else point for s in seeds}
    return dict(curves=curves, comparisons=comparisons, sampling_identity=identity,
                summaries={str(s): summaries[s] for s in seeds},
                endpoint_detail={str(s): {n: {str(u): dict(
                    batch_norm_layers=None if row_at(runs[s][n], u)['held_out']['batch_norm'] is None
                    else row_at(runs[s][n], u)['held_out']['batch_norm']['layers'],
                    policy_calibration=row_at(runs[s][n], u)['held_out']['calibration']['policy'],
                    value_calibration=row_at(runs[s][n], u)['held_out']['calibration']['value'],
                    solved_calibration=row_at(runs[s][n], u)['suites']['solved']['calibration'])
                    for u in sorted({0, common, runs[s][n]['updates']})} for n in NAMES} for s in seeds})


def analyze(run_directory, output, design, seeds):
    benchmarks = dict(inference=inference_summary(read(run_directory / 'inference.json')),
                      training=training_summary(read(run_directory / 'training.json')),
                      collection=collection_summary(read(run_directory / 'collection.json'),
                                                    design['collection']['reference_plies']))
    write(output / 'benchmarks.json', benchmarks)
    learned = learning(run_directory, design, seeds)
    write(output / 'learning.json', dict(curves=learned['curves'], sampling_identity=learned['sampling_identity'],
                                         summaries=learned['summaries']))
    write(output / 'comparisons.json', learned['comparisons'])
    write(output / 'endpoint-detail.json', learned['endpoint_detail'])
    return benchmarks, learned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=str(OUTPUT / 'run'))
    parser.add_argument('--output', default=str(RESULTS))
    args = parser.parse_args()
    from pathlib import Path
    run_directory = Path(args.run)
    status = read(run_directory / 'status.json')
    if status['status'] != 'COMPLETE_PENDING_ANALYSIS':
        raise SystemExit(f"Refusing to analyze an incomplete run: {status['status']}")
    analyze(run_directory, Path(args.output), read(RESULTS / 'design.json'), SEEDS)


if __name__ == '__main__':
    main()
