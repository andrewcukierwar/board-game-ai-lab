"""Evidence-only analysis, frozen before training. Reads the run directory; queries no model.

Every contrast is ResNet 3x64 minus CNN. Uncertainty is a paired cluster
bootstrap within seed (opening families for matches, Phase 4D.4G clusters for
position suites) and an equal-weight pool of the two seeds. Intervals are
conditional on these four trained networks: they do not include run-to-run
training variation.

The run's status and integrity checks are carried into the output. Verdicts
are computed whenever their evidence exists, and `verdicts_valid` says whether
the run completed with every integrity check passing; nothing is overridden.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from .common import ARMS, NAMES, OUTPUT, RESULTS, SEEDS, arm_label, read, write

RAW_METRICS = (  # (name, suite, row key, category filter, higher is better)
    ('solved_optimal_set_ce', 'solved', 'ce', None, False),
    ('solved_optimal_accuracy', 'solved', 'accuracy', None, True),
    ('solved_optimal_probability', 'solved', 'optimal_mass', None, True),
    ('solved_exact_value_mse', 'solved', 'se', None, False),
    ('solved_policy_deep_mistake', 'solved', 'policy_deep_mistake', None, False),
    ('deep_optimal_set_ce', 'deep', 'ce', None, False),
    ('deep_optimal_accuracy', 'deep', 'accuracy', None, True),
    ('deep_exact_value_mse', 'deep', 'se', None, False),
    ('deep_policy_deep_mistake', 'deep', 'policy_deep_mistake', None, False),
    ('tactical_policy_immediate_win', 'tactical', 'policy_correct', 'unique_win', True),
    ('tactical_policy_safe_response', 'tactical', 'policy_correct', 'unique_safe', True),
    ('tactical_policy_avoidable_one_ply_error', 'tactical', 'policy_avoidable', None, False),
    ('tactical_policy_missed_win', 'tactical', 'policy_missed_win', 'unique_win', False))
PUCT_METRICS = (
    ('tactical_puct_immediate_win', 'tactical', 'puct_correct', 'unique_win', True),
    ('tactical_puct_safe_response', 'tactical', 'puct_correct', 'unique_safe', True),
    ('tactical_puct_avoidable_one_ply_error', 'tactical', 'puct_avoidable', None, False),
    ('tactical_puct_missed_win', 'tactical', 'puct_missed_win', 'unique_win', False),
    ('solved_puct_optimal_accuracy', 'solved', 'puct_accuracy', None, True),
    ('solved_puct_deep_mistake', 'solved', 'puct_deep_mistake', None, False),
    ('deep_puct_optimal_accuracy', 'deep', 'puct_accuracy', None, True),
    ('deep_puct_deep_mistake', 'deep', 'puct_deep_mistake', None, False))


# Statistics -----------------------------------------------------------------------------------

def paired_rows(a_rows, b_rows, key, category=None, identity='id', cluster='cluster'):
    """[(cluster, a, b)] paired on row identity; refuses any membership mismatch."""
    select = (lambda r: category is None or r.get('category') == category)
    a = {r[identity]: r for r in a_rows if select(r)}
    b = {r[identity]: r for r in b_rows if select(r)}
    if set(a) != set(b) or not a:
        raise ValueError('Paired rows differ in membership')
    return [(a[i][cluster], float(a[i][key]), float(b[i][key])) for i in sorted(a)]


def contrast(rows_by_seed, label, replicates, seed):
    """Paired cluster bootstrap per seed and the equal-weight pooled contrast (a minus b)."""
    stream = int(hashlib.sha256(f'{seed}:{label}'.encode()).hexdigest(), 16) % 2 ** 63
    rng = np.random.default_rng(stream)
    per_seed, draws, points = {}, [], []
    for key in sorted(rows_by_seed, key=int):
        rows = rows_by_seed[key]
        clusters = sorted({c for c, _, _ in rows})
        index = {c: i for i, c in enumerate(clusters)}
        count, difference = np.zeros(len(clusters)), np.zeros(len(clusters))
        for c, a, b in rows:
            count[index[c]] += 1
            difference[index[c]] += a - b
        point = float(difference.sum() / count.sum())
        picks = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
        boot = difference[picks].sum(1) / count[picks].sum(1)
        low, high = np.quantile(boot, [0.025, 0.975])
        per_seed[str(key)] = dict(a=float(np.mean([a for _, a, _ in rows])), b=float(np.mean([b for _, _, b in rows])),
                                  difference=point, interval95=[float(low), float(high)], rows=len(rows),
                                  clusters=len(clusters))
        draws.append(boot)
        points.append(point)
    pooled = np.mean(draws, axis=0)
    low, high = np.quantile(pooled, [0.025, 0.975])
    return dict(label=label, per_seed=per_seed,
                pooled=dict(difference=float(np.mean(points)), interval95=[float(low), float(high)]))


def direction(result, higher_is_better):
    """Supporting rule: both seeds agree in sign and the pooled interval excludes zero."""
    sign = 1 if higher_is_better else -1
    seeds = [sign * v['difference'] for v in result['per_seed'].values()]
    low, high = sorted(sign * x for x in result['pooled']['interval95'])
    if all(s > 0 for s in seeds) and low > 0:
        return 'BETTER'
    if all(s < 0 for s in seeds) and high < 0:
        return 'WORSE'
    return 'INCONCLUSIVE'


def strength_verdict(final, consistency, mean_score, rule):
    seeds = [v['difference'] for v in final['per_seed'].values()]
    pooled, (low, high) = final['pooled']['difference'], final['pooled']['interval95']
    if not rule['saturation'][0] <= mean_score <= rule['saturation'][1]:
        return 'INCONCLUSIVE', 'endpoint saturated'
    if (all(s > 0 for s in seeds) and pooled >= rule['practical_margin'] and low > 0
            and consistency['pooled']['difference'] >= 0):
        return 'PASS', 'all PASS conditions met'
    if all(s < 0 for s in seeds) and high < 0:
        return 'FAIL', 'both seeds negative and the pooled interval excludes zero'
    return 'INCONCLUSIVE', 'neither the PASS nor the FAIL conditions are met'


def tactical_verdict(categories, rule):
    if any(c['pooled']['difference'] <= rule['fail_point'] and c['pooled']['interval95'][1] < 0 for c in categories):
        return 'FAIL'
    if all(c['pooled']['difference'] >= rule['pass_point'] and c['pooled']['interval95'][0] >= rule['pass_lower']
           for c in categories):
        return 'PASS'
    return 'INCONCLUSIVE'


def value_verdict(result):
    return direction(result, higher_is_better=False)


def compute_claim(ratio, rule):
    if rule['approximately_equal'][0] <= ratio <= rule['approximately_equal'][1]:
        return 'APPROXIMATELY_EQUAL'
    if rule['qualified'][0] <= ratio <= rule['qualified'][1]:
        return 'ROUGHLY_COMPARABLE'
    return 'NOT_EQUAL'


def recommendation(strength, tactical, claim):
    if strength == 'FAIL' or tactical == 'FAIL':
        return 'KEEP_CNN'
    if strength == 'PASS' and claim != 'NOT_EQUAL':
        return 'REPLACE_WITH_RESNET3X64'
    return 'NO_DEMONSTRATED_ADVANTAGE_KEEP_CNN_DEFAULT'


def reliability(pairs, bins=10, low=0.0, high=1.0):
    """Equal-width bins; count-weighted |mean target - mean prediction|."""
    rows, error = [], 0.0
    for b in range(bins):
        chosen = [(p, t) for p, t in pairs if min(bins - 1, int((p - low) / (high - low) * bins)) == b]
        if chosen:
            predicted, observed = (sum(x) / len(chosen) for x in zip(*chosen))
            error += len(chosen) / len(pairs) * abs(observed - predicted)
            rows.append(dict(bin=b, count=len(chosen), predicted=predicted, observed=observed))
    return dict(error=error, bins=rows)


def mean(rows, key, category=None):
    chosen = [r[key] for r in rows if category is None or r.get('category') == category]
    return sum(chosen) / len(chosen)


# Evidence loading -----------------------------------------------------------------------------

class Run:
    def __init__(self, root):
        self.root = Path(root)
        self.status = read(self.root / 'status.json')
        self.design = read(self.root / 'design.json')
        self.tasks = {t['label']: t for t in self.status['tasks']}

    def output(self, seed, name, kind, generation=None):
        label = arm_label(seed, name) + (f'-g{generation}' if generation is not None else '') + f'-{kind}'
        task = self.tasks.get(label)
        return Path(task['output']) if task and task['state'] == 'COMPLETE' else None

    def train(self, seed, name, generation):
        path = self.output(seed, name, 'train', generation)
        return read(path / 'complete.json') if path else None

    def suite(self, seed, name, generation, suite):
        path = self.output(seed, name, 'positions', generation)
        return read(path / f'{suite}.json')['rows'] if path else None

    def matches(self, seed, name, generation):
        path = self.output(seed, name, 'matches', generation)
        return read(path / 'matches.json')['records'] if path else None

    def dynamics(self, seed, name):
        path = self.output(seed, name, 'dynamics')
        return read(path / 'dynamics.json') if path else None


def training_accounting(run):
    """Realized work and computation per arm and generation, and the ResNet/CNN compute ratio."""
    arms = {}
    for seed, name in ARMS:
        rows, totals = [], dict(games=0, updates=0, positions=0, collection_seconds=0.0, update_seconds=0.0,
                                compute_seconds=0.0, task_wall_seconds=0.0, task_cpu_seconds=0.0, neural_positions=0)
        for generation in range(1, run.design['generations'] + 1):
            done = run.train(seed, name, generation)
            if done is None:
                break
            result, timing = done['result'], done['timing']
            task = run.tasks[f'{arm_label(seed, name)}-g{generation}-train']['resources']
            metrics = result['metrics']
            quarter = max(1, len(metrics) // 4)
            norms = [m['gradient_norm'] for m in metrics]
            compute = timing['collection_seconds'] + timing['update_seconds']
            for key, value in (('games', result['games']), ('updates', result['updates']),
                               ('positions', result['new_positions']), ('collection_seconds', timing['collection_seconds']),
                               ('update_seconds', timing['update_seconds']), ('compute_seconds', compute),
                               ('task_wall_seconds', task['wall_seconds']), ('task_cpu_seconds', task['cpu_seconds']),
                               ('neural_positions', result['collection']['neural_positions'])):
                totals[key] += value
            rows.append(dict(
                generation=generation, games=result['games'], updates=result['updates'], positions=result['new_positions'],
                cumulative=dict(totals), collection_seconds=timing['collection_seconds'],
                collection_cpu_over_wall=timing['collection_cpu_seconds'] / timing['collection_seconds'],
                update_seconds=timing['update_seconds'], compute_seconds=compute, task_wall_seconds=task['wall_seconds'],
                task_cpu_seconds=task['cpu_seconds'], peak_rss_mib=task['peak_rss_mib'],
                games_per_second=result['games'] / timing['collection_seconds'],
                updates_per_second=result['updates'] / timing['update_seconds'],
                ms_per_update=1e3 * timing['update_seconds'] / result['updates'],
                neural_positions=result['collection']['neural_positions'],
                neural_positions_per_second=result['collection']['neural_positions'] / timing['collection_seconds'],
                mean_inference_batch=result['collection']['neural_positions'] / result['collection']['forward_calls'],
                ms_per_ply=1e3 * timing['collection_seconds'] / result['new_positions'],
                mean_game_length=result['new_positions'] / result['games'],
                policy_loss=sum(m['policy_loss'] for m in metrics) / len(metrics),
                value_loss=sum(m['value_mse'] for m in metrics) / len(metrics),
                combined_first_quarter=sum(m['combined_loss'] for m in metrics[:quarter]) / quarter,
                combined_last_quarter=sum(m['combined_loss'] for m in metrics[-quarter:]) / quarter,
                gradient_norm_mean=sum(norms) / len(norms), gradient_norm_max=max(norms),
                clipped_fraction=sum(m['clipped'] for m in metrics) / len(metrics),
                boundary_sha256=done['sha256'], weights_sha256=done['after']['model'],
                collection_weights_sha256=result['collection_weights_sha256'],
                sampling=result['diagnostics']['sampling'], search=result['collection']['search']))
        arms[arm_label(seed, name)] = dict(generations=rows, totals=totals, complete=len(rows) == run.design['generations'])
    ratios = {}
    for seed in SEEDS:
        a, b = arms[arm_label(seed, 'resnet3x64')], arms[arm_label(seed, 'cnn')]
        if a['complete'] and b['complete']:
            ratios[str(seed)] = dict(
                compute=a['totals']['compute_seconds'] / b['totals']['compute_seconds'],
                collection=a['totals']['collection_seconds'] / b['totals']['collection_seconds'],
                update=a['totals']['update_seconds'] / b['totals']['update_seconds'],
                task_wall=a['totals']['task_wall_seconds'] / b['totals']['task_wall_seconds'],
                by_generation=[x['cumulative']['compute_seconds'] / y['cumulative']['compute_seconds']
                               for x, y in zip(a['generations'], b['generations'])])
    result = dict(arms=arms, ratio_resnet_over_cnn=ratios)
    if len(ratios) == len(SEEDS):
        totals = {name: sum(arms[arm_label(s, name)]['totals']['compute_seconds'] for s in SEEDS) for name in NAMES}
        result['pooled_compute_ratio'] = totals['resnet3x64'] / totals['cnn']
        result['compute_seconds_by_architecture'] = totals
        result['claim'] = compute_claim(result['pooled_compute_ratio'], run.design['analysis']['equal_compute'])
        result['claim_per_seed'] = {s: compute_claim(r['compute'], run.design['analysis']['equal_compute'])
                                    for s, r in ratios.items()}
    return result


def trajectories(run, dynamics):
    """Every checkpoint of every arm: raw-network measures at all generations, PUCT and matches where evaluated."""
    table = {}
    for seed, name in ARMS:
        label, rows = arm_label(seed, name), []
        data = dynamics.get(label)
        arm = run.design['arms'][name]
        for generation in range(run.design['generations'] + 1):
            row = dict(generation=generation, cumulative_games=generation * arm['games_per_generation'],
                       cumulative_updates=generation * arm['optimizer_updates'])
            if data:
                raw = data['rows'][generation]['raw']
                row['raw'] = {metric: mean(raw[suite], key, category) for metric, suite, key, category, _ in RAW_METRICS}
                solved = raw['solved']
                row['raw']['solved_constant_zero_value_mse'] = sum(r['exact'] ** 2 for r in solved) / len(solved)
                row['raw']['solved_value_calibration_error'] = reliability(
                    [(r['value'], r['exact']) for r in solved], low=-1.0, high=1.0)['error']
                row['raw']['solved_policy_confidence_error'] = reliability(
                    [(r['confidence'], r['accuracy']) for r in solved])['error']
                row['raw']['solved_saturated_value_share'] = sum(abs(r['value']) > 0.95 for r in solved) / len(solved)
                row['cumulative_positions'] = data['rows'][generation]['counters']['total_positions']
            suites = {s: run.suite(seed, name, generation, s) for s in ('solved', 'tactical', 'deep')}
            if all(v is not None for v in suites.values()):
                row['puct'] = {metric: mean(suites[suite], key, category) for metric, suite, key, category, _ in PUCT_METRICS}
            games = run.matches(seed, name, generation)
            if games is not None:
                row['match'] = dict(games=len(games), score=sum(g['score'] for g in games) / len(games),
                                    wins=sum(g['result'] == 'win' for g in games),
                                    draws=sum(g['result'] == 'draw' for g in games),
                                    losses=sum(g['result'] == 'loss' for g in games),
                                    score_as_first=mean([g for g in games if g['agent_color'] == 0], 'score'),
                                    score_as_second=mean([g for g in games if g['agent_color'] == 1], 'score'),
                                    mean_plies=sum(len(g['moves']) - g['prefix_length'] for g in games) / len(games))
            rows.append(row)
        table[label] = rows
    return table


def contrasts_for(run, dynamics, cnn_generation, resnet_generation, config):
    """ResNet at one generation minus CNN at another: raw, PUCT and match contrasts that the evidence supports."""
    replicates, seed = config['bootstrap_replicates'], config['bootstrap_seed']
    tag = f'resnet-g{resnet_generation}-minus-cnn-g{cnn_generation}'
    out = {}
    raw = {s: (dynamics.get(arm_label(s, 'resnet3x64')), dynamics.get(arm_label(s, 'cnn'))) for s in SEEDS}
    if all(a and b for a, b in raw.values()):
        for metric, suite, key, category, better in RAW_METRICS:
            rows = {s: paired_rows(a['rows'][resnet_generation]['raw'][suite], b['rows'][cnn_generation]['raw'][suite],
                                   key, category) for s, (a, b) in raw.items()}
            result = contrast(rows, f'{tag}:{metric}', replicates, seed)
            out[metric] = dict(result, higher_is_better=better, verdict=direction(result, better))
    suites = {s: {x: (run.suite(s, 'resnet3x64', resnet_generation, x), run.suite(s, 'cnn', cnn_generation, x))
                  for x in ('solved', 'tactical', 'deep')} for s in SEEDS}
    if all(a is not None and b is not None for per in suites.values() for a, b in per.values()):
        for metric, suite, key, category, better in PUCT_METRICS:
            rows = {s: paired_rows(*suites[s][suite], key, category) for s in SEEDS}
            result = contrast(rows, f'{tag}:{metric}', replicates, seed)
            out[metric] = dict(result, higher_is_better=better, verdict=direction(result, better))
    games = {s: (run.matches(s, 'resnet3x64', resnet_generation), run.matches(s, 'cnn', cnn_generation)) for s in SEEDS}
    if all(a is not None and b is not None for a, b in games.values()):
        key = lambda g: f'{g["opening_id"]}:{g["game_index"]}'
        rows = {}
        for s, (a, b) in games.items():
            a = [dict(g, id=key(g)) for g in a]
            b = [dict(g, id=key(g)) for g in b]
            rows[s] = paired_rows(a, b, 'score', cluster='family')
        result = contrast(rows, f'{tag}:match_score', replicates, seed)
        out['match_score'] = dict(result, higher_is_better=True, verdict=direction(result, True),
                                  by_opening_family={str(s): family_differences(*games[s]) for s in SEEDS})
    return out


def family_differences(a, b):
    """Paired score difference (ResNet minus CNN) for every opening family, both colors summed."""
    def by_family(games):
        table = {}
        for g in games:
            table.setdefault(g['family'], []).append(g['score'])
        return {f: sum(v) / len(v) for f, v in table.items()}
    x, y = by_family(a), by_family(b)
    return {f: dict(resnet=x[f], cnn=y[f], difference=x[f] - y[f]) for f in sorted(x)}


def dynamics_summary(dynamics):
    """Compact per-generation diagnostics; the raw per-position rows stay in the run evidence."""
    table = {}
    for label, data in dynamics.items():
        rows = []
        for row in data['rows']:
            rows.append({k: row[k] for k in ('generation', 'counters', 'weights_sha256', 'boundary_sha256', 'replay',
                                             'self_play', 'train_probe', 'newest_generation_after_training',
                                             'heldout_next_generation', 'generalization_gap') if k in row})
            norm = row.get('batch_norm')
            if norm:
                rows[-1]['batch_norm'] = {k: v for k, v in norm.items() if k != 'layers'}
        table[label] = rows
    return table


def analyze(root, output):
    run = Run(root)
    design, config = run.design, run.design['analysis']
    final, middle = config['decision_checkpoint'], config['consistency_checkpoint']
    dynamics = {arm_label(s, n): d for s, n in ARMS if (d := run.dynamics(s, n)) is not None}
    accounting = training_accounting(run)
    table = trajectories(run, dynamics)
    equal_compute = {str(g): contrasts_for(run, dynamics, g, g, config) for g in range(design['generations'] + 1)}
    pairs = [(4, 6), (2, 3)] if not design['smoke'] else [(1, 2)]
    equal_games = {f'cnn-g{c}-resnet-g{r}': contrasts_for(run, dynamics, c, r, config) for c, r in pairs}
    integrity = run.status.get('integrity', {})
    valid = (run.status['status'] == 'COMPLETE_PENDING_ANALYSIS'
             and all(v == 'PASS' or v.startswith('NOT APPLICABLE') for v in integrity.values()) and bool(integrity))
    verdicts = {}
    decisive, earlier = equal_compute[str(final)], equal_compute[str(middle)]
    if 'match_score' in decisive and 'match_score' in earlier:
        scores = [table[arm_label(s, n)][final]['match']['score'] for s, n in ARMS]
        verdict, reason = strength_verdict(decisive['match_score'], earlier['match_score'], sum(scores) / len(scores),
                                           config['strength'])
        verdicts['strength'] = dict(verdict=verdict, reason=reason, mean_final_score=sum(scores) / len(scores),
                                    final=decisive['match_score']['pooled'], per_seed=decisive['match_score']['per_seed'],
                                    consistency_checkpoint=earlier['match_score']['pooled'])
    for method in ('puct', 'policy'):
        names = [f'tactical_{method}_immediate_win', f'tactical_{method}_safe_response']
        if all(n in decisive for n in names):
            verdicts[f'tactical_{method}'] = dict(
                verdict=tactical_verdict([decisive[n] for n in names], config['tactical']),
                binding=method == 'puct', categories={n: decisive[n]['pooled'] for n in names})
    if 'solved_exact_value_mse' in decisive:
        verdicts['value'] = dict(verdict=value_verdict(decisive['solved_exact_value_mse']),
                                 pooled=decisive['solved_exact_value_mse']['pooled'])
    if 'claim' in accounting:
        verdicts['equal_compute_claim'] = dict(claim=accounting['claim'], ratio=accounting['pooled_compute_ratio'],
                                               per_seed=accounting['claim_per_seed'])
    if {'strength', 'tactical_puct', 'equal_compute_claim'} <= set(verdicts):
        verdicts['recommendation_rule'] = recommendation(verdicts['strength']['verdict'], verdicts['tactical_puct']['verdict'],
                                                         verdicts['equal_compute_claim']['claim'])
    result = dict(
        root=str(run.root), smoke=design['smoke'], status=run.status['status'], integrity=integrity,
        external_commits=run.status.get('external_commits'), verdicts_valid=valid, design_commit=run.status['design_commit'],
        elapsed_seconds=run.status['elapsed_seconds'], training_seconds=run.status.get('training_end_epoch', 0) - run.status['start_epoch'],
        children_cpu_seconds=run.status.get('children_cpu_seconds'), children_peak_rss_mib=run.status.get('children_peak_rss_mib'),
        tasks=dict(total=len(run.status['tasks']), complete=sum(t['state'] == 'COMPLETE' for t in run.status['tasks']),
                   failed=[t['label'] for t in run.status['tasks'] if t['state'] != 'COMPLETE'], retries=run.status['retries'],
                   wall_seconds_by_kind={k: sum(t.get('wall_seconds', 0) for t in run.status['tasks'] if t['kind'] == k)
                                         for k in ('train', 'positions', 'matches', 'dynamics')},
                   cpu_seconds_by_kind={k: sum(t['resources']['cpu_seconds'] for t in run.status['tasks']
                                               if t['kind'] == k and t['state'] == 'COMPLETE')
                                        for k in ('train', 'positions', 'matches', 'dynamics')}),
        work=run.status['work'], boundaries=run.status['boundaries'], computation=accounting, trajectories=table,
        equal_compute_contrasts=equal_compute, equal_games_contrasts=equal_games, verdicts=verdicts,
        dynamics=dynamics_summary(dynamics))
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = OUTPUT / ('smoke' if args.smoke else 'run')
    result = analyze(root, (root / 'analysis.json') if args.smoke else (RESULTS / 'analysis.json'))
    print({k: (v['verdict'] if 'verdict' in v else v.get('claim')) if isinstance(v, dict) else v
           for k, v in result['verdicts'].items()}, 'valid:', result['verdicts_valid'], 'status:', result['status'])


if __name__ == '__main__':
    main()
