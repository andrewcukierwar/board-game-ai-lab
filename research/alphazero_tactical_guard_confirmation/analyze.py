"""Prospectively frozen analysis: paired family bootstrap, activation, correctness, latency, verdict."""
from collections import Counter, defaultdict
import numpy as np
from research.alphazero_tactical_guard.independent_verify import ReferenceBoard, reference_selection
from .common import RESULTS, REPO, read, write, SEEDS
JA_TASKS = REPO / 'research/alphazero_tactical_guard/results/tasks'

SEED, B, MARGIN = 4420004, 10000, 0.02
COLORS = (0, 1)
PHASES = ((0, 13, 'opening'), (14, 27, 'middle'), (28, 41, 'late'))


def interval(draws): return [float(x) for x in np.quantile(draws, [.025, .975])]


def timings(values):
    x = np.asarray(values, float)
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)), p95=float(np.quantile(x, .95)), worst=float(x.max()))


def phase(occ): return next(name for lo, hi, name in PHASES if lo <= occ <= hi)


def main(RESULTS=RESULTS):
    design = read(RESULTS / 'design.json'); opps = (3, read(RESULTS / 'calibration.json')['opponent_b_depth']); families = [r['family'] for r in read(RESULTS / 'openings.json')['confirmation']]
    stratum = {r['family']: r['stratum'] for r in read(RESULTS / 'openings.json')['confirmation']}
    moves_of = {r['family']: r['moves'] for r in read(RESULTS / 'openings.json')['confirmation']}
    conditions = [(s, d, c) for s in SEEDS for d in opps for c in COLORS]
    F, index = len(families), {f: i for i, f in enumerate(families)}
    scores = {v: np.full((F, len(conditions)), np.nan) for v in ('control', 'guarded')}
    tallies = defaultdict(Counter); activation = Counter(); by = defaultdict(Counter); timing = defaultdict(list)
    occupancy_timing = defaultdict(list); changed_games, changed_families, mismatches, pair_first = set(), set(), [], []
    games_with_change = defaultdict(list)
    for s in SEEDS:
        for d in opps:
            base = RESULTS / 'tasks' / f'{s}-negamax{d}'; first = {}
            for v in ('control', 'guarded'):
                data = read(base / f'{v}-matches.json'); assert data['complete'] and len(data['records']) == 2 * F
                for r in data['records']:
                    j = conditions.index((s, d, r['agent_color'])); i = index[r['family']]
                    assert np.isnan(scores[v][i, j]); scores[v][i, j] = r['score']
                    tallies[(s, d, v)][r['result']] += 1
                    mine = [(k, x) for k, x in enumerate(r['decisions']) if x['color'] == r['agent_color']]
                    first.setdefault((r['family'], r['agent_color']), {})[v] = (mine[0][1]['original_action'], mine[0][1]['visits'])
                    if v != 'guarded': continue
                    prefix = len(moves_of[r['family']])
                    for k, x in mine:
                        g = x['guard']; ref = ReferenceBoard.from_moves(r['moves'][:prefix + k])
                        chosen, reason, _ = reference_selection(ref, x['visits'], x['original_action'])
                        if (chosen, reason) != (g['final_action'], g['reason']) or x['move'] != g['final_action'] or g['original_visits'] != x['visits']:
                            mismatches.append(dict(task=f'{s}-negamax{d}', family=r['family'], color=r['agent_color'], ply=prefix + k))
                        keys = [('all',), ('opponent', f'negamax{d}'), ('seed', s), ('stratum', stratum[r['family']]), ('phase', phase(g['occupancy']))]
                        for key in keys:
                            by[key]['calls'] += 1; by[key][g['reason']] += 1; by[key]['changed'] += g['changed']
                            by[key]['zero_visit_selected'] += g['selected_zero_visits']
                        if g['changed']:
                            changed_games.add((s, d, r['family'], r['agent_color'])); changed_families.add(r['family'])
                            games_with_change[(s, d, r['family'], r['agent_color'])].append(dict(ply=prefix + k, reason=g['reason'],
                                original=g['original_action'], final=g['final_action'], original_visits=g['original_visits'], selected_visits=g['selected_visits']))
                        for f in ('search_seconds', 'total_seconds'): timing[f].append(x[f])
                        for f in ('guard_seconds', 'classification_seconds', 'candidate_count', 'opponent_replies_inspected'): timing[f].append(g[f])
                        timing['relative_overhead'].append(g['guard_seconds'] / x['search_seconds'])
                        occupancy_timing[g['occupancy'] // 7].append(g['guard_seconds'])
            for key, pair in first.items():
                pair_first.append(pair['control'] == pair['guarded'])
    assert not np.isnan(scores['control']).any() and not np.isnan(scores['guarded']).any()
    D = scores['guarded'] - scores['control']  # families x conditions
    rng = np.random.Generator(np.random.PCG64(SEED)); idx = rng.integers(0, F, (B, F))
    def est(cols, rows=None):
        rows = np.arange(F) if rows is None else np.asarray(rows)
        sub = D[np.ix_(rows, cols)]
        local = idx if rows.size == F else np.random.Generator(np.random.PCG64(SEED + 1 + int(rows[0]))).integers(0, rows.size, (B, rows.size))
        draws = sub[local].mean(axis=(1, 2))
        return dict(difference=float(sub.mean()), ci95=interval(draws), families=int(rows.size), paired_games=int(sub.size),
                    control_score=float(scores['control'][np.ix_(rows, cols)].mean()), guarded_score=float(scores['guarded'][np.ix_(rows, cols)].mean())), draws
    col = lambda pred: [j for j, c in enumerate(conditions) if pred(*c)]
    pooled, pooled_draws = est(list(range(len(conditions))))
    out = dict(contract='phase4d4j-b-analysis-v1', bootstrap=dict(replicates=B, seed=SEED, unit='whole opening family (all seeds, opponents, colors, arms together)',
               interval='percentile 95%'), families=F, pooled=pooled, seeds={}, opponents={}, cells={}, colors={}, strata={}, heterogeneity={})
    seed_draws, opp_draws = {}, {}
    for s in SEEDS: out['seeds'][str(s)], seed_draws[s] = est(col(lambda a, b, c: a == s))
    for d in opps: out['opponents'][f'negamax{d}'], opp_draws[d] = est(col(lambda a, b, c: b == d))
    for s in SEEDS:
        for d in opps:
            out['cells'][f'{s}-negamax{d}'] = est(col(lambda a, b, c: a == s and b == d))[0]
            out['cells'][f'{s}-negamax{d}'].update({v: dict(tallies[(s, d, v)]) for v in ('control', 'guarded')})
    for c in COLORS: out['colors']['XO'[c]] = est(col(lambda a, b, cc: cc == c))[0]
    for st in sorted(set(stratum.values())):
        rows = [index[f] for f in families if stratum[f] == st]
        out['strata'][st] = est(list(range(len(conditions))), rows)[0]
        out['strata'][st]['by_opponent'] = {f'negamax{d}': est(col(lambda a, b, c: b == d), rows)[0] for d in opps}
    a, b = opps
    out['heterogeneity']['opponent_b_minus_a'] = dict(difference=out['opponents'][f'negamax{b}']['difference'] - out['opponents'][f'negamax{a}']['difference'],
                                                      ci95=interval(opp_draws[b] - opp_draws[a]))
    out['heterogeneity']['seed_314159_minus_42'] = dict(difference=out['seeds']['314159']['difference'] - out['seeds']['42']['difference'],
                                                        ci95=interval(seed_draws[314159] - seed_draws[42]))
    sd = [out['strata'][k]['difference'] for k in sorted(out['strata'])]
    out['heterogeneity']['strata_range'] = dict(min=min(sd), max=max(sd), note='descriptive; four strata of 24 families each')
    out['families_with_nonzero_pair_difference'] = int((np.abs(D).sum(1) > 0).sum())
    out['paired_games_changed_outcome'] = int((D != 0).sum())
    # Activation, correctness and latency.
    out['activation'] = {' '.join(map(str, k)): dict(v) for k, v in sorted(by.items(), key=lambda kv: str(kv[0]))}
    out['changed'] = dict(decisions=by[('all',)]['changed'], games=len(changed_games), families=len(changed_families),
                          rate=by[('all',)]['changed'] / by[('all',)]['calls'],
                          records={f'{s}-negamax{d}:{f}:{"XO"[c]}': v for (s, d, f, c), v in sorted(games_with_change.items())})
    label = {1.: 'win', .5: 'draw', 0.: 'loss'}
    out['changed_games_outcomes'] = {v: dict(Counter(label[float(scores[v][index[f], conditions.index((s, d, c))])] for (s, d, f, c) in changed_games))
                                     for v in ('control', 'guarded')}
    out['correctness'] = dict(match_decision_mismatches=mismatches, paired_first_decision_identical=all(pair_first), pairs=len(pair_first))
    out['latency'] = {k: timings(v) for k, v in timing.items()}
    out['latency_by_occupancy'] = {f'{7*k}-{7*k+6}': timings(v) for k, v in sorted(occupancy_timing.items())}
    lat = out['latency']; limits = design['latency']
    out['latency_pass'] = lat['guard_seconds']['p95'] <= limits['p95_guard_seconds_max'] and lat['guard_seconds']['worst'] <= limits['worst_guard_seconds_max'] \
        and lat['relative_overhead']['mean'] <= limits['mean_relative_overhead_max']
    # Regression suites (4D.4J-A tactical200/solved300/deep128), exact reproduction against 4D.4J-A saved rows.
    suites = {}; wdl_regressions = []; tactical_fail = []; reproduction = Counter()
    volatile = {'search_seconds', 'total_seconds', 'guard'}
    for s in SEEDS:
        base = RESULTS / 'tasks' / f'{s}-negamax3'
        for suite in ('tactical', 'solved', 'deep'):
            rows = read(base / f'{suite}.json')['rows']; old = {r['id']: r for r in read(JA_TASKS / str(s) / f'{suite}.json')['rows']}
            for r in rows:
                for v in ('control', 'guarded'):
                    same = all(r[v][k] == old[r['id']][v][k] for k in r[v] if k not in volatile)
                    if v == 'guarded':
                        g, h = r[v]['guard'], old[r['id']][v]['guard']
                        same = same and all(g[k] == h[k] for k in g if not k.endswith('seconds'))
                    reproduction['identical' if same else 'different'] += 1
                if suite == 'tactical' and (r['guarded']['avoidable'] or r['guarded']['missed_win']): tactical_fail.append((s, r['id']))
                if suite != 'tactical' and r['guarded']['chosen_value'] < r['control']['chosen_value']: wdl_regressions.append((s, suite, r['id']))
            if suite == 'tactical':
                suites[f'{s}-{suite}'] = {v: dict(correct=sum(r[v]['correct'] for r in rows), avoidable=sum(r[v]['avoidable'] for r in rows),
                                                  missed_win=sum(r[v]['missed_win'] for r in rows)) for v in ('control', 'guarded')}
            else:
                suites[f'{s}-{suite}'] = {v: dict(correct=sum(r[v]['correct'] for r in rows), mistakes=sum(r[v]['mistake'] for r in rows)) for v in ('control', 'guarded')}
                suites[f'{s}-{suite}']['improved'] = sum(r['guarded']['chosen_value'] > r['control']['chosen_value'] for r in rows)
            suites[f'{s}-{suite}']['n'] = len(rows)
    out['regression_suites'] = dict(summary=suites, wdl_regressions=wdl_regressions, tactical_failures=tactical_fail, reproduction_vs_4d4ja=dict(reproduction))
    # Frozen decision hierarchy.
    lo, hi = pooled['ci95']; seeds_ok = all(out['seeds'][str(s)]['difference'] > -MARGIN for s in SEEDS)
    opp_regression = [k for k, v in out['opponents'].items() if v['difference'] <= -MARGIN and v['ci95'][1] < 0]
    failures = []
    if mismatches or tactical_fail or not all(pair_first): failures.append('correctness')
    if wdl_regressions: failures.append('exact_wdl_regression')
    if reproduction['different']: failures.append('regression_reproduction')
    if not out['latency_pass']: failures.append('latency')
    if (pooled['difference'] <= -MARGIN and hi < 0) or opp_regression: failures.append('material_strength_regression')
    if failures: verdict = 'FAIL'
    elif pooled['difference'] > 0 and lo > 0 and seeds_ok and not opp_regression: verdict = 'CONFIRMED_POSITIVE_EFFECT'
    else: verdict = 'INCONCLUSIVE'
    signs = {np.sign(v['difference']) for v in list(out['seeds'].values()) + list(out['opponents'].values())}
    out['verdict'] = dict(strength=verdict, failures=failures, practically_meaningful=pooled['difference'] >= MARGIN,
                          small_effect=verdict == 'CONFIRMED_POSITIVE_EFFECT' and pooled['difference'] < MARGIN,
                          qualified_heterogeneity=len(signs) > 1, seeds_not_materially_negative=seeds_ok, opponent_material_regressions=opp_regression)
    write(RESULTS / 'analysis.json', out)
    print(out['verdict'], pooled)


if __name__ == '__main__':
    main()
