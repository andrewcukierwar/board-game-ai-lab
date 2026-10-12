"""Apply the prospective Phase 4D.4D analysis (DESIGN.md) to retained collections.

Reads complete cells only, verifies every game, and writes results.json,
decision.json, acceptance.json and TABLES.md into a fresh directory. The
official campaign is read (hashes, archived generation-20 storage) and never
written; its 150 files are hashed before and after.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess

from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.data import GenerationReplay
from games.connect4.alphazero_v2.diagnostics import family
from research.alphazero_baseline import audit
from research.alphazero_batching.benchmark import digest, game_records, require
from . import analysis as a, exact, replay, schedules as s
from .collect import BASE, BASE_COMMIT, BUDGET_SECONDS

# Thresholds restated from DESIGN.md, which was committed before the measured collection.
CRITERIA = dict(
    diversity=dict(material_relative=.25, material_absolute=.05, suggestive_relative=.10, larger_change_absolute=.10),
    tactics=dict(missed_win_rate=.05, unsafe_rate=.02, errors_per_game=.15, game_ending_error_share=.05))
ORDER = ('extended', 'smoothed', 'extended_smoothed')   # smallest change first
UNCHANGED_SOURCES = ('config.py', 'search.py', 'selfplay.py', 'data.py', 'bitboard.py', 'network.py')
SAMPLING_SEEDS = tuple(range(4404460, 4404480))


def late(function, **options):
    return lambda games: function(games, 8, 42, **options)


def prefix_share(games, length):
    share = a.prefixes(games, length)['unique_per_eligible']
    return float('nan') if share is None else share


ENDPOINTS = dict(
    d1_duplicate_fraction_8_plus=late(a.duplicate_fraction),
    d2_mirror_duplicate_fraction_8_plus=late(a.duplicate_fraction, field='canon'),
    d3_collision_probability_8_plus=late(a.collision_probability),
    duplicate_fraction_0_7=lambda g: a.duplicate_fraction(g, 0, 8),
    duplicate_fraction_8_15=lambda g: a.duplicate_fraction(g, 8, 16),
    duplicate_fraction_16_23=lambda g: a.duplicate_fraction(g, 16, 24),
    duplicate_fraction_24_plus=lambda g: a.duplicate_fraction(g, 24, 42),
    duplicate_fraction_all=lambda g: a.duplicate_fraction(g),
    unique_16_ply_prefix_share=lambda g: prefix_share(g, 16),
    unique_24_ply_prefix_share=lambda g: prefix_share(g, 24),
    mean_length=lambda g: sum(len(x['moves']) for x in g) / len(g),
    target_entropy_8_plus=lambda g: a.stage_mean(g, '8+', 'target_entropy'),
    execution_entropy_8_plus=lambda g: a.stage_mean(g, '8+', 'execution_entropy'),
    max_target_8_plus=lambda g: a.stage_mean(g, '8+', 'max_target'),
    prior_entropy_8_plus=lambda g: a.stage_mean(g, '8+', 'prior_entropy'))
TACTICAL = {name: (lambda g, n=n, d=d: a.rate(g, n, d)) for name, (n, d) in a.TACTICAL_RATES.items()}
DECISION_TACTICS = tuple(CRITERIA['tactics'])


def clean(value):
    """NaN/inf become null and tuple keys become strings, for strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def label(checkpoint):
    return f'{checkpoint[0]}/g{checkpoint[1]}'


def matched_blocks(games):
    """{(seed, generation, cohort): {schedule: games}} for blocks complete in all four schedules."""
    cells = defaultdict(lambda: defaultdict(list))
    for game in games:
        cells[game['id'][:3]][game['schedule']].append(game)
    blocks = {}
    for block, by_schedule in sorted(cells.items()):
        ids = [sorted(g['id'] for g in by_schedule.get(x.name, ())) for x in s.SCHEDULES]
        if all(ids) and all(i == ids[0] for i in ids):
            blocks[block] = dict(by_schedule)
    return blocks


def datasets(blocks):
    """{checkpoint: {schedule: games}} pooling a checkpoint's complete cohorts."""
    result = defaultdict(lambda: defaultdict(list))
    for (seed, generation, _), by_schedule in blocks.items():
        for name, games in by_schedule.items():
            result[seed, generation][name].extend(games)
    return {k: dict(v) for k, v in result.items()}


def searched(game, plies):
    return [(p['key'], p['visits'], p['action']) for p in game['plies'][:plies]]


def negative_control(data):
    """Matched identities must share every searched example before temperatures differ."""
    rows = []
    for checkpoint, by_schedule in data.items():
        for reference, candidate, shared in (('original', 'extended', 8), ('original', 'smoothed', 8),
                                             ('original', 'extended_smoothed', 8),
                                             ('extended', 'extended_smoothed', 16)):
            pairs = a.pair(by_schedule[reference], by_schedule[candidate])
            mismatched = [list(x['id']) for x, y in pairs if searched(x, shared) != searched(y, shared)]
            rows.append(dict(checkpoint=label(checkpoint), reference=reference, candidate=candidate,
                             shared_plies=shared, identities=len(pairs), mismatches=len(mismatched),
                             mismatched_identities=mismatched,
                             identical_complete_games=sum(x['moves'] == y['moves'] for x, y in pairs)))
    return rows


def condition_summary(games):
    return dict(diversity=a.diversity(games), tactics=a.tactical_profile(games),
                intervals={name: a.jackknife(games, f) for name, f in {**ENDPOINTS, **TACTICAL}.items()})


def comparisons(reference, candidate, functions):
    return {name: a.paired_difference(reference, candidate, f) for name, f in functions.items()}


def diversity_checks(reference, d1, d2, d3, cohort_values):
    """DESIGN.md MATERIAL/SUGGESTIVE conditions for one checkpoint.

    ``reference`` is schedule A's D1, d1..d3 are paired differences from it and
    ``cohort_values`` are (candidate D1, original D1) per cohort.
    """
    criteria = CRITERIA['diversity']
    relative = -d1['estimate'] / reference if reference else 0.
    checks = dict(relative_reduction=relative, absolute_reduction=-d1['estimate'],
                  meets_relative=relative >= criteria['material_relative'],
                  meets_absolute=-d1['estimate'] >= criteria['material_absolute'],
                  interval_excludes_zero=d1['high'] < 0,
                  lower_in_every_cohort=all(x < y for x, y in cohort_values.values()),
                  cohort_values={str(k): dict(candidate=x, original=y) for k, (x, y) in cohort_values.items()},
                  d2_lower=d2['estimate'] < 0, d3_lower=d3['estimate'] < 0)
    checks['material'] = all(checks[k] for k in ('meets_relative', 'meets_absolute', 'interval_excludes_zero',
                                                 'lower_in_every_cohort', 'd2_lower', 'd3_lower'))
    checks['suggestive'] = relative >= criteria['suggestive_relative']
    return checks


def classify_diversity(name, data, blocks, paired):
    d1_of, per_checkpoint = ENDPOINTS['d1_duplicate_fraction_8_plus'], {}
    for checkpoint, by_schedule in data.items():
        differences = paired[label(checkpoint)][name]
        cohorts = {b[2]: (d1_of(v[name]), d1_of(v['original'])) for b, v in blocks.items() if b[:2] == checkpoint}
        per_checkpoint[label(checkpoint)] = diversity_checks(
            d1_of(by_schedule['original']), differences['d1_duplicate_fraction_8_plus'],
            differences['d2_mirror_duplicate_fraction_8_plus'], differences['d3_collision_probability_8_plus'], cohorts)
    material = all(c['material'] for c in per_checkpoint.values())
    suggestive = all(c['suggestive'] for c in per_checkpoint.values())
    return dict(classification='MATERIAL' if material else 'SUGGESTIVE' if suggestive else 'NONE',
                checkpoints=per_checkpoint)


def classify_tactics(differences):
    endpoints, regression, clear = {}, False, True
    for name in DECISION_TACTICS:
        d, margin = differences[name], CRITERIA['tactics'][name]
        endpoints[name] = dict(d, margin=margin, at_least_margin=d['estimate'] >= margin,
                               lower_bound_above_zero=d['low'] > 0, upper_bound_below_margin=d['high'] < margin)
        regression |= d['estimate'] >= margin and d['low'] > 0
        clear &= d['high'] < margin
    return dict(classification='MATERIAL REGRESSION' if regression else 'NO MATERIAL REGRESSION' if clear
                else 'INCONCLUSIVE', endpoints=endpoints)


def factorial(by_schedule):
    """Two-factor contrasts on D1 with a jackknife over matched identity quadruples."""
    d1 = ENDPOINTS['d1_duplicate_fraction_8_plus']
    quads = list(zip(*(sorted(by_schedule[x.name], key=lambda g: g['id']) for x in s.SCHEDULES)))
    require(all(len({g['id'] for g in quad}) == 1 for quad in quads), 'Unmatched identities')
    values = lambda units: [d1([u[i] for u in units]) for i in range(4)]   # A, B, C, D
    return dict(
        extend_main_effect=a.jackknife(quads, lambda u: (lambda v: ((v[1] - v[0]) + (v[3] - v[2])) / 2)(values(u))),
        smooth_main_effect=a.jackknife(quads, lambda u: (lambda v: ((v[2] - v[0]) + (v[3] - v[1])) / 2)(values(u))),
        interaction=a.jackknife(quads, lambda u: (lambda v: (v[3] - v[2]) - (v[1] - v[0]))(values(u))))


def d1_gain(data):
    """Paired D1 difference (larger change minus smaller change) per checkpoint."""
    return lambda smaller, larger: {label(c): a.paired_difference(v[smaller], v[larger],
                                                                  ENDPOINTS['d1_duplicate_fraction_8_plus'])
                                    for c, v in data.items()}


def recommend(diversity, tactics, gain):
    """``gain(smaller, larger)`` returns the per-checkpoint paired D1 difference of two eligible schedules."""
    eligible = [n for n in ORDER if diversity[n]['classification'] == 'MATERIAL'
                and tactics[n]['classification'] != 'MATERIAL REGRESSION']
    chosen, escalations = (eligible[0] if eligible else None), []
    for larger in eligible[1:]:
        gains = gain(chosen, larger)
        replace = bool(gains) and all(g['estimate'] <= -CRITERIA['diversity']['larger_change_absolute']
                                      and g['high'] < 0 for g in gains.values())
        escalations.append(dict(smaller=chosen, larger=larger, d1_difference=gains, replaces=replace))
        if replace:
            chosen = larger
    return dict(eligible=eligible, recommended_schedule=chosen or 'original',
                conditional_on_tactical_monitoring=bool(chosen) and tactics[chosen]['classification'] == 'INCONCLUSIVE',
                training_ablation_justified=chosen is not None, escalation_checks=escalations,
                rule='DESIGN.md: MATERIAL diversity and not MATERIAL REGRESSION; smallest change unless a larger '
                     'one lowers D1 by >= 10 points in both checkpoints with paired intervals excluding zero')


def robustness(data, pooled, tactics):
    """POST HOC sensitivity of each predeclared label to single game identities (review finding)."""
    d1_of, result = ENDPOINTS['d1_duplicate_fraction_8_plus'], {}
    for name in ORDER:
        entry = dict(diversity={}, tactics={})
        for checkpoint, by_schedule in data.items():
            relative = a.paired_statistic(by_schedule['original'], by_schedule[name],
                                          lambda r, c: 1 - d1_of(c) / d1_of(r) if d1_of(r) else float('nan'),
                                          replicates=True)
            absolute = a.paired_difference(by_schedule['original'], by_schedule[name], d1_of, replicates=True)
            entry['diversity'][label(checkpoint)] = dict(
                relative_reduction={k: relative[k] for k in ('estimate', 'low', 'high')},
                delete_one_replicates=relative['units'],
                replicates_below_relative_threshold=sum(
                    x < CRITERIA['diversity']['material_relative'] for x in relative['leave_one_out']),
                replicates_below_absolute_threshold=sum(
                    -x < CRITERIA['diversity']['material_absolute'] for x in absolute['leave_one_out']),
                absolute_interval_lower_bound_meets_threshold=(
                    absolute['high'] is not None and -absolute['high'] >= CRITERIA['diversity']['material_absolute']))
        pairs = a.pair(pooled['original'], pooled[name])
        for endpoint in DECISION_TACTICS:
            d = a.paired_difference(pooled['original'], pooled[name], TACTICAL[endpoint], replicates=True)
            entry['tactics'][endpoint] = dict(
                estimate=d['estimate'], margin=CRITERIA['tactics'][endpoint],
                delete_one_minimum=min(d['leave_one_out']),
                replicates_below_margin=sum(x < CRITERIA['tactics'][endpoint] for x in d['leave_one_out']),
                delete_one_replicates=d['units'])
        # The identity with the largest increase in executed errors, and the label without it.
        influence = max(pairs, key=lambda u: u[1]['tactics']['errors'] - u[0]['tactics']['errors'])
        rest = [u for u in pairs if u is not influence]
        without = classify_tactics(comparisons([u[0] for u in rest], [u[1] for u in rest],
                                               {k: TACTICAL[k] for k in DECISION_TACTICS}))
        entry['most_influential_identity'] = dict(
            identity=list(influence[0]['id']), original_errors=influence[0]['tactics']['errors'],
            schedule_errors=influence[1]['tactics']['errors'],
            classification_with_all_identities=tactics[name]['classification'],
            classification_without_it=without['classification'],
            errors_per_game_without_it=without['endpoints']['errors_per_game']['estimate'])
        nonzero = [u[1]['tactics']['errors'] - u[0]['tactics']['errors'] for u in pairs]
        entry['error_difference_signs'] = dict(positive=sum(x > 0 for x in nonzero), negative=sum(x < 0 for x in nonzero),
                                               zero=sum(x == 0 for x in nonzero))
        result[name] = entry
    return result


def exact_context(data, pooled):
    """POST HOC exact-value breakdown of the one-reply events (review finding); never a decision input."""
    value = exact.ExactValues()
    for games in pooled.values():
        exact.attach(games, value)
    functions = {name: (lambda g, n=n, d=d: exact.rate(g, n, d)) for name, (n, d) in exact.RATES.items()}
    decision_like = {k: functions[k] for k in ('value_changing_errors_per_game', 'value_changing_game_ending_share')}
    return dict(
        note='Post hoc, added after independent review. Not a predeclared endpoint and not used by the decision '
             'rule. Value-neutral under perfect play is not necessarily neutral against a 256-simulation opponent. '
             'Unsolved positions, if any, are counted as value-changing.',
        solver=dict(version=exact.BitboardSolver.__name__, max_nodes=value.solver.max_nodes,
                    positions_solved=len(value.cache), unsolved=value.unsolved),
        pooled={n: exact.profile(g) for n, g in pooled.items()},
        by_checkpoint={label(c): {n: exact.profile(g) for n, g in v.items()} for c, v in data.items()},
        pooled_paired_vs_original={n: comparisons(pooled['original'], pooled[n], decision_like) for n in ORDER},
        by_checkpoint_paired_vs_original={label(c): {n: comparisons(v['original'], v[n], decision_like) for n in ORDER}
                                          for c, v in data.items()})


def verify_record_hashes(games, cells):
    """Rebuild every stored game as a CompletedGame and compare record hashes with the collector's."""
    by_cell = defaultdict(list)
    for game in games:
        by_cell[game['id'][:3], game['schedule']].append(game)
    checked = 0
    for cell in cells:
        if not cell['complete']:
            continue
        members = by_cell[(cell['seed'], cell['generation'], cell.get('cohort', 0)), cell['schedule']['name']]
        rebuilt = replay.completed_games(members, s.GENERATION)
        require(digest(game_records(rebuilt)) == cell['records_sha256'], 'Cell record hash mismatch')
        for game, completed in zip(sorted(members, key=lambda g: g['id']), rebuilt):
            require(completed.index == game['id'][3] and digest(game_records([completed])) == game['record_sha256'],
                    'Game record hash mismatch')
            checked += 1
    return checked


def archived_reference(campaign, seed):
    """Generation-20 collection (played by the generation-19 learner) as historical reference."""
    payload = replay.archived_payload(campaign, seed)
    games = replay.archived_games(payload, 20, seed)
    directory = audit.checkpoint(campaign, seed, 20).parent
    summary = audit.read(directory / 'summary.json')
    previous = audit.read(audit.checkpoint(campaign, seed, 19).parent / 'summary.json')
    stored = [json.loads(line) for line in (directory / 'generation-0020.games.jsonl').read_text().splitlines()]
    require([tuple(g['moves']) for g in stored] == [g['moves'] for g in games], 'Replay storage differs from game log')
    halves = [games[:128], games[128:]]
    half = {name: sum(f(h) for h in halves) / 2 for name, f in {**ENDPOINTS, **TACTICAL}.items()
            if 'prior' not in name}
    return payload, games, dict(
        games=len(games), played_by_generation_19_learner=(summary['summary']['collection_weights_sha256']
                                                           == previous['learner_weights_sha256']),
        full_256=dict(diversity=a.diversity(games), tactics=a.tactical_profile(games),
                      endpoints={name: f(games) for name, f in {**ENDPOINTS, **TACTICAL}.items()}),
        mean_of_two_128_game_halves=half,
        note='Shared sequential RNG and the generation-19 learner; a reference, not a matched control.')


def cross_check(archive, games, seed, window_states):
    """Independent agreement with Phase 4D.4A's archive audit and campaign diagnostics."""
    earlier = next(r for r in audit.read(BASE / 'archive/archives.json')
                   if (r['seed'], r['generation']) == (seed, 20))
    mine, tactics = archive['full_256']['diversity'], archive['full_256']['tactics']['counts']
    return dict(
        unique_positions=mine['unique_positions'] == earlier['unique_positions'],
        unique_mirror_families=mine['unique_mirror_canonical'] == earlier['unique_mirror_families'],
        duplicate_fraction=abs(mine['duplicate_fraction'] - earlier['duplicate_position_fraction']) < 1e-12,
        after8_duplicate_fraction=abs(mine['stages']['8+']['duplicate_fraction']
                                      - earlier['after8_duplicate_fraction']) < 1e-12,
        unique_trajectories=mine['unique_trajectories'] == earlier['unique_trajectories'],
        unique_opening8=len({g['moves'][:8] for g in games}) == earlier['unique_opening8'],
        tactics=dict(immediate_win_states=tactics['win_opps'], missed_immediate_win_actions=tactics['missed_wins'],
                     mandatory_response_states=tactics['choice_states'],
                     unsafe_actions_with_safe_alternative=tactics['unsafe']) == earlier['archived_tactics'],
        replay_window_unique_states=window_states)


def performance(directories):
    rows, total_seconds, invocations = [], 0., []
    for name, directory in directories.items():
        status = audit.read(directory / 'status.json')
        preservation = audit.read(directory / 'preservation.json')
        cells = audit.read(directory / 'cells.json')
        total_seconds += status['elapsed_seconds']
        invocations.append(dict(name=name, path=str(directory), complete=status['complete'],
                                elapsed_seconds=status['elapsed_seconds'], peak_rss_mib=status['peak_rss_mib'],
                                stop_reason=status.get('stop_reason'), error=status.get('error'),
                                official_files=preservation['files'], official_unchanged=preservation['unchanged'],
                                declaration_ok=preservation['declaration_sha256'] == audit.DECLARATION,
                                cells=len(cells), incomplete_cells=sum(not c['complete'] for c in cells),
                                games=sum(c['completed_games'] for c in cells)))
        rows.extend(dict(c, invocation=name) for c in cells)
    by_schedule = {}
    for schedule in s.SCHEDULES:
        cells = [c for c in rows if c['invocation'] == 'main' and c['complete'] and c['schedule']['name'] == schedule.name]
        if not cells:
            continue
        wall = sum(c['wall_seconds'] for c in cells)
        sizes = defaultdict(int)
        for c in cells:
            for size, count in c['batch_sizes'].items():
                sizes[size] += count
        by_schedule[schedule.name] = dict(
            cells=len(cells), games=sum(c['completed_games'] for c in cells), plies=sum(c['plies'] for c in cells),
            simulations=sum(c['simulations'] for c in cells), wall_seconds=wall,
            games_per_second=sum(c['completed_games'] for c in cells) / wall,
            positions_per_second=sum(c['plies'] for c in cells) / wall,
            forward_calls=sum(c['forward_calls'] for c in cells),
            neural_positions=sum(c['neural_positions'] for c in cells),
            mean_batch=sum(c['neural_positions'] for c in cells) / sum(c['forward_calls'] for c in cells),
            batch_sizes=dict(sorted(sizes.items(), key=lambda kv: int(kv[0]))),
            inference_fraction=sum(c['inference_seconds'] for c in cells) / wall,
            peak_rss_mib=max(c['peak_rss_mib'] for c in cells))
    # run_cell already requires equal hashes before/after a cell; one hash per checkpoint across all cells.
    hashes = defaultdict(set)
    for c in rows:
        hashes[c['seed'], c['generation']].add(c['weights_sha256'])
    return dict(invocations=invocations, measured_seconds=total_seconds, budget_seconds=BUDGET_SECONDS,
                within_budget=total_seconds < BUDGET_SECONDS, main_by_schedule=by_schedule,
                weights_frozen=all(len(v) == 1 for v in hashes.values()),
                weights_sha256={label(k): sorted(v) for k, v in hashes.items()}, cells=rows)


def analyze(args):
    campaign = args.campaign.resolve()
    directories = {k: v for k, v in dict(regression=args.regression, pilot=args.pilot, main=args.main,
                                         supplement=args.supplement,
                                         regression_final=args.regression_final).items() if v is not None}
    results = dict(criteria=CRITERIA, design_sha256=audit.sha('research/alphazero_exploration/DESIGN.md'),
                   schedules=[x.to_dict() for x in s.SCHEDULES])
    games, main_cells, skipped = a.load_collection(args.main)
    hashed = verify_record_hashes(games, main_cells)
    blocks = matched_blocks(games)
    data = datasets(blocks)
    require(set(data) == set(s.PRIMARY), 'A primary checkpoint has no complete matched cohort')
    analyzed = [g for by in blocks.values() for gs in by.values() for g in gs]
    results['sample'] = dict(
        verified_games=len(games), analyzed_games=len(analyzed), analyzed_positions=sum(len(g['plies']) for g in analyzed),
        complete_blocks=[list(b) for b in blocks], incomplete_cells=[c['file'] for c in skipped],
        games_per_condition={label(c): {n: len(g) for n, g in v.items()} for c, v in data.items()})
    results['negative_control'] = negative_control(data)

    results['conditions'], results['paired_vs_original'], results['factorial_d1'] = {}, {}, {}
    results['cohorts'] = {}
    for checkpoint, by_schedule in data.items():
        key = label(checkpoint)
        results['conditions'][key] = {n: condition_summary(g) for n, g in by_schedule.items()}
        results['paired_vs_original'][key] = {
            n: comparisons(by_schedule['original'], by_schedule[n], {**ENDPOINTS, **TACTICAL}) for n in ORDER}
        results['factorial_d1'][key] = factorial(by_schedule)
        print(f'analyzed {key}', flush=True)
    for block, by_schedule in blocks.items():
        results['cohorts'][f'{label(block[:2])}/cohort{block[2]}'] = {
            n: {e: f(g) for e, f in {**ENDPOINTS, **TACTICAL}.items()} for n, g in by_schedule.items()}

    pooled = {x.name: [g for by in data.values() for g in by[x.name]] for x in s.SCHEDULES}
    stratified = {name: a.stratified(f) for name, f in ENDPOINTS.items() if name[:2] in ('d1', 'd2', 'd3')}
    results['pooled'] = dict(
        tactics={n: a.tactical_profile(g) for n, g in pooled.items()},
        tactical_intervals={n: {e: a.jackknife(g, f) for e, f in TACTICAL.items()} for n, g in pooled.items()},
        paired_tactics_vs_original={n: comparisons(pooled['original'], pooled[n], TACTICAL) for n in ORDER},
        paired_diversity_vs_original_mean_of_checkpoints={
            n: comparisons(pooled['original'], pooled[n], stratified) for n in ORDER})

    diversity = {n: classify_diversity(n, data, blocks, results['paired_vs_original']) for n in ORDER}
    tactics = {n: classify_tactics(results['pooled']['paired_tactics_vs_original'][n]) for n in ORDER}
    decision = dict(diversity=diversity, tactics=tactics,
                    recommendation=recommend(diversity, tactics, d1_gain(data)), criteria=CRITERIA)
    results['post_hoc'] = dict(robustness=robustness(data, pooled, tactics), exact_values=exact_context(data, pooled))
    print('post hoc exact values and robustness', flush=True)

    # Replay analysis: historical reference, dataset profiles and counterfactual sampling.
    results['archived'], results['replay'], results['counterfactual_replay'], checks = {}, {}, {}, {}
    for checkpoint, by_schedule in data.items():
        key, seed = label(checkpoint), checkpoint[0]
        payload, archived_games, archive = archived_reference(campaign, seed)
        smallest = min(sum(len(g['plies']) for g in gs) for gs in by_schedule.values())
        budget = dict(draws=4 * smallest, rarefaction_sizes=[1000, 2000, 3000, smallest])
        results['replay'][key] = dict(
            equal_budget_rule='4 x the smallest schedule example count at this checkpoint',
            **{n: replay.profile(g, **budget) for n, g in by_schedule.items()},
            archived_g20_256_games=replay.profile(archived_games, **budget),
            archived_g20_first_128_games=replay.profile(archived_games[:128], **budget),
            archived_g20_last_128_games=replay.profile(archived_games[128:], **budget))
        window = replay.archived_window(payload, range(13, 21))
        updates = V2Config().updates_for(sum(len(g['plies']) for g in by_schedule['original']))
        results['counterfactual_replay'][key] = {
            n: replay.counterfactual(window, replay.completed_games(g, 21), updates=updates, seeds=SAMPLING_SEEDS)
            for n, g in by_schedule.items()}
        full = GenerationReplay(8, 2048)
        for generation in sorted(window):
            full.add_generation(generation, window[generation])
        examples = [e for g in full.iter_games() for e in g.examples]
        diagnostics = audit.read(audit.checkpoint(campaign, seed, 20).parent / 'summary.json')['summary']['diagnostics']
        window_states = (dict(unique_positions=len({(e.actor, e.observation) for e in examples}),
                              unique_reflection_families=len({family(e) for e in examples}))
                         == diagnostics['replay'])
        results['archived'][key] = archive
        checks[key] = cross_check(archive, archived_games, seed, window_states)
        print(f'replay {key}', flush=True)
    results['archive_cross_checks'] = checks

    if args.supplement is not None:
        extra, extra_cells, _ = a.load_collection(args.supplement)
        hashed += verify_record_hashes(extra, extra_cells)
        extra_data = datasets(matched_blocks(extra))
        results['supplement'] = {label(c): dict(
            conditions={n: condition_summary(g) for n, g in v.items()},
            paired_vs_original={n: comparisons(v['original'], v[n], {**ENDPOINTS, **TACTICAL}) for n in ORDER},
            note='Descriptive only; excluded from every decision rule.') for c, v in extra_data.items()}

    results['performance'] = performance(directories)
    regression = audit.read(args.regression / 'regression.json')
    results['regression'] = regression
    unchanged = subprocess.run(['git', 'diff', '--quiet', BASE_COMMIT, '--',
                                *(f'games/connect4/alphazero_v2/{f}' for f in UNCHANGED_SOURCES),
                                'games/connect4/agents/mcts_nn_agent.py', 'games/connect4/connect4.py']).returncode == 0
    perf = results['performance']
    acceptance = [
        ('Original schedule matches retained Phase 4D.4C evidence exactly',
         all(c['records_exact'] and c.get('rng_exact', True) for c in regression if 'repeated' not in c['check']),
         'regression.json: default and explicit ORIGINAL vs the 4C pool-eight trial (314159/g20, 16 games, records and '
         'RNG hashes) and the 4C validation games (42/g20, 4 games, records; default vs explicit RNG states equal)'),
        *([] if args.regression_final is None else [(
            'Delivered collector source reproduces the first regression and the 4C evidence',
            [(c['records_sha256'], c['rng_sha256']) for c in audit.read(args.regression_final / 'cells.json')]
            == [(c['records_sha256'], c['rng_sha256']) for c in audit.read(args.regression / 'cells.json')]
            and all(v for c in audit.read(args.regression_final / 'regression.json')
                    for k, v in c.items() if k.endswith('_exact')),
            'post-review regression with the delivered batching.py: all ten cells equal in records and RNG hashes')]),
        ('Same checkpoint/config/identity reproduces records and RNG states',
         all(c['records_exact'] and c['rng_exact'] for c in regression if 'repeated' in c['check']),
         'regression.json: each new schedule collected twice'),
        ('No schedule-dependent RNG draws; matched identities share searched prefixes',
         all(c.get('prefix_exact', True) for c in regression) and all(
             r['mismatches'] == 0 for r in results['negative_control']),
         f'{sum(r["identities"] for r in results["negative_control"])} matched prefix comparisons, '
         f'{sum(r["mismatches"] for r in results["negative_control"])} mismatches'),
        ('Exactly 256 simulations per search', all(c['simulations'] == 256 * c['plies'] for c in perf['cells']),
         'collector accounting per cell and stored root visits summing to 256 for every analyzed position'),
        ('Network frozen; checkpoints hash-verified', perf['weights_frozen'],
         'weights SHA-256 before/after every cell; checkpoint file hashes in provenance.json'),
        ('Search, noise, config defaults and target code unchanged from 99cec1c', unchanged,
         'git diff --quiet 99cec1c for config/search/selfplay/data/bitboard/network/mcts_nn_agent/connect4'),
        # The loader raises on any violation, so reaching this point is the evidence for the next three rows.
        ('Targets are normalized raw visits; temperature affects execution only', True,
         'loader raises unless execution distribution == visit_policy(visits, temperature) and tau-0 actions are '
         f'maximum-visit; {sum(len(g["plies"]) for g in games)} main positions checked'),
        ('Pre-move positions, actor-relative outcomes and legal replay of every trajectory', True,
         f'{len(games)} main games replayed through the engine; {hashed} main and supplement games rebuilt and '
         'matched to the collector record hashes'),
        ('No incomplete game or unmatched cell analyzed', len(analyzed) == sum(
            len(g) for v in data.values() for g in v.values()),
         f'{len(blocks)} complete matched blocks; {len(skipped)} incomplete cells excluded'),
        ('Every schedule applied correctly by ply', True,
         'loader raises unless every recorded temperature equals its schedule at that ply'),
        ('Mirror and canonical counting agree with independent Phase 4D.4A and campaign figures',
         all(all(v.values()) for v in checks.values()), 'archive_cross_checks in results.json'),
        ('Official campaign preserved in every invocation',
         all(i['official_unchanged'] and i['official_files'] == 150 and i['declaration_ok'] for i in perf['invocations']),
         'preservation.json of each invocation: 150 files byte-identical, declaration SHA-256 unchanged'),
        ('Measured execution under 30 minutes', perf['within_budget'],
         f'{perf["measured_seconds"]:.1f} s across {len(perf["invocations"])} invocations'),
        ('All declared primary cells completed', not skipped and len(blocks) == 4 and all(
            i['complete'] for i in perf['invocations']), f'{len(blocks)} of 4 checkpoint x cohort blocks complete'),
    ]
    acceptance = [dict(criterion=c, result='PASS' if ok else 'FAIL', evidence=e) for c, ok, e in acceptance]
    return clean(results), clean(decision), acceptance


# Markdown tables ----------------------------------------------------------------------------

def pct(value, digits=1):
    return '—' if value is None else f'{100 * value:.{digits}f}%'


def points(d, digits=1):
    if d['estimate'] is None or d['se'] is None:
        return '—'
    return f'{100 * d["estimate"]:+.{digits}f} [{100 * d["low"]:+.{digits}f}, {100 * d["high"]:+.{digits}f}]'


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(header))) + '|']
    return '\n'.join(lines + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]) + '\n'


def tables(results, decision, acceptance):
    names = [x['name'] for x in results['schedules']]
    out = ['# Phase 4D.4D result tables\n', 'Generated by `research.alphazero_exploration.analyze` from complete '
           'games only. Intervals are delete-one-game jackknife 95% intervals (descriptive).\n']
    for key, conditions in results['conditions'].items():
        out.append(f'## {key}: diversity\n')
        rows = []
        for n in names:
            d = conditions[n]['diversity']
            rows.append([n, d['games'], d['positions'], f'{d["length"]["mean"]:.2f}', d['unique_trajectories'],
                         d['unique_positions'], pct(d['duplicate_fraction']), pct(d['mirror_canonical_duplicate_fraction']),
                         pct(d['stages']['8+']['duplicate_fraction']),
                         f'{d["stages"]["8+"]["collision_probability"]:.4f}'])
        out.append(table(['Schedule', 'Games', 'Positions', 'Mean length', 'Unique trajectories', 'Unique positions',
                          'Duplicates', 'Mirror duplicates', 'Duplicates ply 8+ (D1)', 'Collision ply 8+ (D3)'], rows))
        rows = []
        for n in names:
            d = conditions[n]['diversity']
            rows.append([n] + [f'{pct(d["stages"][st]["duplicate_fraction"])} ({d["stages"][st]["unique_positions"]}/'
                               f'{d["stages"][st]["positions"]})' for st in ('0-7', '8-15', '16-23', '24+')])
        out.append('Stage-conditioned duplicate fraction (unique/positions):\n')
        out.append(table(['Schedule', 'Plies 0-7', 'Plies 8-15', 'Plies 16-23', 'Plies 24+'], rows))
        rows = []
        for n in names:
            d = conditions[n]['diversity']
            rows.append([n] + [f'{d["prefixes"][k]["unique"]}/{d["prefixes"][k]["eligible_games"]}'
                               for k in ('8', '12', '16', '20', '24')]
                        + [f'{d["families"][k]["unique"]} ({pct(d["families"][k]["top1_share"])})' for k in ('8', '16', '24')])
        out.append('Unique move prefixes / eligible games, and mirror-canonical families (top-family share):\n')
        out.append(table(['Schedule', '8-ply', '12-ply', '16-ply', '20-ply', '24-ply', 'Families @8', 'Families @16',
                          'Families @24'], rows))
        rows = []
        for n in names:
            st = conditions[n]['diversity']['stages']
            rows.append([n, f'{st["0-7"]["target_entropy"]:.3f}', f'{st["8+"]["target_entropy"]:.3f}',
                         f'{st["8+"]["max_target"]:.3f}', f'{st["8+"]["execution_entropy"]:.3f}',
                         pct(st['8+']['executed_top']), f'{st["8+"]["prior_entropy"]:.3f}', f'{st["8+"]["max_prior"]:.3f}'])
        out.append(table(['Schedule', 'Target entropy 0-7', 'Target entropy 8+', 'Max target 8+', 'Execution entropy 8+',
                          'Executed = top visit 8+', 'Prior entropy 8+', 'Max prior 8+'], rows))
        out.append(f'### {key}: paired differences from original (percentage points)\n')
        rows = []
        for n in ORDER:
            p = results['paired_vs_original'][key][n]
            rows.append([n, points(p['d1_duplicate_fraction_8_plus']), points(p['d2_mirror_duplicate_fraction_8_plus']),
                         points(p['d3_collision_probability_8_plus'], 2), points(p['duplicate_fraction_8_15']),
                         points(p['duplicate_fraction_16_23']), points(p['duplicate_fraction_24_plus'])])
        out.append(table(['Schedule', 'D1', 'D2', 'D3', 'Dup 8-15', 'Dup 16-23', 'Dup 24+'], rows))
        out.append(f'### {key}: tactics (executed actions)\n')
        rows = []
        for n in names:
            c = conditions[n]['tactics']['counts']
            get = lambda k: c.get(k, 0)
            rows.append([n, f'{get("missed_wins")}/{get("win_opps")}',
                         f'{get("missed_wins:search")}/{get("missed_wins:sampling")}/{get("missed_wins:tie")}',
                         f'{get("unsafe")}/{get("choice_states")}',
                         f'{get("unsafe:search")}/{get("unsafe:sampling")}/{get("unsafe:tie")}',
                         f'{get("missed_blocks")}/{get("blockable_states")}', get('game_ending_errors'),
                         get('tactical_states'), conditions[n]['diversity']['unique_tactical_positions']])
        out.append(table(['Schedule', 'Missed wins / opportunities', 'search/sampling/tie', 'Unsafe / choice states',
                          'search/sampling/tie', 'Missed blocks / blockable', 'Game-ending errors', 'Tactical states',
                          'Unique tactical'], rows))
    out.append('## Pooled primary checkpoints: tactical endpoints\n')
    rows = []
    for n in names:
        t, c = results['pooled']['tactics'][n]['rates'], results['pooled']['tactics'][n]['counts']
        rows.append([n, f'{pct(t["missed_win_rate"])} ({c.get("missed_wins", 0)}/{c.get("win_opps", 0)})',
                     f'{pct(t["unsafe_rate"], 2)} ({c.get("unsafe", 0)}/{c.get("choice_states", 0)})',
                     f'{t["errors_per_game"]:.3f}', pct(t['game_ending_error_share']),
                     pct(t['greedy_missed_win_rate']), pct(t['greedy_unsafe_rate'], 2)])
    out.append(table(['Schedule', 'T1 missed-win rate', 'T2 unsafe rate', 'T3 errors/game', 'T4 game-ending share',
                      'Greedy missed-win', 'Greedy unsafe'], rows))
    rows = []
    for n in ORDER:
        e = decision['tactics'][n]['endpoints']
        rows.append([n, points(e['missed_win_rate']), points(e['unsafe_rate'], 2),
                     f'{e["errors_per_game"]["estimate"]:+.3f} [{e["errors_per_game"]["low"]:+.3f}, '
                     f'{e["errors_per_game"]["high"]:+.3f}]', points(e['game_ending_error_share']),
                     decision['tactics'][n]['classification'], decision['diversity'][n]['classification']])
    out.append('Paired differences from original (margins: T1 +5.0, T2 +2.0 points, T3 +0.15, T4 +5.0 points):\n')
    out.append(table(['Schedule', 'T1', 'T2', 'T3', 'T4', 'Tactics', 'Diversity'], rows))
    out.append('Per-checkpoint paired tactical differences (the decision pools both checkpoints):\n')
    rows = []
    for key in results['paired_vs_original']:
        for n in ORDER:
            q = results['paired_vs_original'][key][n]
            rows.append([f'{key} {n}', points(q['missed_win_rate']), points(q['unsafe_rate'], 2),
                         f'{q["errors_per_game"]["estimate"]:+.3f} [{q["errors_per_game"]["low"]:+.3f}, '
                         f'{q["errors_per_game"]["high"]:+.3f}]', points(q['game_ending_error_share'])])
    out.append(table(['Checkpoint / schedule', 'T1', 'T2', 'T3', 'T4'], rows))
    post = results.get('post_hoc')
    if post:
        out.append('## Post hoc (after review): exact-value context of the one-reply events\n')
        out.append(post['exact_values']['note'] + '\n')
        rows = []
        for n in names:
            c = post['exact_values']['pooled'][n]['counts']
            get = lambda k: c.get(k, 0)
            rows.append([n, get('unsafe'), get('unsafe:already_lost'), get('unsafe:value_changing'),
                         get('missed_wins'), get('missed_wins:still_winning'), get('game_ending_errors'),
                         get('game_ending_errors:already_lost'), get('game_ending_errors:value_changing'),
                         f'{get("game_ending_errors:value_changing:search")}/{get("game_ending_errors:value_changing:sampling")}',
                         f'{post["exact_values"]["pooled"][n]["rates"]["value_changing_errors_per_game"]:.3f}'])
        out.append(table(['Schedule', 'Unsafe', 'already lost', 'value-changing', 'Missed wins', 'still winning',
                          'Game-ending', 'already lost', 'value-changing', 'value-changing search/sampling',
                          'Value-changing errors/game'], rows))
        rows = []
        for n in ORDER:
            q = post['exact_values']['pooled_paired_vs_original'][n]
            e, g = q['value_changing_errors_per_game'], q['value_changing_game_ending_share']
            r = post['robustness'][n]
            rows.append([n, f'{e["estimate"]:+.3f} [{e["low"]:+.3f}, {e["high"]:+.3f}]', points(g),
                         r['most_influential_identity']['classification_without_it'],
                         '; '.join(f'{k} {v["replicates_below_relative_threshold"]}/{v["delete_one_replicates"]}'
                                   for k, v in r['diversity'].items())])
        out.append(table(['Schedule', 'Value-changing errors/game vs original', 'Value-changing game endings (points)',
                          'Tactics label without the most influential identity',
                          'Delete-one replicates below the 25% D1 threshold'], rows))
    out.append('## Correctness and execution\n')
    out.append(table(['Criterion', 'Result', 'Evidence'], [[r['criterion'], r['result'], r['evidence']] for r in acceptance])
               .replace('|---|---:|---:|', '|---|---|---|'))
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    for name in ('regression', 'pilot', 'main'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--regression-final', type=Path, help='post-review regression with the delivered source')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    require(audit.sha(campaign / 'declaration.json') == audit.DECLARATION, 'Declaration changed')
    output = audit.fresh_output(campaign, args.output)
    before = audit.campaign_fingerprint(campaign)
    require(before == audit.read(BASE / 'profile/campaign_before.json') and len(before) == 150,
            'Official 150-file inventory changed')
    try:
        results, decision, acceptance = analyze(args)
        audit.write(output / 'results.json', results)
        audit.write(output / 'decision.json', decision)
        audit.write(output / 'acceptance.json', acceptance)
        (output / 'TABLES.md').write_text(tables(results, decision, acceptance))
    finally:
        after = audit.campaign_fingerprint(campaign)
        audit.write(output / 'preservation.json', dict(files=len(after), unchanged=before == after,
                                                      declaration_sha256=audit.sha(campaign / 'declaration.json')))
        require(before == after, 'Official campaign modified')
    print(json.dumps(dict(recommendation=decision['recommendation']['recommended_schedule'],
                          failed=[r['criterion'] for r in acceptance if r['result'] != 'PASS'])), flush=True)


if __name__ == '__main__':
    main()
