"""Opening-cluster analysis; no Elo conversion or outcome-driven stopping."""
import argparse
import itertools
import json
import random
import statistics
from pathlib import Path

from scripts.evaluate_mcts_strength import atomic_json, digest, load_rows, validate_rows


def percentile(values, q):
    ordered = sorted(values)
    offset = (len(ordered) - 1) * q
    lo = int(offset)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (offset - lo)


def bootstrap(values, strata, seed, repetitions=10000):
    """Resample independent opening IDs within fixed opening-length strata.

    Each value already contains both color games. Joint budget differences
    enter as one value per shared opening, preserving condition dependence.
    """
    if not values:
        return None
    rng = random.Random(seed)
    groups = [[values[key] for key in sorted(values) if strata[key] == group]
              for group in sorted({strata[key] for key in values})]
    samples = [sum(sum(rng.choices(group, k=len(group))) for group in groups) / len(values)
               for _ in range(repetitions)]
    return dict(mean=statistics.mean(values.values()),
                ci95=[percentile(samples, .025), percentile(samples, .975)],
                ci98_75=[percentile(samples, .00625), percentile(samples, .99375)],
                independent_openings=len(values))


def summarize(config, rows):
    validate_rows(config, rows)
    strata = {o['id']: o['length'] for o in config['openings']}
    seed = config['analysis']['bootstrap_seed']
    reps = config['analysis']['replicates']
    summary, pair_values = [], {}
    for matchup in config['matchups']:
        games = [r for r in rows if r['matchup'] == matchup['id']]
        grouped = {}
        for game in games:
            grouped.setdefault(game['opening'], []).append(game)
        complete = {key: group for key, group in grouped.items() if len(group) == 2}
        values = {key: statistics.mean(r['challenger_score'] for r in group)
                  for key, group in complete.items()}
        pair_values[matchup['id']] = values
        interval = bootstrap(values, strata, seed, reps)
        wdl = {label: sum(r['challenger_score'] == score for r in games)
               for label, score in [('wins', 1), ('draws', .5), ('losses', 0)]}
        colors = {str(c): statistics.mean(r['challenger_score'] for r in games
                    if r['challenger_color'] == c) if any(r['challenger_color'] == c for r in games) else None
                  for c in (0, 1)}
        times = {}
        for role in ('challenger', 'opponent'):
            moves = [m for r in games for m in r['moves'] if m['role'] == role]
            wall = [m['wall_seconds'] for m in moves]
            times[role] = dict(search_calls=len(moves), wall_seconds=sum(wall),
                cpu_seconds=sum(m['cpu_seconds'] for m in moves),
                median_ms=1000 * statistics.median(wall) if wall else None,
                mean_ms=1000 * statistics.mean(wall) if wall else None,
                p95_ms=1000 * percentile(wall, .95) if wall else None,
                maximum_ms=1000 * max(wall) if wall else None,
                tactical_bypasses=sum(m['simulations'] == 0 for m in moves),
                executed_simulations=sum(m['simulations'] or 0 for m in moves))
        summary.append(dict(matchup=matchup['id'], primary=matchup['primary'], planned_pairs=matchup['pairs'],
            completed_games=len(games), complete_pairs=len(complete), unpaired_games=len(games) - 2 * len(complete),
            **wdl, empirical_all_game_score=statistics.mean(r['challenger_score'] for r in games) if games else None,
            color_score_rates=colors, paired_score=interval,
            paired_advantage={key: 2 * val - 1 for key, val in values.items()},
            paired_advantage_ci95=[2 * x - 1 for x in interval['ci95']] if interval else None,
            primary_improvement_familywise=bool(interval and matchup['primary'] and interval['ci98_75'][0] > .5),
            compute=times))
    comparisons = []
    for opponent in ('mcts-400', 'negamax-4', 'negamax-6'):
        keys = [m['id'] for m in config['matchups'] if m['id'].endswith(f'-vs-{opponent}')]
        for low, high in itertools.combinations(keys, 2):
            shared = pair_values[low].keys() & pair_values[high].keys()
            differences = {key: pair_values[high][key] - pair_values[low][key] for key in sorted(shared)}
            comparisons.append(dict(lower=low, higher=high,
                paired_score_difference=bootstrap(differences, strata, seed, reps)))
    return dict(experiment_sha256=digest(config), method=config['analysis'], matchups=summary,
                budget_comparisons=comparisons, complete_games=len(rows),
                total_search_calls=sum(len(r['moves']) for r in rows),
                total_search_wall_seconds=sum(m['wall_seconds'] for r in rows for m in r['moves']),
                total_game_seconds=sum(r['elapsed_seconds'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads((args.directory / 'experiment.json').read_text())
    result = summarize(config, load_rows(args.directory / 'results.jsonl'))
    atomic_json(args.directory / 'analysis.json', result)
    for row in result['matchups']:
        print(row['matchup'], row['wins'], row['draws'], row['losses'], row['paired_score'])


if __name__ == '__main__':
    main()
