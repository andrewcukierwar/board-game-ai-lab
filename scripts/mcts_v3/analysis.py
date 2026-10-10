"""Opening-cluster analysis for MCTS v3 studies; no Elo, no outcome-driven stopping.

    python -m scripts.mcts_v3.analysis --study pilot1

Kept apart from the harness so analysis changes never alter the pinned
game-playing sources of a declared study.
"""
import argparse
import json
import random
import statistics

from scripts.evaluate_mcts_strength import atomic_json, digest, load_rows
from scripts.mcts_v3.harness import FAMILY, ROOT, schedule, validate_rows


def role_of(config_row, opening, index):
    """Role that made move ``index`` of a game from ``opening``."""
    mover = (len(opening['history']) + index) % 2
    return 'challenger' if mover == config_row['challenger_color'] else 'opponent'


def percentile(values, q):
    ordered = sorted(values)
    offset = (len(ordered) - 1) * q
    low = int(offset)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (offset - low)


def bootstrap(values, strata, seed, replicates, family=FAMILY):
    """Resample whole openings within opening-length strata."""
    rng = random.Random(seed)
    groups = [[values[key] for key in sorted(values) if strata[key] == group]
              for group in sorted({strata[key] for key in values})]
    samples = [sum(sum(rng.choices(group, k=len(group))) for group in groups) / len(values)
               for _ in range(replicates)]
    tail = 0.025 / family
    return dict(mean=statistics.mean(values.values()), openings=len(values),
                ci95=[percentile(samples, .025), percentile(samples, .975)],
                ci_family=[percentile(samples, tail), percentile(samples, 1 - tail)],
                family=family)


def sign_flip_p(advantages, seed, replicates):
    """Two-sided randomisation test of mean per-opening advantage = 0."""
    rng = random.Random(seed)
    observed = abs(sum(advantages))
    extreme = sum(abs(sum(a if rng.random() < .5 else -a for a in advantages)) >= observed - 1e-12
                  for _ in range(replicates))
    return (extreme + 1) / (replicates + 1)


def timing(games, openings, role):
    wall, simulations, searched_wall = [], [], 0
    for row in games:
        for index, (us, sims) in enumerate(zip(row['wall_us'], row['simulations'])):
            if role_of(row, openings[row['opening']], index) == role:
                wall.append(us / 1000)
                if sims:
                    simulations.append(sims)
                    searched_wall += us
    if not wall:
        return None
    return dict(decisions=len(wall), total_wall_seconds=sum(wall) / 1000,
                mean_ms=statistics.mean(wall), median_ms=statistics.median(wall),
                p95_ms=percentile(wall, .95), max_ms=max(wall),
                searched_decisions=len(simulations),
                mean_simulations=statistics.mean(simulations) if simulations else None,
                simulations_per_second=(sum(simulations) / (searched_wall / 1e6)
                                        if searched_wall else None))


def analyze(name):
    directory = ROOT / name
    config = json.loads((directory / 'study.json').read_text())
    rows = load_rows(directory / 'results.jsonl')
    validate_rows(config, rows)
    openings = {o['id']: o for o in config['openings']}
    strata = {key: o['length'] for key, o in openings.items()}
    seed, replicates = config['analysis']['bootstrap_seed'], config['analysis']['replicates']
    summary, pair_scores = [], {}
    for m in config['matchups']:
        games = [r for r in rows if r['matchup'] == m['id']]
        grouped = {}
        for game in games:
            grouped.setdefault(game['opening'], []).append(game['challenger_score'])
        values = {key: statistics.mean(group) for key, group in grouped.items() if len(group) == 2}
        pair_scores[m['id']] = values
        if not values:
            continue
        interval = bootstrap(values, strata, seed, replicates)
        times = {role: timing(games, openings, role) for role in ('challenger', 'opponent')}
        summary.append(dict(
            matchup=m['id'], primary=m['primary'], challenger=m['challenger'], opponent=m['opponent'],
            planned_pairs=m['pairs'], complete_pairs=len(values), games=len(games),
            wins=sum(r['challenger_score'] == 1 for r in games),
            draws=sum(r['challenger_score'] == .5 for r in games),
            losses=sum(r['challenger_score'] == 0 for r in games),
            score=interval, sign_flip_p=sign_flip_p([2 * v - 1 for v in values.values()], seed, replicates),
            color_scores={str(c): statistics.mean(r['challenger_score'] for r in games
                                                  if r['challenger_color'] == c) for c in (0, 1)},
            length_scores={str(length): statistics.mean(v for k, v in values.items() if strata[k] == length)
                           for length in sorted({strata[k] for k in values})},
            timing=times,
            time_ratio=times['challenger']['total_wall_seconds'] / times['opponent']['total_wall_seconds']))
    result = dict(study=name, study_sha256=digest(config), complete_games=len(rows),
                  planned_games=len(list(schedule(config))), matchups=summary,
                  total_game_seconds=sum(r['elapsed_seconds'] for r in rows))
    atomic_json(directory / 'analysis.json', result)
    return result, pair_scores, strata, config


def table(result):
    lines = ['| Matchup | W / D / L | Score | 95% interval | Family interval | Time ratio | Chal. ms | Opp. ms |',
             '| --- | --- | ---: | --- | --- | ---: | ---: | ---: |']
    for row in result['matchups']:
        s, t = row['score'], row['timing']
        lines.append(
            f'| {row["matchup"]} | {row["wins"]} / {row["draws"]} / {row["losses"]} '
            f'| {100 * s["mean"]:.1f}% | {100 * s["ci95"][0]:.1f}–{100 * s["ci95"][1]:.1f}% '
            f'| {100 * s["ci_family"][0]:.1f}–{100 * s["ci_family"][1]:.1f}% '
            f'| {row["time_ratio"]:.3f} | {t["challenger"]["mean_ms"]:.2f} | {t["opponent"]["mean_ms"]:.2f} |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    args = parser.parse_args()
    result = analyze(args.study)[0]
    print(table(result))
    print(f'{result["complete_games"]}/{result["planned_games"]} games')


if __name__ == '__main__':
    main()
