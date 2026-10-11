"""Rebuild compact JSON and Markdown tables from a completed baseline audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from .audit import fresh_output, read, safe_moves_scan, winning_moves_scan, write


def summarize(profile, positions, ladder, archive, state):
    searches = read(profile/'profile.json')
    exclusive = Counter()
    inclusive = Counter()
    for r in searches:
        exclusive.update(r['timing']['exclusive_seconds'])
        exclusive['other'] += r['timing']['other_seconds']
        inclusive.update(r['timing']['inclusive_seconds'])
    total = sum(exclusive.values())
    performance = []
    for position in dict.fromkeys(r['position'] for r in searches):
        for budget in (256,512):
            rows = [r for r in searches if r['position']==position and r['simulations']==budget]
            performance.append(dict(position=position,simulations=budget,
                median_of_checkpoint_medians_ms=statistics.median(r['wall']['median'] for r in rows)*1000,
                checkpoint_median_range_ms=[min(r['wall']['median'] for r in rows)*1000,
                                           max(r['wall']['median'] for r in rows)*1000],
                mean_inference_calls=statistics.mean(t['inference_calls'] for r in rows for t in r['trials']),
                median_simulations_per_second=statistics.median(t['simulations_per_second'] for r in rows for t in r['trials'])))
    position_summary = []
    for r in read(positions/'positions_assessment.json'):
        values = r['solved_records']
        tactics = r['tactical_records']
        position_summary.append(dict(seed=r['seed'],generation=r['generation'],
            immediate_win=r['tactics']['unique_win']['accuracy'],safe_response=r['tactics']['unique_safe']['accuracy'],
            nn_immediate_win=r['tactics']['unique_win']['nn_accuracy'],nn_safe_response=r['tactics']['unique_safe']['nn_accuracy'],
            solved_class_balanced_mse=statistics.mean(x['mse'] for x in r['values'].values()),
            nn_solved_optimal_fraction=statistics.mean(x['nn_optimal'] for x in values),
            tactical_avoidable_errors=sum(x['avoidable_error'] for x in tactics),
            policy_entropy_tactical=statistics.mean(x['policy_entropy'] for x in tactics),
            search_move_counts=dict(Counter(x['move'] for x in tactics)),
            nn_move_counts=dict(Counter(x['nn_move'] for x in tactics))))
    ladder_records = read(ladder/'ladder.json') if (ladder/'ladder.json').exists() else [
        json.loads(line) for line in (ladder/'ladder.jsonl').read_text().splitlines()]
    ladder_summary = []
    for r in ladder_records:
        errors, decisions = Counter(), 0
        for game in r['records']:
            if game['abandoned']:
                continue
            moves = game['moves']
            for ply in range(game['prefix_length'],len(moves)):
                if ply % 2 != game['agent_color']:
                    continue
                decisions += 1
                wins = winning_moves_scan(moves[:ply])
                safe = safe_moves_scan(moves[:ply])
                if wins:
                    errors['immediate_win_opportunities'] += 1
                    errors['missed_immediate_wins'] += moves[ply] not in wins
                elif safe and len(safe) < 7-sum(moves[:ply].count(a)==6 for a in range(7)):
                    errors['safe_response_opportunities'] += 1
                    errors['unsafe_responses'] += moves[ply] not in safe
        ladder_summary.append(dict(seed=r['seed'],generation=r['generation'],opponent=r['opponent'],
                                    **r['summary'],agent_decisions=decisions,agent_tactics=dict(errors)))
    archived = read(archive/'archives.json')
    archive_summary = []
    for seed in sorted({r['seed'] for r in archived}):
        rows = [r for r in archived if r['seed']==seed]
        collection = sum(r['resources']['collection_seconds'] for r in rows)
        training = sum(r['resources']['training_seconds'] for r in rows)
        archive_summary.append(dict(seed=seed,games=sum(r['summary']['games'] for r in rows),
            positions=sum(r['positions'] for r in rows),collection_seconds=collection,training_seconds=training,
            collection_fraction=collection/(collection+training),
            snapshots=[dict(seed=seed,generation=r['generation'],length_mean=r['length']['mean'],
                unique_positions=r['unique_positions'],unique_mirror_families=r['unique_mirror_families'],
                duplicate_position_fraction=r['duplicate_position_fraction'],after8_duplicate_fraction=r['after8_duplicate_fraction'],
                unique_opening8=r['unique_opening8'],top_opening8_count=r['top_opening8_count'],
                unique_trajectories=r['unique_trajectories'],max_trajectory_multiplicity=r['max_trajectory_multiplicity'],
                policy_loss=r['summary']['mean_policy_loss'],value_mse=r['summary']['mean_value_mse'],
                replay_positions=r['summary']['replay_positions'],updates=r['summary']['updates'],
                sampled_positions=r['summary']['sampled_positions'],archived_tactics=r['archived_tactics'])
                for r in rows if r['generation'] in (1,10,15,20)]))
    return dict(schema='phase4d4a-baseline-summary-v1',performance=performance,
        exclusive_runtime_percent={k:v/total*100 for k,v in exclusive.items()},
        inclusive_runtime_percent={k:v/total*100 for k,v in inclusive.items()},
        instrumentation_overhead_ratio_median=statistics.median(r['timing']['instrumented_wall_seconds']/r['wall']['median'] for r in searches),
        cpu_wall_ratio_median=statistics.median(t['cpu_seconds']/t['wall_seconds'] for r in searches for t in r['trials']),
        profiles= len(searches),latency_trials=sum(len(r['trials']) for r in searches),
        positions=position_summary,ladder=ladder_summary,archives=archive_summary,
        official_development=read(archive/'official_development.json'),
        replay_exploration=read(archive/'replay_exploration.json'),state=read(state/'state_microbenchmark.json'),
        memory=read(profile/'memory.json'),forward_batches=read(profile/'forward_batches.json'))


def tables(summary):
    lines = ['# Measured baseline tables','',
             'Latency: median of eight checkpoint medians, five uninstrumented trials per cell.','',
             '| Position | 256 ms | 512 ms | Mean forward calls at 512 |',
             '|---|---:|---:|---:|']
    for position in dict.fromkeys(r['position'] for r in summary['performance']):
        a,b = [r for r in summary['performance'] if r['position']==position]
        lines.append(f'| {position} | {a["median_of_checkpoint_medians_ms"]:.2f} | {b["median_of_checkpoint_medians_ms"]:.2f} | {b["mean_inference_calls"]:.1f} |')
    lines += ['', '| Seed | Gen | Win accuracy | Safe accuracy | NN win | NN safe | Value MSE | NN solved optimal |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['positions']:
        lines.append(f'| {r["seed"]} | {r["generation"]} | {r["immediate_win"]:.0%} | {r["safe_response"]:.0%} | '
                     f'{r["nn_immediate_win"]:.0%} | {r["nn_safe_response"]:.0%} | {r["solved_class_balanced_mse"]:.4f} | {r["nn_solved_optimal_fraction"]:.1%} |')
    lines += ['', 'New ladder: score = (wins + 0.5 draws) / games. Tiny samples; no significance claim.','',
              '| Seed | Gen | Opponent | W/D/L | X score | O score | Score | Family bootstrap 95% |',
              '|---|---:|---|---|---:|---:|---:|---|']
    for r in summary['ladder']:
        interval = r['interval95']
        lines.append(f'| {r["seed"]} | {r["generation"]} | {r["opponent"]} | {r["wins"]}/{r["draws"]}/{r["losses"]} | '
                     f'{r["by_color"]["X"]["score"]:.0%} | {r["by_color"]["O"]["score"]:.0%} | {r["score"]:.1%} | '
                     f'{interval[0]:.0%}–{interval[1]:.0%} |')
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profile','positions','ladder','archive','state','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    campaign = read(args.profile/'provenance.json')['parameters']['campaign']
    fresh_output(campaign,args.output)
    result = summarize(args.profile,args.positions,args.ladder,args.archive,args.state)
    write(args.output/'summary.json',result)
    (args.output/'TABLES.md').write_text(tables(result))


if __name__ == '__main__':
    main()
