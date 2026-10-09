"""Trajectory-cluster inference and cost summaries for Phase 3A."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path

from scripts.analyze_mcts_strength import percentile
from scripts.evaluate_mcts_strength import atomic_json, digest, load_rows
from scripts.evaluate_negamax_depths import validate


def interval(values, openings, seed, repetitions=10000):
    if not values:
        return None
    groups = {}
    for key,value in values.items():
        o=openings[key]
        groups.setdefault(o['cohort'],{}).setdefault(o['family'],[]).append(value)
    rng=random.Random(seed)
    samples=[]
    cohorts=list(groups.values())
    for _ in range(repetitions):
        selected=[v for cohort in cohorts for family in rng.choices(list(cohort.values()),k=len(cohort)) for v in family]
        samples.append(statistics.mean(selected))
    weights=[len(f)/len(values) for c in cohorts for f in c.values()]
    # Supplementary bounded-variable interval guards degenerate bootstrap CIs.
    # Valid only under independent trajectory generation, not a cure for selection.
    radius=math.sqrt(math.log(2/.025)*sum(w*w for w in weights)/2)
    mean=statistics.mean(values.values())
    return dict(mean=mean, ci95=[percentile(samples,.025),percentile(samples,.975)],
                ci97_5=[percentile(samples,.0125),percentile(samples,.9875)],
                conservative97_5=[max(0,mean-radius),min(1,mean+radius)],
                openings=len(values), clusters=sum(len(c) for c in cohorts),
                clusters_by_cohort={c:len(g) for c,g in groups.items()},
                caution='Approximate percentile coverage; small/degenerate strata can understate uncertainty')


def describe(games):
    return dict(games=len(games), wins=sum(g['challenger_score']==1 for g in games),
                draws=sum(g['challenger_score']==.5 for g in games), losses=sum(g['challenger_score']==0 for g in games),
                score=statistics.mean(g['challenger_score'] for g in games) if games else None,
                red=statistics.mean(g['challenger_score'] for g in games if g['challenger_color']==0) if any(g['challenger_color']==0 for g in games) else None,
                yellow=statistics.mean(g['challenger_score'] for g in games if g['challenger_color']==1) if any(g['challenger_color']==1 for g in games) else None)


def summarize(config,rows):
    validate(config,rows)
    openings={o['id']:o for o in config['openings']}
    seed,reps=config['analysis']['seed'],config['analysis']['replicates']
    results, pair_values = [],{}
    for m in config['matchups']:
        games=[r for r in rows if r['matchup']==m['id']]
        grouped={}
        for g in games:
            grouped.setdefault(g['opening'],[]).append(g)
        values={key:statistics.mean(g['challenger_score'] for g in group) for key,group in grouped.items() if len(group)==2}
        pair_values[m['id']]=values
        strata={}
        for dimension in ('cohort','stage','tactics'):
            strata[dimension]={}
            for label in sorted({o[dimension] for o in openings.values()}):
                selected=[g for g in games if openings[g['opening']][dimension]==label]
                subset={k:v for k,v in values.items() if openings[k][dimension]==label}
                strata[dimension][label]=dict(**describe(selected), paired=interval(subset,openings,seed,reps))
        # Cross cohort/stage is primary descriptive view, preceding pooled estimate.
        cross={f'{cohort}/{stage}':describe([g for g in games if openings[g['opening']]['cohort']==cohort and openings[g['opening']]['stage']==stage])
               for cohort in ('random','agent') for stage in ('early','midgame','late')}
        ci=interval(values,openings,seed,reps)
        results.append(dict(matchup=m['id'], primary=m['primary'], planned_pairs=m['pairs'],
                            **describe(games), complete_pairs=len(values), unpaired_games=len(games)-2*len(values),
                            strata=strata, cohort_stage=cross, paired_score=ci,
                            pair_scores=values, primary_bootstrap_improvement=bool(m['primary'] and ci and ci['ci97_5'][0]>.5),
                            primary_conservative_improvement=bool(m['primary'] and ci and ci['conservative97_5'][0]>.5)))
    contrasts=[]
    for a,b in (('n8-vs-m2000','n10-vs-m2000'),('n10-vs-m2000','n10-vs-m5000')):
        if a not in pair_values or b not in pair_values:
            continue
        shared=pair_values[a].keys() & pair_values[b].keys()
        # Transform [-1,1] differences to [0,1] for bounded interval, then restore.
        ci=interval({k:(pair_values[b][k]-pair_values[a][k]+1)/2 for k in shared},openings,seed,reps)
        if ci:
            ci={**ci, 'mean':2*ci['mean']-1, **{k:[2*x-1 for x in ci[k]] for k in ('ci95','ci97_5','conservative97_5')}}
        contrasts.append(dict(lower=a,higher=b,paired_score_difference=ci))
    cost={}
    for r in rows:
        for move in r['moves']:
            a=r[move['role']+'_config']
            name='n'+str(a['depth']) if a['type']=='negamax' else 'm'+str(a['simulations'])
            cost.setdefault(name,[]).append(move)
    cost={name:dict(calls=len(moves),wall_seconds=sum(m['wall_seconds'] for m in moves),
                   cpu_seconds=sum(m['cpu_seconds'] for m in moves),
                   median_ms=1000*statistics.median(m['wall_seconds'] for m in moves),
                   p95_ms=1000*percentile([m['wall_seconds'] for m in moves],.95),
                   maximum_ms=1000*max(m['wall_seconds'] for m in moves),
                   max_nodes=max((m['negamax_stats']['nodes'] for m in moves if m['negamax_stats']),default=0),
                   max_tt_entries=max((m['negamax_stats']['entries'] for m in moves if m['negamax_stats']),default=0))
          for name,moves in cost.items()}
    return dict(experiment_sha256=digest(config), matchups=results, contrasts=contrasts, compute=cost,
                games=len(rows), game_seconds=sum(r['elapsed_seconds'] for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads((args.directory/'experiment.json').read_text())
    result=summarize(config,load_rows(args.directory/'results.jsonl'))
    atomic_json(args.directory/'analysis.json',result)
    for row in result['matchups']:
        print(row['matchup'],row['wins'],row['draws'],row['losses'],row['paired_score'])


if __name__=='__main__':
    main()
