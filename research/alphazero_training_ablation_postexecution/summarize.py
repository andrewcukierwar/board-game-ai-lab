"""Descriptive mechanism/resource tables from frozen evidence, no new endpoints."""
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import time
import numpy as np
from research.alphazero_training_ablation.common import ROOT,OUTPUT,read,write
from research.alphazero_training_ablation.analyze import bootstrap,paired

def main():
    started=time.monotonic(); status=read(OUTPUT/'execution/status.json')
    analysis=read(ROOT/'results/analysis.json'); rows=[]
    require=lambda condition: None if condition else (_ for _ in ()).throw(ValueError('Invalid descriptive source'))
    totals=Counter(); collection_seconds=optimization_stage_seconds=0.
    for seed in (42,314159):
        for arm in ('control','treatment'):
            for generation in (21,22):
                base=OUTPUT/f'execution/{seed}-{arm}-g{generation}'
                complete=read(base/'complete.json'); games=read(base/'games.json')['games']
                extra=complete['extra']; result=complete['result']; metrics=result['metrics']
                n=result['new_positions']; samples=result['diagnostics']['sampling']['samples_by_source_generation']
                replay=extra['replay_counts']; replay_positions=sum(v['positions'] for v in replay.values())
                labels=Counter()
                for game in games:
                    for ply in range(len(game['moves'])):
                        labels[str(0 if game['winner']==-1 else 1 if game['winner']==ply%2 else -1)]+=1
                require(sum(labels.values())==n and len(metrics)==len(games)==64)
                training={field:dict(mean=float(np.mean([m[field] for m in metrics])),
                    first16=float(np.mean([m[field] for m in metrics[:16]])),last16=float(np.mean([m[field] for m in metrics[-16:]])),
                    min=float(min(m[field] for m in metrics)),max=float(max(m[field] for m in metrics)))
                    for field in ('combined_loss','policy_loss','value_mse','gradient_norm')}
                stages=complete['stage_resources']
                optimize=stages['before_checkpoint']['wall_seconds']-stages['before_update']['wall_seconds']
                collect=result['resources']['collection_seconds']
                forwards=result['collection']['forward_calls']; neural=result['collection']['neural_positions']
                require(sum(samples.values())==8192)
                rows.append(dict(seed=seed,arm=arm,generation=generation,games=64,positions=n,mean_length=n/64,
                    duplicates_rate=extra['duplicates']/n,mirror_duplicates_rate=extra['mirror_duplicates']/n,
                    unique_positions=n-extra['duplicates'],mirror_unique_positions=n-extra['mirror_duplicates'],
                    post8_duplicate_rate=extra['post_ply8_duplicates']/extra['post_ply8_positions'],
                    post8_mirror_duplicate_rate=extra['post_ply8_mirror_duplicates']/extra['post_ply8_positions'],
                    opening_families=extra['opening_families'],opening_concentration=extra['opening_concentration'],
                    policy_target_entropy=extra['target_entropy'],outcomes=dict(first_win=extra['first_player_wins'],second_win=extra['second_player_wins'],draw=extra['draws']),
                    outcome_labels=dict(labels),avoidable_oneply=extra['avoidable_one_reply_errors'],
                    missed_win=extra['missed_immediate_wins'],missed_safe=extra['missed_safe_responses'],
                    avoidable_rate=extra['avoidable_one_reply_errors']/n,missed_win_rate=extra['missed_immediate_wins']/n,
                    replay_positions=replay_positions,replay_generations=list(map(int,replay)),
                    experimental_replay_positions=extra['experimental_replay_positions'],
                    experimental_replay_fraction=extra['experimental_replay_positions']/replay_positions,
                    sampled_by_generation=samples,samples=8192,new_sample_fraction=extra['experimental_sample_exposure']/8192,
                    new_sample_appearances=extra['experimental_sample_exposure'],current_generation_sample_fraction=samples[str(generation)]/8192,
                    updates=64,training=training,gradients=result['diagnostics']['gradients'],
                    collection_seconds=collect,optimization_stage_seconds=optimize,
                    optimization_stage_definition='First before_update to before_checkpoint:64 updates plus summary/RNG bookkeeping, not pure optimizer timing',
                    collection_forward_calls=forwards,neural_positions=neural,mean_batch_occupancy=neural/forwards,
                    neural_positions_per_collection_second=neural/collect,
                    generation_task=read(base/'task-complete.json'),evaluations=analysis['learning'][str(seed)][f'{arm}-g{generation}']))
                totals.update(games=64,positions=n,updates=64,samples=8192,new_samples=extra['experimental_sample_exposure'],
                    collection_forward_calls=forwards,neural_positions=neural)
                collection_seconds+=collect; optimization_stage_seconds+=optimize
    resources=dict(wall_seconds=status['elapsed_seconds'],start_epoch=status['start_epoch'],
        finish_epoch=status['start_epoch']+status['elapsed_seconds'],
        start_eastern=datetime.fromtimestamp(status['start_epoch'],ZoneInfo('America/New_York')).isoformat(),
        finish_eastern_approx=datetime.fromtimestamp(status['start_epoch']+status['elapsed_seconds'],ZoneInfo('America/New_York')).isoformat(),
        finish_time_note='Finish epoch reconstructed from recorded start epoch plus monotonic elapsed; no separate exit timestamp retained',
        child_wall_seconds=sum(t['resources']['wall_seconds'] for t in status['tasks']),
        child_cpu_seconds=sum(t['resources']['cpu_seconds'] for t in status['tasks']),
        max_child_peak_rss_mib=max(t['resources']['peak_rss_mib'] for t in status['tasks']),
        collection_seconds=collection_seconds,optimization_stage_seconds=optimization_stage_seconds,
        task_wall_by_type={kind:sum(t['resources']['wall_seconds'] for t in status['tasks'] if t['task']==kind) for kind in ('train','positions','matches')},
        task_cpu_by_type={kind:sum(t['resources']['cpu_seconds'] for t in status['tasks'] if t['task']==kind) for kind in ('train','positions','matches')},
        max_generation_attempt_seconds=max(t['resources']['wall_seconds'] for t in status['tasks'] if t['task']=='train'),
        failed_attempts=0,retries=0,tasks=status['tasks'])
    resources['supervisor_and_process_wall_overhead']=resources['wall_seconds']-resources['child_wall_seconds']
    effects=[]
    for seed in (42,314159):
        for gen in (21,22):
            c=next(r for r in rows if (r['seed'],r['arm'],r['generation'])==(seed,'control',gen))
            t=next(r for r in rows if (r['seed'],r['arm'],r['generation'])==(seed,'treatment',gen))
            effects.append(dict(seed=seed,generation=gen,**{f:t[f]-c[f] for f in ('positions','mean_length','duplicates_rate',
                'mirror_duplicates_rate','post8_duplicate_rate','post8_mirror_duplicate_rate','opening_families','policy_target_entropy',
                'experimental_replay_fraction','new_sample_fraction','avoidable_rate','missed_win_rate')}))
    write(ROOT/'results/mechanism-resources.json',dict(rows=rows,totals=dict(totals),treatment_control_descriptive=effects,
        resources=resources,postexecution_summary_seconds=time.monotonic()-started,
        caveats=['Descriptive mechanism contrasts are postexecution, without new acceptance thresholds or causal claims.',
                 'Training losses target sampled visit distributions/selfplay outcomes; solved-suite metrics use exact optimal sets/WDL. Their difference is not a valid generalization-gap estimate.',
                 'Peak RSS is per child, not simultaneous whole-machine peak; supervisor CPU/startup CPU before worker timer are not fully measured.']))
    behaviors=[]
    for seed in (42,314159):
        parent={s:read(OUTPUT/f'execution/parent-{seed}/{s}.json')['rows'] for s in ('tactical','solved','deep')}
        for arm,generation in (('parent',20),('control',21),('treatment',21),('control',22),('treatment',22)):
            suites=parent if arm=='parent' else {s:read(OUTPUT/f'execution/{seed}-{arm}-g{generation}-evaluation/{s}.json')['rows'] for s in parent}
            record=dict(seed=seed,arm=arm,generation=generation,tactical={},deep={})
            for method in ('policy','puct'):
                field=f'{method}_correct'
                for category in ('unique_win','unique_safe'):
                    selected=[r for r in suites['tactical'] if r['category']==category]
                    original=[r for r in parent['tactical'] if r['category']==category]
                    record['tactical'][f'{method}-{category}']=dict(metrics=bootstrap(selected,(field,))[field],
                        correct=sum(r[field] for r in selected),n=len(selected),
                        change_from_parent=bootstrap(paired(selected,original,(field,)),(field,))[field])
                for kind in ('avoidable','missed_win'):
                    field=f'{method}_{kind}'
                    record['tactical'][f'{method}-{kind}']=dict(count=sum(r[field] for r in suites['tactical']),
                        metrics=bootstrap(suites['tactical'],(field,))[field])
                for suite in ('solved','deep'):
                    field=f'{method}_deep_mistake'
                    record['deep'][f'{method}-{suite}']=dict(metrics=bootstrap(suites[suite],(field,))[field],
                        count=sum(r[field] for r in suites[suite]),
                        change_from_parent=bootstrap(paired(suites[suite],parent[suite],(field,)),(field,))[field])
            for category in ('unique_win','unique_safe'):
                selected=[dict(id=r['id'],cluster=r['cluster'],search_minus_policy=r['puct_correct']-r['policy_correct'])
                          for r in suites['tactical'] if r['category']==category]
                record['tactical'][f'search_minus_policy-{category}']=bootstrap(selected,('search_minus_policy',))['search_minus_policy']
            behaviors.append(record)
    write(ROOT/'results/checkpoint-behavior.json',dict(rows=behaviors,
        note='Same frozen endpoints/records/clusters and10000-family-bootstrap; additional descriptive per-checkpoint/parent summaries, no new decision thresholds.'))
    print(dict(totals=dict(totals),runtime=resources['wall_seconds']))

if __name__=='__main__': main()
