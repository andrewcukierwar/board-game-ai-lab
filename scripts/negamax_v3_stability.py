"""Descriptive paired-sample stability; no inferential confidence interval."""
import argparse
import copy
import statistics
from scripts import benchmark_negamax_v3 as bench


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True)
    args=parser.parse_args()
    directory=bench.ROOT/args.phase
    data=bench.rows(directory)
    variants=[v for v in data[0]['variants'] if v!='direct']
    omissions=[]
    for omit in range(7):
        subset=copy.deepcopy(data)
        for r in subset:
            for b in r['variants'].values():
                samples=[s for i,s in enumerate(b['samples']) if i!=omit]
                b['median_wall_seconds']=statistics.median(s['wall_seconds'] for s in samples)
                b['median_cpu_seconds']=statistics.median(s['cpu_seconds'] for s in samples)
        omissions.append(bench.analyze(subset))
    result={v:dict(leave_one_repetition_out_geometric_range=[min(a[v]['geometric_speedup'] for a in omissions),max(a[v]['geometric_speedup'] for a in omissions)],
                   leave_one_repetition_out_wall_sum_range=[min(a[v]['wall_sum_ratio'] for a in omissions),max(a[v]['wall_sum_ratio'] for a in omissions)],
                   accepted_in_omissions=sum(a[v]['accepted'] for a in omissions)) for v in variants}
    conditions=[]
    for r in data:
        for v in variants:
            a,b=r['variants']['direct'],r['variants'][v]
            ratios=[aa['wall_seconds']/bb['wall_seconds'] for aa,bb in zip(a['samples'],b['samples'])]
            conditions.append(dict(position=r['position'],depth=r['depth'],variant=v,paired_speedup_min=min(ratios),paired_speedup_max=max(ratios),
                                   baseline_wall_min=min(s['wall_seconds'] for s in a['samples']),baseline_wall_max=max(s['wall_seconds'] for s in a['samples'])))
    bench.write_new(directory/'stability.json',dict(method='Seven descriptive leave-one-repetition-out analyses, not confidence intervals. Correlated conditions and host activity remain.',variants=result,conditions=conditions))
    print(result)


if __name__=='__main__':
    main()
