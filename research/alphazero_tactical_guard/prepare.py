"""Prospective design and final freeze, before any measured experiment."""
import argparse
from .common import *
from .evaluate import fixtures,openings

def main():
    runtime();p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    if args.freeze:
        if read(RESULTS/'prelaunch-review.json')['disposition']!='PASS':raise ValueError('Review pending')
        write(RESULTS/'freeze.json',dict(files=source_map(),runtime=runtime(),checkpoints=[parent_entry(s,6) for s in (42,314159)]));return
    # Task began 2026-10-14 02:53:39 UTC, client date October13.
    origin=__import__('datetime').datetime(2026,10,14,2,53,39,tzinfo=__import__('datetime').timezone.utc).timestamp()
    from datetime import datetime,timezone
    assert datetime.fromtimestamp(origin,timezone.utc).strftime('%Y-%m-%d %H:%M:%S')=='2026-10-14 02:53:39'
    data=fixtures();op=openings();pilot=read(RESULTS/'pilot.json')
    write(RESULTS/'protected-before.json',protected())
    write(RESULTS/'design.json',dict(contract='phase4d4j-a-protocol-v1',base=BASE,branch=BRANCH,
        seeds=[42,314159],generation=6,architecture=hc.IDENTITY['cnn'],checkpoints=[parent_entry(s,6) for s in (42,314159)],
        simulations=256,exploration=1.41,root_noise=False,terminal_values='exact actor-relative unchanged',
        original_ties='historical seeded maximum visits',override_ties='maximum ORIGINAL visits then lowest physical column; unchanged if original safe',
        search_seed=4404901,match_seed=4404902,analysis_seed=4410903,bootstrap_replicates=10000,
        fixture_sha256=sha(REPO/'research/alphazero_resnet_selfplay/fixtures.json'),
        opening_sha256=sha(REPO/'games/connect4/alphazero_v2/frozen/openings-development.json'),
        fixtures={suite:dict(n=len(rows),ids=[r['id'] for r in rows],clusters=sorted({r['cluster'] for r in rows})) for suite,rows in data.items() if suite in ('tactical','solved','deep')},
        openings=[dict(id=r['id'],family=r['family'],moves=r['moves']) for r in op],opponent=dict(name='reference-negamax',depth=3,ties='seeded uniform'),
        variants=['control','guarded'],expected_seed_tasks=2,fixture_searches=2512,fixture_paired_records=1256,matches=320,
        secondary='OMITTED before freeze to protect primary and review budget',resnet='OMITTED',training=False,selfplay=False,
        match_rules=dict(PASS='pooled difference>=0.05 AND both seed differences>=0 AND pooled 95% lower>0',
            FAIL='pooled difference<=-0.05 AND pooled 95% upper<0',otherwise='INCONCLUSIVE'),
        tactical_rule='any reproducible violation FAIL; immediate wins selected, safe action if available, legal and inactive unchanged',
        deeper_rule='any confirmed oracle WDL regression FAIL and blocks positive publication pending independent investigation',
        latency=dict(p95_guard_seconds_max=.001,worst_guard_seconds_max=.005,mean_relative_overhead_max=.05),
        clusters='whole opening families both colors; fixture ancestry/reflection; row weighted; pooled fixed seed average per row then common clusters',
        task_origin_epoch=origin,absolute_deadline_epoch=origin+2700,hard_maximum_seconds=2700,
        lifecycle_computation_reserve_seconds=180,projected_seconds=pilot['primary_projected_seconds'],
        no_retry=True,no_sealed_access=True,model_reproduction='six prelaunch preservation roots only; results review uses all saved records'))
    print('Design prepared')
if __name__=='__main__':main()
