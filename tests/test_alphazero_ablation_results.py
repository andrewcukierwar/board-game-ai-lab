"""Evidence-only regression checks of audited raw records and independent math."""
import math
import pytest
from research.alphazero_training_ablation.common import ROOT,OUTPUT,read,sha,source_map

def test_frozen_sources_and_all_checkpoint_files():
    assert source_map()==read(ROOT/'results/freeze.json')['files']
    audited=read(ROOT/'results/integrity-audit.json')
    assert audited['status']=='PASS' and len(audited['boundaries']['records'])==12
    for record in audited['boundaries']['records'].values():
        assert sha(record['path'])==record['file_sha256']
        assert record['source_runtime_strict_reload']
    assert len(audited['official_files'])==150

def test_completed_budgets_and_no_retries():
    status=read(OUTPUT/'execution/status.json'); audit=read(ROOT/'results/integrity-audit.json')['structure']
    assert len(status['tasks'])==22 and all(t['state']=='COMPLETE' for t in status['tasks'])
    assert audit['counts']==dict(games=512,updates=512,matches=256)
    assert status['elapsed_seconds']<10800
    assert all(t['resources']['wall_seconds']<1200 for t in status['tasks'] if t['task']=='train')
    assert audit['selfplay_simulations']+audit['position_evaluation_simulations']+audit['strength_simulations']==11137280

def test_raw_probability_learning_contrasts():
    fixtures=read(ROOT/'fixtures.json')['solved']; effects=[]
    analysis=read(ROOT/'results/analysis.json')
    for seed in (42,314159):
        arms={a:read(OUTPUT/f'execution/{seed}-{a}-g22-evaluation/solved.json')['rows'] for a in ('control','treatment')}
        reconstructed={}
        for arm,rows in arms.items():
            assert [r['id'] for r in rows]==[f['id'] for f in fixtures]
            reconstructed[arm]=[(-math.log(sum(r['policy'][a] for a in f['optimal_actions'])),
                int(r['policy_action'] in f['optimal_actions']),(r['value']-f['value'])**2) for r,f in zip(rows,fixtures)]
        delta=[sum(t[i]-c[i] for t,c in zip(reconstructed['treatment'],reconstructed['control']))/300 for i in range(3)]
        for field,value in zip(('ce','accuracy','mse'),delta):
            assert value==pytest.approx(analysis['learning'][str(seed)]['contrast-g22'][field]['mean'],abs=1e-12)
        effects.append(delta)
    assert sum(v[1] for v in effects)/2==-5/600

def test_independent_bootstrap_and_decisions():
    main=read(ROOT/'results/analysis.json'); independent=read(ROOT/'results/independent-recomputed.json')
    for seed in ('42','314159'):
        for field in ('ce','accuracy','mse','optimal_mass'):
            x=main['learning'][seed]['contrast-g22'][field]; y=independent['learning'][seed]['contrast-g22'][field]
            assert x['mean']==pytest.approx(y['mean'],abs=1e-12)
            assert x['ci95']==pytest.approx(y['ci95'],abs=1e-12)
    for field in ('neural_learning','tactical_reliability','playing_strength'):
        assert main['decisions'][field]==independent['decisions'][field]
    assert main['decisions']['neural_learning']=='FAIL'
    assert main['decisions']['tactical_reliability']=='FAIL'
    assert main['decisions']['playing_strength']=='INCONCLUSIVE'

def test_artifact_manifest_and_exposure_accounting():
    manifest=read(ROOT/'results/evidence-manifest.json')
    for relative,expected in manifest['files'].items(): assert sha(ROOT/relative)==expected
    diagnostics=read(ROOT/'results/mechanism-resources.json')
    assert diagnostics['totals']['new_samples']==3397
    assert diagnostics['totals']['samples']==65536
    for row in diagnostics['rows']:
        assert sum(row['sampled_by_generation'].values())==8192
        assert row['new_sample_fraction']==row['new_sample_appearances']/8192
        assert row['experimental_replay_fraction']==row['experimental_replay_positions']/row['replay_positions']
