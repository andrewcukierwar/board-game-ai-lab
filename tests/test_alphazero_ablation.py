"""Meaningful frozen-fixture, paired-cluster and owned-watchdog checks."""
import subprocess
import sys
from pathlib import Path
import time
import pytest

def test_execution_entrypoints_import():
    from research.alphazero_training_ablation import worker,prepare,publish_evidence,evaluate
    assert callable(worker.main) and callable(prepare.main) and callable(publish_evidence.main)

def test_family_bootstrap_and_directional_margins():
    from research.alphazero_training_ablation.analyze import bootstrap,paired,noninferiority
    a=[dict(id=str(i),cluster=str(i//2),x=.03) for i in range(8)]
    b=[dict(id=str(i),cluster=str(i//2),x=0.) for i in range(8)]
    result=bootstrap(paired(a,b,('x',)),('x',))['x']
    assert result['families']==4 and result['mean']==pytest.approx(.03)
    assert result['ci95']==pytest.approx([.03,.03])
    assert noninferiority(result,.05)=='PASS'
    assert noninferiority(result,.02)=='FAIL'
    assert noninferiority(result,.05,False)=='FAIL'
    with pytest.raises(ValueError): paired(a,b[:-1],('x',))

def test_cluster_ancestry_and_mirror_linkage():
    from research.alphazero_training_ablation.freeze_fixtures import clusters
    rows=[dict(id='a',family='a',moves=[0,1,2,3,4,5]),
          dict(id='b',family='b',moves=[0,1,2,3,4,5,2,2]),
          dict(id='c',family='c',moves=[6,5,4,3,2,1,0,1]),
          dict(id='d',family='d',moves=[3,3,3,3,0,0])]
    result=clusters(rows)
    assert result[0]['cluster']==result[1]['cluster']==result[2]['cluster']
    assert result[0]['cluster']!=result[3]['cluster']

def test_frozen_fixture_coverage_and_match_budget():
    from research.alphazero_training_ablation.common import ROOT,read
    from research.alphazero_training_ablation.freeze_fixtures import validate_solved
    fixture=read(ROOT/'fixtures.json')
    for row in fixture['solved']+fixture['deep']: validate_solved(row)
    assert len(fixture['solved'])==300 and len(fixture['tactical'])==200
    assert len(fixture['deep'])==128 and len(fixture['openings'])==32
    assert len({r['family'] for r in fixture['openings']})==32
    assert all(r['moves'] and r['id'].endswith('-base') for r in fixture['openings'])
    assert len({r['cluster'] for r in fixture['solved']})<150
    assert len({r['cluster'] for r in fixture['tactical']})<100

def test_watchdog_stops_owned_child_and_keeps_failure_log(tmp_path):
    from research.alphazero_training_ablation.manage import execute
    started=time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        execute([sys.executable,'-c','import time; print("retained",flush=True); time.sleep(30)'],4,tmp_path/'attempt.log')
    assert time.monotonic()-started<4
    assert 'retained' in (tmp_path/'attempt.log').read_text()

def test_order_and_training_caps():
    from research.alphazero_training_ablation.manage import ORDER
    assert len(ORDER)==8 and len(set(ORDER))==8
    assert {g for _,_,g in ORDER}=={21,22}
    assert len(ORDER)*64==512
    assert ORDER[:2]==((42,'control',21),(42,'treatment',21))
    assert ORDER[4:6]==((42,'treatment',22),(42,'control',22))
