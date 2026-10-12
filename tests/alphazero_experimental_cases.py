"""Bounded scratch-only training and adversarial generation-boundary validation."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from games.connect4.alphazero_v2 import experimental as E
from games.connect4.alphazero_v2 import artifacts, batching
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.generation import GenerationRunner, load_resume_boundary
from games.connect4.alphazero_v2.experimental_import import fork_verified, read_historical
from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime


@pytest.fixture(autouse=True, scope='module')
def runtime():
    configure_deterministic_runtime(1)


def config(**changes):
    base = V2Config(seed=123, games_per_generation=8, self_play_simulations=1, batch_size=2,
                    replay_generations=2, replay_max_games=16, max_generations=4)
    return E.ExperimentalConfig(replace(base, **changes), optimizer_updates=1)


def runner(**changes):
    return E.ExperimentalRunner(config(**changes), lineage='test', arm='control')


def rehash(p):
    p['integrity'] = E.digest({k:v for k,v in p.items() if k != 'integrity'})
    return p


def assert_equal(a,b):
    ac,bc=a._content(),b._content()
    ac.pop('resources'); bc.pop('resources')
    assert E.digest(ac) == E.digest(bc)
    assert a.factory.hash(a.model) == b.factory.hash(b.model)
    assert E.digest(a.trainer.optimizer.state_dict()) == E.digest(b.trainer.optimizer.state_dict())
    assert a.replay.digest() == b.replay.digest()
    assert list(a.replay.iter_games()) == list(b.replay.iter_games())
    assert a.counters() == b.counters() and a.history == b.history
    assert E.digest(a.global_rng) == E.digest(b.global_rng)
    assert a.sampling_rng.getstate() == b.sampling_rng.getstate()
    assert a.augmenter.rng.getstate() == b.augmenter.rng.getstate()


@pytest.fixture
def completed(tmp_path):
    r = runner()
    r.run_generation(tmp_path/'complete.pt')
    return r, tmp_path/'complete.pt'


def test_exact_resume_pool_eight_and_rng(completed,tmp_path):
    r,path = completed
    loaded = E.load_experimental(path, expected_sha256=artifacts.file_sha256(path), expected_config=r.config)
    assert_equal(r,loaded)
    assert r.history[-1]['collection']['max_active'] == 8
    assert r.history[-1]['collection']['batch_sizes'][8] > 0
    random.random(); np.random.rand(); torch.rand(4)
    E.load_experimental(path)
    assert E.digest(E.rng_payload()) == E.digest(r.global_rng)
    a,b=r.run_generation(tmp_path/'a.pt'),loaded.run_generation(tmp_path/'b.pt')
    for result in (a,b):
        result.pop('resources'); result.pop('boundary_sha256')
    assert a==b
    assert_equal(r,loaded)


def test_fresh_process_resume(completed,tmp_path):
    r,path=completed
    r.run_generation(tmp_path/'expected.pt')
    script=f'''
from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime
configure_deterministic_runtime(1)
from games.connect4.alphazero_v2.experimental import load_experimental
r=load_experimental({str(path)!r})
r.run_generation({str(tmp_path/'actual.pt')!r})
'''
    subprocess.run([sys.executable,'-B','-c',script],check=True,cwd=Path(__file__).resolve().parents[1],timeout=60)
    assert_equal(r,E.load_experimental(tmp_path/'actual.pt'))


def test_repeated_deterministic_collection(completed,tmp_path):
    _,path=completed
    a,b=E.load_experimental(path),E.load_experimental(path)
    a.run_generation(tmp_path/'a.pt'); b.run_generation(tmp_path/'b.pt')
    assert_equal(a,b)


@pytest.mark.parametrize('point', ['collection','game_finalized','validation','before_replay',
    'after_replay','before_update','after_update','before_checkpoint','boundary_validation'])
def test_interruption_recovers_exactly(completed,tmp_path,point):
    _,path=completed
    expected,failed=E.load_experimental(path),E.load_experimental(path)
    expected.run_generation(tmp_path/'expected.pt')
    def interrupt(event):
        if event==point:
            raise KeyboardInterrupt(point)
    with pytest.raises(KeyboardInterrupt):
        failed.run_generation(tmp_path/'interrupted.pt',progress=interrupt)
    assert not (tmp_path/'interrupted.pt').exists()
    assert failed.failed
    with pytest.raises(RuntimeError): failed.save(tmp_path/'bad.pt')
    with pytest.raises(RuntimeError): failed.run_generation(tmp_path/'bad.pt')
    recovered=E.load_experimental(path)
    recovered.run_generation(tmp_path/'recovered.pt')
    assert_equal(expected,recovered)


def test_mid_search_interruption(completed,tmp_path):
    _,path=completed
    r=E.load_experimental(path)
    checks=0
    def stop():
        nonlocal checks
        checks+=1
        if checks==11: raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): r.run_generation(tmp_path/'bad.pt',check=stop)
    assert not (tmp_path/'bad.pt').exists()


def test_game_finalization_failure(tmp_path,monkeypatch):
    r=runner()
    def fail(*a,**k): raise KeyboardInterrupt('finalization')
    monkeypatch.setattr(batching,'finalize_game',fail)
    with pytest.raises(KeyboardInterrupt): r.run_generation(tmp_path/'bad.pt')
    assert r.failed and not (tmp_path/'bad.pt').exists()


@pytest.mark.parametrize('operation',['serialization','validation','fsync'])
def test_checkpoint_interruption(completed,tmp_path,monkeypatch,operation):
    _,path=completed
    r=E.load_experimental(path)
    def fail(*a,**k): raise OSError(operation)
    if operation=='serialization': monkeypatch.setattr(torch,'save',fail)
    if operation=='validation': monkeypatch.setattr(E,'load_experimental',fail)
    if operation=='fsync': monkeypatch.setattr(artifacts.os,'fsync',fail)
    with pytest.raises(OSError): r.run_generation(tmp_path/'bad.pt')
    assert r.failed and not (tmp_path/'bad.pt').exists()
    assert not list(tmp_path.glob('.*.tmp'))


def test_atomic_publication_and_overwrite(completed,tmp_path,monkeypatch):
    r,path=completed
    before=artifacts.file_sha256(path)
    with pytest.raises(FileExistsError): r.save(path)
    assert artifacts.file_sha256(path)==before
    link=artifacts.os.link
    observed=[]
    def checked(src,dst):
        assert not Path(dst).exists()
        E.load_experimental(src,restore_global_rng=False)
        observed.append(True)
        return link(src,dst)
    monkeypatch.setattr(artifacts.os,'link',checked)
    r.save(tmp_path/'new.pt')
    assert observed==[True]


def test_post_publication_fsync_failure_leaves_complete_boundary(completed,tmp_path,monkeypatch):
    _,path=completed
    r=E.load_experimental(path)
    fsync=artifacts.os.fsync
    calls=0
    def fail_directory(fd):
        nonlocal calls
        calls+=1
        if calls==2: raise OSError('directory durability uncertain')
        return fsync(fd)
    monkeypatch.setattr(artifacts.os,'fsync',fail_directory)
    with pytest.raises(OSError): r.run_generation(tmp_path/'published.pt')
    assert r.failed
    assert E.load_experimental(tmp_path/'published.pt').completed_generations==2


@pytest.mark.parametrize('mutation', ['temperature','visits','winner','move','schedule','missing_schedule',
    'protocol','replay_digest','replay_identity','optimizer_groups','optimizer_moment','optimizer_step',
    'architecture','runtime','source','rng','augmentation','counters','model','config',
    'history_positions','history_steps','history_weights'])
def test_adversarial_boundary_validation(completed,tmp_path,mutation):
    r,path=completed
    p=torch.load(path,weights_only=True)
    entry=p['replay']['generations'][0]; game=entry['games'][0]
    if mutation=='temperature': game['action_temperatures'][0]=0.5
    if mutation=='visits': game['visits'][0][0]+=1
    if mutation=='winner': game['winner']=9
    if mutation=='move': game['moves'][0]=7
    if mutation=='schedule': entry['schedule']=E.EXTENDED.to_dict()
    if mutation=='missing_schedule': del entry['schedule']
    if mutation=='protocol': entry['protocol']='unknown'
    if mutation=='replay_digest': p['replay']['digest']='0'*64
    if mutation=='replay_identity': entry['generation']=2
    if mutation=='optimizer_groups': p['optimizer']['param_groups'][0]['lr']=0.1
    if mutation=='optimizer_moment': p['optimizer']['state'][0]['exp_avg']=torch.zeros(1)
    if mutation=='optimizer_step': p['optimizer']['state'][0]['step']=torch.tensor(9.)
    if mutation=='architecture': p['config']['architecture']='resnet-unimplemented'
    if mutation=='runtime': p['runtime']['intra_op_threads']=99
    if mutation=='source': p['source']['execution']['sha256']='0'*64
    if mutation=='rng': p['global_rng']['torch']=torch.zeros(1,dtype=torch.uint8)
    if mutation=='augmentation': p['augmentation']['reflected']+=1
    if mutation=='counters': p['counters']['total_games']+=1
    if mutation=='model': p['model']['conv1.weight'][0,0,0,0]+=1
    if mutation=='history_positions':
        p['history'][0]['new_positions']+=1; p['counters']['total_positions']+=1
    if mutation=='history_steps': p['history'][0]['trainer_steps']+=1
    if mutation=='history_weights': p['history'][0]['collection_weights_sha256']='0'*64
    if mutation=='config': p['config']['rng_protocol']='unknown'
    torch.save(rehash(p),tmp_path/'tampered.pt')
    with pytest.raises((ValueError,RuntimeError,TypeError,IndexError)):
        E.load_experimental(tmp_path/'tampered.pt')


def test_integrity_and_file_corruption(completed,tmp_path):
    _,path=completed
    with pytest.raises(ValueError): E.load_experimental(path,expected_sha256='0'*64)
    p=torch.load(path,weights_only=True); p['schedule']=E.EXTENDED.to_dict()
    torch.save(p,tmp_path/'tampered.pt')
    with pytest.raises(ValueError,match='integrity'): E.load_experimental(tmp_path/'tampered.pt')
    (tmp_path/'truncated.pt').write_bytes(path.read_bytes()[:64])
    with pytest.raises(Exception): E.load_experimental(tmp_path/'truncated.pt')
    with pytest.raises(ValueError): E.load_experimental(path,expected_config=config(seed=999))


@pytest.fixture
def synthetic_parent(tmp_path):
    historical=GenerationRunner(config().training)
    historical.run_generation()
    payload=historical.boundary_payload()
    historical.save_resume_boundary(tmp_path/'legacy.pt')
    # Current original loader still verifies its unmodified contract on scratch data.
    verified=load_resume_boundary(tmp_path/'legacy.pt',expected_sha256=artifacts.file_sha256(tmp_path/'legacy.pt'))
    assert verified.state_sha256()==payload['state_sha256']
    return dict(original=payload, verified_state=payload['state_sha256'], verified_replay=payload['replay']['digest'])


def fork(parent,arm,schedule):
    c=E.ExperimentalConfig(replace(V2Config.from_dict(parent['original']['config']),max_generations=5),1)
    return fork_verified(parent,c,parent_sha256='1'*64,schedule=schedule,lineage='synthetic-fork',arm=arm)


def test_identical_arm_initialization_and_mixed_schedule_expiration(synthetic_parent,tmp_path):
    control=fork(synthetic_parent,'control',E.ORIGINAL)
    treatment=fork(synthetic_parent,'treatment',E.EXTENDED)
    assert control.initial_state_sha256()==treatment.initial_state_sha256()
    assert control.replay.digest()==synthetic_parent['verified_replay']
    for r in (control,treatment):
        r.save(tmp_path/f'{r.arm}-fork.pt')
        restored=E.load_experimental(tmp_path/f'{r.arm}-fork.pt')
        assert_equal(r,restored)
    treatment.run_generation(tmp_path/'mixed.pt')
    loaded=E.load_experimental(tmp_path/'mixed.pt')
    assert loaded.metadata[1]['protocol']==E.LEGACY_PROTOCOL
    assert loaded.metadata[2]['schedule']==E.EXTENDED.to_dict()
    assert_equal(treatment,loaded)
    treatment.run_generation(tmp_path/'expired.pt')
    loaded=E.load_experimental(tmp_path/'expired.pt')
    assert loaded.replay.generations==(2,3) and set(loaded.metadata)=={2,3}
    assert all(m['protocol']==E.PROTOCOL for m in loaded.metadata.values())


@pytest.mark.parametrize('mutation',['historical_temperature','legacy_schedule','parent_attestation','optimizer','training'])
def test_fork_rejections(synthetic_parent,tmp_path,mutation):
    p=deepcopy(synthetic_parent)
    if mutation=='historical_temperature': p['original']['replay']['generations'][0]['games'][0]['action_temperatures'][0]=0.2
    if mutation=='parent_attestation': p['verified_state']='0'*64
    if mutation=='optimizer': p['original']['optimizer_state_dict']['state'][0]['exp_avg']=torch.zeros(1)
    if mutation=='training':
        c=E.ExperimentalConfig(replace(config().training,learning_rate=0.1),1)
        with pytest.raises(ValueError): fork_verified(p,c,parent_sha256='1'*64,schedule=E.ORIGINAL,lineage='test',arm='x')
        return
    if mutation=='legacy_schedule':
        r=fork(p,'control',E.ORIGINAL); r.save(tmp_path/'fork.pt')
        payload=torch.load(tmp_path/'fork.pt',weights_only=True)
        payload['replay']['generations'][0]['schedule']=E.EXTENDED.to_dict()
        torch.save(rehash(payload),tmp_path/'bad.pt')
        with pytest.raises(ValueError,match='Historical'): E.load_experimental(tmp_path/'bad.pt')
        return
    with pytest.raises(ValueError): fork(p,'control',E.ORIGINAL)


def test_historical_hash_and_checkout_rejection(tmp_path):
    path=tmp_path/'parent.pt'; path.write_bytes(b'bad')
    with pytest.raises(ValueError,match='file hash'):
        read_historical(path,expected_sha256='0'*64,historical_checkout=tmp_path)
    with pytest.raises(ValueError,match='revision'):
        read_historical(path,expected_sha256=artifacts.file_sha256(path),historical_checkout=Path(__file__).resolve().parents[1])


def test_evaluation_hook_is_rng_neutral(completed):
    r,path=completed
    previous=E.digest(E.rng_payload())
    def evaluator(model,counters):
        random.random(); np.random.rand(); torch.rand(2)
        assert not model.training and all(not p.requires_grad for p in model.parameters())
        return counters
    assert E.evaluate_boundary(path,evaluator,expected_sha256=artifacts.file_sha256(path))==r.counters()
    assert E.digest(E.rng_payload())==previous


def test_identity_changes_poison_continuation(completed,tmp_path,monkeypatch):
    r,_=completed
    monkeypatch.setattr(E,'execution_source_identity',lambda: dict(sha256='changed'))
    with pytest.raises(RuntimeError): r.run_generation(tmp_path/'bad.pt')
    assert not (tmp_path/'bad.pt').exists()


def test_original_contract_and_official_artifact_isolation():
    root=Path(__file__).resolve().parents[1]
    base='a1639463ef5d66f0f12ab090674d5b861612a11c'
    for name in ('generation','batching','data','config','network','training','provenance','artifacts'):
        relative=f'games/connect4/alphazero_v2/{name}.py'
        original=subprocess.run(['git','show',f'{base}:{relative}'],check=True,capture_output=True).stdout
        assert (root/relative).read_bytes()==original
    assert E.CONTRACT != __import__('games.connect4.alphazero_v2.config',fromlist=['RESUME_CONTRACT']).RESUME_CONTRACT


def test_campaign_output_guard_including_symlink(tmp_path):
    official=tmp_path/'official'; official.mkdir()
    (official/'declaration.json').write_text('{}')
    link=tmp_path/'alias'; link.symlink_to(official,target_is_directory=True)
    r=runner()
    for root in (official,link):
        with pytest.raises(ValueError,match='campaign'):
            r.save(root/'accidental.pt')
        with pytest.raises(ValueError,match='campaign'):
            r.run_generation(root/'accidental.pt')
    assert sorted(p.name for p in official.iterdir())==['declaration.json']


def test_factory_owns_optimizer_restoration():
    factory=E.model_factory(E.ARCHITECTURE)
    assert callable(factory.restore_optimizer)
    with pytest.raises(ValueError,match='Unsupported'):
        E.model_factory('resnet')


def test_reader_has_no_assert_or_relaxed_validation_flags():
    from games.connect4.alphazero_v2.experimental_import import READER
    import ast
    tree=ast.parse(READER)
    assert not any(isinstance(n,ast.Assert) for n in ast.walk(tree))
    loader=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='load_resume_boundary')
    assert not any(k.arg in ('strict_source','strict_runtime') for k in loader.keywords)
