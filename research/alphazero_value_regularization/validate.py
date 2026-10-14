"""Focused correctness and strict fresh-process continuation tests; disposable work only."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time
import torch
from torch.nn import functional as F
from games.connect4.alphazero_v2.training import training_loss, batch_tensors
from games.connect4.alphazero_v2.experimental import digest, restore_rng, rng_payload
from games.connect4.alphazero_v2.provenance import runtime_identity
from .common import runtime, OUTPUT, RESULTS, write, sha, hc
from .checkpoint import initial, sample, state, save, load
from .objective import loss, step


def expect_error(fn):
    try: fn()
    except (ValueError,FileExistsError,KeyError,RuntimeError): return
    raise AssertionError('Expected strict rejection')


def child(path,seed,arm,updates,out):
    r,p=load(path,expected_sha256=sha(path),seed=seed,arm=arm,updates=updates)
    batch,trace=sample(r);step(r,batch,arm)
    write(out,dict(state=digest(state(r)),trace=trace))


def main():
    start=time.perf_counter();runtime();checks=[]
    # Analytic independent formula and derivative; all seven raw logits.
    logits=torch.tensor([[1.,2.,3.,4.,5.,6.,7.],[7.,6.,5.,4.,3.,2.,1.]],requires_grad=True)
    v=torch.tensor([[.25],[-.75]],requires_grad=True)
    pi=torch.tensor([[0.,0.,.5,.5,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.]])
    z=torch.tensor([[1.],[-1.]])
    original=training_loss(logits,v,pi,z)
    c=loss(logits,v,pi,z,'control');t=loss(logits,v,pi,z,'treatment')
    assert all(torch.equal(a,b) for a,b in zip(c[:3],original))
    assert torch.equal(t[1],-(pi*F.log_softmax(logits,dim=1)).sum(1).mean())
    assert torch.allclose(t[3],torch.tensor(.0625))
    g=torch.autograd.grad(t[3],v,retain_graph=True)[0]
    assert torch.allclose(g,.4*v/2)
    grads=torch.autograd.grad(t[0],(logits,v));assert all(torch.isfinite(g).all() for g in grads)
    checks += ['control_exact_original_loss','treatment_mean_square_analytic','policy_formula_unchanged','finite_policy_and_value_gradients']
    for seed in (42,314159):
        a,e=initial(seed); initial_state=digest(state(a));b,_=initial(seed)
        assert digest(state(b))==initial_state
        assert a.trainer.steps==b.trainer.steps==720 and a.completed_generations==5
        assert digest(a.trainer.optimizer.state_dict())==digest(b.trainer.optimizer.state_dict())
        ba,ta=sample(a);bb,tb=sample(b);assert ba==bb and ta==tb
        # Canonical trainer versus control: not only loss, exact entire numerical trajectory.
        a.trainer.step(ba);step(b,bb,'control');assert digest(state(a))==digest(state(b))
        checks += [f'{seed}_identical_initial_state_optimizer',f'{seed}_identical_examples_augmentation',f'{seed}_control_exact_original_update']
        for arm in ('control','treatment'):
            r,_=initial(seed)
            before=digest(state(r))
            batch,trace=sample(r)
            # Full network gradients from penalty reach trunk; no policy-head direct gradient.
            if arm=='treatment':
                tensors=batch_tensors(batch);r.model.train();_,values=r.model(tensors[0]);(.2*values.square().mean()).backward()
                names=dict(r.model.named_parameters())
                trunk=[p for n,p in names.items() if n.startswith('conv')]
                assert trunk and any(p.grad is not None and p.grad.abs().sum()>0 for p in trunk)
                r.trainer.optimizer.zero_grad(set_to_none=True);r.model.eval()
                checks += [f'{seed}_penalty_reaches_shared_trunk']
            step(r,batch,arm)
            path=OUTPUT/'validation-v3'/f'{seed}-{arm}-1.pt'
            h=save(r,seed,arm,1,[digest(trace)],path)
            live_state=digest(state(r))
            reloaded,p=load(path,expected_sha256=h,seed=seed,arm=arm,updates=1)
            assert digest(state(reloaded))==live_state
            batch,next_trace=sample(r);step(r,batch,arm);expected=digest(state(r))
            out=OUTPUT/'validation-v3'/f'{seed}-{arm}-fresh.json'
            cmd=[sys.executable,'-m','research.alphazero_value_regularization.validate','child',str(path),str(seed),arm,'1',str(out)]
            subprocess.run(cmd,env=hc.thread_environment(),check=True,timeout=30)
            from .common import read
            q=read(out);assert q['state']==expected and q['trace']==next_trace
            expect_error(lambda: save(r,seed,arm,2,['0'*64]*2,path))
            expect_error(lambda: load(path,expected_sha256='0'*64,seed=seed,arm=arm,updates=1))
            expect_error(lambda: load(path,expected_sha256=h,seed=seed,arm=('treatment' if arm=='control' else 'control'),updates=1))
            expect_error(lambda: load(path,expected_sha256=h,seed=seed,arm=arm,updates=2))
            checks += [f'{seed}_{arm}_strict_roundtrip',f'{seed}_{arm}_fresh_process_exact_continuation',f'{seed}_{arm}_wrong_identity_and_overwrite_rejection']
        # A historical path is rejected before save even if it is a nonexistent filename.
        from .common import parent_entry, output_path
        expect_error(lambda: output_path(Path(parent_entry(seed,5)['path']).parent/'forbidden-new.pt'))
        checks += [f'{seed}_historical_output_rejected']
    # Malformed-state failures recompute integrity, so rejection is semantic, not only hash-based.
    original=torch.load(OUTPUT/'validation-v3/42-control-1.pt',weights_only=True)
    from .checkpoint import load as strict_load
    mutations={
        'trace':lambda q:q.update(trace=[]),
        'schema':lambda q:q.update(unexpected=1),
        'contract':lambda q:q['contract'].update(version=999),
        'config':lambda q:q['config']['training'].update(learning_rate=.001),
        'counter':lambda q:q['state']['counters'].update(trainer_steps=999),
        'replay':lambda q:q['state']['replay'].update(digest='wrong'),
        'runtime':lambda q:q['runtime'].update(intra_op_threads=2),
        'source':lambda q:q['source'].update(unexpected='bad'),
        'design':lambda q:q.update(design_sha256='0'*64),
        'parent':lambda q:q['parent'].update(sha256='0'*64),
        'optimizer':lambda q:q['state']['optimizer']['param_groups'][0].update(lr=.001),
        'sampling':lambda q:q['state'].update(sampling=initial(42)[0]._content()['sampling']),
        'augmentation':lambda q:q['state']['augmentation'].update(reflected=0),
        'global_rng':lambda q:q['state']['global_rng']['torch'].fill_(0),
    }
    for name,mutate in mutations.items():
        q=deepcopy(original);mutate(q);q['integrity']=digest({k:v for k,v in q.items() if k!='integrity'})
        path=OUTPUT/'validation-v3'/f'malformed-{name}.pt';torch.save(q,path)
        expect_error(lambda:strict_load(path,expected_sha256=sha(path),seed=42,arm='control',updates=1))
        checks.append('strict_'+name+'_semantic_rejection')
    from .common import source_map,parent_entry,read,strict_parent_load,REPO
    design=read(REPO/'research/alphazero_resnet_selfplay/results/design.json')
    entry=parent_entry(42,5)
    expect_error(lambda:strict_parent_load(entry['path'],'0'*64,design,42,'cnn',restore_global_rng=False))
    wrong=deepcopy(design);wrong['training_overrides']['learning_rate']=.001
    expect_error(lambda:strict_parent_load(entry['path'],entry['sha256'],wrong,42,'cnn',restore_global_rng=False))
    historical=torch.load(entry['path'],weights_only=True);historical['source']['execution']['sha256']='0'*64
    historical['integrity']=digest({k:v for k,v in historical.items() if k!='integrity'})
    path=OUTPUT/'validation-v3/original-source-mismatch.pt';torch.save(historical,path)
    expect_error(lambda:strict_parent_load(path,sha(path),design,42,'cnn',restore_global_rng=False))
    checks += ['original_wrong_sha_rejected','original_wrong_config_rejected','original_wrong_source_rejected']
    write(RESULTS/'validation.json',dict(passed=True,checks=checks,wall_seconds=time.perf_counter()-start,runtime=runtime_identity(),source_files=source_map(),
        prior_disposable_failures=['v1 JSON tuple/list trace serialization comparison (fixed)', 'v2 source freeze correctly rejected concurrent source edits (final tests run with stable sources)']))
    print('Focused validation passed',len(checks),'checks',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='child':
        runtime();child(Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4],int(sys.argv[5]),Path(sys.argv[6]))
    else: main()
