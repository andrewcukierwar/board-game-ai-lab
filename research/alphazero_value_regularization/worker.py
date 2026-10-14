"""Fresh process per training segment/evaluation; no collection capability."""
import argparse
import time
from pathlib import Path
from games.connect4.alphazero_v2.experimental import digest, rng_payload
from .common import runtime, enforce, read, write, OUTPUT, RESULTS, Deadline, sha, hc
from .checkpoint import initial, state, save, load, sample
from .objective import step


def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['initialize','train','evaluate'])
    p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['control','treatment'])
    p.add_argument('--updates',type=int,default=0);p.add_argument('--previous',type=int,default=0)
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    start=time.perf_counter();cpu=hc.cpu_seconds();runtime();enforce();check=Deadline(args.deadline);check()
    design=read(RESULTS/'design.json')
    if args.seed not in design['seeds'] or args.updates not in design['milestones']:
        raise ValueError('Task outside frozen seed/milestone memberships')
    if args.task=='initialize':
        records={}
        for arm in ('control','treatment'):
            r,e=initial(args.seed)
            records[arm]=dict(state_sha256=digest(state(r)),parent=e,
                optimizer_sha256=digest(r.trainer.optimizer.state_dict()),counters=r.counters(),
                checkpoint_sha256=save(r,args.seed,arm,0,[],OUTPUT/'checkpoints'/f'{args.seed}-{arm}-0.pt'))
        if records['control']['state_sha256']!=records['treatment']['state_sha256']:
            raise ValueError('Different initial computational states')
        write(OUTPUT/'records'/f'{args.seed}-initial.json',records)
    else:
        path=OUTPUT/'checkpoints'/f'{args.seed}-{args.arm}-{args.previous if args.task=="train" else args.updates}.pt'
        before_updates=args.previous if args.task=='train' else args.updates
        r,payload=load(path,expected_sha256=sha(path),seed=args.seed,arm=args.arm,updates=before_updates)
        if args.task=='train':
            if args.previous not in design['milestones'] or args.previous>=args.updates or design['milestones'].index(args.updates)!=design['milestones'].index(args.previous)+1:
                raise ValueError('Training segment not frozen')
            replay=r.replay.digest();before=digest(state(r));trace=payload['trace'].copy();losses=[]
            for index in range(args.previous,args.updates):
                check();batch,decision=sample(r);trace.append(digest(decision));losses.append(step(r,batch,args.arm))
                if (index+1)%256==0:print(args.seed,args.arm,'updates',index+1,flush=True)
            r.global_rng=rng_payload()
            if r.replay.digest()!=replay:raise ValueError('Training changed retained replay')
            path=OUTPUT/'checkpoints'/f'{args.seed}-{args.arm}-{args.updates}.pt'
            h=save(r,args.seed,args.arm,args.updates,trace,path)
            write(OUTPUT/'records'/f'{args.seed}-{args.arm}-{args.updates}-train.json',
                dict(complete=True,seed=args.seed,arm=args.arm,updates=args.updates,previous=args.previous,
                    steps=len(losses),checkpoint_sha256=h,trace=trace,trace_sha256=digest(trace),
                    before_state=before,after_state=digest(state(r)),replay_sha256=replay,
                    optimizer_sha256=digest(r.trainer.optimizer.state_dict()),counters=r.counters(),
                    sampling_sha256=digest(r.sampling_rng.getstate()),augmentation_sha256=digest(state(r)['augmentation']),
                    losses=losses,wall_seconds=time.perf_counter()-start,cpu_seconds=hc.cpu_seconds()-cpu))
        else:
            from .evaluate import evaluate
            label=f'{args.seed}-initial-0' if args.updates==0 else f'{args.seed}-{args.arm}-{args.updates}'
            evaluate(r,OUTPUT/'tasks'/label,check)
            write(OUTPUT/'tasks'/label/'resource.json',dict(wall_seconds=time.perf_counter()-start,
                cpu_seconds=hc.cpu_seconds()-cpu,peak_rss_mib=hc.peak_rss_mib(),checkpoint_sha256=sha(path)))
    check();enforce()

if __name__=='__main__':main()
