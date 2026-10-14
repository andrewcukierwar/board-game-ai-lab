"""One sequential seed task; immutable raw evidence, no retry or training."""
import argparse,time
from .common import runtime,enforce,load,state,OUTPUT,write,hc,Deadline
from .evaluate import positions,matches

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    runtime();freeze=enforce();check=Deadline(args.deadline);check()
    output=OUTPUT/'tasks'/str(args.seed)
    if output.exists():raise FileExistsError('Task cannot be retried')
    output.mkdir(parents=True)
    start=time.time();cpu=hc.cpu_seconds();runner,model,inference,entry=load(args.seed)
    before=state(runner);weights=runner.factory.hash(model)
    positions(inference,args.seed,check,output)
    for variant in ('control','guarded'):matches(inference,args.seed,check,output,variant)
    check();enforce()
    assert before==state(runner) and weights==runner.factory.hash(model)
    write(output/'identity.json',dict(complete=True,seed=args.seed,architecture=hc.IDENTITY['cnn'],generation=6,
        checkpoint=entry,weights_sha256=weights,before=before,after=state(runner),
        seconds=time.time()-start,cpu_seconds=hc.cpu_seconds()-cpu,rss_mib=hc.peak_rss_mib(),
        fixture_searches=1256,matches=160))
    print(f'Completed seed {args.seed}',flush=True)
if __name__=='__main__':main()
