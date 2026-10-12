"""One bounded read-only task; strict loading and prospective source/runtime gate."""
import argparse
from pathlib import Path
import signal
import time
import resource
from .common import RESULTS, load, runtime, enforce, write, sha
from .interventions import Intervention
from . import evaluate, values


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=('positions','matches','values'))
    parser.add_argument('--arm',required=True)
    parser.add_argument('--generation',type=int,default=6)
    parser.add_argument('--variant',default='A')
    parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    runtime(); freeze=enforce()
    started=time.monotonic();cpu=time.process_time()
    output=RESULTS/'tasks'/args.label
    output.mkdir(parents=True,exist_ok=False)
    def stop(*_):
        raise InterruptedError('Owned worker cancellation')
    signal.signal(signal.SIGTERM,stop)
    def check():
        if time.time()>=args.deadline or (RESULTS/'CANCEL').exists():
            raise TimeoutError('Frozen hard stop reached')
    check()
    plan=__import__('json').loads((RESULTS/'design.json').read_text())
    if args.task=='values':
        values.diagnostics(args.arm,freeze,output,check)
    else:
        runner, model, base, entry=load(args.arm,args.generation,freeze)
        weights=runner.factory.hash(model)
        wrapper=Intervention(base,args.variant)
        getattr(evaluate,args.task)(wrapper,plan,output,check)
        assert runner.factory.hash(model)==weights and sha(entry['path'])==entry['sha256']
        write(output/'identity.json',dict(checkpoint=entry,weights_sha256=weights,variant=args.variant,
                                         arm=args.arm,generation=args.generation))
    check()
    write(output/'complete.json',dict(complete=True,task=args.task,arm=args.arm,variant=args.variant,
          generation=args.generation,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2))

if __name__=='__main__':
    main()
