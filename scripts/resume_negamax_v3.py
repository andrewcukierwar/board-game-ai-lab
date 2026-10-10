"""Finish only missing complete conditions without altering frozen evidence.

A partial/corrupt existing JSON fails closed: preserve it under a new WIP name
before retrying; never silently replace or treat it as completed acceptance data.
"""
import argparse
import copy
import json
from pathlib import Path
import time
from scripts import benchmark_negamax_v3 as bench


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--stop',type=int,default=32)
    args=parser.parse_args()
    directory=args.directory or bench.ROOT/args.phase
    bench.require_lock()
    frozen_check=bench.check
    config=frozen_check(directory)
    began=time.monotonic()
    for i in range(args.start,min(args.stop,len(config['positions']))):
        for depth in config['depths']:
            path=directory/f'condition-{i:02d}-{depth:02d}.json'
            if path.exists():
                saved=json.loads(path.read_text())
                assert saved['position']==config['positions'][i]['id'] and saved['depth']==depth
                assert set(saved['variants'])==set(config['variants'])
                assert all(len(b['samples'])==config['samples'] and len(b['memory'])==config['memory_repetitions']
                           for b in saved['variants'].values())
                print('preserved',path.name,flush=True)
                continue
            assert time.monotonic()-began < config['maximum_batch_seconds'], 'Resume batch budget exhausted'
            # Select a declared condition after validating every frozen byte;
            # no histories, sample counts, sources or acceptance gates change.
            def selected_check(d):
                selected=copy.deepcopy(frozen_check(d))
                selected['depths']=[depth]
                return selected
            bench.check=selected_check
            try:
                bench.run(directory,i,i+1)
            finally:
                bench.check=frozen_check


if __name__=='__main__':main()
