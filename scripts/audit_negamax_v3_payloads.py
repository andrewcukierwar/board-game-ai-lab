"""Exact per-node-only audit: captured TT equality and counted tree invariants."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from scripts import benchmark_negamax_v3 as bench
from scripts.negamax_v3_variants import load_source


def checksum(entries):
    h=hashlib.sha256()
    for key,value in entries.items():
        h.update(f'{key}:{value}\n'.encode())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',default='tuning')
    args=parser.parse_args()
    directory=bench.ROOT/args.phase
    owner=bench.require_lock()
    config=bench.check(directory)
    declaration=json.loads((directory/'payload-declaration.json').read_text())
    assert declaration['script_sha256']==bench.digest(Path(__file__).read_bytes())
    modules={v:load_source((directory/f'{v}-source.py').read_text(),v) for v in config['variants']}
    data=bench.rows(directory)
    assert len(data)==len(config['positions'])*len(config['depths'])
    evidence=[]
    started=time.monotonic()
    for r in data:
        assert time.monotonic()-started < declaration['budget_seconds']
        fixture=next(p for p in config['positions'] if p['id']==r['position'])
        with bench.capture(modules['direct']) as held:
            ref=bench.decision(modules['direct'],bench.position(fixture['history']),r['depth'])
        entries=held[0].entries
        bench.same(r['variants']['direct']['samples'][0],ref,True)
        digest=checksum(entries)
        candidates={}
        for v,module in modules.items():
            if v=='direct':continue
            assert r['variants'][v]['counted']['diagnostics']==r['variants']['direct']['counted']['diagnostics']
            with bench.capture(module) as held:
                result=bench.decision(module,bench.position(fixture['history']),r['depth'])
            bench.same(ref,result,True)
            assert entries==held[0].entries, (r['position'],r['depth'],v,'TT payload mismatch')
            assert checksum(held[0].entries)==digest
            candidates[v]=dict(exact_table_equal=True,source_sha256=config['sources'][f'{v}-source.py'])
        evidence.append(dict(position=r['position'],depth=r['depth'],entries=len(entries),
                             packed_tt_sha256=digest,variants=candidates))
    bench.write_new(directory/'payload-audit.json',dict(passed=True,conditions=len(evidence),
                    elapsed_seconds=time.monotonic()-started,lock_owner=owner,results=evidence))
    print('Exact TT/tree audit passed',len(evidence),'conditions')


if __name__=='__main__':main()
