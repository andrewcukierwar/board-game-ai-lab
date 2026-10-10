"""Source-frozen PVS scout/re-search counts outside acceptance timing."""
import json
from pathlib import Path
from scripts import benchmark_negamax_v3 as bench
from scripts.negamax_v3_variants import load_source,replace


def main():
    directory=bench.ROOT/'pvs'
    owner=bench.require_lock()
    config=bench.check(directory)
    declaration=json.loads((directory/'diagnostic-declaration.json').read_text())
    assert declaration['script_sha256']==bench.digest(Path(__file__).read_bytes())
    source=(directory/'pvs-diagnostic.py').read_text()
    source=replace(source,'        self.nodes = 0','        self.scouts = self.researches = 0\n        self.nodes = 0')
    source=replace(source,'                value = -negamax(state, depth - 1, -alpha - 1, -alpha, table)',
        '                if table is not None:\n                    table.scouts += 1\n'
        '                value = -negamax(state, depth - 1, -alpha - 1, -alpha, table)')
    source=replace(source,'                if alpha < value < beta:',
        '                if alpha < value < beta:\n                    if table is not None:\n                        table.researches += 1')
    results=[]
    for i,fixture in enumerate(config['positions']):
        if fixture['id'] not in declaration['positions']:
            continue
        for depth in declaration['depths']:
            module=load_source(source,'pvs-counted')
            with bench.capture(module) as held:
                result=bench.decision(module,bench.position(fixture['history']),depth)
            primary=json.loads((directory/f'condition-{i:02d}-{depth:02d}.json').read_text())
            bench.same(primary['variants']['pvs']['samples'][0],result,True)
            results.append(dict(position=fixture['id'],depth=depth,scouts=held[0].scouts,
                                researches=held[0].researches,**result))
    bench.write_new(directory/'scout-diagnostics.json',dict(runtime=bench.runtime(),lock_owner=owner,results=results))
    print('PVS scout diagnostics',len(results))


if __name__=='__main__':
    main()
