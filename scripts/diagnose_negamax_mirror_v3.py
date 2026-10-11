"""Post-timing explanatory diagnostics, never acceptance samples."""
import cProfile
import json
from pathlib import Path
import pstats

from scripts import benchmark_negamax_v3 as bench
from scripts.negamax_v3_variants import load_source, replace


def main():
    directory = bench.ROOT / 'mirror'
    config = bench.check(directory)
    owner = bench.require_lock()
    declaration = json.loads((directory / 'diagnostic-declaration.json').read_text())
    assert declaration['script_sha256'] == bench.digest(Path(__file__).read_bytes())
    results = []
    for variant in ('mirror','mirror-selective'):
        source = (directory/f'{variant}-diagnostic.py').read_text()
        source = replace(source,'        self.nodes = 0',
            '        self.orientations = {}\n        self.cross_orientation_hits = 0\n        self.nodes = 0')
        source = replace(source,'        table.reflected_hits += int(mirrored)',
            '        table.cross_orientation_hits += int(table.orientations[key] != mirrored)\n'
            '        table.reflected_hits += int(mirrored)')
        source = replace(source,'        table.entries[key] = pack_entry',
            '        table.orientations[key] = mirrored\n        table.entries[key] = pack_entry')
        for fixture in config['positions']:
            if fixture['id'] not in declaration['positions']:
                continue
            for depth in declaration['depths']:
                module=load_source(source,variant+'-orientation')
                profiler=cProfile.Profile()
                with bench.capture(module) as held:
                    profiler.enable()
                    result=bench.decision(module,bench.position(fixture['history']),depth)
                    profiler.disable()
                primary=json.loads((directory/f"condition-{config['positions'].index(fixture):02d}-{depth:02d}.json").read_text())
                bench.same(primary['variants'][variant]['samples'][0],result,True)
                stats=pstats.Stats(profiler)
                canonical=[dict(function=k[2],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3])
                           for k,v in stats.stats.items() if k[2] in ('identity','reflect')]
                results.append(dict(position=fixture['id'],depth=depth,variant=variant,
                    cross_orientation_hits=held[0].cross_orientation_hits,
                    reflected_hits=held[0].reflected_hits,canonical_profile=canonical,**result))
    bench.write_new(directory/'orientation-diagnostics.json',dict(lock_owner=owner,runtime=bench.runtime(),results=results))
    print('orientation diagnostics complete',len(results))


if __name__=='__main__':
    main()
