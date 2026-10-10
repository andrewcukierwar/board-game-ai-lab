"""Bounded source-frozen profile diagnostic; never timing acceptance evidence."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time

from scripts import benchmark_negamax_v3 as bench
from scripts.negamax_v3_variants import load_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='profile')
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    owner = bench.require_lock()
    directory = bench.ROOT / args.phase
    design = (directory / 'DESIGN.md').read_bytes()
    declaration = json.loads((directory / 'manifest.json').read_text())
    assert declaration['design_sha256'] == bench.digest(design)
    assert declaration['source_sha256'] == bench.digest(args.source.read_bytes())
    fixture_manifest = json.loads((bench.ROOT / 'mirror/manifest.json').read_text())
    module = load_source(args.source.read_text(),'profile-v3')
    fixtures = [p for p in fixture_manifest['positions'] if p['id'] in declaration['positions']]
    assert len(fixtures) == len(declaration['positions'])
    for _ in range(64):
        module.SearchTable()
    profiler = cProfile.Profile()
    decisions = []
    start = time.monotonic()
    for fixture in fixtures:
        assert time.monotonic()-start < declaration['budget_seconds']
        for _ in range(declaration['repetitions']):
            profiler.enable()
            result = bench.decision(module,bench.position(fixture['history']),declaration['depth'])
            profiler.disable()
            decisions.append(dict(position=fixture['id'],**result))
    stats = pstats.Stats(profiler)
    functions = [dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],
                      self_seconds=v[2],cumulative_seconds=v[3]) for k,v in stats.stats.items()]
    functions.sort(key=lambda v:v['self_seconds'],reverse=True)
    bench.write_new(directory/'profile.json',dict(runtime=bench.runtime(),lock_owner=owner,
                       elapsed_seconds=time.monotonic()-start,decisions=decisions,functions=functions))
    print(json.dumps(functions[:15],indent=2))


if __name__ == '__main__':
    main()
