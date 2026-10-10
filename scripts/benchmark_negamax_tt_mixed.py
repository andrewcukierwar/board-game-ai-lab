"""Predeclared supplemental B2: mix index bits, preserve complete stored identity."""
import argparse
import json
from pathlib import Path
import types

from games.connect4.agents.negamax_tt import DirectMappedEntries
from scripts import benchmark_negamax_tt as bench
from scripts.negamax_tt_variants import BASELINE, AGENT_PATH, load_variant, variant_source

MULTIPLIER = 11400714819323198485
WORD_MASK = (1 << 64) - 1
ROOT = bench.ROOT / 'mixed-index'


class MixedEntries(DirectMappedEntries):
    __slots__ = ()

    def __init__(self, capacity):
        super().__init__(capacity)
        if capacity & (capacity - 1):
            raise ValueError('mixed capacity must be a power of two')

    def get(self, key):
        slot = ((hash(key) * MULTIPLIER) & WORD_MASK) >> (65 - self.capacity.bit_length())
        return self.values[slot] if self.keys[slot] == key else None

    def __setitem__(self, key, value):
        slot = ((hash(key) * MULTIPLIER) & WORD_MASK) >> (65 - self.capacity.bit_length())
        old = self.keys[slot]
        if old is None:
            self.occupancy += 1
        elif old == key:
            self.replacements += 1
        else:
            self.evictions += 1
        self.keys[slot], self.values[slot] = key, value


def load_mixed_variant(variant, capacity=None):
    if capacity is None:
        return load_variant(variant)
    source = variant_source(variant, bounded=True)
    module = types.ModuleType(f'negamax_tt_{variant}_mixed_{capacity}')
    exec(compile(source, f'{BASELINE}:{AGENT_PATH}:{variant}:mixed:{capacity}', 'exec'), module.__dict__)
    original = module.SearchTable

    class MixedSearchTable(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.entries = MixedEntries(capacity)

    module.SearchTable = MixedSearchTable
    return module


def declare(directory, parent):
    # Reuse only original immutable A evidence; do not rerun or pool A.
    config = bench.declare(directory)
    for name in ('A-results.jsonl', 'A-analysis.json', 'A-microbenchmark.json'):
        with (directory / name).open('xb') as stream:
            stream.write((parent / name).read_bytes())
    bench.write_new(directory / 'mixed-manifest.json', dict(
        design_sha256=bench.digest((directory / 'DESIGN.md').read_bytes()),
        runner_sha256=bench.digest(Path(__file__).read_bytes()),
        A_results_sha256=bench.digest((parent / 'A-results.jsonl').read_bytes()),
        A_analysis_sha256=bench.digest((parent / 'A-analysis.json').read_bytes()),
        parent_manifest_sha256=bench.digest((parent / 'manifest.json').read_bytes()),
        strategy='high bits of 64-bit multiplicative hash; full stored key equality',
        multiplier=MULTIPLIER, word_bits=64, capacities=config['capacities']))


def run(directory):
    frozen = json.loads((directory / 'mixed-manifest.json').read_text())
    assert frozen['runner_sha256'] == bench.digest(Path(__file__).read_bytes())
    assert frozen['design_sha256'] == bench.digest((directory / 'DESIGN.md').read_bytes())
    assert frozen['A_results_sha256'] == bench.digest((directory / 'A-results.jsonl').read_bytes())
    assert frozen['A_analysis_sha256'] == bench.digest((directory / 'A-analysis.json').read_bytes())
    # All source, coverage, acceptance and parity checks in the original runner
    # stay active. Patch only the table factory and fresh-process worker command.
    bench.load_variant = load_mixed_variant

    def fresh(variant, capacity, history, depth):
        import subprocess
        import sys
        return json.loads(subprocess.check_output([sys.executable, '-m',
            'scripts.benchmark_negamax_tt_mixed', 'rss-worker', '--variant', variant,
            '--capacity', str(capacity or 0), '--history', json.dumps(history), '--depth', str(depth)]))

    bench.fresh_rss = fresh
    bench.run(directory, 'B')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['declare', 'run', 'rss-worker'])
    parser.add_argument('--directory', type=Path, default=ROOT)
    parser.add_argument('--parent', type=Path, default=bench.ROOT)
    parser.add_argument('--variant')
    parser.add_argument('--capacity', type=int, default=0)
    parser.add_argument('--history')
    parser.add_argument('--depth', type=int)
    args = parser.parse_args()
    if args.command == 'declare':
        declare(args.directory, args.parent)
    elif args.command == 'rss-worker':
        bench.load_variant = load_mixed_variant
        print(json.dumps(bench.rss_worker(args.variant, args.capacity or None,
                                         json.loads(args.history), args.depth)))
    else:
        run(args.directory)


if __name__ == '__main__':
    main()
