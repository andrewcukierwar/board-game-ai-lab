"""Explicit one-generation launch/recovery; fork-only commands never train."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from .artifacts import file_sha256
from .config import V2Config
from .experimental import ExperimentalConfig, ORIGINAL, EXTENDED, load_experimental, check_output_path
from .experimental_import import read_historical, fork_verified
from .provenance import configure_deterministic_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fork = commands.add_parser('fork-pair')
    fork.add_argument('--parent', required=True)
    fork.add_argument('--parent-sha256', required=True)
    fork.add_argument('--historical-checkout', required=True)
    fork.add_argument('--output', required=True)
    fork.add_argument('--lineage', required=True)
    fork.add_argument('--games', type=int, default=64)
    fork.add_argument('--updates', type=int, required=True)
    fork.add_argument('--new-generations', type=int, default=2)
    for command in ('inspect', 'run-one'):
        sub = commands.add_parser(command)
        sub.add_argument('--boundary', required=True)
        sub.add_argument('--sha256', required=True)
        if command == 'run-one':
            sub.add_argument('--output', required=True)
    args = parser.parse_args()
    configure_deterministic_runtime(1)
    if args.command == 'fork-pair':
        verified = read_historical(args.parent, expected_sha256=args.parent_sha256,
                                   historical_checkout=args.historical_checkout)
        old = V2Config.from_dict(verified['original']['config'])
        config = ExperimentalConfig(replace(old, games_per_generation=args.games,
                    max_generations=20+args.new_generations), args.updates)
        root = Path(args.output)
        check_output_path(root)
        root.mkdir(parents=True, exist_ok=False)
        arms = [fork_verified(verified, config, parent_sha256=args.parent_sha256,
                              schedule=schedule, lineage=args.lineage, arm=arm)
                for arm, schedule in (('control', ORIGINAL), ('treatment', EXTENDED))]
        if arms[0].initial_state_sha256() != arms[1].initial_state_sha256():
            raise ValueError('Initial arm states differ')
        result = dict(initial_state_sha256=arms[0].initial_state_sha256(), arms={})
        for runner in arms:
            directory = root/runner.arm
            directory.mkdir()
            path = directory/'fork-0020.pt'
            result['arms'][runner.arm] = dict(path=str(path), sha256=runner.save(path))
        (root/'fork-manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    else:
        runner = load_experimental(args.boundary, expected_sha256=args.sha256)
        if args.command == 'inspect':
            result = dict(counters=runner.counters(), replay_digest=runner.replay.digest(),
                          initial_state_sha256=runner.initial_state_sha256(), parent=runner.parent['file_sha256'] if runner.parent else None)
        else:
            output = Path(args.output)
            check_output_path(output)
            output.mkdir(parents=True, exist_ok=False)
            try:
                result = runner.run_generation(output/f'fork-generation-{runner.completed_generations+1:04d}.pt')
            except KeyboardInterrupt:
                raise SystemExit('Generation abandoned; reload the last published boundary')
            (output/'resources.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
