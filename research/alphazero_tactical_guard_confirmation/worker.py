"""One (seed, opponent) task: optional regression suites, control matches, guarded matches; no retry."""
import argparse, time
from .common import runtime, enforce, load, state, OUTPUT, write, hc, Deadline
from .evaluate import matches, positions, openings


def main():
    p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, required=True); p.add_argument('--depth', type=int, required=True)
    p.add_argument('--suites', action='store_true'); p.add_argument('--deadline', type=float, required=True); args = p.parse_args()
    runtime(); enforce(); check = Deadline(args.deadline); check()
    output = OUTPUT / 'tasks' / f'{args.seed}-negamax{args.depth}'
    if output.exists(): raise FileExistsError('Task cannot be retried')
    output.mkdir(parents=True)
    start, cpu = time.time(), hc.cpu_seconds()
    runner, model, inference, entry = load(args.seed); before = state(runner); weights = runner.factory.hash(model)
    if args.suites:
        positions(inference, args.seed, check, output)  # 4D.4J-A regression suites, identical streams
    for variant in ('control', 'guarded'):
        matches(inference, args.seed, args.depth, variant, check, output)
    check(); enforce()
    assert before == state(runner) and weights == runner.factory.hash(model)
    write(output / 'identity.json', dict(complete=True, seed=args.seed, opponent=f'negamax{args.depth}', generation=6,
        architecture=hc.IDENTITY['cnn'], checkpoint=entry, weights_sha256=weights, before=before, after=state(runner),
        suites=args.suites, matches=4 * len(openings()), seconds=time.time() - start, cpu_seconds=hc.cpu_seconds() - cpu,
        rss_mib=hc.peak_rss_mib()))
    print(f'Completed {args.seed} negamax{args.depth}', flush=True)


if __name__ == '__main__':
    main()
