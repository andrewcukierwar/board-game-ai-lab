"""Prospective design and final freeze, before any measured confirmation match."""
import argparse
from .common import *


def main():
    runtime(); p = argparse.ArgumentParser(); p.add_argument('--freeze', action='store_true'); args = p.parse_args()
    if args.freeze:
        if read(RESULTS / 'prelaunch-review.json')['disposition'] != 'PASS': raise ValueError('Review pending')
        write(RESULTS / 'freeze.json', dict(files=source_map(), runtime=runtime(), guard_sha256=sha(GUARD),
              checkpoints=[ja.parent_entry(s, 6) for s in SEEDS])); return
    openings, calibration, pilot = read(RESULTS / 'openings.json'), read(RESULTS / 'calibration.json'), None
    assert sha(GUARD) == GUARD_SHA256 and read(RESULTS / 'validation.json')['disposition'] == 'PASS'
    depth = calibration['opponent_b_depth']; F = len(openings['confirmation'])
    write(RESULTS / 'protected-before.json', protected())
    write(RESULTS / 'design.json', dict(contract='phase4d4j-b-protocol-v1', base=BASE, branch=BRANCH,
        guard=dict(path='research/alphazero_tactical_guard/guard.py', sha256=GUARD_SHA256, contract='connect4-tactical-root-guard-v1',
                   decision_contract='connect4-guarded-decision-v1', frozen_in='ffb8009b014ab6c84b9eb00ccab986fcb6bc4b14', modified=False),
        seeds=list(SEEDS), generation=6, architecture=hc.IDENTITY['cnn'], checkpoints=[ja.parent_entry(s, 6) for s in SEEDS],
        simulations=256, exploration=1.41, root_noise=False, final_selection='deterministic temperature 0 (historical seeded max-visit ties)',
        variants=['control', 'guarded'], opponents=[dict(name='negamax3', depth=3, role='A continuity'),
                                                     dict(name=f'negamax{depth}', depth=depth, role='B stronger, selected by declared calibration rule')],
        secondary_generation20='OMITTED before freeze: different official-campaign loader not verified strict-compatible; +768 games (~470 CPU-s) would compete with primary and review budget',
        resnet='OMITTED', training=False, selfplay=False,
        families=F, opening_sha256=openings['confirmation_sha256'], openings_file_sha256=sha(RESULTS / 'openings.json'),
        strata={s: sum(r['stratum'] == s for r in openings['confirmation']) for s in sorted({r['stratum'] for r in openings['confirmation']})},
        match_namespace='phase4d4j-b-confirmation:negamax{depth}', match_seed=4420003,
        tasks=[dict(seed=s, depth=d, suites=d == 3) for s in SEEDS for d in (3, depth)],
        matches=2 * 2 * 2 * F * 2, matches_per_task=2 * F * 2, regression_searches=2 * 2 * 628,
        regression_suites='4D.4J-A tactical200/solved300/deep128 with identical 4D.4J-A streams (regression only, not confirmation evidence)',
        analysis=dict(seed=4420004, replicates=10000, unit='whole opening family; all seeds, opponents, colors and arms kept together',
                      interval='percentile 95%', pooled='equal-weight mean over all 768 paired game differences (balanced 4 seed x opponent cells)',
                      strata='within-stratum family resampling', margin=0.02),
        decision_rules=dict(
            FAIL='any reproducible tactical correctness violation (guarded match decision disagrees with independent reference, tactical200 guarded avoidable/missed win, or first paired decision differs between arms); '
                 'any solved300/deep128 guarded exact-WDL below control; regression-suite non-reproduction of 4D.4J-A; checkpoint/protocol integrity failure; '
                 'latency failure; or demonstrated material regression: pooled difference <= -0.02 with 95% upper < 0, or any single opponent difference <= -0.02 with its 95% upper < 0',
            CONFIRMED_POSITIVE_EFFECT='no FAIL condition AND pooled difference > 0 AND pooled 95% lower > 0 AND each seed difference > -0.02 AND no opponent material regression',
            PRACTICALLY_MEANINGFUL='reported separately: pooled point estimate >= +0.02 (no +5pp requirement)',
            SMALL_EFFECT='CONFIRMED_POSITIVE_EFFECT with pooled point estimate < +0.02',
            INCONCLUSIVE='otherwise; a positive point estimate whose interval spans zero is never called confirmed',
            QUALIFIED='if seed or opponent point estimates differ in sign, the interpretation is qualified'),
        latency=dict(p95_guard_seconds_max=.001, worst_guard_seconds_max=.005, mean_relative_overhead_max=.05),
        timing_note='four single-thread tasks run in parallel on the 10-core M4; guard timing is a CPU-local nanosecond clock separate from inference',
        task_origin_epoch=TASK_ORIGIN_EPOCH, hard_maximum_seconds=HARD_MAXIMUM_SECONDS,
        absolute_deadline_epoch=TASK_ORIGIN_EPOCH + HARD_MAXIMUM_SECONDS,
        compute_deadline_epoch=TASK_ORIGIN_EPOCH + HARD_MAXIMUM_SECONDS - 1200,
        projected_cpu_seconds=calibration['eligibility'][str(depth)]['projected_total_cpu_seconds'] if str(depth) in calibration['eligibility'] else None,
        no_retry=True, no_sealed_access=True))
    print('Design prepared')


if __name__ == '__main__':
    main()
