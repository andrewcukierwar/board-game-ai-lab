"""Declared studies for the MCTS v3 research; see docs/search-mcts-v3/DESIGN.md.

Each function returns the arguments of ``harness.declare``. A study is added
here and committed before its first game is played.
"""
import json
import math

from scripts.mcts_v3.harness import ROOT, matchup, mcts, negamax

PRIMARY_BUDGETS = (400, 2000)
SINGLES = dict(
    aa={},
    r1=dict(rollout='decisive'),
    r2=dict(rollout='safe'),
    s=dict(solver=True),
    e=dict(expansion='center'),
    c050=dict(exploration=0.5),
    c070=dict(exploration=0.7),
    c100=dict(exploration=1.0),
    c200=dict(exploration=2.0),
)


def research(simulations, config):
    """Research agent spec; the A/A control uses the research class at defaults."""
    return dict(type='research', simulations=simulations, config=dict(config))


def preflight():
    """Harness smoke run on a separately seeded set excluded from inference."""
    rows = [matchup(f'{name}-{b}', research(b, config), mcts(b))
            for b in PRIMARY_BUDGETS for name, config in SINGLES.items()]
    rows += [matchup(f'base-{b}-vs-negamax-{d}', mcts(b), negamax(d))
             for b in PRIMARY_BUDGETS for d in (4, 6, 8)]
    return dict(opening_set='preflight', matchups=rows, cap_seconds=600,
                note='Smoke and cost projection only; never used for inference.')


def pilot1():
    """Single components versus the baseline at equal simulations (dev set)."""
    rows = [matchup(f'{name}-{b}', research(b, config), mcts(b))
            for b in PRIMARY_BUDGETS for name, config in SINGLES.items()]
    return dict(opening_set='dev', matchups=rows, cap_seconds=2400,
                note='Exploratory pilot 1: single components, equal simulations.')


def equal_time_budget(study, matchup_id, budget):
    """Declared rule: floor(min(1, 0.90 * r) * B), r = baseline / candidate wall time.

    ``r`` comes from the realised game timings of an already analysed
    development-set study, so the budget is frozen before it is used.
    """
    rows = json.loads((ROOT / study / 'analysis.json').read_text())['matchups']
    ratio = next(row['time_ratio'] for row in rows if row['matchup'] == matchup_id)
    return math.floor(min(1.0, 0.90 / ratio) * budget)


def pilot2():
    """Passing structural components versus the baseline at equal time (dev set)."""
    rows = [matchup(f'{name}t-{b}', research(equal_time_budget('pilot1', f'{name}-{b}', b),
                                             SINGLES[name]), mcts(b))
            for b in PRIMARY_BUDGETS for name in ('r1', 'r2', 's')]
    return dict(opening_set='dev', matchups=rows, cap_seconds=900,
                note='Exploratory pilot 2: R1, R2, S at equal-time budgets from pilot 1 timings. '
                     'E failed the pilot-1 pass rule (pooled 51.4% < 52%).')


COMBOS = dict(
    r1s=dict(rollout='decisive', solver=True),
    r2s=dict(rollout='safe', solver=True),
)


def pilot3a():
    """Both rollout levels combined with the solver, equal simulations (dev set)."""
    rows = [matchup(f'{name}-{b}', research(b, config), mcts(b))
            for b in PRIMARY_BUDGETS for name, config in COMBOS.items()]
    return dict(opening_set='dev', matchups=rows, cap_seconds=600,
                note='Exploratory pilot 3a (design amendment 1): R1+S and R2+S, equal simulations.')


def pilot3b():
    """Both combinations versus the baseline at equal time (dev set)."""
    rows = [matchup(f'{name}t-{b}', research(equal_time_budget('pilot3a', f'{name}-{b}', b), config),
                    mcts(b))
            for b in PRIMARY_BUDGETS for name, config in COMBOS.items()]
    return dict(opening_set='dev', matchups=rows, cap_seconds=600,
                note='Exploratory pilot 3b: R1+S and R2+S at equal-time budgets from pilot 3a timings.')


def pilot3c():
    """Exploration-constant sweep on R2+S, head-to-head against R2+S at 1.41 (dev set)."""
    rows = [matchup(f'r2s-c{round(100 * c):03d}-{b}',
                    research(b, dict(COMBOS['r2s'], exploration=c)), research(b, COMBOS['r2s']))
            for b in PRIMARY_BUDGETS for c in (0.5, 0.7, 1.0, 2.0)]
    return dict(opening_set='dev', matchups=rows, cap_seconds=1200,
                note='Exploratory pilot 3c: R2+S had the higher pooled equal-time score in 3b '
                     '(61.6% vs 61.0%), so the constant sweep runs on R2+S, equal simulations.')


R2S_C050 = dict(COMBOS['r2s'], exploration=0.5)


def pilot3d():
    """R2+S with c = 0.5 versus the baseline at equal simulations (dev set)."""
    rows = [matchup(f'r2sc-{b}', research(b, R2S_C050), mcts(b)) for b in PRIMARY_BUDGETS]
    return dict(opening_set='dev', matchups=rows, cap_seconds=300,
                note='Exploratory pilot 3d: c = 0.5 scored 53.65% pooled head-to-head in 3c '
                     '(rule: >= 52%), so it replaces 1.41 on R2+S. Equal simulations, for timing.')


def pilot3e():
    """R2+S with c = 0.5 versus the baseline at equal time (dev set)."""
    rows = [matchup(f'r2sct-{b}', research(equal_time_budget('pilot3d', f'r2sc-{b}', b), R2S_C050),
                    mcts(b)) for b in PRIMARY_BUDGETS]
    return dict(opening_set='dev', matchups=rows, cap_seconds=300,
                note='Exploratory pilot 3e: R2+S c = 0.5 at equal-time budgets from pilot 3d timings.')


# Finalists frozen after the pilots and before any held-out game. Equal-time
# budgets follow the declared rule from development-set timings; the literals
# guard against the rule or its inputs drifting after the freeze.
FINALISTS = dict(f1=(R2S_C050, 'pilot3d', 'r2sc'), f2=(COMBOS['r1s'], 'pilot3a', 'r1s'))
FROZEN_BUDGETS = dict(f1={400: 187, 2000: 1023}, f2={400: 255, 2000: 1415})


def finalist(name, budget, equal_time):
    config, study, prefix = FINALISTS[name]
    if not equal_time:
        return research(budget, config)
    simulations = equal_time_budget(study, f'{prefix}-{budget}', budget)
    if simulations != FROZEN_BUDGETS[name][budget]:
        raise ValueError('Equal-time budget differs from the frozen value')
    return research(simulations, config)


def confirm_primary():
    """Held-out primary comparisons: both finalists, both budgets, both modes."""
    rows = [matchup(f'{name}-{mode}-{b}', finalist(name, b, mode == 'time'), mcts(b), primary=True)
            for name in FINALISTS for b in PRIMARY_BUDGETS for mode in ('sims', 'time')]
    return dict(opening_set='holdout', matchups=rows, cap_seconds=4500,
                note='Confirmatory primary family of 8 on the held-out set.')


def confirm_negamax():
    """Held-out secondary: baseline and equal-time finalists versus Negamax 4/6/8."""
    rows = []
    for b in PRIMARY_BUDGETS:
        for d in (4, 6, 8):
            rows.append(matchup(f'base-{b}-vs-negamax-{d}', mcts(b), negamax(d), pairs=96))
            rows += [matchup(f'{name}-time-{b}-vs-negamax-{d}', finalist(name, b, True), negamax(d),
                             pairs=96) for name in FINALISTS]
    return dict(opening_set='holdout', matchups=rows, cap_seconds=2000,
                note='Confirmatory secondary (exploratory): common-opponent comparison on the '
                     'first 96 held-out openings.')


# Paired differences reported by the analysis: (label, candidate matchup, reference matchup).
CONTRASTS = dict(confirm_negamax=[
    (f'{name} minus baseline at {b} vs Negamax {d}',
     f'{name}-time-{b}-vs-negamax-{d}', f'base-{b}-vs-negamax-{d}')
    for name in FINALISTS for b in PRIMARY_BUDGETS for d in (4, 6, 8)])

SCALING_BUDGETS = (100, 800, 5000)


def pilot4():
    """F1 at the secondary budgets, equal simulations, for equal-time calibration (dev set)."""
    rows = [matchup(f'r2sc-{b}', research(b, R2S_C050), mcts(b)) for b in SCALING_BUDGETS]
    return dict(opening_set='dev', matchups=rows, cap_seconds=600,
                note='Development-set timing calibration for the declared scaling secondary; '
                     'scores are exploratory.')


def confirm_scaling():
    """Held-out secondary: F1 at equal time across 100 / 800 / 5,000."""
    rows = [matchup(f'f1-time-{b}', research(equal_time_budget('pilot4', f'r2sc-{b}', b), R2S_C050),
                    mcts(b), pairs=128) for b in SCALING_BUDGETS]
    return dict(opening_set='holdout', matchups=rows, cap_seconds=900,
                note='Confirmatory secondary (exploratory): budget scaling of F1 at equal time, '
                     'first 128 held-out openings.')


def confirm_empty():
    """Secondary: F1 at equal time from the empty board, 64 independent seed pairs."""
    rows = [matchup(f'f1-time-{b}', finalist('f1', b, True), mcts(b)) for b in PRIMARY_BUDGETS]
    return dict(opening_set='empty', matchups=rows, cap_seconds=600,
                note='Secondary (exploratory): real starting position; clusters are seed pairs.')


def strict_budget(budget):
    """Follow-up rule: shrink F1's equal-time budget by its empty-board time overrun.

    floor(B' * 0.90 / ratio), with ratio the realised F1 / baseline search time
    in the completed ``confirm_empty`` games at baseline budget ``budget``.
    """
    rows = json.loads((ROOT / 'confirm_empty' / 'analysis.json').read_text())['matchups']
    ratio = next(row['time_ratio'] for row in rows if row['matchup'] == f'f1-time-{budget}')
    return math.floor(FROZEN_BUDGETS['f1'][budget] * 0.90 / ratio)


def followup_strict_empty():
    """F1 at strict-latency budgets from the empty board, fresh seed pairs."""
    rows = [matchup(f'f1-strict-{b}', research(strict_budget(b), R2S_C050), mcts(b))
            for b in PRIMARY_BUDGETS]
    return dict(opening_set='empty2', matchups=rows, cap_seconds=600,
                note='Follow-up A (design amendment 2): strict-latency budgets on 64 fresh '
                     'empty-board seed pairs. Calibrated on confirm_empty, evaluated here.')


def followup_strict_holdout():
    """F1 at strict-latency budgets on the held-out set."""
    rows = [matchup(f'f1-strict-{b}', research(strict_budget(b), R2S_C050), mcts(b))
            for b in PRIMARY_BUDGETS]
    return dict(opening_set='holdout', matchups=rows, cap_seconds=900,
                note='Follow-up A (design amendment 2): strict-latency budgets on the held-out '
                     'set. No selection is made from these results.')


def followup_equivalence():
    """F1 at its frozen equal-time budgets versus much larger baseline budgets."""
    f1 = FROZEN_BUDGETS['f1']
    rows = [matchup(f'f1-{f1[small]}-vs-base-{large}', research(f1[small], R2S_C050), mcts(large),
                    pairs=128)
            for small, large in ((400, 2000), (400, 5000), (2000, 5000), (2000, 10000))]
    return dict(opening_set='holdout', matchups=rows, cap_seconds=1200,
                note='Follow-up C (design amendment 3, exploratory): how much baseline compute '
                     'F1 replaces. First 128 held-out openings; nothing is selected from it.')


STUDIES = dict(preflight=preflight, pilot1=pilot1, pilot2=pilot2, pilot3a=pilot3a,
               pilot3b=pilot3b, pilot3c=pilot3c, pilot3d=pilot3d, pilot3e=pilot3e,
               confirm_primary=confirm_primary, confirm_negamax=confirm_negamax,
               pilot4=pilot4, confirm_scaling=confirm_scaling, confirm_empty=confirm_empty,
               followup_strict_empty=followup_strict_empty,
               followup_strict_holdout=followup_strict_holdout,
               followup_equivalence=followup_equivalence)
