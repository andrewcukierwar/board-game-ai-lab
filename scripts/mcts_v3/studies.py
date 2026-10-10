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


STUDIES = dict(preflight=preflight, pilot1=pilot1, pilot2=pilot2, pilot3a=pilot3a,
               pilot3b=pilot3b, pilot3c=pilot3c)
