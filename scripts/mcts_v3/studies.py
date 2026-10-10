"""Declared studies for the MCTS v3 research; see docs/search-mcts-v3/DESIGN.md.

Each function returns the arguments of ``harness.declare``. A study is added
here and committed before its first game is played.
"""
from scripts.mcts_v3.harness import matchup, mcts, negamax

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


STUDIES = dict(preflight=preflight, pilot1=pilot1)
