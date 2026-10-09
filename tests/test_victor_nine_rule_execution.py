"""Local response transitions and complete adversarial continuations.

Local unit tests deliberately bypass cover acceptance to inspect one rule's
response mechanism. They make no outcome claim. Public policy construction is
tested separately with independently checked whole-board witnesses.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from games.connect4.victor import Position, RuleName, SearchBudget, search_nine_rule_cover
from games.connect4.victor.composite import Component, CompositeCandidate, solution_clauses
from games.connect4.victor.execution import NineRulePolicy, Obligation, PolicyState, RuleState, _compile
from games.connect4.victor.exact import Bits
from games.connect4.victor.geometry import Square
from games.connect4.victor.nine_rules import RuleEvidence
from victor_validation.exact_oracle import replay, solve
from victor_validation.nine_rule_reference import DIAGRAMS
from test_victor_functional_solver import endgames

S = Square.from_name
CL, BI, VE = RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL
AE, LI, HI = RuleName.AFTEREVEN, RuleName.LOWINVERSE, RuleName.HIGHINVERSE
BC, BE, SB = RuleName.BASECLAIM, RuleName.BEFORE, RuleName.SPECIALBEFORE


def local(name, rule, names=()):
    raw = replay(DIAGRAMS[name])
    p = Position.from_board(raw.board, 0)
    from games.connect4.victor.nine_rules import analyze_nine_rules
    candidates = [e.candidate for e in analyze_nine_rules(p).evidence if e.candidate.rule == rule]
    c = next(c for c in candidates if {S(n) for n in names} <= c.affected_squares)
    policy = object.__new__(NineRulePolicy)
    clauses = solution_clauses(c, p)
    covered = tuple(g for g in p.potential_groups(0) if any(cl.solves(g) for cl in clauses))
    policy.witness = SimpleNamespace(evidence=(RuleEvidence(c, clauses, covered),))
    policy.initial = Bits.from_position(p)
    policy.start = PolicyState(policy.initial, (RuleState(0, 'waiting', _compile(c)),))
    return policy, c


def test_aftereven_timing_stops_at_black_completion():
    policy, c = local('6.4', AE, ('f1', 'f2'))
    d = policy.select((5,))
    assert d.column == 5 and d.kind == 'terminal_win'
    terminal = policy.select((5, 5))
    assert terminal.status == 'terminal'
    assert policy.select((5, 5, 0)).status == 'illegal_continuation'


def test_lowinverse_releases_other_column_as_vertical_and_preserves_identity():
    policy, c = local('6.6', LI, ('c2', 'c3', 'd2', 'd3'))
    d = policy.select((2,))
    assert d.column == 2 and d.kind == 'forced_response'
    assert d.state.rules[0].phase == 'activated'
    assert d.state.rules[0].index == 0
    assert all(o.kind == VE for o in d.state.rules[0].obligations)
    d = policy.select((2, 2, 3))
    assert d.column == 3  # residual d2-d3 Vertical
    assert policy.select((2, 1)).status == 'policy_violated'


@pytest.mark.parametrize('second_white,second_black', [(2,3), (3,2)])
def test_highinverse_residual_claimeven_and_conditional_baseinverse(second_white, second_black):
    policy, c = local('6.6', HI, ('c2','c3','c4','d2','d3','d4'))
    d = policy.select((2,))
    assert d.column == 2
    kinds = {o.kind for o in d.state.rules[0].obligations}
    assert kinds == {CL, BI}
    assert policy.select((2,2,second_white)).column == second_black


@pytest.mark.parametrize('white,black,follow_white,follow_black', [
    (1,4,2,2), (2,4,1,2), (4,2,1,2),
])
def test_baseclaim_ordered_role_alternatives(white, black, follow_white, follow_black):
    policy, c = local('6.7', BC, ('b1','c1','c2','e1'))
    # Several role assignments use the same squares. Pin the intended one.
    c = CompositeCandidate(BC, roles=(S('b1'),S('c1'),S('e1')))
    p = Position.from_board(policy.initial.board, 0)
    clauses = solution_clauses(c,p)
    policy.witness = SimpleNamespace(evidence=(RuleEvidence(c,clauses,
        tuple(g for g in p.potential_groups(0) if any(cl.solves(g) for cl in clauses))),))
    policy.start = PolicyState(policy.initial,(RuleState(0,'waiting',_compile(c)),))
    assert policy.select((white,)).column == black
    assert policy.select((white,black,follow_white)).column == follow_black


def test_before_even_upper_vertical_and_specialbefore_extra_square():
    policy,c = local('6.9',BE,('e1','e2','b4','b5'))
    assert any(k.rule == VE and k.upper.is_even for k in c.components)
    assert policy.select((4,)).column == 4
    policy,c = local('6.10',SB,('e2','d3','f1','f2','g1','g2'))
    assert policy.select((4,)).column == 3
    assert policy.select((3,)).column == 4
    assert policy.select((5,)).column == 5  # component Claimeven
    policy,c = local('6.10',BE,('e2','e3','f1','f2','g1','g2'))
    assert {k.rule for k in c.components} == {CL,VE}
    assert policy.select((5,)).column == 5
    assert policy.select((5,5,4)).column == 4


def test_highinverse_unplayable_original_lower_does_not_create_baseinverse():
    policy,c = local('6.6',HI,('a2','a3','a4','c4','c5','c6'))
    # Set up a legal local branch with a1 White / c2 Black; a2 is now playable.
    state = policy._black(policy._white(policy.start,0),2)
    state = policy._white(state,0)
    assert policy._decision(state).column == 0
    assert {o.kind for o in state.rules[0].obligations} == {CL}


def test_ambiguous_and_unplayable_obligations_are_explicit():
    policy,c = local('6.6',LI,('c2','c3','d2','d3'))
    state = policy._white(policy.start,2)
    assert policy._decision(replace(state,pending=(S('c3'),S('d2')))).status == 'conflicting_obligations'
    assert policy._decision(replace(state,pending=(S('g6'),))).status == 'unplayable_obligation'
    full_forbidden = tuple(Obligation(CL,(S(chr(97+c)+'1'),S(chr(97+c)+'2'))) for c in (0,1,4,5,6))
    # Cover all current landings, including the even inverse lowers.
    full_forbidden += (Obligation(LI,(S('c2'),S('c3'),S('d2'),S('d3'))),)
    p=policy.initial.drop(0)
    # Add a1-a2? a2 is now landing, use HI/LI prohibition to cover it as well.
    obligations=full_forbidden+(Obligation(LI,(S('a2'),S('a3'),S('b2'),S('b3'))),)
    d=policy._decision(PolicyState(p,(RuleState(0,'waiting',obligations),)))
    assert d.status == 'no_permitted_spare'


def test_public_policy_acceptance_and_cutoffs_are_not_certificates():
    raw = replay(DIAGRAMS['6.10'])
    p=Position.from_board(raw.board,0)
    cover=search_nine_rule_cover(p)
    policy=NineRulePolicy(cover.witness)
    assert len(policy.start.rules) == len(cover.witness.evidence)
    assert policy.audit(SearchBudget(nodes=0,max_remaining=24)).status == 'unknown_remaining_cap'
    with pytest.raises((ValueError,AttributeError)):
        NineRulePolicy(SimpleNamespace(context=SimpleNamespace(position=p)))
    assert policy.select((True,)).status == 'invalid_continuation'


def test_adversarial_endgames_all_white_continuations_compare_to_oracle():
    found=composite=0
    for history,p in endgames(seed=6010,count=100,remaining=(8,10)):
        position=Position.from_board(p.board,p.turn)
        cover=search_nine_rule_cover(position)
        if cover.witness is None:
            continue
        found+=1
        composite+=any(e.candidate.rule not in (CL,BI,VE) for e in cover.witness.evidence)
        policy=NineRulePolicy(cover.witness)
        audit=policy.audit(SearchBudget(nodes=100_000,seconds=None,max_remaining=10))
        assert audit.status == 'verified_policy_nonloss', (history,audit)
        assert solve(p,max_remaining=10).mover_value <= 0
        cutoff=policy.audit(SearchBudget(nodes=0,seconds=None,max_remaining=10))
        assert cutoff.status == 'unknown_node_budget' and not cutoff.counterexample
        # Select on each immediate White branch; responses are actual legal moves.
        for col in p.legal_columns:
            d=policy.select((col,))
            assert d.column in p.drop(col).legal_columns
    assert found >= 5 and composite >= 2
