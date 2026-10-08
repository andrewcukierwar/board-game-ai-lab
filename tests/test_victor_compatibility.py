"""Exact six-entry §7.4 fragment; fixtures are constructed unless labelled thesis."""
from copy import copy
from itertools import combinations_with_replacement

import pytest

from games.connect4.victor import (
    Position, RuleCandidate, RuleName, Square, compatible, enumerate_candidates,
    required_constraints,
)
from games.connect4.victor.contracts import CompatibilityConstraint

CL, BI, VE = RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL


def pair(rule, *names):
    return RuleCandidate(rule, tuple(Square.from_name(n) for n in names))


@pytest.mark.parametrize('first,second,conflict', [
    (pair(CL, 'a1', 'a2'), pair(CL, 'b1', 'b2'), pair(CL, 'a1', 'a2')),
    (pair(CL, 'a1', 'a2'), pair(BI, 'b1', 'c1'), pair(BI, 'a1', 'b1')),
    (pair(CL, 'a3', 'a4'), pair(VE, 'b2', 'b3'), pair(VE, 'a2', 'a3')),
    (pair(BI, 'a1', 'b1'), pair(BI, 'c1', 'd1'), pair(BI, 'b1', 'c1')),
    (pair(BI, 'a2', 'b1'), pair(VE, 'c2', 'c3'), pair(VE, 'a2', 'a3')),
    (pair(VE, 'a2', 'a3'), pair(VE, 'b2', 'b3'), pair(VE, 'a2', 'a3')),
])
def test_every_supported_pair_acceptance_rejection_and_symmetry(first, second, conflict):
    assert required_constraints(first.rule, second.rule) == (CompatibilityConstraint.DISJOINT_SQUARES,)
    assert required_constraints(second.rule, first.rule) == required_constraints(first.rule, second.rule)
    assert compatible(first, second) and compatible(second, first)
    assert not compatible(first, conflict) and not compatible(conflict, first)


@pytest.mark.parametrize('rule,names', [(CL, ('a1', 'a2')), (BI, ('a1', 'b1')), (VE, ('a2', 'a3'))])
def test_duplicate_instances_conflict(rule, names):
    candidate = pair(rule, *names)
    assert not compatible(candidate, copy(candidate))


def test_thesis_section_5_3_conflicting_responses_use_claimeven_lower_trigger():
    # §5.3 p.34 explicitly contrasts a1-b1 with Claimeven b1-b2.
    cl, bi = pair(CL, 'b1', 'b2'), pair(BI, 'a1', 'b1')
    assert cl.coverage_squares.isdisjoint(bi.coverage_squares)
    assert not compatible(cl, bi)
    # Same issue with standalone Vertical's upper / Claimeven's lower.
    cl, ve = pair(CL, 'a3', 'a4'), pair(VE, 'a2', 'a3')
    assert cl.coverage_squares.isdisjoint(ve.coverage_squares)
    assert not compatible(cl, ve)


@pytest.mark.parametrize('first,second', [
    (a, b) for a, b in combinations_with_replacement(RuleName, 2)
    if a not in (CL, BI, VE) or b not in (CL, BI, VE)
])
def test_all_unimplemented_matrix_entries_fail_closed(first, second):
    for a, b in ((first, second), (second, first)):
        with pytest.raises(ValueError, match='not implemented'):
            required_constraints(a, b)


def test_corrupted_or_untyped_candidates_rejected():
    candidate = pair(CL, 'a1', 'a2')
    for rule in ('claimeven', RuleName.AFTEREVEN):
        corrupted = copy(candidate)
        object.__setattr__(corrupted, 'rule', rule)
        with pytest.raises(ValueError):
            compatible(candidate, corrupted)
    with pytest.raises(ValueError):
        compatible(candidate, None)
    with pytest.raises(ValueError):
        required_constraints('claimeven', CL)


def test_entire_small_universe_against_raw_disjointness_reference_and_mirror():
    p = Position.from_board([[' '] * 7 for _ in range(6)], 0)
    candidates = enumerate_candidates(p)
    for a, b in combinations_with_replacement(candidates, 2):
        expected = not any(x == y for x in a.squares for y in b.squares)
        assert compatible(a, b) == compatible(b, a) == expected
        assert compatible(a.reflected(), b.reflected()) == expected
