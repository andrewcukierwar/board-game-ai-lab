"""Allis §7.4 constraints 1–4 across all nine rules (intrinsic shapes, no board).

Thesis cases cite §§5.3, 6.7, 7.1–7.4. Interpretations (constraint 2 'entirely
above'; N.B.(ii) special squares) are documented in
docs/victor-nine-rule-implementation.md and backed by exact-oracle counterexamples
in test_victor_nine_rule_search.py.
"""
from copy import copy
from random import Random

import pytest

from games.connect4.connect4 import Connect4
from games.connect4.victor import Group, Position, RuleCandidate, RuleName, Square, compatible
from games.connect4.victor.compatibility import failed_constraints
from games.connect4.victor.composite import Component, CompositeCandidate
from games.connect4.victor.contracts import CompatibilityConstraint as K
from games.connect4.victor.nine_rules import analyze_nine_rules, conflict_masks, pairwise_conflicts
from victor_validation import nine_rule_reference as R

S = Square.from_name
CL, BI, VE = RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL
AE, LI, HI = RuleName.AFTEREVEN, RuleName.LOWINVERSE, RuleName.HIGHINVERSE
BC, BE, SB = RuleName.BASECLAIM, RuleName.BEFORE, RuleName.SPECIALBEFORE


def line(first, last):
    a, b = S(first), S(last)
    dr, dc = (b.row_index - a.row_index) // 3, (b.column - a.column) // 3
    return Group(tuple(Square(a.row_index + i * dr, a.column + i * dc) for i in range(4)))


def pair(rule, a, b):
    return RuleCandidate(rule, (S(a), S(b)))


def cl(a, b):
    return Component(CL, S(a), S(b))


def ve(a, b):
    return Component(VE, S(a), S(b))


def li(*pairs):
    return CompositeCandidate(LI, components=tuple(ve(*p.split('-')) for p in pairs))


def hi(*names):
    return CompositeCandidate(HI, roles=tuple(S(n) for n in names))


def both(a, b):
    """Compatibility is symmetric; return it once after checking both orders."""
    assert compatible(a, b) == compatible(b, a)
    assert failed_constraints(a, b) == failed_constraints(b, a)
    return compatible(a, b)


# §7.1 / diagram 7.2: Claimevens below an inverse change Zugzwang; above is fine.
def test_claimeven_must_not_be_below_or_across_an_inverse():
    low = li('a4-a5', 'b4-b5')
    assert not both(pair(CL, 'a1', 'a2'), low)
    assert failed_constraints(pair(CL, 'a1', 'a2'), low) == (K.NO_CLAIMEVEN_BELOW_INVERSE,)
    assert both(pair(CL, 'a5', 'a6'), li('a2-a3', 'b2-b3'))  # "can be used above".
    assert both(pair(CL, 'e1', 'e2'), li('a2-a3', 'b2-b3'))  # Diagram 7.1: other column.
    # Overlap from above or below: never "above" (constraint 2 only, no constraint 1).
    assert not both(pair(CL, 'a3', 'a4'), li('a2-a3', 'b2-b3'))
    assert not both(pair(CL, 'a1', 'a2'), li('a2-a3', 'b2-b3'))
    assert not both(pair(CL, 'a3', 'a4'), hi('a2', 'a3', 'a4', 'b2', 'b3', 'b4'))
    assert both(pair(CL, 'a5', 'a6'), hi('a2', 'a3', 'a4', 'b2', 'b3', 'b4'))
    assert not both(pair(CL, 'a1', 'a2'), hi('a4', 'a5', 'a6', 'b4', 'b5', 'b6'))


def test_aftereven_pairs():
    f = CompositeCandidate(AE, line('c2', 'f2'), (cl('f1', 'f2'),))
    fg = CompositeCandidate(AE, line('d2', 'g2'), (cl('f1', 'f2'), cl('g1', 'g2')))
    assert both(f, fg)  # §7.3: two Afterevens sharing a Claimeven.
    assert not both(f, pair(CL, 'f1', 'f2'))  # Constraint 1: sharing is "not allowed".
    above = CompositeCandidate(BE, line('d3', 'g3'), (ve('f3', 'f4'),))
    assert both(f, above)  # Column f: {f1,f2} vs {f3,f4} are disjoint.
    crossing = CompositeCandidate(BE, line('c2', 'f2'), (ve('f2', 'f3'),))
    assert not both(f, crossing)  # §7.3: a Vertical meets a Claimeven.
    assert failed_constraints(f, crossing) == (K.COLUMNWISE_DISJOINT_OR_EQUAL,)
    # With an inverse: disjoint squares AND no Claimeven below the inverse.
    assert failed_constraints(f, li('f4-f5', 'g2-g3')) == (K.NO_CLAIMEVEN_BELOW_INVERSE,)
    assert both(f, li('a2-a3', 'b2-b3'))
    assert not both(f, CompositeCandidate(BC, roles=(S('a1'), S('f1'), S('g1'))))


def test_equal_square_sets_of_different_component_kinds_are_column_equal():
    # Constraint 3 compares square sets: a Before's even-upper Vertical f1-f2 and an
    # Aftereven's Claimeven f1-f2 combine (Black simply never plays f1 first).
    vertical = CompositeCandidate(BE, line('c1', 'f1'), (ve('f1', 'f2'),))
    claimeven = CompositeCandidate(AE, line('c2', 'f2'), (cl('f1', 'f2'),))
    assert both(vertical, claimeven)


def test_before_lowinverse_share_equal_vertical_but_no_claimeven_below():
    inverse = li('a2-a3', 'b2-b3')
    sharing = CompositeCandidate(BE, line('a2', 'd2'), (ve('a2', 'a3'), ve('b2', 'b3'),
                                                        ve('c2', 'c3'), ve('d2', 'd3')))
    assert both(sharing, inverse)  # §7.3: a Vertical equal to a Lowinverse column.
    crossing = CompositeCandidate(BE, line('a4', 'd4'), (cl('a3', 'a4'), ve('b4', 'b5'),
                                                         ve('c4', 'c5'), ve('d4', 'd5')))
    assert set(failed_constraints(crossing, inverse)) == {
        K.NO_CLAIMEVEN_BELOW_INVERSE, K.COLUMNWISE_DISJOINT_OR_EQUAL}
    beneath = CompositeCandidate(BE, line('a2', 'd2'), (cl('a1', 'a2'), ve('b2', 'b3'),
                                                        ve('c2', 'c3'), ve('d2', 'd3')))
    assert failed_constraints(beneath, li('a4-a5', 'e2-e3')) == (K.NO_CLAIMEVEN_BELOW_INVERSE,)
    # Highinverse with Before is 1&2: even an equal pair conflicts.
    assert not both(sharing, hi('a2', 'a3', 'a4', 'e2', 'e3', 'e4'))


def test_specialbefore_special_squares_are_never_equal():
    sb = CompositeCandidate(SB, line('d2', 'g2'), (cl('f1', 'f2'), cl('g1', 'g2')), (S('e2'), S('d3')))
    same_specials = CompositeCandidate(SB, line('b2', 'e2'), (ve('b2', 'b3'),), (S('e2'), S('d3')))
    assert not both(sb, same_specials)  # Column parts {e2} and {d3} are equal but special.
    aftereven = CompositeCandidate(AE, line('d2', 'g2'), (cl('f1', 'f2'), cl('g1', 'g2')))
    assert both(sb, aftereven)  # N.B.(ii): shares exactly some Claimevens.
    other = CompositeCandidate(BE, line('d2', 'g2'), (ve('d2', 'd3'), ve('e2', 'e3'),
                                                      cl('f1', 'f2'), cl('g1', 'g2')))
    assert not both(sb, other)  # Its d2-d3 / e2-e3 meet the special squares.
    assert not both(sb, pair(CL, 'f1', 'f2'))  # Constraint 1 with a Claimeven rule.
    assert both(sb, pair(CL, 'e3', 'e4')) and both(sb, pair(BI, 'a1', 'b1'))  # §6.9 narrative.
    plain = CompositeCandidate(BE, line('d2', 'g2'), (ve('e2', 'e3'), cl('f1', 'f2'), cl('g1', 'g2')))
    assert not both(plain, pair(CL, 'e3', 'e4'))  # §6.9: the problem Specialbefore fixes.


def test_inverse_pairs_constraint_four():
    assert both(li('a2-a3', 'b2-b3'), li('a4-a5', 'b4-b5'))  # Equal column sets, stacked.
    assert both(li('a2-a3', 'b2-b3'), li('c2-c3', 'd2-d3'))  # Disjoint column sets.
    assert not both(li('a2-a3', 'b2-b3'), li('b4-b5', 'c2-c3'))  # Partially shared columns.
    assert failed_constraints(li('a2-a3', 'b2-b3'), li('b4-b5', 'c2-c3')) == (
        K.DISJOINT_SQUARES_AND_DISJOINT_OR_EQUAL_COLUMN_SETS,)
    assert not both(li('a2-a3', 'b2-b3'), li('a2-a3', 'c2-c3'))  # Shared squares.
    assert both(li('a2-a3', 'b2-b3'), hi('a4', 'a5', 'a6', 'b4', 'b5', 'b6'))
    assert not both(hi('a2', 'a3', 'a4', 'b2', 'b3', 'b4'), hi('b4', 'b5', 'b6', 'c2', 'c3', 'c4'))
    assert both(hi('a2', 'a3', 'a4', 'b2', 'b3', 'b4'), hi('c2', 'c3', 'c4', 'd4', 'd5', 'd6'))


def test_constraint_one_pairs_with_shared_constituents_are_not_allowed():
    before = CompositeCandidate(BE, line('b4', 'e1'), (ve('b4', 'b5'), ve('e1', 'e2')))
    assert not both(before, pair(VE, 'b4', 'b5'))  # §7.2: equal Vertical, still not allowed.
    assert both(before, pair(VE, 'b2', 'b3'))
    bc = CompositeCandidate(BC, roles=(S('b1'), S('c1'), S('e1')))
    assert not both(bc, pair(BI, 'c1', 'e1'))  # §7.2: Baseinverse/Baseclaim overlap.
    assert not both(pair(CL, 'c1', 'c2'), pair(BI, 'c1', 'e1'))  # §6.7 narrative.
    assert both(bc, pair(BI, 'a1', 'd1'))
    assert both(bc, CompositeCandidate(BC, roles=(S('a1'), S('d1'), S('g1'))))
    assert not both(bc, bc.reflected())  # f1, e1, c1: shares c1 and e1.
    assert not both(bc, CompositeCandidate(BC, roles=(S('e1'), S('c1'), S('b1'))))
    assert not both(bc, li('c4-c5', 'd2-d3'))  # Baseclaim's Claimeven c1-c2 is below.
    assert both(bc, li('d2-d3', 'f4-f5'))


@pytest.mark.parametrize('candidate', [
    pair(CL, 'a1', 'a2'), li('a2-a3', 'b2-b3'), hi('a2', 'a3', 'a4', 'b2', 'b3', 'b4'),
    CompositeCandidate(BE, line('b4', 'e1'), (ve('b4', 'b5'), ve('e1', 'e2'))),
    CompositeCandidate(SB, line('d2', 'g2'), (cl('f1', 'f2'), cl('g1', 'g2')), (S('e2'), S('d3'))),
    CompositeCandidate(AE, line('d2', 'g2'), (cl('f1', 'f2'), cl('g1', 'g2'))),
])
def test_duplicates_conflict(candidate):
    assert not compatible(candidate, copy(candidate))


def test_corrupted_composites_rejected():
    good = CompositeCandidate(BE, line('b4', 'e1'), (ve('b4', 'b5'), ve('e1', 'e2')))
    for field, value in (('rule', 'before'), ('rule', BI), ('components', [ve('b4', 'b5')]),
                         ('components', (ve('e1', 'e2'), ve('b4', 'b5'))), ('group', None),
                         ('roles', (S('a1'),)), ('group', 'b4-e1')):
        forged = copy(good)
        object.__setattr__(forged, field, value)
        with pytest.raises(ValueError):
            compatible(good, forged)
    with pytest.raises(ValueError):
        compatible(good, None)


def _positions(seed, count):
    rng = Random(seed)
    out = []
    while len(out) < count:
        game = Connect4()
        for _ in range(rng.randrange(6, 30)):
            if game.is_game_over():
                break
            game.make_move(rng.choice(game.get_valid_moves()))
        p = Position.from_board(game.board, game.current_player)
        if not p.terminal:
            out.append(p)
    return out


@pytest.mark.parametrize('p', _positions(74, 6))
def test_random_universes_match_reference_symmetry_mirror_and_fast_conflicts(p):
    rng = Random(len(p.board[5]))
    candidates = tuple(e.candidate for e in analyze_nine_rules(p).evidence)
    identities = [R.identity_of(c) for c in candidates]
    for _ in range(3000):
        i, j = rng.randrange(len(candidates)), rng.randrange(len(candidates))
        a, b = candidates[i], candidates[j]
        expected = R.compatible(identities[i], identities[j])
        assert compatible(a, b) == compatible(b, a) == expected
        assert compatible(a.reflected(), b.reflected()) == expected
    useful = tuple(e.candidate for e in analyze_nine_rules(p).evidence if e.conditional_solved_groups)
    fast, count = conflict_masks(useful)
    reference = [1 << i for i in range(len(useful))]
    for i, j in pairwise_conflicts(useful):
        reference[i] |= 1 << j
        reference[j] |= 1 << i
    assert fast == reference and count == len(pairwise_conflicts(useful))
