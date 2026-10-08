"""Allis §§6.4–6.9 composite rule definitions; never game values.

Fixtures labelled with a diagram number are legal reconstructions of the
visually checked thesis boards (``nine_rule_reference.DIAGRAMS``); all other
positions are constructed here. Expected groups are written out by hand or
derived by the independent test reference, never by the producer.
"""
from dataclasses import FrozenInstanceError
from random import Random

import pytest

from games.connect4.connect4 import Connect4
from games.connect4.victor import ALL_GROUPS, Group, Position, RuleCandidate, RuleName, Square
from games.connect4.victor.composite import (
    COMPOSITE_RULES, Component, CompositeCandidate, SolutionClause, enumerate_afterevens,
    enumerate_all_candidates, enumerate_baseclaims, enumerate_befores, enumerate_composites,
    enumerate_highinverses, enumerate_lowinverses, enumerate_specialbefores,
    prerequisite_failures, solution_clauses,
)
from games.connect4.victor.nine_rules import analyze_nine_rules
from victor_validation import nine_rule_reference as R

S = Square.from_name
CL, BI, VE = RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL
AE, LI, HI = RuleName.AFTEREVEN, RuleName.LOWINVERSE, RuleName.HIGHINVERSE
BC, BE, SB = RuleName.BASECLAIM, RuleName.BEFORE, RuleName.SPECIALBEFORE


def play(moves):
    game = Connect4()
    for column in moves:
        assert game.make_move(column)
    return Position.from_board(game.board, game.current_player)


def diagram(name):
    return play(R.DIAGRAMS[name])


def line(first, last):
    a, b = S(first), S(last)
    dr, dc = (b.row_index - a.row_index) // 3, (b.column - a.column) // 3
    return Group(tuple(Square(a.row_index + i * dr, a.column + i * dc) for i in range(4)))


def cl(lower, upper):
    return Component(CL, S(lower), S(upper))


def ve(lower, upper):
    return Component(VE, S(lower), S(upper))


def sq(*names):
    return tuple(S(n) for n in names)


def white_targets(p):
    return {g for g in ALL_GROUPS if all(p.board[s.row_index][s.column] != 'O' for s in g.squares)}


def solved(p, candidate):
    clauses = solution_clauses(candidate, p)
    return {g for g in white_targets(p) if any(c.solves(g) for c in clauses)}


def having(p, *required_sets):
    """Independent expectation: White groups meeting every required name set."""
    return {g for g in white_targets(p)
            if all({s.name for s in g.squares} & set(names) for names in required_sets)}


# ------------------------------------------------------------ diagram replays

@pytest.mark.parametrize('name', sorted(R.DIAGRAM_BOARDS))
def test_diagram_replays_match_visually_transcribed_boards(name):
    p = diagram(name)
    assert p.board == tuple(tuple(row) for row in R.DIAGRAM_BOARDS[name])
    assert p.player_to_move == 0 and not p.terminal  # Every diagram: White to move.


# ------------------------------------------------------------------ Aftereven

def test_diagram_6_4_afterevens_and_inherited_claimeven_coverage():
    p = diagram('6.4')
    c2f2 = CompositeCandidate(AE, line('c2', 'f2'), (cl('f1', 'f2'),))
    b2e2 = CompositeCandidate(AE, line('b2', 'e2'), (cl('b1', 'b2'),))
    assert set(enumerate_afterevens(p)) >= {c2f2, b2e2}
    assert prerequisite_failures(c2f2, p) == prerequisite_failures(b2e2, p) == ()
    # "solves all groups which need a square in the range f3-f6 ... also f2 itself".
    assert solved(p, c2f2) == having(p, ('f2', 'f3', 'f4', 'f5', 'f6'))
    assert solved(p, b2e2) == having(p, ('b2', 'b3', 'b4', 'b5', 'b6'))
    assert line('c3', 'f3') in solved(p, c2f2)  # White's c3-e3 needs only f3.
    assert [c.label for c in solution_clauses(c2f2, p)] == ['aftereven-columns', 'claimeven:f1-f2']


def test_diagram_6_5_multi_column_aftereven_needs_every_column():
    p = diagram('6.5')
    d2g2 = CompositeCandidate(AE, line('d2', 'g2'), (cl('f1', 'f2'), cl('g1', 'g2')))
    assert d2g2 in enumerate_afterevens(p)
    f_above, g_above = ('f3', 'f4', 'f5', 'f6'), ('g3', 'g4', 'g5', 'g6')
    timing = having(p, f_above, g_above)
    assert solved(p, d2g2) == timing | having(p, ('f2',)) | having(p, ('g2',))
    assert line('c3', 'f3') not in solved(p, d2g2)  # Only the f column: thesis example.
    assert line('d6', 'g3') in timing  # f4 and g3: one square above in BOTH columns.
    # "there is no rule which can be used to solve this problem" (all nine rules).
    report = analyze_nine_rules(p)
    assert not any(line('c3', 'f3') in e.conditional_solved_groups for e in report.evidence)


@pytest.mark.parametrize('moves,group', [
    ([2, 2, 3, 3, 2, 4, 3, 4, 4, 0], 'c2-f2'),
])
def test_aftereven_requires_every_empty_square_even_with_empty_square_below(moves, group):
    p = play(moves)
    groups = {e.group for e in enumerate_afterevens(p)}
    assert line(*group.split('-')) in groups
    # Odd empty squares (row 1 or 3) or a directly playable even square never qualify.
    for candidate in enumerate_afterevens(p):
        for square in candidate.group.squares:
            if p.is_empty(square):
                assert square.is_even and not p.is_playable(square)
    # Diagram 6.8: f2 is the missing square but f1 is occupied, so no Aftereven there.
    assert line('d2', 'g2') not in {e.group for e in enumerate_afterevens(diagram('6.8'))}


def test_aftereven_shapes_and_prerequisites_rejected():
    p = diagram('6.4')
    with pytest.raises(ValueError):
        CompositeCandidate(AE, line('c2', 'f2'), (ve('f1', 'f2'),))  # Not a Claimeven.
    with pytest.raises(ValueError):
        CompositeCandidate(AE, None, (cl('f1', 'f2'),))
    with pytest.raises(ValueError):
        CompositeCandidate(AE, line('c2', 'f2'), ())
    with pytest.raises(ValueError):
        CompositeCandidate(AE, line('c3', 'f3'), (cl('f1', 'f2'),))  # f2 not in group.
    with pytest.raises(ValueError):
        CompositeCandidate(AE, line('a1', 'a4'), (cl('a1', 'a2'),))  # Lower inside group.
    # Wrong board: White stone in the group; components not covering every hole.
    white = CompositeCandidate(AE, line('c1', 'f4'), (cl('f3', 'f4'),))  # c1, d2: mixed.
    assert 'group contains an opponent stone' in prerequisite_failures(white, p)
    missing = CompositeCandidate(AE, line('d2', 'g2'), (cl('f1', 'f2'),))
    assert 'components do not handle exactly the empty group squares' in prerequisite_failures(
        missing, diagram('6.5'))


# ----------------------------------------------------------------- Lowinverse

def test_diagram_6_6_lowinverses_including_unaligned_and_unplayable_lowers():
    p = diagram('6.6')
    found = set(enumerate_lowinverses(p))
    li = CompositeCandidate(LI, components=(ve('c2', 'c3'), ve('d2', 'd3')))
    assert li in found
    assert CompositeCandidate(LI, components=(ve('c4', 'c5'), ve('d4', 'd5'))) in found
    assert CompositeCandidate(LI, components=(ve('a2', 'a3'), ve('c4', 'c5'))) in found
    assert not p.is_playable(S('a2'))  # "not necessary for the lower squares to be playable".
    # Both upper squares, or both squares of either constituent Vertical.
    assert solved(p, li) == (having(p, ('c3',), ('d3',)) | having(p, ('c2',), ('c3',))
                             | having(p, ('d2',), ('d3',)))
    assert line('a3', 'd3') in solved(p, li) and line('c3', 'c6') not in solved(p, li)
    assert line('c1', 'c4') not in solved(p, li)  # c1 is Black: not a White group.


@pytest.mark.parametrize('components', [
    (ve('c1', 'c2'), ve('d1', 'd2')),  # Even upper squares.
    (ve('c2', 'c3'), ve('c4', 'c5')),  # Same column.
    (ve('c2', 'c3'),),                 # One column.
    (ve('c2', 'c3'), ve('d2', 'd3'), ve('e2', 'e3')),
    (cl('c1', 'c2'), ve('d2', 'd3')),  # Claimeven part.
])
def test_lowinverse_invalid_shapes(components):
    with pytest.raises(ValueError):
        CompositeCandidate(LI, components=components)


def test_lowinverse_occupied_square_fails_prerequisites():
    li = CompositeCandidate(LI, components=(ve('a2', 'a3'), ve('d2', 'd3')))
    assert prerequisite_failures(li, play([])) == ()
    assert prerequisite_failures(li, play([3, 3, 3])) == ('occupied rule squares: d3 d2',)
    assert li not in enumerate_lowinverses(play([3, 3, 3]))


# ---------------------------------------------------------------- Highinverse

def test_diagram_6_6_highinverse_with_both_lowers_playable():
    p = diagram('6.6')
    hi = CompositeCandidate(HI, roles=sq('c2', 'c3', 'c4', 'd2', 'd3', 'd4'))
    assert hi in enumerate_highinverses(p)
    assert CompositeCandidate(HI, roles=sq('a2', 'a3', 'a4', 'c4', 'c5', 'c6')) in enumerate_highinverses(p)
    labels = [c.label for c in solution_clauses(hi, p)]
    assert labels == ['upper-pair', 'middle-pair', 'vertical:c3-c4', 'vertical:d3-d4',
                      'lower-first+upper-second', 'lower-second+upper-first']
    # Thesis: the conditional pairs (c2,d4) and (d2,c4) are "no use in this position".
    assert solved(p, hi) == (having(p, ('c4',), ('d4',)) | having(p, ('c3',), ('d3',))
                             | having(p, ('c3',), ('c4',)) | having(p, ('d3',), ('d4',)))


def test_highinverse_conditional_coverage_requires_direct_playability_now():
    hi = CompositeCandidate(HI, roles=sq('c2', 'c3', 'c4', 'e2', 'e3', 'e4'))
    both = play([2, 4])      # c1 and e1 occupied: both lower squares playable.
    only_e = play([4, 6])    # e1 occupied, c1 empty: only e2 playable.
    for p in (both, only_e):
        assert hi in enumerate_highinverses(p)
    assert {line('b1', 'e4'), line('c2', 'f5'), line('b5', 'e2'), line('c4', 'f1')} <= solved(p := both, hi)
    assert line('b5', 'e2') in solved(only_e, hi) and line('c4', 'f1') in solved(only_e, hi)
    assert line('b1', 'e4') not in solved(only_e, hi)  # Requires c2 to be playable NOW.
    assert line('c2', 'f5') not in solved(only_e, hi)
    assert 'lower-first+upper-second' not in [c.label for c in solution_clauses(hi, only_e)]


@pytest.mark.parametrize('roles', [
    ('c1', 'c2', 'c3', 'd1', 'd2', 'd3'),  # Odd upper squares.
    ('c2', 'c3', 'c5', 'd2', 'd3', 'd4'),  # Not consecutive.
    ('c2', 'c3', 'c4', 'c4', 'c5', 'c6'),  # Same column.
    ('c4', 'c3', 'c2', 'd2', 'd3', 'd4'),  # Reversed roles.
    ('c2', 'c3', 'c4'),
])
def test_highinverse_invalid_shapes(roles):
    with pytest.raises(ValueError):
        CompositeCandidate(HI, roles=sq(*roles))


def test_highinverse_column_order_is_normalized():
    a = CompositeCandidate(HI, roles=sq('d2', 'd3', 'd4', 'c2', 'c3', 'c4'))
    assert a == CompositeCandidate(HI, roles=sq('c2', 'c3', 'c4', 'd2', 'd3', 'd4'))
    assert a.roles[0] == S('c2')


# ------------------------------------------------------------------ Baseclaim

def test_diagram_6_7_baseclaim_roles():
    p = diagram('6.7')
    bc = CompositeCandidate(BC, roles=sq('b1', 'c1', 'e1'))
    assert bc in enumerate_baseclaims(p) and bc.baseclaim_square == S('c2')
    assert {line('b1', 'e4'), line('c1', 'f1')} <= solved(p, bc)
    assert solved(p, bc) == having(p, ('b1',), ('c2',)) | having(p, ('c1',), ('e1',))
    # Reversed first/third roles solve different groups.
    reversed_roles = CompositeCandidate(BC, roles=sq('e1', 'c1', 'b1'))
    assert reversed_roles in enumerate_baseclaims(p)
    assert line('b1', 'e4') not in solved(p, reversed_roles)
    assert line('b1', 'e1') in solved(p, reversed_roles)
    assert solved(p, reversed_roles) == having(p, ('e1',), ('c2',)) | having(p, ('c1',), ('b1',))


def test_baseclaim_invalid_shapes_and_prerequisites():
    with pytest.raises(ValueError):
        CompositeCandidate(BC, roles=sq('b1', 'c2', 'e1'))  # Second square even.
    with pytest.raises(ValueError):
        CompositeCandidate(BC, roles=sq('b1', 'b3', 'e1'))  # Shared column.
    with pytest.raises(ValueError):
        CompositeCandidate(BC, roles=sq('b1', 'c1'))
    p = diagram('6.7')
    assert prerequisite_failures(CompositeCandidate(BC, roles=sq('b2', 'c1', 'e1')), p) == (
        'b2 is not directly playable',)
    assert prerequisite_failures(CompositeCandidate(BC, roles=sq('b1', 'd1', 'e1')), p)[0].startswith(
        'occupied rule squares')
    # Every enumerated Baseclaim: three distinct playable squares, odd second square.
    for bc in enumerate_baseclaims(p):
        assert all(p.is_playable(s) for s in bc.roles) and not bc.roles[1].is_even
    assert len(enumerate_baseclaims(p)) == sum(
        1 for a in p.playable_squares for b in p.playable_squares for c in p.playable_squares
        if len({a, b, c}) == 3 and b.row % 2 == 1)


# --------------------------------------------------------------------- Before

def test_diagram_6_8_before_solves_what_no_earlier_rule_can():
    p = diagram('6.8')
    before = CompositeCandidate(BE, line('d2', 'g2'), (ve('f2', 'f3'),))
    assert before in enumerate_befores(p) and prerequisite_failures(before, p) == ()
    assert solved(p, before) == having(p, ('f3',))  # Successor f3; Vertical f2-f3.
    report = analyze_nine_rules(p)
    solvers = {e.candidate.rule for e in report.evidence if line('d3', 'g3') in e.conditional_solved_groups}
    assert solvers == {BE}


def test_diagram_6_9_before_with_even_upper_vertical_and_claimeven_alternative():
    p = diagram('6.9')
    verticals = CompositeCandidate(BE, line('b4', 'e1'), (ve('b4', 'b5'), ve('e1', 'e2')))
    claim = CompositeCandidate(BE, line('b4', 'e1'), (cl('b3', 'b4'), ve('e1', 'e2')))
    assert {verticals, claim} <= set(enumerate_befores(p))
    with pytest.raises(ValueError):
        RuleCandidate(VE, sq('e1', 'e2'))  # Not a standalone Vertical (even upper).
    # "The only group which needs both [b5 and e2] is b5-e2" (pure geometry).
    assert [g for g in ALL_GROUPS if {S('b5'), S('e2')} <= set(g.squares)] == [line('b5', 'e2')]
    assert line('b5', 'e2') in solved(p, verticals) and line('b5', 'e2') in solved(p, claim)
    assert solved(p, verticals) == (having(p, ('b5',), ('e2',)) | having(p, ('b4',), ('b5',))
                                    | having(p, ('e1',), ('e2',)))
    assert solved(p, claim) == (having(p, ('b5',), ('e2',)) | having(p, ('b4',))
                                | having(p, ('e1',), ('e2',)))


def test_before_invalid_shapes_and_prerequisites():
    group = line('b4', 'e1')
    with pytest.raises(ValueError, match='Aftereven'):
        CompositeCandidate(BE, line('a2', 'd2'), (cl('a1', 'a2'), cl('b1', 'b2')))
    with pytest.raises(ValueError):
        CompositeCandidate(BE, line('a6', 'd6'), (ve('a5', 'a6'),))  # Handles a5, not in group.
    with pytest.raises(ValueError):
        CompositeCandidate(BE, group, (ve('b4', 'b5'), ve('b5', 'b6')))  # Overlap.
    with pytest.raises(ValueError):
        CompositeCandidate(BE, group, (ve('b4', 'b5'), ve('b4', 'b5')))
    with pytest.raises(ValueError):
        CompositeCandidate(BE, None, (ve('b4', 'b5'),))
    with pytest.raises(ValueError):  # Upper-row group square.
        CompositeCandidate(BE, line('a6', 'd6'), (cl('a5', 'a6'), ve('b5', 'b6')))
    p = diagram('6.9')
    partial = CompositeCandidate(BE, group, (ve('b4', 'b5'),))
    assert prerequisite_failures(partial, p) == (
        'components do not handle exactly the empty group squares',)
    white = CompositeCandidate(BE, line('a1', 'd1'), (ve('a1', 'a2'), ve('b1', 'b2')))  # d1 White.
    assert prerequisite_failures(white, p) == ('group contains an opponent stone',)


def test_vertical_before_group_cannot_use_overlapping_components():
    # Stacked group squares in one column would need overlapping pairs.
    p = play([0, 6, 0, 6])
    assert not any(len({s.column for s in c.group.squares}) == 1 and len(c.components) > 1
                   for c in enumerate_befores(p))


# --------------------------------------------------------------- Specialbefore

def test_diagram_6_10_specialbefore_versus_plain_before():
    p = diagram('6.10')
    special = CompositeCandidate(SB, line('d2', 'g2'), (cl('f1', 'f2'), cl('g1', 'g2')), sq('e2', 'd3'))
    plain = CompositeCandidate(BE, line('d2', 'g2'), (ve('e2', 'e3'), cl('f1', 'f2'), cl('g1', 'g2')))
    assert special in enumerate_specialbefores(p) and plain in enumerate_befores(p)
    assert prerequisite_failures(special, p) == ()
    assert line('d3', 'g3') in solved(p, special) and line('d3', 'g3') in solved(p, plain)
    assert line('c4', 'f1') in solved(p, special)  # Both playable squares e2 and d3.
    assert solved(p, special) == (having(p, ('d3',), ('e3',), ('f3',), ('g3',))
                                  | having(p, ('e2',), ('d3',)) | having(p, ('f2',)) | having(p, ('g2',)))
    assert [c.label for c in solution_clauses(special, p)][:2] == ['successors+extra', 'playable-pair']
    assert special.special_squares == {S('e2'), S('d3')}
    assert plain.special_squares == frozenset()


@pytest.mark.parametrize('components,roles', [
    ((cl('f1', 'f2'), cl('g1', 'g2')), ('e2', 'e3')),   # Extra square in the same column.
    ((cl('f1', 'f2'), cl('g1', 'g2')), ('e2', 'f2')),   # Extra square in the group.
    ((cl('f1', 'f2'), cl('g1', 'g2')), ('d3', 'e2')),   # Playable square outside the group.
    ((cl('f1', 'f2'), cl('g1', 'g2')), ('e2', 'f1')),   # Extra square reused by a component.
    ((ve('e2', 'e3'), cl('f1', 'f2'), cl('g1', 'g2')), ('e2', 'd3')),  # Playable square doubled.
    ((cl('f1', 'f2'), cl('g1', 'g2')), ('e2',)),
])
def test_specialbefore_invalid_shapes(components, roles):
    with pytest.raises(ValueError):
        CompositeCandidate(SB, line('d2', 'g2'), components, sq(*roles))


def test_specialbefore_playability_prerequisites():
    p = diagram('6.10')
    not_playable_extra = CompositeCandidate(
        SB, line('d2', 'g2'), (cl('f1', 'f2'), cl('g1', 'g2')), sq('e2', 'b2'))
    assert prerequisite_failures(not_playable_extra, p) == ('b2 is not directly playable',)
    raised_group = CompositeCandidate(SB, line('d2', 'g2'), (ve('e2', 'e3'), cl('g1', 'g2')), sq('f2', 'd3'))
    assert 'f2 is not directly playable' in prerequisite_failures(raised_group, p)


# -------------------------------------------------------- shared structure

def test_component_shapes():
    assert ve('e1', 'e2').before_square == S('e1') and cl('b3', 'b4').before_square == S('b4')
    for bad in (lambda: Component(CL, S('a2'), S('a3')), lambda: Component(VE, S('a2'), S('a1')),
                lambda: Component(VE, S('a1'), S('b2')), lambda: Component(BI, S('a1'), S('a2')),
                lambda: Component('vertical', S('a1'), S('a2')), lambda: Component(VE, 'a1', S('a2'))):
        with pytest.raises(ValueError):
            bad()


def test_composite_type_and_rule_errors():
    for rule in (CL, BI, VE, 'before', None):
        with pytest.raises(ValueError):
            CompositeCandidate(rule, line('d2', 'g2'), (ve('f2', 'f3'),))
    with pytest.raises(ValueError):
        CompositeCandidate(BE, line('d2', 'g2'), [ve('f2', 'f3')])
    with pytest.raises(ValueError):
        CompositeCandidate(BE, line('d2', 'g2'), (('vertical', S('f2'), S('f3')),))
    with pytest.raises(ValueError, match='two-square'):
        RuleCandidate(BE, sq('f2', 'f3'))
    candidate = CompositeCandidate(BE, line('d2', 'g2'), (ve('f2', 'f3'),))
    with pytest.raises(FrozenInstanceError):
        candidate.rule = AE
    with pytest.raises(ValueError):
        solution_clauses(None, play([]))
    with pytest.raises(ValueError):
        candidate.solution_clauses(None)


def test_component_order_is_normalized_and_identity_is_canonical():
    a = CompositeCandidate(BE, line('d2', 'g2'), (cl('g1', 'g2'), ve('e2', 'e3'), cl('f1', 'f2')))
    b = CompositeCandidate(BE, line('d2', 'g2'), (ve('e2', 'e3'), cl('f1', 'f2'), cl('g1', 'g2')))
    assert a == b and hash(a) == hash(b)
    assert [k.lower.column for k in a.components] == [4, 5, 6]


def test_solution_clause_semantics():
    clause = SolutionClause('x', (frozenset(sq('a1', 'a2')), frozenset(sq('b1',))))
    assert clause.solves(line('a1', 'd1'))
    assert not clause.solves(line('c1', 'f1'))
    assert SolutionClause('empty', ()).solves(line('a1', 'd1'))
    assert not SolutionClause('none', (frozenset(),)).solves(line('a1', 'd1'))


@pytest.mark.parametrize('moves', [[0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 2, 1, 2, 1]])
def test_terminal_positions_have_no_composites(moves):
    p = play(moves)
    assert p.terminal and enumerate_composites(p) == () and enumerate_composites(p, 0) == ()


def test_controller_selects_the_groups_without_opponent_stones():
    p = diagram('6.4')
    assert all(all(p.board[s.row_index][s.column] != 'X' for s in c.group.squares)
               for c in enumerate_afterevens(p, 1) + enumerate_befores(p, 1))
    assert all(all(p.board[s.row_index][s.column] != 'O' for s in c.group.squares)
               for c in enumerate_afterevens(p, 0) + enumerate_befores(p, 0))
    assert enumerate_lowinverses(p) == tuple(c for c in enumerate_composites(p, 0) if c.rule == LI)
    with pytest.raises(ValueError):
        enumerate_composites(p, 2)


def _random_positions(seed, count, max_plies=34):
    rng = Random(seed)
    out = []
    while len(out) < count:
        game = Connect4()
        for _ in range(rng.randrange(0, max_plies)):
            if game.is_game_over():
                break
            game.make_move(rng.choice(game.get_valid_moves()))
        p = Position.from_board(game.board, game.current_player)
        if not p.terminal:
            out.append(p)
    return out


@pytest.mark.parametrize('p', _random_positions(1988, 12))
def test_enumeration_and_coverage_match_independent_reference(p):
    report = analyze_nine_rules(p)
    identities = [R.identity_of(e.candidate) for e in report.evidence]
    assert len(identities) == len(set(identities))
    assert set(identities) == R.reference_identities(p.board)
    targets = R.white_groups(p.board)
    for evidence, identity in zip(report.evidence, identities):
        expected = {g for g in targets if R.solved(identity, p.board, g)}
        got = {frozenset((s.column, 6 - s.row_index) for s in g.squares)
               for g in evidence.conditional_solved_groups}
        assert got == expected, identity
        if type(evidence.candidate) is CompositeCandidate:
            assert prerequisite_failures(evidence.candidate, p) == ()
    assert analyze_nine_rules(p) == report  # Deterministic.


@pytest.mark.parametrize('p', _random_positions(7, 8))
def test_mirror_symmetry_of_enumeration_and_coverage(p):
    mirror = Position.from_board([list(reversed(row)) for row in p.board], p.player_to_move)
    original, reflected = analyze_nine_rules(p), analyze_nine_rules(mirror)
    by_candidate = {e.candidate: e for e in reflected.evidence}
    assert {e.candidate.reflected() for e in original.evidence} == set(by_candidate)
    for evidence in original.evidence:
        assert evidence.candidate.reflected().reflected() == evidence.candidate
        other = by_candidate[evidence.candidate.reflected()]
        assert {g.reflected() for g in evidence.conditional_solved_groups} == set(other.conditional_solved_groups)


def test_enumeration_order_and_counts_on_empty_board():
    p = play([])
    candidates = enumerate_all_candidates(p)
    rules = [c.rule for c in candidates]
    assert rules == sorted(rules, key=list(RuleName).index)
    counts = {rule: rules.count(rule) for rule in RuleName}
    # Independently: LI = C(7,2)*2*2 pairs (upper rows 3/5); HI likewise (upper rows 4/6);
    # BC = 7*6*5 ordered playable triples (all second squares on odd row 1); AE = row-2/4/6
    # horizontals (12); BE/SB counted by the reference.
    assert counts[LI] == counts[HI] == 84 and counts[BC] == 210 and counts[AE] == 12
    reference = R.reference_identities(p.board)
    assert counts[BE] == sum(i[0] == 'before' for i in reference) == 196
    assert counts[SB] == sum(i[0] == 'specialbefore' for i in reference) == 224
    assert set(COMPOSITE_RULES) == set(RuleName) - {CL, BI, VE}
