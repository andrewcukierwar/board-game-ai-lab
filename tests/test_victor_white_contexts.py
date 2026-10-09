"""Visually transcribed Allis ch.8 diagrams and explicit source restrictions."""
from dataclasses import replace

import pytest

from games.connect4.victor import Position, Square, SolverBudget, SearchBudget, select_move
from games.connect4.victor.compatibility import footprint
from games.connect4.victor.white import search_white_covers, white_evaluation_contexts
from games.connect4.victor.coverage import SearchStatus

DIAGRAMS = {
    '8.1': ['       ', '       ', '   OX  ', ' XXXO  ', ' XOOO  ', 'XOXXO  '],
    '8.2': ['       ', '       ', '   OX  ', ' XXXO  ', ' XOOO  ', ' OXXO X'],
    '8.3': ['   OO  ', '   XX  ', '   OO  ', '   XX  ', '   OX  ', '   XO X'],
    '8.7': ['       ', '   XO  ', '   OX  ', '   XX  ', '   XO  ', '  OXO  '],
}
S = Square.from_name


def diagram(name):
    return Position.from_board([list(r) for r in DIAGRAMS[name]], 1)


@pytest.mark.parametrize('name', ['8.1', '8.2'])
def test_odd_threat_reserved_column_targets_and_complete_restricted_cover(name):
    p = diagram(name)
    contexts = white_evaluation_contexts(p)
    assert len(contexts) == 1
    c = contexts[0]
    assert c.kind == 'odd_threat' and c.crossing == S('a3')
    assert c.reserved_columns == (0,)
    assert all(s.column != 0 for s in c.permitted_squares)
    # Every Black group needing a3 or higher is conditionally irrelevant.
    assert all(g not in c.target_groups for g in p.potential_groups(1)
               if any(s.column == 0 and s.row >= 3 for s in g.squares))
    if name == '8.2':
        assert not any(cl.label == 'crossing-odd:a1' for cl in c.claims)
    cover, = search_white_covers(p)
    assert cover.status is SearchStatus.FOUND
    assert cover.certification == 'conditional_source_coverage_only'
    assert all(footprint(e.candidate).squares <= c.permitted_squares for e in cover.evidence)
    assert all(any(g in e.conditional_solved_groups for e in cover.evidence) for g in c.target_groups)


def test_combination_even_above_odd_all_claims_and_baseinverse_pair_shift():
    c = next(c for c in white_evaluation_contexts(diagram('8.3')) if c.crossing == S('f3'))
    assert c.kind == 'even_above_odd' and c.odd == S('g3') and c.even == S('g4')
    assert c.reserved_columns == (5, 6)
    claims = {cl.label: cl.requirements for cl in c.claims}
    assert 'crossing-odd:f1' not in claims
    assert claims['crossing-odd:f3'] == (frozenset((S('f3'),)),)
    assert claims['crossing-successor+odd'] == (frozenset((S('f4'),)), frozenset((S('g3'),)))
    assert claims['reserved-baseinverse'] == (frozenset((S('f1'),)), frozenset((S('g2'),)))
    assert not any(label.startswith('reserved-vertical') for label in claims)  # starts at g3
    assert claims['both-above'] == (frozenset(S('f'+str(r)) for r in (4,5,6)),
                                   frozenset(S('g'+str(r)) for r in (4,5,6)))


def test_odd_above_even_two_variants_are_distinct():
    p = diagram('8.7')
    c, = white_evaluation_contexts(p)
    assert c.kind == 'odd_above_even_unplayable'
    assert 'reserved-baseinverse' in {cl.label for cl in c.claims}
    rows = [list(row) for row in p.board]
    rows[5][6], rows[4][2] = 'X', 'O'  # g2 becomes playable; counts preserved
    p = Position.from_board(rows, 1)
    c = next(c for c in white_evaluation_contexts(p) if c.crossing == S('f3'))
    assert c.kind == 'odd_above_even_playable'
    assert {cl.label for cl in c.claims} == {'crossing-odd:f3', 'crossing-odd:f5', 'both-above'}


def test_source_contexts_do_not_claim_a_white_win_and_turn_is_not_reversed():
    p = diagram('8.1')
    budget = SolverBudget(exact=SearchBudget(nodes=0), strategic_children=0)
    r = select_move(p.board, 1, budget)
    assert r.white_context_count == 1 and r.white_covers[0].status is SearchStatus.FOUND
    assert r.exact_value is r.bound is None
    assert r.move in [s.column for s in p.playable_squares]
    rows = [list(row) for row in p.board]
    rows[5][0] = ' '  # remove a1: now White to move
    assert white_evaluation_contexts(Position.from_board(rows, 0)) == ()
    assert search_white_covers(p, context_budget=0) == ()


@pytest.mark.parametrize('name', DIAGRAMS)
def test_reflection_preserves_regions_targets_and_claims(name):
    p = diagram(name)
    q = Position.from_board(tuple(tuple(reversed(row)) for row in p.board), 1)
    left, right = white_evaluation_contexts(p), white_evaluation_contexts(q)
    assert len(left) == len(right)
    for c in left:
        d = next(d for d in right if d.kind == c.kind and d.crossing == c.crossing.reflected()
                 and d.odd == (c.odd.reflected() if c.odd else None))
        assert {g.reflected() for g in c.target_groups} == set(d.target_groups)
        assert {s.reflected() for s in c.permitted_squares} == set(d.permitted_squares)


@pytest.mark.parametrize('kwargs', [dict(node_budget=-1), dict(context_budget=True)])
def test_invalid_white_budgets(kwargs):
    with pytest.raises(ValueError):
        search_white_covers(diagram('8.1'), **kwargs)
