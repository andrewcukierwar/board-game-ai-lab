"""Local mathematical definitions, not a game-solving or compatibility test suite.

Only fixtures labelled 'thesis' reproduce original diagrams. All other fixtures
and all replay sequences below are constructed here; the diagram replays are
legal reconstructions, not move lists attributed to Allis.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from itertools import combinations
from random import Random

import pytest

from games.connect4.connect4 import Connect4, _WINDOWS
from games.connect4.grounding.analysis import GROUPS
from games.connect4.victor import (
    ALL_GROUPS, Group, Position, RuleCandidate, RuleName, Square, analyze_candidates,
    enumerate_baseinverses, enumerate_candidates, enumerate_claimevens, enumerate_verticals,
)

S = Square.from_name
CL, BI, VE = RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL


def play(moves):
    game = Connect4()
    for column in moves:
        assert game.make_move(column)
    return Position.from_board(game.board, game.current_player)


def pair(rule, *names):
    return RuleCandidate(rule, tuple(S(n) for n in names))


def line(first, last):
    """Independent endpoint interpolation for expected groups."""
    a, b = S(first), S(last)
    dr = (b.row_index - a.row_index) // 3
    dc = (b.column - a.column) // 3
    return Group(tuple(Square(a.row_index + i * dr, a.column + i * dc) for i in range(4)))


def coverage(report, candidate):
    return next(e.conditional_solved_groups for e in report.evidence if e.candidate == candidate)


@pytest.mark.parametrize('row', range(1, 7))
@pytest.mark.parametrize('column', range(7))
def test_coordinate_round_trip_and_reflection(row, column):
    square = Square(6 - row, column)
    assert square.row == row
    assert square.name == 'abcdefg'[column] + str(row)
    assert S(square.name) == square
    assert square.is_even == (row in (2, 4, 6))
    assert square.reflected() == Square(6 - row, 6 - column)
    assert square.reflected().reflected() == square


@pytest.mark.parametrize('coords', [(-1, 0), (6, 0), (0, -1), (0, 7), (True, 0), (0, 1.0)])
def test_invalid_coordinates(coords):
    with pytest.raises(ValueError):
        Square(*coords)


@pytest.mark.parametrize('name', ['a0', 'a7', 'h1', 'A1', 'a11', '', None, 1])
def test_invalid_names(name):
    with pytest.raises(ValueError):
        S(name)


def test_all_69_groups_against_independent_endpoint_geometry_and_both_existing_scans():
    # Enumerate pairs of endpoints, not the production directional window loops.
    squares = [Square(r, c) for r in range(6) for c in range(7)]
    expected = set()
    for a, b in combinations(squares, 2):
        dr, dc = b.row_index - a.row_index, b.column - a.column
        if (dr, dc) in ((0, 3), (3, 0), (3, 3), (3, -3)):
            expected.add(line(a.name, b.name))
    assert set(ALL_GROUPS) == expected
    assert len(ALL_GROUPS) == len(set(ALL_GROUPS)) == 69
    assert ALL_GROUPS == tuple(sorted(ALL_GROUPS))
    cells = {frozenset((s.row_index, s.column) for s in g.squares) for g in ALL_GROUPS}
    assert cells == {frozenset(g) for g in GROUPS} == {frozenset(g) for g in _WINDOWS}
    assert sum(len({s.row for s in g.squares}) == 1 for g in ALL_GROUPS) == 24
    assert sum(len({s.column for s in g.squares}) == 1 for g in ALL_GROUPS) == 21
    assert {g.reflected() for g in ALL_GROUPS} == set(ALL_GROUPS)
    assert Group(tuple(reversed(ALL_GROUPS[0].squares))) == ALL_GROUPS[0]


@pytest.mark.parametrize('names', [('a1',), ('a1',) * 4, ('a1', 'b1', 'c1', 'e1'),
                                 ('a1', 'b2', 'c3', 'd3')])
def test_invalid_groups(names):
    with pytest.raises(ValueError):
        Group(tuple(S(n) for n in names))


def test_empty_board_candidates_and_exact_order():
    p = play([])
    expected_cl = tuple(pair(CL, f'{c}{r}', f'{c}{r+1}') for c in 'abcdefg' for r in (1, 3, 5))
    expected_ve = tuple(pair(VE, f'{c}{r}', f'{c}{r+1}') for c in 'abcdefg' for r in (2, 4))
    expected_bi = tuple(pair(BI, a + '1', b + '1') for a, b in combinations('abcdefg', 2))
    assert enumerate_claimevens(p) == expected_cl
    assert enumerate_baseinverses(p) == expected_bi
    assert enumerate_verticals(p) == expected_ve
    result = enumerate_candidates(p)
    assert result == expected_cl + expected_bi + expected_ve
    assert len(result) == len(set(result)) == 56
    assert enumerate_candidates(p) == result
    # Both unsupported pairs and edge/top-row pairs remain valid geometry.
    assert pair(CL, 'g5', 'g6') in result
    assert pair(VE, 'a4', 'a5') in result
    assert not p.is_playable(S('a4'))


def test_thesis_diagram_6_1_exact_list():
    # §6.1, pp.36–37: visually checked original diagram and printed list.
    p = play([2, 3, 3, 3, 3, 3, 3, 4])
    assert p.board == tuple(tuple(r) for r in [
        '   X   ', '   O   ', '   X   ', '   O   ', '   X   ', '  XOO  '])
    expected = ['a1-a2', 'a3-a4', 'a5-a6', 'b1-b2', 'b3-b4', 'b5-b6',
                'c3-c4', 'c5-c6', 'e3-e4', 'e5-e6', 'f1-f2', 'f3-f4',
                'f5-f6', 'g1-g2', 'g3-g4', 'g5-g6']
    assert enumerate_claimevens(p) == tuple(pair(CL, *v.split('-')) for v in expected)
    assert not p.is_playable(S('e4'))
    report = analyze_candidates(p, 1)
    assert line('d4', 'g4') in coverage(report, pair(CL, 'e3', 'e4'))
    assert line('b1', 'e4') not in coverage(report, pair(CL, 'e3', 'e4'))  # Black d3.
    assert line('e3', 'e6') in coverage(report, pair(CL, 'e3', 'e4'))
    # Even this thesis example's overall draw argument is NOT implemented.
    assert report.status == 'candidates_only' and report.unverified_obligations


def test_thesis_diagram_6_2_useful_and_useless_baseinverses():
    # §6.2, pp.37–38: printed seven useful pairs and three explicit coverage lists.
    p = play([2, 4, 2, 2, 3, 3])
    assert p.board == tuple(tuple(r) for r in [
        '       ', '       ', '       ', '  O    ', '  XO   ', '  XXO  '])
    report = analyze_candidates(p, 1)
    useful = {e.candidate for e in report.evidence
              if e.candidate.rule == BI and e.conditional_solved_groups}
    expected = ['a1-b1', 'c4-d3', 'c4-e2', 'c4-f1', 'd3-e2', 'd3-f1', 'e2-f1']
    # The printed list omits b1-d3: b1-c2-d3-e4 is unblocked (c2 is White).
    # Follow the formal definition, without silently treating its list as exhaustive.
    assert useful == {pair(BI, *v.split('-')) for v in expected + ['b1-d3']}
    assert set(coverage(report, pair(BI, 'b1', 'd3'))) == {line('b1', 'e4')}
    assert set(coverage(report, pair(BI, 'a1', 'b1'))) == {line('a1', 'd1')}
    assert set(coverage(report, pair(BI, 'c4', 'd3'))) == {
        line('a6', 'd3'), line('b5', 'e2'), line('c4', 'f1')}
    assert set(coverage(report, pair(BI, 'd3', 'f1'))) == {line('c4', 'f1')}
    # Thesis explicitly calls these possible but useless, for different reasons.
    assert coverage(report, pair(BI, 'a1', 'c4')) == ()  # No common geometric group.
    assert coverage(report, pair(BI, 'f1', 'g1')) == ()  # e1 blocks d1-g1 for White.
    assert line('d1', 'g1') not in report.opponent_groups
    assert pair(BI, 'c4', 'd3') == pair(BI, 'd3', 'c4')
    assert pair(BI, 'a1', 'c3') not in enumerate_baseinverses(p)  # Occupied.
    assert pair(BI, 'a1', 'c5') not in enumerate_baseinverses(p)  # Unsupported.


def test_thesis_diagram_6_3_vertical_coverage():
    # §6.3, pp.38–39: e4-e5 solves e2-e5 and e3-e6; e1 is Black.
    p = play([2, 4, 2, 2, 3, 3, 4, 2, 4, 2])
    assert p.board == tuple(tuple(r) for r in [
        '       ', '  O    ', '  O    ', '  O X  ', '  XOX  ', '  XXO  '])
    report = analyze_candidates(p, 1)
    assert set(coverage(report, pair(VE, 'e4', 'e5'))) == {
        line('e2', 'e5'), line('e3', 'e6')}
    assert pair(CL, 'e3', 'e4') not in enumerate_claimevens(p)
    assert pair(BI, 'c6', 'e4') in enumerate_baseinverses(p)
    assert all(len({s.column for s in g.squares}) == 1
               for e in report.evidence if e.candidate.rule == VE for g in e.conditional_solved_groups)


@pytest.mark.parametrize('rule,names', [
    (CL, ('a2', 'a3')), (VE, ('a1', 'a2')),  # Wrong upper parity.
    (CL, ('a1', 'a4')), (VE, ('a2', 'a5')),  # Not adjacent.
    (CL, ('a1', 'b2')), (VE, ('a2', 'b3')),  # Different columns.
    (CL, ('a2', 'a1')), (VE, ('a3', 'a2')),  # Reversed roles.
    (BI, ('a1', 'a1')), (BI, ('a1', 'a2')),  # Not two distinct landing squares.
])
def test_superficially_similar_but_invalid_shapes(rule, names):
    with pytest.raises(ValueError):
        pair(rule, *names)


@pytest.mark.parametrize('rule', list(RuleName)[3:])
def test_remaining_rules_explicitly_unimplemented(rule):
    with pytest.raises(ValueError, match='not implemented'):
        pair(rule, 'a1', 'b1')


@pytest.mark.parametrize('height', range(7))
def test_all_column_heights_and_occupied_pairs(height):
    p = play([0] * height)
    candidates = enumerate_candidates(p)
    assert {s.name for s in p.playable_squares if s.column == 0} == (
        {f'a{height+1}'} if height < 6 else set())
    assert {(v.squares[0].row, v.squares[1].row) for v in enumerate_claimevens(p)
            if v.squares[0].column == 0} == {(l, l+1) for l in (1, 3, 5) if l > height}
    assert {(v.squares[0].row, v.squares[1].row) for v in enumerate_verticals(p)
            if v.squares[0].column == 0} == {(l, l+1) for l in (2, 4) if l > height}
    assert sum(any(s.column == 0 for s in v.squares) for v in enumerate_baseinverses(p)) == (
        6 if height < 6 else 0)
    assert all(p.is_empty(s) for v in candidates for s in v.squares)


def test_claimeven_covers_upper_without_lower_but_vertical_requires_both():
    report = analyze_candidates(play([]), 1)
    cl = pair(CL, 'a3', 'a4')
    ve = pair(VE, 'a2', 'a3')
    assert line('a4', 'd4') in coverage(report, cl)
    assert line('a3', 'd3') not in coverage(report, cl)
    assert line('a3', 'd3') not in coverage(report, ve)
    assert set(coverage(report, ve)) == {line('a1', 'a4'), line('a2', 'a5')}
    assert line('a3', 'a6') not in coverage(report, ve)
    assert cl.affected_squares == {S('a3'), S('a4')}
    assert cl.prerequisites.directly_playable_squares == ()
    assert cl.prerequisites.upper_parity == 'even'
    assert cl.depends_on_zugzwang and not ve.depends_on_zugzwang
    bi = pair(BI, 'a1', 'b1')
    assert bi.prerequisites.directly_playable_squares == bi.squares
    assert not bi.depends_on_zugzwang


def test_defender_filter_and_turn_are_separate_from_candidate_geometry():
    p = play([0])
    white, black = analyze_candidates(p, 0), analyze_candidates(p, 1)
    assert white.opponent_groups == p.potential_groups(1)
    assert black.opponent_groups == p.potential_groups(0)
    g = line('a1', 'd1')
    assert g not in white.opponent_groups and g in black.opponent_groups
    assert tuple(e.candidate for e in white.evidence) == tuple(e.candidate for e in black.evidence)
    assert white.status == black.status == 'candidates_only'
    assert white.unverified_obligations == black.unverified_obligations
    assert not hasattr(white, 'outcome') and not hasattr(white, 'proven')
    for invalid in (True, -1, 2, 'X'):
        with pytest.raises(ValueError):
            analyze_candidates(p, invalid)
        with pytest.raises(ValueError):
            p.potential_groups(invalid)


def test_invalid_boards_rejected():
    empty = [[' '] * 7 for _ in range(6)]
    cases = [(empty[:-1], 0), ([r[:-1] for r in empty], 0), (empty, 1), (empty, True),
             (['       '] * 6, 0)]
    for value, row, column, player in [('?', 5, 0, 0), ('X', 4, 0, 1), ('O', 5, 0, 0)]:
        board = deepcopy(empty)
        board[row][column] = value
        cases.append((board, player))
    for rows, player in [(['       '] * 4 + ['OOOOXXX', 'XXXXOOO'], 0),
                         (['       '] * 4 + ['OXOOXX ', 'XXXXOOO'], 1)]:
        cases.append(([list(r) for r in rows], player))
    for board, player in cases:
        with pytest.raises(ValueError):
            Position.from_board(board, player)
        with pytest.raises(ValueError):
            Position(board, player)


@pytest.mark.parametrize('moves', [[0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 2, 1, 2, 1]])
def test_terminal_wins_have_no_candidates(moves):
    p = play(moves)
    assert p.terminal
    assert p.playable_squares  # Physical geometry is still distinct from legal future play.
    for enumerate_rule in (enumerate_claimevens, enumerate_baseinverses, enumerate_verticals,
                           enumerate_candidates):
        assert enumerate_rule(p) == ()
    assert analyze_candidates(p, 0).evidence == analyze_candidates(p, 1).evidence == ()


def test_terminal_draw_has_no_candidates():
    p = Position.from_board([list(r) for r in ['XXOOXXO', 'OOXXOOX'] * 3], 0)
    assert p.terminal and p.playable_squares == ()
    assert enumerate_candidates(p) == ()


def test_snapshot_and_results_are_immutable():
    game = Connect4()
    before = deepcopy(game.board)
    p = Position.from_board(game.board, 0)
    report = analyze_candidates(p, 1)
    assert game.board == before
    game.make_move(3)
    assert p.board == tuple(tuple(r) for r in before)
    assert analyze_candidates(p, 1) == report
    with pytest.raises(FrozenInstanceError):
        p.player_to_move = 1
    with pytest.raises(TypeError):
        p.board[5][3] = 'X'
    with pytest.raises(FrozenInstanceError):
        report.status = 'proven'
    with pytest.raises(FrozenInstanceError):
        report.evidence[0].candidate.rule = VE


def test_small_deterministic_positions_against_independent_predicate_oracle_and_mirrors():
    # Exhaust all unordered empty-square pairs on small sampled legal positions.
    # The oracle uses bottom-based coordinate arithmetic and raw cells, not
    # candidate properties, playable_squares, potential_groups or coverage_squares.
    rng = Random(1988)
    for _ in range(6):
        game = Connect4()
        for ply in range(25):
            if game.is_game_over():
                break
            if ply % 4 == 0:
                p = Position.from_board(game.board, game.current_player)
                raw = p.board
                empties = [(c, r) for c in range(7) for r in range(1, 7) if raw[6-r][c] == ' ']
                expected = set()
                for a, b in combinations(empties, 2):
                    ac, ar = a
                    bc, br = b
                    if ac == bc and br == ar + 1:
                        kind = CL if br % 2 == 0 else VE
                        expected.add((kind, frozenset((a, b))))
                    if ac != bc and all(r == 1 or raw[7-r][c] != ' ' for c, r in (a, b)):
                        expected.add((BI, frozenset((a, b))))
                actual = enumerate_candidates(p)
                assert {(v.rule, frozenset((s.column, s.row) for s in v.squares))
                        for v in actual} == expected
                assert len(actual) == len(set(actual))
                mirrored = Position.from_board([list(reversed(row)) for row in raw], p.player_to_move)
                assert {v.reflected() for v in actual} == set(enumerate_candidates(mirrored))
                for defender in (0, 1):
                    report = analyze_candidates(p, defender)
                    mirror_report = analyze_candidates(mirrored, defender)
                    blocker = 'XO'[defender]
                    for evidence in report.evidence:
                        v = evidence.candidate
                        # Infer coverage from rule definition without its coverage property.
                        required = ({max(v.squares, key=lambda s: s.row)} if v.rule == CL
                                    else set(v.squares))
                        wanted = {g for g in ALL_GROUPS
                                  if required <= set(g.squares)
                                  and all(raw[s.row_index][s.column] != blocker for s in g.squares)}
                        assert set(evidence.conditional_solved_groups) == wanted
                        assert evidence.conditional_solved_groups == tuple(sorted(wanted))
                        assert {g.reflected() for g in wanted} == set(coverage(mirror_report, v.reflected()))
            assert game.make_move(rng.choice(game.get_valid_moves()))
