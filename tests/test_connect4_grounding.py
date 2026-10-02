"""Known tactical positions, primary-source examples, and false-positive guards."""
from copy import deepcopy
import json
from random import Random

import pytest

from games.connect4.connect4 import Connect4
from games.connect4.grounding import analyze_move, analyze_position
from games.connect4.grounding.analysis import GROUPS
from games.connect4.grounding.knowledge import knowledge_catalog, retrieve_knowledge


def position(moves):
    game = Connect4()
    for c in moves:
        assert not game.is_game_over()
        assert game.is_valid_move(c)
        game.make_move(c)
    return game


def analyze(moves):
    game = position(moves)
    return analyze_position(game.board, game.current_player)


def named(squares):
    return {s['square']['name'] for s in squares}


def test_empty_position_coordinates_and_no_claim_of_safety():
    game = Connect4()
    facts = analyze_position(game.board, 0)
    assert len(GROUPS) == 69
    assert facts['legal_columns'] == list(range(7))
    assert facts['defense']['status'] == 'no_immediate_threat'
    assert facts['immediate_winning_columns'] == []
    assert facts['alternatives'][3]['landing_square'] == {
        'row_index': 5, 'column': 3, 'row': 1, 'name': 'd1'}


def test_immediate_vertical_win_and_terminal_no_future_moves():
    game = position([0, 1, 0, 1, 0, 2])
    original = deepcopy(game.board)
    facts = analyze_position(game.board, 0)
    assert facts['immediate_winning_columns'] == [0]
    assert named(facts['winning_squares'][0]['squares']) == {'a4'}
    last = analyze_move(game.board, 0, 0)
    assert last['was_immediate_win']
    assert last['landing_square']['row_index'] == 2
    terminal = analyze_position(last['board_after'], 1)
    assert terminal['outcome'] == {'status': 'win', 'winner': 0}
    assert terminal['legal_columns'] == terminal['alternatives'] == []
    assert terminal['immediate_winning_columns'] == []
    assert terminal['defense']['status'] == 'not_applicable_terminal'
    assert game.board == original
    with pytest.raises(ValueError):
        analyze_move(last['board_after'], 1, 3)


def test_mandatory_block_for_only_survival_move():
    facts = analyze([0, 1, 0, 1, 2, 1])
    assert facts['immediate_winning_columns'] == []
    assert named(facts['opponent_immediate_winning_squares']) == {'b4'}
    assert facts['defense'] == {'status': 'mandatory_block', 'mandatory_column': 1,
                                'survival_columns': [1]}
    game = position([0, 1, 0, 1, 2, 1])
    last = analyze_move(game.board, 0, 1)
    assert last['was_mandatory_block']
    assert named(last['winning_square_changes'][1]['removed']) == {'b4'}


def test_thesis_diagram_3_9_two_distinct_playable_threats():
    # Allis §3.4, thesis/PDF p.22: d1,d2,c1,d3,e1; Black to move.
    facts = analyze([3, 3, 2, 3, 4])
    assert named(facts['opponent_immediate_winning_squares']) == {'b1', 'f1'}
    assert facts['defense']['status'] == 'unavoidable_loss_next_reply'
    assert facts['defense']['mandatory_column'] is None
    assert facts['defense']['survival_columns'] == []


def test_threat_creation_from_thesis_tactical_example():
    game = position([3, 3, 2, 3])
    last = analyze_move(game.board, 0, 4)
    assert named(last['winning_square_changes'][0]['created']) == {'b1', 'f1'}
    assert not last['was_immediate_win']
    assert last['outcome']['status'] == 'ongoing'


def test_thesis_diagram_3_1_floating_winning_squares_are_not_immediate_wins():
    # Allis §3.1, thesis/PDF pp.16-17, exact listed play sequence.
    facts = analyze([3, 3, 3, 4, 3, 4, 3, 3, 2, 2, 2, 0,
                     2, 0, 2, 2, 4, 0, 0, 0, 4, 0, 4, 4])
    white = facts['winning_squares'][0]['squares']
    assert named(white) == {'b2', 'b3', 'b4', 'b5', 'b6', 'f2', 'f3', 'f4', 'f5', 'f6'}
    assert not any(s['playable'] for s in white)
    assert facts['immediate_winning_columns'] == []
    assert facts['opponent_immediate_winning_squares'] == []
    assert facts['defense']['status'] == 'no_immediate_threat'
    # This analyzer deliberately does not infer the thesis's eventual result.
    assert facts['outcome']['status'] == 'ongoing'
    assert facts['legal_columns'] == [1, 5, 6]
    assert next(s for s in white if s['square']['name'] == 'b3')['parity'] == 'odd'
    assert next(s for s in white if s['square']['name'] == 'b2')['parity'] == 'even'


def test_own_win_means_opponent_threat_does_not_make_block_mandatory():
    facts = analyze([0, 2, 2, 0, 3, 3, 1, 2, 3, 0, 5, 0, 4])
    assert facts['immediate_winning_columns']
    assert facts['opponent_immediate_winning_squares']
    assert facts['defense']['status'] == 'immediate_win_available'
    assert facts['defense']['mandatory_column'] is None


def test_apparent_block_can_support_another_winning_square():
    seq = [5, 6, 5, 4, 2, 3, 0, 1, 3, 0, 0, 5, 0, 2, 5, 2, 1, 2, 3, 0, 6, 6, 2]
    game = position(seq)
    facts = analyze_position(game.board, game.current_player)
    threat, = facts['opponent_immediate_winning_squares']
    block = threat['square']['column']
    attempted = next(a for a in facts['alternatives'] if a['column'] == block)
    assert not attempted['avoids_immediate_loss']
    assert attempted['opponent_winning_replies']
    assert facts['defense']['status'] == 'unavoidable_loss_next_reply'
    assert facts['defense']['mandatory_column'] is None
    last = analyze_move(game.board, game.current_player, block)
    assert not last['was_mandatory_block']
    assert last['winning_square_changes'][1 - game.current_player]['newly_playable']


def test_two_lines_sharing_one_square_are_one_defensible_threat():
    facts = analyze([3, 1, 2, 5, 1, 1, 5, 2, 4, 2, 4, 3, 6, 1, 1, 1, 0, 4, 0, 0])
    threat, = facts['opponent_immediate_winning_squares']
    assert len(threat['groups']) == 2
    assert facts['defense']['status'] == 'mandatory_block'
    assert facts['defense']['mandatory_column'] == threat['square']['column']


@pytest.mark.parametrize('mirror', [False, True])
def test_both_diagonal_directions(mirror):
    seq = [3, 1, 2, 5, 1, 1, 5, 2, 4, 2, 4, 3, 6, 1, 1, 1, 0, 4, 0, 0,
           5, 2, 4, 5, 2, 2, 6, 4, 4, 6]
    if mirror:
        seq = [6 - c for c in seq]
    facts = analyze(seq)
    assert facts['immediate_winning_columns']
    assert any(len({s['column'] for s in g}) == 4 and len({s['row'] for s in g}) == 4
               for t in facts['winning_squares'][0]['squares'] if t['playable'] for g in t['groups'])


def test_full_column_and_draw():
    facts = analyze([3] * 6)
    assert 3 not in facts['legal_columns']
    game = position([3] * 6)
    with pytest.raises(ValueError):
        analyze_move(game.board, 0, 3)
    board = [list(row) for row in ['XXOOXXO', 'OOXXOOX'] * 3]
    facts = analyze_position(board, 0)
    assert facts['outcome'] == {'status': 'draw', 'winner': None}
    assert facts['legal_columns'] == []


@pytest.mark.parametrize('invalid', ['shape', 'piece', 'gravity', 'counts', 'turn', 'both_win', 'old_win'])
def test_reject_invalid_positions(invalid):
    board = [[' '] * 7 for _ in range(6)]
    player = 0
    if invalid == 'shape': board.pop()
    if invalid == 'piece': board[5][0] = 'R'
    if invalid == 'gravity': board[0][0] = 'X'; player = 1
    if invalid == 'counts': board[5][0] = 'O'
    if invalid == 'turn': player = True
    if invalid == 'both_win':
        board[5] = list('XXXXOOO'); board[4] = list('OOOOXXX')
    if invalid == 'old_win':
        board[5] = list('XXXXOOO'); board[4] = list('OXOOXX '); player = 1
    with pytest.raises(ValueError):
        analyze_position(board, player)


def test_tactical_results_agree_with_actual_engine_moves_and_replies():
    rng = Random(1988)
    for _ in range(12):
        game = Connect4()
        while not game.is_game_over():
            facts = analyze_position(game.board, game.current_player)
            assert facts['legal_columns'] == sorted(game.get_valid_moves())
            expected_wins = []
            for alternative in facts['alternatives']:
                candidate = Connect4(game.board, game.current_player)
                candidate.make_move(alternative['column'])
                if candidate.check_winner() == game.current_player:
                    expected_wins.append(alternative['column'])
                replies = []
                if not candidate.is_game_over():
                    for c in candidate.get_valid_moves():
                        reply = Connect4(candidate.board, candidate.current_player)
                        reply.make_move(c)
                        if reply.check_winner() == candidate.current_player:
                            replies.append(c)
                assert sorted(replies) == sorted(s['square']['column'] for s in alternative['opponent_winning_replies'])
            assert facts['immediate_winning_columns'] == expected_wins
            game.make_move(rng.choice(game.get_valid_moves()))


def test_curated_catalog_references_and_reference_only_rules():
    entries = knowledge_catalog()
    rules = [e for e in entries if e['kind'] == 'rule']
    assert {e['name'] for e in rules} == {'Claimeven', 'Baseinverse', 'Vertical', 'Aftereven',
                                         'Lowinverse', 'Highinverse', 'Baseclaim', 'Before', 'Specialbefore'}
    expected = {'claimeven': ('6.1', [36, 37]), 'baseinverse': ('6.2', [37, 38]),
                'vertical': ('6.3', [38, 39]), 'aftereven': ('6.4', [39, 40]),
                'lowinverse': ('6.5', [40, 41]), 'highinverse': ('6.6', [41, 42]),
                'baseclaim': ('6.7', [42, 43]), 'before': ('6.8', [43, 45]),
                'specialbefore': ('6.9', [45, 46])}
    for e in rules:
        section, pages = expected[e['id']]
        assert e['references'][0]['section'] == section
        assert e['references'][0]['thesis_pages'] == pages
        assert e['application_status'] == 'reference_only'
        assert e['programmatic_evidence']['status'] == 'not_implemented'
    for e in entries:
        assert e['preconditions'] and e['limitations'] and e['source_url'].endswith('/connect4_thesis.pdf')
        for ref in e['references']:
            assert ref['thesis_pages'] == ref['pdf_pages_1_based']
            assert [x - 1 for x in ref['pdf_pages_1_based']] == ref['pdf_page_indices_0_based']
    a = retrieve_knowledge(['tactics', 'coordinates', 'tactics'])
    assert a == retrieve_knowledge(['coordinates', 'tactics'])
    a[0]['name'] = 'changed'
    assert retrieve_knowledge(['coordinates'])[0]['name'] == 'Board nomenclature'
    with pytest.raises(ValueError):
        retrieve_knowledge(['invented_rule'])
    json.dumps(entries, allow_nan=False)
