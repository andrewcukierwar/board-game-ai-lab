"""Revision-bound evidence assembly. No network calls or agent introspection."""
from copy import deepcopy

from games.connect4.connect4 import Connect4
from .analysis import analyze_move, analyze_position, outcome
from .history import record_move
from .knowledge import SOURCE_TITLE, SOURCE_URL, retrieve_knowledge

COORDINATES = {
    'matrix': '6 rows top-first; row_index 0 is top, 5 is bottom; column 0..6 is left to right',
    'named_squares': 'a1..g6; row 1 is bottom; row = 6 - row_index',
    'players': [{'player': 0, 'piece': 'X', 'thesis_color': 'White'},
                {'player': 1, 'piece': 'O', 'thesis_color': 'Black'}],
    'move_number': '1-based ply: one placed stone; revision 0 is the empty game',
}


def build_explanation_context(*, board, player_to_move, revision, move_history,
                              players, game_id=None, concept_ids=()):
    """Build JSON-ready evidence from a detached, complete session snapshot.

Replay all records from the empty board, checking boards, turns, legal moves,
revisions, agent configuration and outcomes. Refuse mismatches, never silently
explain another revision. Standalone positions use analyze_position instead.
"""
    if type(revision) is not int or revision < 0 or revision != len(move_history):
        raise ValueError('revision must equal the complete history length')
    if len(players) != 2:
        raise ValueError('two player configurations are required')
    game = Connect4()
    records = []
    for index, record in enumerate(move_history):
        data = record.to_dict()
        if game.is_game_over() or type(record.column) is not int or not game.is_valid_move(record.column):
            raise ValueError('history contains an illegal or post-terminal move')
        before = [list(row) for row in game.board]
        player = game.current_player
        game.make_move(record.column)
        expected = record_move(before, game.board, player, players[player], record.column, index)
        if data != expected.to_dict():
            raise ValueError('history does not match a legal session replay')
        records.append(data)
    if [list(row) for row in board] != game.board or player_to_move != game.current_player:
        raise ValueError('position does not match the recorded history')
    facts = analyze_position(board, player_to_move)
    last = (analyze_move(records[-1]['board_before'], records[-1]['player'], records[-1]['column'])
            if records else None)
    relevant = {'coordinates', 'tactics', 'rule_framework', *concept_ids}
    observations = []
    has_move_threats = last is not None and any(
        change[key] for change in last['winning_square_changes']
        for key in ('created', 'removed', 'newly_playable'))
    if any(t['squares'] for t in facts['winning_squares']) or has_move_threats:
        relevant.update(('winning_square', 'parity'))
        observations.append({
            'concept_id': 'parity', 'status': 'context_only',
            'statement': 'Winning-square parity is recorded; no strategic ownership or eventual winner follows from parity alone.'})
    return {
        'schema_version': '1.0',
        'provenance': {'method': 'deterministic_post_hoc_analysis', 'game_id': game_id,
                       'revision': revision, 'history_verified_by_replay': True,
                       'agent_reasoning_available': False},
        'coordinates': deepcopy(COORDINATES),
        'position': {'board': [list(row) for row in board], 'player_to_move': player_to_move,
                     'players': deepcopy(players), 'outcome': outcome(board)},
        'move_history': records,
        'confirmed_tactical_facts': facts,
        'last_move_facts': last,
        'supported_allis_rule_applications': [],
        'general_strategic_observations': observations,
        'unknown_or_unproven': [
            {'topic': 'agent_intent', 'reason': 'No search trace or internal decision rationale was recorded. Negamax is not assumed to use Allis rules.'},
            {'topic': 'strategic_result', 'reason': 'No long-term game-theoretic value, optimality ranking, Zugzwang control, nine-rule application, compatibility or coverage proof is computed.'},
            {'topic': 'victor_agent', 'reason': 'The experimental VictorAgent is not a complete solver and supplies no evidence to this payload.'},
        ],
        'analysis_limits': {
            'lookahead': 'Each legal move and all immediate winning replies; no deeper search.',
            'survival': 'Avoids a loss on the next reply only, not a promise of a draw or win.',
            'winning_square_changes': 'Geometric square-set changes, not a complete diff of all groups; terminal patterns are not future legal moves.',
            'formal_allis_rules_implemented': [],
        },
        'knowledge': {'source_title': SOURCE_TITLE, 'source_url': SOURCE_URL,
                      'pagination': 'Thesis page references follow its contents. In this supplied PDF they equal 1-based PDF page numbers. Body pages have no visible printed folios; zero-based indices are separately recorded.',
                      'retrieval': 'Deterministic concept IDs and verified fact presence; retrieval does not assert rule applicability.',
                      'entries': retrieve_knowledge(relevant)},
    }
